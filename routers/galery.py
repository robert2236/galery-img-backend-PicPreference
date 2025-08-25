from fastapi import APIRouter, HTTPException, Depends,status
from database.galery import (get_all_images,create_image, get_one_image,delete_image)
from models.galery import Image,UpdateImage,ImageResponse,InteractionUpdate
from models.users import User
from database.databases import user,coleccion
from utils.auth.oauth import get_current_user
import base64
from datetime import datetime
import random
from datetime import datetime
from fastapi import HTTPException
from utils.feature_extractor import FeatureExtractor  # Asegúrate de importar tu extractor
import numpy as np
from utils.auth.oauth import extract_user_id


feature_extractor = FeatureExtractor()

galery= APIRouter()

@galery.get('/api/images')
async def get_images():
    response = await get_all_images()
    return response



@galery.post('/api/create_image', response_model=Image)
async def save_image(img: Image):
    # 1. Verificar si el usuario existe
    user_exists = await user.find_one({"user_id": img.user_id})
    user_data = await user.find_one({"user_id": (img.user_id)})
    if not user_exists:
        raise HTTPException(status_code=404, detail="Usuario inválido")
    
    # 2. Verificar si la imagen ya existe
    if img.image_url:
        imgFound = await get_one_image(img.image_url)
        if imgFound:
            raise HTTPException(409, "Image already exists")
    
    # 3. Extraer características visuales (FEATURES)
    try:
        # Ejemplo: img.image_url contiene la ruta o URL de la imagen
        features = feature_extractor.extract(img.image_url)
        features_array = np.array(features)
        features_list = features_array.tolist()
    except Exception as e:
        raise HTTPException(500, f"Error al extraer características: {str(e)}")
    
    # 4. Preparar los datos de la imagen para guardar
    img_data = img.dict(by_alias=True, exclude_none=True)
    img_data.update({
        "upload_date": datetime.utcnow(),  # Fecha de subida
        "username": user_data["username"],
        "image_id": random.randint(1000, 9999),  # ID aleatorio (¿o usar UUID?)
        "features": features_list,  
        "title": img.title
    })
    
    # 5. Guardar en la base de datos
    response = await create_image(img_data)
    
    if response:
        return response
    raise HTTPException(400, "Something went wrong")

@galery.put('/api/images/{id}', response_model=Image)
async def put_img(id: str, data: UpdateImage):
    data.id = id
    response = UpdateImage(**data.dict())
    if response:
        return response
    raise HTTPException(404, f"There is no task with the id {id}")

@galery.delete('/api/images/{id}')
async def remove_image(id: str):
    response = await delete_image(id)
    if response:
        return "Successfully image delete from the database"
    raise HTTPException(404, f"There is no task with the id {id}")

@galery.post("/api/upload-base64/{user_id}")
async def upload_base64_image(
    image: Image,
    user_id: int,
):
    try:
        header, encoded = image.image_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        if len(image_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="La imagen excede el tamaño máximo de 5MB")
        
        image_data = {
            "user_id": user_id,  # <-- ahora sí existe
            "image_url": image.image_url,
            "category": image.category,
            "upload_date": datetime.utcnow()
        }
        result = await coleccion.insert_one(image_data)
        await user.update_one(
            {"user_id": user_id},
            {"$push": {"images": result.inserted_id}}
        )
        return {"id": str(result.inserted_id), "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@galery.put("/api/images/{image_id}/interactions", response_model=Image)
async def update_interactions(
    image_id: int,
    interaction: InteractionUpdate,
    user_data: dict = Depends(extract_user_id)
):
    user_id = user_data["user_id"]
    print(user_id)
    usuario = await user.find_one({"username": user_id})
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # 1. Validar acción permitida (corregí el typo "likes")
    valid_actions = ["likes", "downloads", "views"]
    if interaction.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Acción no válida. Use: {', '.join(valid_actions)}"
        )

    # 2. Verificar si la imagen existe primero
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    # 3. Preparar operación de actualización para la imagen
    update_operation = {
        "$inc": {f"interactions.{interaction.action}": interaction.increment},
        "$set": {"interactions.last_interaction": datetime.utcnow()}
    }
    
    # 4. Si es like, agregar al array liked_by
    if interaction.action == "likes":
        if "liked_by" in image and user_id in image["liked_by"]:
            raise HTTPException(
                status_code=400,
                detail="Ya has dado like a esta imagen"
            )
        update_operation["$addToSet"] = {"liked_by": user_id}

    # 5. Actualizar la imagen
    update_result = await coleccion.update_one(
        {"image_id": image_id},
        update_operation
    )

    if update_result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    # 6. Actualizar el usuario - VERSIÓN CORREGIDA
    update_result = await user.update_one(
    {"username": user_id},
    {
        "$addToSet": {"images": image_id},
        "$set": {"last_updated": datetime.utcnow()}
    }
    )

    
    # Verificar resultado
    if update_result.modified_count == 1:
        print("Imagen agregada exitosamente")
    else:
        print("No se realizaron cambios")
        
    user_data = await user.find_one({"user_id": user_id})
    print("Lista actual de imágenes:", user_data)
            

    # 7. Devolver la imagen actualizada
    updated_image = await coleccion.find_one({"image_id": image_id})
    return Image(**updated_image)