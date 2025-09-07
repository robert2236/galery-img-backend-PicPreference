from fastapi import APIRouter, HTTPException, Depends,status
from database.galery import (get_all_images,create_image, get_one_image,delete_image)
from models.galery import Image,UpdateImage,ImageResponse,InteractionUpdate, CommentCreate
from models.users import User
from database.databases import user,coleccion
from utils.auth.oauth import get_current_user
import base64
from datetime import datetime
import random
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from utils.feature_extractor import FeatureExtractor  # Asegúrate de importar tu extractor
import numpy as np
from utils.auth.oauth import extract_user_id
import asyncio
from typing import Optional, List
from models.Pagination import PaginationParams


feature_extractor = FeatureExtractor()

galery= APIRouter()

image_cache = None
cache_timestamp = None
CACHE_DURATION = 10  # 5 minutos en segundos



@galery.get('/api/images')
async def get_images():
    global image_cache, cache_timestamp
    
    # Verificar cache
    if (image_cache is not None and 
        cache_timestamp is not None and 
        (datetime.now() - cache_timestamp).total_seconds() < CACHE_DURATION):
        
        print("✅ Sirviendo desde cache")
        return JSONResponse(content=image_cache)
    
    # Obtener datos frescos
    print("🔄 Obteniendo datos frescos")
    images = await get_all_images()
    
    # Convertir a formato serializable manualmente
    serializable_response = []
    for image in images:
        image_dict = image.dict()
        
        # Convertir datetime en comentarios
        if 'comments' in image_dict and image_dict['comments']:
            for comment in image_dict['comments']:
                if 'created_at' in comment and isinstance(comment['created_at'], datetime):
                    comment['created_at'] = comment['created_at'].isoformat()
        
        serializable_response.append(image_dict)
    
    # Actualizar cache
    image_cache = serializable_response
    cache_timestamp = datetime.now()
    
    return JSONResponse(content=serializable_response)

@galery.get('/test')
async def test():
    return {"message": "✅ Funciona"}

from fastapi import HTTPException, Depends, Query
from typing import Optional
from datetime import datetime

# Clase para manejar los parámetros de paginación
class PaginationParams:
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Número de elementos a saltar"),
        limit: int = Query(20, ge=1, le=100, description="Límite de resultados por página"),
        page: int = Query(1, ge=1, description="Número de página (alternativo a skip)")
    ):
        self.limit = limit
        # Si se proporciona page, calcular skip automáticamente
        if page > 1:
            self.skip = (page - 1) * limit
        else:
            self.skip = skip
        self.page = page if page > 1 else (skip // limit) + 1

@galery.get('/api/images/search')
async def search_images(
    q: Optional[str] = None,
    tags: Optional[str] = None,
    min_likes: Optional[int] = None,
    min_views: Optional[int] = None,
    pagination: PaginationParams = Depends()
):
    try:
        # Obtener todas las imágenes usando la función existente
        all_images = await get_all_images()
        
        # Convertir a lista de diccionarios
        images_dicts = []
        for image in all_images:
            # Convertir el modelo Pydantic a diccionario
            image_dict = image.dict()
            
            # Asegurar que los campos existan y no sean None
            image_dict.setdefault('interactions', {})
            image_dict['interactions'].setdefault('likes', 0)
            image_dict['interactions'].setdefault('views', 0)
            image_dict.setdefault('tags', [])
            image_dict.setdefault('title', '')  # Asegurar que title no sea None
            image_dict.setdefault('category', '')
            image_dict.setdefault('username', '')
            
            # Convertir valores None a string vacío
            if image_dict['title'] is None:
                image_dict['title'] = ''
            if image_dict['username'] is None:
                image_dict['username'] = ''
            
            # Convertir datetime en comentarios
            if 'comments' in image_dict and image_dict['comments']:
                for comment in image_dict['comments']:
                    if 'created_at' in comment and isinstance(comment['created_at'], datetime):
                        comment['created_at'] = comment['created_at'].isoformat()
            
            images_dicts.append(image_dict)
        
        # Aplicar filtros paso a paso
        filtered_images = images_dicts
        
        # 1. Filtrar por texto de búsqueda (q)
        if q:
            q_lower = q.lower().strip()
            temp_filtered = []
            
            for img in filtered_images:
                # Asegurar que los campos no sean None antes de usar .lower()
                title = (img.get('title') or '').lower()
                category = (img.get('category') or '').lower()
                username = (img.get('username') or '').lower()
                image_tags = [tag.lower() for tag in (img.get('tags') or []) if tag is not None]
                
                # Buscar coincidencias
                title_match = q_lower in title
                category_match = q_lower in category
                username_match = q_lower in username
                tags_match = any(q_lower in tag for tag in image_tags)
                
                if title_match or category_match or username_match or tags_match:
                    temp_filtered.append(img)
            
            filtered_images = temp_filtered
        
        # 2. Filtrar por tags específicos
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(',') if tag.strip()]
            temp_filtered = []
            for img in filtered_images:
                image_tags = [tag.lower() for tag in (img.get('tags') or []) if tag is not None]
                if any(tag in image_tags for tag in tag_list):
                    temp_filtered.append(img)
            filtered_images = temp_filtered
        
        # 3. Filtrar por mínimo de likes
        if min_likes is not None:
            filtered_images = [
                img for img in filtered_images 
                if img['interactions'].get('likes', 0) >= min_likes
            ]
        
        # 4. Filtrar por mínimo de views
        if min_views is not None:
            filtered_images = [
                img for img in filtered_images 
                if img['interactions'].get('views', 0) >= min_views
            ]
        
        # Aplicar paginación
        total_count = len(filtered_images)
        
        # Calcular información de paginación
        total_pages = (total_count + pagination.limit - 1) // pagination.limit if pagination.limit > 0 else 1
        has_next = (pagination.skip + pagination.limit) < total_count
        has_prev = pagination.skip > 0
        
        paginated_images = filtered_images[pagination.skip:pagination.skip + pagination.limit]
        
        return {
            "results": paginated_images,
            "total": total_count,
            "limit": pagination.limit,
            "skip": pagination.skip,
            "page": pagination.page,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev,
            "next_page": pagination.page + 1 if has_next else None,
            "prev_page": pagination.page - 1 if has_prev else None,
            "search_query": q,
            "filters_applied": {
                "text_search": q is not None,
                "tags_filter": tags is not None,
                "min_likes": min_likes,
                "min_views": min_views
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")
    
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
        """ if "liked_by" in image and user_id in image["liked_by"]:
            raise HTTPException(
                status_code=400,
                detail="Ya has dado like a esta imagen"
            ) """
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


@galery.put("/api/images/{image_id}/comments", response_model=Image)
async def add_comment(
    image_id: int,
    comment_data: CommentCreate,
    user_data: dict = Depends(extract_user_id)
):
    user_id = user_data["user_id"]
    
    # Verificar que el usuario existe
    usuario = await user.find_one({"username": user_id})
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar que la imagen existe
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Generar ID único para el comentario
    comment_id = datetime.utcnow().timestamp()
    
    # Estructura del comentario
    new_comment = {
        "comment_id": int(comment_id),
        "user_id": user_id,
        "comment": comment_data.comment,
        "created_at": datetime.utcnow(),
        "parent_comment_id": comment_data.parent_comment_id,
        "likes": 0,
        "replies": []
    }
    
    # Si es un comentario padre, agregarlo al array principal
    if comment_data.parent_comment_id is None:
        update_operation = {
            "$push": {"comments": new_comment},
            "$inc": {"interactions.comments": 1},
            "$set": {"interactions.last_interaction": datetime.utcnow()}
        }
    else:
        # Si es una respuesta, agregarlo al comentario padre
        update_operation = {
            "$push": {"comments.$[comment].replies": new_comment},
            "$inc": {"interactions.comments": 1},
            "$set": {"interactions.last_interaction": datetime.utcnow()}
        }
    
    # Actualizar la imagen
    if comment_data.parent_comment_id is None:
        update_result = await coleccion.update_one(
            {"image_id": image_id},
            update_operation
        )
    else:
        update_result = await coleccion.update_one(
            {"image_id": image_id, "comments.comment_id": comment_data.parent_comment_id},
            update_operation,
            array_filters=[{"comment.comment_id": comment_data.parent_comment_id}]
        )
    
    if update_result.modified_count == 0:
        if comment_data.parent_comment_id:
            raise HTTPException(
                status_code=404, 
                detail="Comentario padre no encontrado"
            )
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Actualizar el usuario
    await user.update_one(
        {"username": user_id},
        {
            "$addToSet": {"commented_images": image_id},
            "$set": {"last_updated": datetime.utcnow()}
        }
    )
    
    # Devolver la imagen actualizada
    updated_image = await coleccion.find_one({"image_id": image_id})
    return updated_image

@galery.delete("/api/images/{image_id}/likes", response_model=Image)
async def remove_like(
    image_id: int,
    user_data: dict = Depends(extract_user_id)
):
    user_id = user_data["user_id"]
    
    # Verificar si el usuario existe
    usuario = await user.find_one({"username": user_id})
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar si la imagen existe
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Verificar si el usuario tiene like en esta imagen
    liked_by = image.get("liked_by", [])
    if user_id not in liked_by:
        raise HTTPException(
            status_code=400,
            detail="No tienes like en esta imagen"
        )
    
    # Remover el like
    update_operation = {
        "$pull": {"liked_by": user_id},
        "$inc": {"interactions.likes": -1},
        "$set": {"interactions.last_interaction": datetime.utcnow()}
    }
    
    # Actualizar la imagen
    update_result = await coleccion.update_one(
        {"image_id": image_id},
        update_operation
    )
    
    if update_result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    
    # Devolver la imagen actualizada
    updated_image = await coleccion.find_one({"image_id": image_id})
    return Image(**updated_image)



