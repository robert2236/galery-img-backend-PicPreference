from database.databases import coleccion
from models.galery import UpdateImage, Image,ImageResponse
from bson import ObjectId
import datetime

async def get_all_images():
    """Obtiene todas las imágenes y convierte liked_by a strings"""
    cursor = coleccion.find()
    img_list = []
    
    async for document in cursor:
        # Convertir todos los liked_by a strings
        if "liked_by" in document and document["liked_by"]:
            document["liked_by"] = [str(user_id) for user_id in document["liked_by"]]
        
        img_list.append(ImageResponse(**document))
    
    return img_list

async def get_one_image(find):
    img = await coleccion.find_one({"image_url": find})
    return img

async def create_image(image):
    new_image = await coleccion.insert_one(image)
    created_image = await coleccion.find_one({"_id": new_image.inserted_id})
    return created_image

async def update_image(id: str, data: UpdateImage):
    image = {k: v for k, v in data.dict().items() if v is not None}
    await coleccion.update_one({"_id": ObjectId(id)}, {"$set": image})
    document = await coleccion.find_one({"_id": ObjectId(id)})
    return document

async def delete_image(id):
    await coleccion.delete_one({"_id": ObjectId(id)})
    return True