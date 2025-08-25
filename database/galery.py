from database.databases import coleccion
from models.galery import UpdateImage, Image,ImageResponse
from bson import ObjectId
import datetime

async def get_all_images():
    img_list = []
    cursor = coleccion.find({}, {"features": 0})
    async for document in cursor:
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