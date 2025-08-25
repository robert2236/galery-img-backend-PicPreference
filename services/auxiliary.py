from database.databases import coleccion, user

async def get_user_viewed_images(user_id: int):
    """Obtener IDs de imágenes que el usuario ya ha visto"""
    viewed_images = await coleccion.find({
        "interactions.user_id": user_id
    }, {"_id": 1}).to_list(None)
    
    return [img["_id"] for img in viewed_images]

async def get_popular_images(limit: int = 10):
    """Imágenes populares para cold start"""
    return await coleccion.find().sort("interactions.likes", -1).limit(limit).to_list(None)