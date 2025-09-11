# generate_interactions_for_base64.py
import asyncio
import random
from database.databases import coleccion, user
from datetime import datetime, timedelta

async def generate_interactions_for_base64():
    """Genera interacciones para imágenes base64 existentes"""
    print("🎭 Generando interacciones para imágenes base64...")
    
    # Obtener sólo imágenes con base64 (que tienen data:image en image_url)
    images = await coleccion.find({
        "image_url": {"$regex": "^data:image", "$options": "i"}
    }).to_list(None)
    
    users = await user.find().to_list(None)
    
    if not images:
        print("❌ No hay imágenes base64 en la base de datos")
        print("   Ejecuta primero: python generate_base64_images.py")
        return
    
    if not users:
        print("❌ No hay usuarios")
        return
    
    print(f"📊 Generando interacciones para {len(images)} imágenes base64 y {len(users)} usuarios")
    
    total_interactions = 0
    for user_data in users:
        user_id = user_data.get("user_id")
        if not user_id:
            continue
            
        # Cada usuario interactúa con 10-20 imágenes
        num_interactions = random.randint(10, 20)
        images_to_interact = random.sample(images, min(num_interactions, len(images)))
        
        for image in images_to_interact:
            image_id = image.get("image_id")
            if not image_id:
                continue
            
            # 70% like, 30% view
            action = "like" if random.random() < 0.7 else "view"
            
            # Actualizar imagen
            await coleccion.update_one(
                {"image_id": image_id},
                {
                    "$inc": {f"interactions.{action}s": 1},
                    "$addToSet": {"liked_by": user_id},
                    "$set": {"interactions.last_interaction": datetime.utcnow() - timedelta(days=random.randint(0, 30))}
                }
            )
            
            # Actualizar usuario
            await user.update_one(
                {"user_id": user_id},
                {
                    "$addToSet": {"images": image_id},
                    "$set": {"last_updated": datetime.utcnow()}
                }
            )
            
            total_interactions += 1
            
            if total_interactions % 50 == 0:
                print(f"📈 {total_interactions} interacciones creadas...")
    
    print(f"🎉 {total_interactions} interacciones creadas para imágenes base64")

if __name__ == "__main__":
    asyncio.run(generate_interactions_for_base64())