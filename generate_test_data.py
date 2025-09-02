# generate_test_data.py - Versión mejorada
import asyncio
import random
import numpy as np
from datetime import datetime, timedelta
from database.databases import coleccion, user
from bson import ObjectId

async def generate_realistic_interactions():
    """Genera interacciones REALISTAS y COHERENTES para testing"""
    print("🔄 Generando interacciones realistas...")
    
    try:
        # Obtener todas las imágenes
        images = await coleccion.find().to_list(None)
        users = await user.find().to_list(None)
        
        if not images or not users:
            print("⚠️ No hay imágenes o usuarios para generar interacciones")
            return

        # Crear clusters de usuarios que interactúan de manera similar
        user_clusters = {}
        for user_data in users:
            user_id = user_data.get("user_id")
            if user_id:
                # Asignar a un cluster aleatorio
                cluster = random.randint(1, 3)
                user_clusters[user_id] = cluster
                print(f"👤 Usuario {user_id} asignado al cluster {cluster}")

        # Crear grupos de imágenes similares (basado en categorías simuladas)
        image_groups = {}
        for i, image in enumerate(images):
            image_id = image.get("image_id")
            # Agrupar imágenes en 4 categorías basadas en su posición
            group = (i % 4) + 1
            image_groups[image_id] = group
            print(f"🖼️ Imagen {image_id} asignada al grupo {group}")

        # Generar interacciones REALISTAS
        for user_data in users:
            user_id = user_data.get("user_id")
            if not user_id:
                continue
                
            user_cluster = user_clusters.get(user_id, 1)
            interacted_images = []
            
            # Usuario interactúa con 10-20 imágenes
            num_interactions = random.randint(10, 20)
            
            for _ in range(num_interactions):
                # Elegir imágenes que coincidan con el cluster del usuario
                suitable_images = [
                    img for img in images 
                    if img.get("image_id") and 
                    image_groups.get(img.get("image_id"), 0) == user_cluster
                ]
                
                if suitable_images:
                    image = random.choice(suitable_images)
                    image_id = image.get("image_id")
                    
                    if image_id not in interacted_images:
                        # Registrar like (70% probabilidad) o view (30%)
                        action = "like" if random.random() < 0.7 else "view"
                        
                        # Actualizar la imagen
                        await coleccion.update_one(
                            {"image_id": image_id},
                            {
                                "$inc": {f"interactions.{action}s": 1},
                                "$addToSet": {"liked_by": user_id},
                                "$set": {"interactions.last_interaction": datetime.utcnow()}
                            }
                        )
                        
                        # Actualizar el usuario
                        await user.update_one(
                            {"user_id": user_id},
                            {
                                "$addToSet": {"images": image_id},
                                "$set": {"last_updated": datetime.utcnow()}
                            }
                        )
                        
                        interacted_images.append(image_id)
                        print(f"✅ Usuario {user_id} {action} imagen {image_id}")
            
            print(f"📊 Usuario {user_id} interactuó con {len(interacted_images)} imágenes")

        print(f"✅ Interacciones realistas generadas para {len(users)} usuarios")

    except Exception as e:
        print(f"❌ Error generando interacciones: {e}")

async def main():
    """Función principal MEJORADA"""
    print("🎭 Generador de Datos de Prueba REALISTAS")
    print("=" * 50)
    
    await generate_realistic_interactions()
    
    print("\n🎉 ¡Datos de prueba REALISTAS generados exitosamente!")

if __name__ == "__main__":
    asyncio.run(main())