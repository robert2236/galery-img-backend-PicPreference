# generate_test_data.py - Versión mejorada para 15 usuarios
import asyncio
import random
from datetime import datetime
from database.databases import coleccion, user

async def generate_realistic_interactions():
    """Genera interacciones REALISTAS para los 15 usuarios de prueba"""
    print("🔄 Generando interacciones realistas para 15 usuarios...")
    
    try:
        # Obtener todas las imágenes
        images = await coleccion.find().to_list(None)
        users = await user.find().to_list(None)
        
        if not images:
            print("⚠️ No hay imágenes para generar interacciones")
            return
        
        if not users:
            print("⚠️ No hay usuarios para generar interacciones")
            return

        # Filtrar solo los 15 usuarios de prueba (testuser1 a testuser15)
        test_users = [u for u in users if u.get("username", "").startswith("testuser")]
        print(f"👥 Encontrados {len(test_users)} usuarios de prueba")
        
        # Mostrar los usuarios encontrados
        for u in test_users:
            print(f"   - {u.get('username')} (ID: {u.get('user_id')})")

        # Generar interacciones REALISTAS para cada usuario de prueba
        for user_data in test_users:
            user_id = user_data.get("user_id")
            username = user_data.get("username", "")
            
            if not user_id:
                continue
                
            interacted_images = []
            
            # Cada usuario interactúa con 3-8 imágenes (número realista para 15 usuarios)
            num_interactions = random.randint(3, 8)
            print(f"\n🎯 Usuario {username} interactuará con {num_interactions} imágenes")
            
            for i in range(num_interactions):
                # Elegir una imagen aleatoria que no haya sido interactuada
                available_images = [img for img in images if img.get("image_id") not in interacted_images]
                
                if not available_images:
                    print(f"   ⚠️ No hay más imágenes disponibles para {username}")
                    break
                    
                image = random.choice(available_images)
                image_id = image.get("image_id")
                
                if not image_id:
                    continue
                
                # Decidir la acción: 60% like, 40% view (más realista)
                action = "like" if random.random() < 0.6 else "view"
                
                # Actualizar la imagen
                update_data = {
                    "$inc": {f"interactions.{action}s": 1},
                    "$set": {"interactions.last_interaction": datetime.utcnow()}
                }
                
                # Solo agregar a liked_by si es un like
                if action == "like":
                    update_data["$addToSet"] = {"liked_by": user_id}
                
                await coleccion.update_one(
                    {"image_id": image_id},
                    update_data
                )
                
                # Actualizar el usuario (solo registrar la imagen interactuada)
                await user.update_one(
                    {"user_id": user_id},
                    {
                        "$addToSet": {"images": image_id},
                        "$set": {"last_updated": datetime.utcnow()}
                    }
                )
                
                interacted_images.append(image_id)
                print(f"   ✅ {action.capitalize()} en imagen {image_id}")
            
            print(f"   📊 {username} interactuó con {len(interacted_images)} imágenes")

        # Mostrar estadísticas finales
        print(f"\n📈 ESTADÍSTICAS FINALES:")
        print(f"   Total usuarios de prueba: {len(test_users)}")
        print(f"   Total imágenes en sistema: {len(images)}")
        
        # Contar interacciones totales
        total_images = await coleccion.count_documents({})
        images_with_interactions = await coleccion.count_documents({
            "$or": [
                {"interactions.likes": {"$gt": 0}},
                {"interactions.views": {"$gt": 0}}
            ]
        })
        
        print(f"   Imágenes con interacciones: {images_with_interactions}/{total_images}")

    except Exception as e:
        print(f"❌ Error generando interacciones: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Función principal"""
    print("🎭 Generador de Datos de Prueba - 15 Usuarios")
    print("=" * 50)
    
    await generate_realistic_interactions()
    
    print("\n🎉 ¡Interacciones realistas generadas exitosamente!")

if __name__ == "__main__":
    asyncio.run(main())