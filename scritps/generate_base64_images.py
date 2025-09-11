# generate_base64_images.py
import asyncio
import random
import base64
import requests
from database.databases import coleccion
from datetime import datetime

async def generate_base64_test_images():
    """Genera imágenes de prueba con base64 real"""
    print("🖼️ Generando imágenes con base64...")
    
    categories = ["naturaleza", "animales", "ciudad", "arte", "deportes", "comida", "tecnologia", "viajes"]
    
    for i in range(50):  # Generar 50 imágenes (base64 es más pesado)
        try:
            # 1. Descargar una imagen real de Picsum
            response = requests.get(f"https://picsum.photos/400/300?random={i}", timeout=10)
            if response.status_code != 200:
                print(f"❌ Error descargando imagen {i}")
                continue
            
            # 2. Convertir a base64
            image_bytes = response.content
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            base64_url = f"data:image/jpeg;base64,{base64_image}"
            
            # 3. Crear datos de la imagen
            image_data = {
                "image_id": random.randint(10000, 99999),
                "title": f"Imagen prueba {i+1}",
                "image_url": base64_url,  # ← ESTA ES LA DIFERENCIA
                "category": random.choice(categories),
                "user_id": random.randint(1, 10),
                "username": f"user{random.randint(1, 10)}",
                "upload_date": datetime.utcnow(),
                "features": [random.random() for _ in range(2048)],  # Características ficticias
                "interactions": {
                    "likes": random.randint(0, 30),
                    "views": random.randint(5, 100),
                    "downloads": random.randint(0, 10),
                    "last_interaction": datetime.utcnow()
                },
                "liked_by": [random.randint(1, 10) for _ in range(random.randint(0, 5))]
            }
            
            # 4. Insertar en la base de datos
            await coleccion.insert_one(image_data)
            print(f"✅ Imagen {i+1} creada: {image_data['image_id']} (tamaño base64: {len(base64_url)//1000}KB)")
            
            # Pequeña pausa para no saturar
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error creando imagen {i}: {e}")
    
    print("🎉 Imágenes base64 creadas")

if __name__ == "__main__":
    asyncio.run(generate_base64_test_images())