# check_base64_images.py
import asyncio
from database.databases import coleccion

async def check_base64_images():
    """Verifica las imágenes base64 en la base de datos"""
    print("🔍 Verificando imágenes base64...")
    
    # Contar total de imágenes
    total_images = await coleccion.count_documents({})
    print(f"📊 Total de imágenes: {total_images}")
    
    # Contar imágenes con base64
    base64_images = await coleccion.count_documents({
        "image_url": {"$regex": "^data:image", "$options": "i"}
    })
    print(f"📊 Imágenes base64: {base64_images}")
    
    # Contar imágenes con URLs normales
    url_images = await coleccion.count_documents({
        "image_url": {"$regex": "^http", "$options": "i"}
    })
    print(f"📊 Imágenes con URL: {url_images}")
    
    # Mostrar algunas imágenes base64
    if base64_images > 0:
        print("\n🔍 Algunas imágenes base64:")
        base64_examples = await coleccion.find({
            "image_url": {"$regex": "^data:image", "$options": "i"}
        }).limit(3).to_list(None)
        
        for i, img in enumerate(base64_examples):
            image_url = img.get("image_url", "")
            print(f"   {i+1}. ID: {img.get('image_id')}")
            print(f"      Título: {img.get('title', 'Sin título')}")
            print(f"      Tipo: {image_url[:50]}...")  # Mostrar solo el inicio
            print(f"      Tamaño: {len(image_url)//1000}KB")
    
    # Recomendaciones
    if base64_images < 20:
        print(f"\n❌ Necesitas más imágenes base64 ({base64_images}/20 mínimo)")
        print("   Ejecuta: python generate_base64_images.py")
    else:
        print(f"\n✅ Tienes suficientes imágenes base64 ({base64_images})")

if __name__ == "__main__":
    asyncio.run(check_base64_images())