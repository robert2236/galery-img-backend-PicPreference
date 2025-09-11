# delete_images_after_25.py
import asyncio
from database.databases import coleccion

async def delete_images_after_25():
    """Elimina las imágenes después de la número 25 basándose en image_id"""
    print("🗑️ Eliminando imágenes después de la número 25...")
    
    try:
        # Obtener todas las imágenes ordenadas por image_id (asumiendo que son secuenciales)
        all_images = await coleccion.find().sort("image_id", 1).to_list(length=None)
        
        if len(all_images) <= 25:
            print(f"ℹ️ Solo hay {len(all_images)} imágenes, no se elimina nada")
            return
        
        # Separar las primeras 25 imágenes del resto
        images_to_keep = all_images[:25]
        images_to_delete = all_images[25:]
        
        print(f"📊 Total de imágenes: {len(all_images)}")
        print(f"✅ Se mantendrán: {len(images_to_keep)} imágenes")
        print(f"🗑️ Se eliminarán: {len(images_to_delete)} imágenes")
        print(f"📋 Image IDs a eliminar: {[img['image_id'] for img in images_to_delete]}")
        
        confirm = input("¿Continuar con la eliminación? (s/n): ")
        
        if confirm.lower() != 's':
            print("❌ Operación cancelada")
            return
        
        # Eliminar las imágenes sobrantes
        ids_to_delete = [img["_id"] for img in images_to_delete]
        result = await coleccion.delete_many({
            "_id": {"$in": ids_to_delete}
        })
        
        print(f"✅ Eliminación completada: {result.deleted_count} imágenes eliminadas")
        print(f"💾 Imágenes restantes: {len(images_to_keep)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(delete_images_after_25())
    