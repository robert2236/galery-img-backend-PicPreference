from database.databases import user, coleccion
import asyncio

async def normalize_liked_by():
    """Convierte todos los liked_by strings a números y reemplaza 'test' por su user_id real"""
    print("🔄 Normalizando liked_by en la base de datos...")
    
    # Primero, encontrar el user_id correspondiente al username "test"
    test_user = await user.find_one({"username": "test"})
    if not test_user:
        print("❌ No se encontró usuario 'test' en la colección de usuarios")
        return
    
    test_user_id = test_user.get("user_id")
    print(f"✅ Usuario 'test' encontrado: user_id = {test_user_id}")
    
    # Actualizar todos los documentos donde liked_by contiene "test"
    cursor = coleccion.find({"liked_by": "test"})
    updated_count = 0
    
    async for image in cursor:
        # Reemplazar "test" con el user_id numérico
        new_liked_by = []
        for item in image.get("liked_by", []):
            if item == "test":
                new_liked_by.append(test_user_id)
                print(f"   🔄 Reemplazando 'test' por {test_user_id} en imagen {image.get('image_id')}")
            else:
                new_liked_by.append(item)
        
        await coleccion.update_one(
            {"image_id": image["image_id"]},
            {"$set": {"liked_by": new_liked_by}}
        )
        updated_count += 1
    
    print(f"✅ {updated_count} imágenes actualizadas (reemplazado 'test' por {test_user_id})")
    
    # Ahora convertir cualquier string numérico a número
    cursor = coleccion.find({"liked_by": {"$exists": True}})
    converted_count = 0
    
    async for image in cursor:
        needs_update = False
        new_liked_by = []
        
        for item in image.get("liked_by", []):
            if isinstance(item, str) and item.isdigit():
                # Convertir string numérico a int
                new_liked_by.append(int(item))
                needs_update = True
                print(f"   🔄 Convirtiendo '{item}' a {int(item)} en imagen {image.get('image_id')}")
            else:
                new_liked_by.append(item)
        
        if needs_update:
            await coleccion.update_one(
                {"image_id": image["image_id"]},
                {"$set": {"liked_by": new_liked_by}}
            )
            converted_count += 1
    
    print(f"✅ {converted_count} imágenes con strings numéricos convertidos")
    print("🎉 Normalización completada!")

# Ejecutar la función async
async def main():
    await normalize_liked_by()

if __name__ == "__main__":
    # Ejecutar el event loop de asyncio
    asyncio.run(main())