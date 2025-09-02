# create_test_user.py
import asyncio
from database.databases import user
from datetime import datetime

async def create_test_user():
    """Crea un usuario de prueba para las tests"""
    test_user = {
        "user_id": 1,
        "username": "testuser",
        "password": "testpassword",  # Deberías hashear esto en producción
        "email": "test@example.com",
        "created_at": datetime.utcnow(),
        "images": [],  # Lista de imágenes interactuadas
        "last_updated": datetime.utcnow()
    }
    
    # Verificar si el usuario ya existe
    existing_user = await user.find_one({"username": "testuser"})
    if existing_user:
        print("✅ Usuario de prueba ya existe")
        return
    
    result = await user.insert_one(test_user)
    print(f"✅ Usuario de prueba creado: {result.inserted_id}")

if __name__ == "__main__":
    asyncio.run(create_test_user())