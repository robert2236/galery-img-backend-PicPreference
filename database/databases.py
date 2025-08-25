import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URL)
db = client["galery"]
coleccion = db["images"]
user = db["users"]
categories = db["categories"]

async def test_connection():
    try:
        await client.server_info()  # Prueba la conexión
        print("✅ Conexión exitosa a MongoDB")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")