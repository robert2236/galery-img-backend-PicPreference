import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Carga las variables de entorno del archivo .env
load_dotenv()

# Obtiene la URL con fallback a localhost si no existe la variable
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "galery_dom")

# serverSelectionTimeoutMS corto: en la nube falla rapido si MongoDB no responde
client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]

coleccion = db["images"]
user = db["users"]
categories = db["categories"]

async def test_connection():
    try:
        await client.server_info()  # Prueba la conexión
        print("[OK] Conexión exitosa a MongoDB")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
