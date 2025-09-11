# create_test_user.py
import asyncio
from database.databases import user
from datetime import datetime
import random
import string

async def create_test_users(num_users=15):
    """Crea múltiples usuarios de prueba para las tests y genera archivo TXT"""
    
    # Lista para almacenar las credenciales
    credentials = []
    
    for i in range(1, num_users + 2):
        # Generar datos únicos para cada usuario
        username = f"testuser{i}"
        email = f"test{i}@example.com"
        
        # Generar password aleatorio
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        
        test_user = {
            "user_id": i,
            "username": username,
            "password": password,  # Deberías hashear esto en producción
            "email": email,
            "created_at": datetime.utcnow(),
            "images": [],  # Lista de imágenes interactuadas
            "last_updated": datetime.utcnow()
        }
        
        # Verificar si el usuario ya existe
        existing_user = await user.find_one({"username": username})
        if existing_user:
            print(f"✅ Usuario {username} ya existe")
            # Obtener la contraseña existente para el archivo
            existing_password = existing_user.get('password', 'N/A')
            credentials.append(f"{username}:{existing_password}")
            continue
        
        result = await user.insert_one(test_user)
        print(f"✅ Usuario {username} creado: {result.inserted_id}")
        credentials.append(f"{username}:{password}")
    
    # Generar archivo TXT con las credenciales
    generate_credentials_file(credentials)
    
    return credentials

def generate_credentials_file(credentials, filename="user_credentials.txt"):
    """Genera un archivo TXT con las credenciales de los usuarios"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("CREDENCIALES DE USUARIOS DE PRUEBA\n")
            file.write("=" * 40 + "\n")
            file.write("Formato: username:password\n")
            file.write("=" * 40 + "\n\n")
            
            for cred in credentials:
                file.write(cred + "\n")
            
            file.write("\n" + "=" * 40 + "\n")
            file.write(f"Total de usuarios: {len(credentials)}\n")
            file.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📄 Archivo '{filename}' generado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error al generar archivo TXT: {e}")

async def main():
    """Función principal para crear 15 usuarios y generar archivo"""
    print("🚀 Creando 15 usuarios de prueba...")
    print("📝 Generando archivo de credenciales...")
    
    credentials = await create_test_users(15)
    
    print("\n📋 Resumen de credenciales:")
    print("-" * 30)
    for cred in credentials:
        print(cred)
    
    print(f"\n🎉 Proceso completado! Se crearon/verificaron {len(credentials)} usuarios")
    print("💾 Las credenciales se guardaron en 'user_credentials.txt'")

if __name__ == "__main__":
    asyncio.run(main())