# test_recommendations.py
import requests
import json
import time
from utils.auth.oauth import extract_user_id

BASE_URL = "http://localhost:8000"  # Ajusta según tu puerto

# Credenciales de prueba
TEST_CREDENTIALS = {
    "username": "test",
    "password": "test"
}

def get_auth_token():
    """Obtiene token JWT usando el endpoint de login"""
    try:
        # Usar FormData para simular OAuth2PasswordRequestForm
        login_data = {
            "username": TEST_CREDENTIALS["username"],
            "password": TEST_CREDENTIALS["password"]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token")
        else:
            print(f"❌ Error en login: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error obteniendo token: {e}")
        return None

def get_auth_headers():
    """Obtiene headers de autenticación con JWT"""
    token = get_auth_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    else:
        print("⚠️ No se pudo obtener token de autenticación")
        return {}

def test_recommendations():
    """Script para probar el sistema de recomendación"""
    print("🚀 Iniciando pruebas del sistema de recomendación...")
    
    # Obtener token primero
    auth_headers = get_auth_headers()
    if not auth_headers:
        print("❌ No se puede continuar sin autenticación")
        return
    
    try:
        # 1. Verificar que la API está funcionando
        print("\n1. 🔍 Verificando conexión con la API...")
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("   ✅ API conectada correctamente")
            health_data = response.json()
            print(f"   📊 Database: {health_data.get('database', 'unknown')}")
            print(f"   📊 Visual Recommender: {health_data.get('visual_recommender', 'unknown')}")
        else:
            print(f"   ❌ Error conectando con la API: {response.status_code}")
            return

        # 2. Obtener el user_id del usuario de prueba
        print("\n2. 👤 Obteniendo información del usuario de prueba...")
        # Primero obtener el token para luego obtener información del usuario
        user_id = None
        user_info_response = requests.get(f"{BASE_URL}/api/profile", headers=auth_headers)
        if user_info_response.status_code == 200:
            user_info = user_info_response.json()
            user_id = user_info.get('user_id')
            print(f"   ✅ Usuario: {user_info.get('username', 'test')} (ID: {user_id})")
        else:
            print(f"   ⚠️ No se pudo obtener información del usuario, usando ID por defecto: 1")
            user_id = 1  # ID por defecto

        # 3. Obtener todas las imágenes
        print("\n3. 📸 Obteniendo imágenes de la galería...")
        response = requests.get(f"{BASE_URL}/api/images", headers=auth_headers)
        if response.status_code == 200:
            images = response.json()
            print(f"   ✅ Encontradas {len(images)} imágenes")
            
            # Mostrar primeras 3 imágenes
            for i, img in enumerate(images[:3]):
                image_id = img.get('image_id', img.get('_id', 'N/A'))
                print(f"      {i+1}. {img.get('title', 'Sin título')} (ID: {image_id})")
        else:
            print(f"   ❌ Error obteniendo imágenes: {response.status_code}")
            print(f"   Response: {response.text}")
            return

        if not images:
            print("   ⚠️ No hay imágenes para probar")
            return

        # 4. Probar recomendaciones visuales 
        print("\n4. 🔍 Probando recomendaciones visuales...")
        for i, image in enumerate(images[:3]):
            # Obtener el ID correcto (image_id o _id)
            image_id = image.get('image_id', image.get('_id', ''))
            image_title = image.get('title', 'Sin título')
            print(f"   📷 Imagen {i+1}: {image_title} (ID: {image_id})")
            
            # Solo probar si el image_id existe
            if image_id:
                response = requests.get(f"{BASE_URL}/recommend/similar/{image_id}?limit=3", headers=auth_headers)
                if response.status_code == 200:
                    similar = response.json()
                    similar_count = len(similar.get('similar_images', []))
                    print(f"   ✅ Similares encontrados: {similar_count}")
                    
                    if similar_count > 0:
                        for j, sim in enumerate(similar.get('similar_images', [])[:2]):
                            print(f"      {j+1}. {sim.get('title', 'Sin título')} (likes: {sim.get('likes', 0)})")
                    else:
                        print("      ⚠️ No se encontraron imágenes similares")
                else:
                    print(f"   ❌ Error en recomendaciones visuales: {response.status_code}")
                    print(f"   Response: {response.text}")
            else:
                print(f"   ⚠️ Saltando - ID de imagen no disponible")
            print()

        # 5. Probar recomendaciones populares
        print("5. 🏆 Probando recomendaciones populares...")
        response = requests.get(f"{BASE_URL}/recommend/popular?limit=3", headers=auth_headers)
        if response.status_code == 200:
            popular = response.json()
            popular_count = len(popular.get('recommendations', []))
            print(f"   ✅ Imágenes populares encontradas: {popular_count}")
            
            if popular_count > 0:
                for i, pop in enumerate(popular.get('recommendations', [])[:3]):
                    print(f"      {i+1}. {pop.get('title', 'Sin título')} (likes: {pop.get('likes', 0)})")
            else:
                print("      ⚠️ No se encontraron imágenes populares")
        else:
            print(f"   ❌ Error en recomendaciones populares: {response.status_code}")

        # 6. Probar optimización de pesos
        print("\n6. ⚙️ Probando optimización de pesos...")
        response = requests.get(f"{BASE_URL}/recommend/optimize-weights", headers=auth_headers)
        if response.status_code == 200:
            weights = response.json()
            print("   ✅ Pesos optimizados obtenidos:")
            total = 0
            for key, value in weights.items():
                if isinstance(value, (int, float)):
                    print(f"      {key}: {value:.3f}")
                    total += value
                else:
                    print(f"      {key}: {value}")
            print(f"      Total: {total:.3f}")
        else:
            print(f"   ❌ Error optimizando pesos: {response.status_code}")
            print(f"   Response: {response.text}")

  

        # 7. Probar recomendaciones para usuario (user_id dinámico)
        print(f"\n8. 👤 Probando recomendaciones para usuario (ID: {user_id})...")
        response = requests.get(f"{BASE_URL}/recommend/user/{user_id}?limit=20", headers=auth_headers)
        if response.status_code == 200:
            user_recs = response.json()
            rec_count = len(user_recs.get('recommendations', []))
            print(f"   ✅ Recomendaciones para usuario {user_id}: {rec_count}")
            
            if rec_count > 0:
                for i, rec in enumerate(user_recs.get('recommendations', [])[:20]):
                    print(f"      {i+1}. {rec.get('title', 'Sin título')} (score: {rec.get('score', 0):.2f})")
            else:
                print("      ⚠️ No se encontraron recomendaciones para el usuario")
        else:
            print(f"   ❌ Error en recomendaciones de usuario: {response.status_code}")
            print(f"   Response: {response.text}")

        # 9. Probar el endpoint principal de recomendaciones (user_id dinámico)
        print(f"\n9. 🌟 Probando endpoint principal de recomendaciones (ID: {user_id})...")
        response = requests.get(f"{BASE_URL}/recommend/{user_id}", headers=auth_headers)
        if response.status_code == 200:
            main_recs = response.json()
            rec_count = len(main_recs.get('recommendations', []))
            strategy = main_recs.get('strategy', 'unknown')
            print(f"   ✅ Recomendaciones principales: {rec_count} (estrategia: {strategy})")
            
            if rec_count > 0:
                for i, rec in enumerate(main_recs.get('recommendations', [])[:3]):
                    print(f"      {i+1}. {rec.get('title', 'Sin título')}")
            else:
                print("      ⚠️ No se encontraron recomendaciones principales")
        else:
            print(f"   ❌ Error en recomendaciones principales: {response.status_code}")
            print(f"   Response: {response.text}")

        print("\n🎉 ¡Pruebas completadas!")

    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar con la API. Asegúrate de que el servidor esté ejecutándose.")
        print("   Ejecuta: uvicorn main:app --reload --port 8000")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Pequeña pausa para asegurar que el servidor esté listo
    print("⏳ Esperando que el servidor esté listo...")
    time.sleep(3)
    test_recommendations()