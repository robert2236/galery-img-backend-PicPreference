from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from decouple import config
from fastapi import FastAPI, BackgroundTasks, HTTPException
from routers.users import users
from routers.galery import galery
from routers.category import category
from routers.recommendations import router as recommendations_router
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from contextlib import asynccontextmanager
from services.genetic import optimize_weights
from services.graph import InteractionGraph
from services.recommender import VisualRecommender
from models.galery import Image
from database.databases import coleccion, user
from bson import ObjectId
from utils.feature_extractor import FeatureExtractor
from fastapi.encoders import jsonable_encoder
import asyncio
from services.graph import graph_recommender  # Importar la instancia global
from services.auxiliary import (
    get_popular_images, 
    get_user_viewed_images,
    get_content_based_recommendations,
    get_graph_based_recommendations,  # Cambiar a esta importación
    get_popular_recommendations,
    remove_duplicates
)

# Instancias globales
graph_recommender = InteractionGraph()
visual_recommender = VisualRecommender()
recommender = VisualRecommender()

async def rebuild_graph_periodically():
    """Reconstruye el grafo periódicamente (ejecutar en background)"""

    while True:
        try:
            print("🔄 Reconstruyendo grafo de interacciones (tarea programada)...")
            graph_recommender.graph.clear()  # Limpiar grafo existente
            
            # Obtener todas las imágenes con likes
            images_with_likes = await coleccion.find({
                "liked_by": {"$exists": True, "$ne": []}
            }).to_list(None)
            
            for img in images_with_likes:
                image_id = img.get("image_id")
                liked_by = img.get("liked_by", [])
                
                for user_id in liked_by:
                    graph_recommender.graph.add_edge(
                        f"user_{user_id}",
                        f"image_{image_id}",
                        weight=img.get("interactions", {}).get("likes", 1)
                    )
            
            print(f"✅ Grafo reconstruido: {graph_recommender.graph.number_of_nodes()} nodos, {graph_recommender.graph.number_of_edges()} aristas")
            
        except Exception as e:
            print(f"❌ Error en reconstrucción periódica: {e}")
        
        # Esperar 1 hora antes de la próxima reconstrucción
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación"""
    # Startup
    print("🔄 Inicializando sistemas de recomendación...")
    
    try:
        # Construir grafo de interacciones
        print("📊 Construyendo grafo de interacciones...")
        await graph_recommender.build_from_db()
        print(f"✅ Grafo de interacciones: {graph_recommender.graph.number_of_nodes()} nodos, {graph_recommender.graph.number_of_edges()} aristas")
        
        # Construir índice visual
        print("📊 Construyendo índice visual...")
        await visual_recommender.build_index()
        print(f"✅ Índice visual: {len(visual_recommender.image_ids) if visual_recommender.image_ids else 0} imágenes indexadas")
        
        # Iniciar tarea de reconstrucción periódica en background
        asyncio.create_task(rebuild_graph_periodically())
        print("✅ Tarea de reconstrucción periódica iniciada")
        
    except Exception as e:
        print(f"❌ Error inicializando sistemas de recomendación: {e}")
    
    yield
    
    # Shutdown (opcional)
    print("🔴 Apagando aplicación...")

app = FastAPI(
    title="Sistema de Recomendación de Galería",
    description="API para sistema de recomendación de imágenes con múltiples estrategias",
    version="1.0.0",
    lifespan=lifespan
)

security = HTTPBasic()

# Configuración de CORS
origins = [
    "https://5j6k3nm7-5173.use2.devtunnels.ms",
    "https://5j6k3nm7-5000.use2.devtunnels.ms", 
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Incluir routers
app.include_router(users)
app.include_router(galery)
app.include_router(category)
app.include_router(recommendations_router)

@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "message": "Sistema de Recomendación de Galería",
        "version": "1.0.0",
        "endpoints": {
            "documentación": "/docs",
            "recomendaciones": "/recommend",
            "galería": "/galery",
            "usuarios": "/users",
            "categorías": "/category"
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del sistema"""
    try:
        # Verificar conexión a la base de datos
        db_status = await coleccion.count_documents({}) >= 0
        graph_status = graph_recommender.graph.number_of_nodes() > 0 if graph_recommender.graph else False
        visual_status = visual_recommender.tree is not None
        
        return {
            "status": "healthy",
            "database": "connected" if db_status else "disconnected",
            "graph_recommender": "ready" if graph_status else "initializing",
            "visual_recommender": "ready" if visual_status else "initializing",
            "total_images": await coleccion.count_documents({}),
            "graph_nodes": graph_recommender.graph.number_of_nodes() if graph_recommender.graph else 0
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.post("/process-image")
async def process_image(image: Image, background_tasks: BackgroundTasks):
    """Endpoint para procesar nuevas imágenes"""
    try:
        img_dict = image.dict(by_alias=True)
        
        # Extraer características en segundo plano
        background_tasks.add_task(
            process_image_features,
            img_dict["image_url"],
            img_dict["_id"]
        )
        
        result = await coleccion.insert_one(img_dict)
        
        return {
            "message": "Imagen en procesamiento",
            "image_id": str(result.inserted_id),
            "status": "processing_features"
        }
    except Exception as e:
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")

async def process_image_features(image_url: str, image_id: str):
    """Tarea en segundo plano para extraer características"""
    try:
        features = FeatureExtractor().extract(image_url)
        await coleccion.update_one(
            {"_id": ObjectId(image_id)},
            {"$set": {"features": features}}
        )
        print(f"✅ Características extraídas para imagen {image_id}")
        
        # Reconstruir el índice con la nueva imagen
        await visual_recommender.build_index()
        
    except Exception as e:
        print(f"❌ Error extrayendo características: {e}")

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int, image_id: Optional[str] = None):
    """
    Recomendaciones PERSONALIZADAS para un usuario.
    Si se proporciona image_id, recomienda basado en esa imagen específica.
    Si no, recomienda basado en el historial del usuario.
    """
    try:
        # Verificar que el usuario existe
        user_exists = await user.find_one({"user_id": user_id})
        if not user_exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        if image_id:
            # Recomendación basada en una imagen específica
            result = await get_image_based_recommendations(user_id, image_id)
        else:
            # Recomendación basada en el historial del usuario
            result = await get_user_based_recommendations(user_id)
        
        # Asegurar que todos los ObjectId sean convertidos a strings
        return jsonable_encoder(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando recomendaciones: {str(e)}")

async def get_image_based_recommendations(user_id: int, image_id: str):
    """Recomendaciones basadas en una imagen específica - CORREGIDO para IDs numéricos"""
    try:
        # Verificar que la imagen existe (por image_id numérico)
        target_image = await coleccion.find_one({"image_id": int(image_id)})
        if not target_image:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")

        similar_images_ids = await recommender.find_similar(image_id, k=10)
        
        # Obtener información completa de las imágenes similares
        similar_images = []
        for img_id in similar_images_ids:
            image = await coleccion.find_one({"image_id": img_id})
            if image:
                similar_images.append({
                    "id": image["image_id"],  # Usar image_id numérico
                    "title": image.get("title", "Sin título"),
                    "image_url": image.get("image_url", ""),
                    "category": image.get("category", ""),
                    "interactions": image.get("interactions", {})
                })
        
        # Filtrar para excluir imágenes que el usuario ya ha visto
        user_viewed = await get_user_viewed_images(user_id)
        filtered_recommendations = [
            img for img in similar_images 
            if img["id"] not in user_viewed
        ]
        
        return {
            "user_id": user_id,
            "based_on_image": image_id,
            "total_recommendations": len(filtered_recommendations),
            "recommendations": filtered_recommendations[:10]
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de imagen debe ser numérico")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en recomendaciones basadas en imagen: {str(e)}")

async def get_user_based_recommendations(user_id: int):
    """Recomendaciones basadas en el historial del usuario - VERSIÓN MEJORADA"""
    try:
        print(f"🔍 Generando recomendaciones para usuario {user_id}")
        
        # 1. Obtener el usuario y sus imágenes interactuadas
        user_data = await user.find_one({"user_id": user_id})
        if not user_data:
            print(f"⚠️ Usuario {user_id} no encontrado")
            return await get_cold_start_recommendations(user_id)
        
        user_viewed_images = user_data.get("images", [])
        print(f"📊 Usuario {user_id} ha interactuado con {len(user_viewed_images)} imágenes")
        
        # 2. Estrategia basada en el número de interacciones
        if len(user_viewed_images) == 0:
            # Usuario sin interacciones - Recomendaciones populares
            print("🎯 Estrategia: Cold Start (populares)")
            return await get_cold_start_recommendations(user_id)
            
        elif len(user_viewed_images) < 5:
            # Usuario con pocas interacciones - Combinar contenido + populares
            print("🎯 Estrategia: Poco historial (híbrido)")
            return await get_hybrid_recommendations(user_id, user_viewed_images)
            
        else:
            # Usuario con buen historial - Recomendaciones personalizadas
            print("🎯 Estrategia: Historial completo (personalizado)")
            return await get_personalized_recommendations(user_id, user_viewed_images)
        
    except Exception as e:
        print(f"❌ Error en recomendaciones de usuario: {e}")
        return await get_cold_start_recommendations(user_id)

async def get_hybrid_recommendations(user_id: int, user_viewed_images: list):
    """Recomendaciones para usuarios con poco historial"""
    all_recommendations = []
    
    # 1. Recomendaciones basadas en contenido (imágenes similares a las que ha visto)
    content_recs = await get_content_based_recommendations(user_viewed_images, limit=8)
    all_recommendations.extend(content_recs)
    
    # 2. Recomendaciones del grafo (comportamiento de usuarios similares)
    try:
        graph_recs = await get_graph_based_recommendations(user_id, user_viewed_images, limit=6)
        all_recommendations.extend(graph_recs)
    except Exception as e:
        print(f"⚠️ Error en recomendaciones del grafo: {e}")
    
    # 3. Recomendaciones populares (fallback)
    if len(all_recommendations) < 10:
        popular_recs = await get_popular_recommendations(user_viewed_images, limit=6)
        all_recommendations.extend(popular_recs)
    
    # Eliminar duplicados y ordenar
    unique_recommendations = remove_duplicates(all_recommendations)
    unique_recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "user_id": user_id,
        "based_on_history": True,
        "strategy": "hybrid_light",
        "interacted_images_count": len(user_viewed_images),
        "total_recommendations": len(unique_recommendations),
        "recommendations": unique_recommendations[:15]
    }

async def get_personalized_recommendations(user_id: int, user_viewed_images: list):
    """Recomendaciones para usuarios con buen historial"""
    all_recommendations = []
    
    # 1. Recomendaciones del grafo (comportamiento - mayor peso)
    graph_recs = await get_graph_based_recommendations(user_id, user_viewed_images, limit=10)
    all_recommendations.extend(graph_recs)
    
    # 2. Recomendaciones basadas en contenido
    content_recs = await get_content_based_recommendations(user_viewed_images[-3:], limit=8)
    all_recommendations.extend(content_recs)
    
    # 3. Recomendaciones populares (solo si es necesario)
    if len(all_recommendations) < 12:
        popular_recs = await get_popular_recommendations(user_viewed_images, limit=4)
        all_recommendations.extend(popular_recs)
    
    # Eliminar duplicados y ordenar
    unique_recommendations = remove_duplicates(all_recommendations)
    unique_recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "user_id": user_id,
        "based_on_history": True,
        "strategy": "personalized",
        "interacted_images_count": len(user_viewed_images),
        "total_recommendations": len(unique_recommendations),
        "recommendations": unique_recommendations[:15]
    }

async def get_cold_start_recommendations(user_id: int):
    """Recomendaciones para usuarios nuevos (cold start)"""
    popular_images = await get_popular_images()
    return {
        "user_id": user_id,
        "based_on_history": False,
        "strategy": "cold_start_popular",
        "interacted_images_count": 0,
        "total_recommendations": len(popular_images[:15]),
        "recommendations": popular_images[:15]
    }

@app.get("/system-status")
async def system_status():
    """Estado detallado del sistema de recomendación"""
    try:
        total_images = await coleccion.count_documents({})
        images_with_features = await coleccion.count_documents({"features": {"$exists": True}})
        images_with_interactions = await coleccion.count_documents({"interactions": {"$exists": True}})
        
        return {
            "total_images": total_images,
            "images_with_features": images_with_features,
            "images_with_interactions": images_with_interactions,
            "feature_coverage": f"{(images_with_features/total_images*100):.1f}%" if total_images > 0 else "0%",
            "interaction_coverage": f"{(images_with_interactions/total_images*100):.1f}%" if total_images > 0 else "0%",
            "graph_nodes": graph_recommender.graph.number_of_nodes() if graph_recommender.graph else 0,
            "graph_edges": graph_recommender.graph.number_of_edges() if graph_recommender.graph else 0,
            "kdtree_built": visual_recommender.tree is not None,
            "kdtree_size": len(visual_recommender.image_ids) if visual_recommender.image_ids else 0
        }
    except Exception as e:
        raise HTTPException(500, f"Error obteniendo estado del sistema: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)