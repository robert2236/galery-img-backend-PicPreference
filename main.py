from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from decouple import config
from fastapi import FastAPI, BackgroundTasks, HTTPException
from routers.users import users
from routers.galery import galery
from routers.category import category
from routers.recommendations import router as recommendations_router
from routers.dashboard import metrics_router
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
from services.recommendation_engine import RecommendationEngine
from services.recommender_evaluator import RecommenderEvaluator
from services.graph_recommender import graph_recommender
from services.simple_evaluator import SimpleEvaluator


simple_evaluator = SimpleEvaluator(k=5)
recommendation_engine = RecommendationEngine()
graph_recommender = InteractionGraph()

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
            
            print(f"📊 Procesando {len(images_with_likes)} imágenes con likes...")
            
            # PRIMERO: Añadir todos los nodos de usuario e imagen
            all_user_ids = set()
            all_image_ids = set()
            
            for img in images_with_likes:
                image_id = img.get("image_id")
                liked_by = img.get("liked_by", [])
                
                if image_id is not None:
                    all_image_ids.add(image_id)
                    all_user_ids.update(liked_by)
            
            # Añadir nodos de usuario
            for user_id in all_user_ids:
                user_node = f"user_{user_id}"
                graph_recommender.graph.add_node(user_node, type="user")
            
            # Añadir nodos de imagen
            for image_id in all_image_ids:
                image_node = f"image_{image_id}"
                graph_recommender.graph.add_node(image_node, type="image")
            
            print(f"✅ Añadidos {len(all_user_ids)} usuarios y {len(all_image_ids)} imágenes como nodos")
            
            # SEGUNDO: Añadir aristas (conexiones)
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
            
            # Debug: mostrar algunos nodos
            user_nodes = [n for n in graph_recommender.graph.nodes() if isinstance(n, str) and n.startswith('user_')]
            image_nodes = [n for n in graph_recommender.graph.nodes() if isinstance(n, str) and n.startswith('image_')]
            
            print(f"   👥 User nodes: {user_nodes[:5]}...")
            print(f"   📷 Image nodes: {image_nodes[:5]}...")
            
        except Exception as e:
            print(f"❌ Error en reconstrucción periódica: {e}")
            import traceback
            traceback.print_exc()
        
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
    "http://192.168.1.103:5173"
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
app.include_router(metrics_router)

@app.on_event("startup")
async def startup_event():
    """Inicializar el sistema al arrancar"""
    try:
        print("🚀 Inicializando sistema de recomendación...")
        await recommendation_engine.initialize()
        print("✅ Sistema de recomendación listo")
    except Exception as e:
        print(f"❌ Error inicializando sistema: {e}")

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
    
@app.get("/api/recommend/{user_id}")
async def get_recommendations(user_id: int, page: int = 1, limit: int = 10):
    """Endpoint principal de recomendaciones con paginación"""
    try:
        # Obtener todas las recomendaciones
        all_recommendations = await recommendation_engine.get_recommendations(user_id)
        
        if not all_recommendations:
            fallback = await get_popular_images(10)
            # Paginar las recomendaciones de fallback
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_fallback = fallback[start_idx:end_idx]
            
            return {
                "status": "warning",
                "message": "No se pudieron generar recomendaciones personalizadas",
                "fallback_recommendations": paginated_fallback,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": len(fallback),
                    "pages": (len(fallback) + limit - 1) // limit
                }
            }
        
        # Paginar las recomendaciones
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_recommendations = all_recommendations[start_idx:end_idx]
        
        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": paginated_recommendations,
            "count": len(paginated_recommendations),
            "pagination": {
                "page": page,
                "limit": limit,
                "total": len(all_recommendations),
                "pages": (len(all_recommendations) + limit - 1) // limit
            }
        }
        
    except Exception as e:
        print(f"❌ Error en endpoint de recomendaciones: {e}")
        fallback = await get_popular_images(5)
        # Paginar las recomendaciones de error
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_fallback = fallback[start_idx:end_idx]
        
        return {
            "status": "error",
            "message": "Error interno del sistema",
            "fallback_recommendations": paginated_fallback,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": len(fallback),
                "pages": (len(fallback) + limit - 1) // limit
            }
        }
        
@app.get("/graph-stats")
async def get_graph_stats():
    """Estadísticas del grafo"""
    return {
        "nodes": graph_recommender.graph.number_of_nodes(),
        "edges": graph_recommender.graph.number_of_edges(),
        "user_nodes": len([n for n in graph_recommender.graph.nodes if graph_recommender.graph.nodes[n].get('type') == 'user']),
        "image_nodes": len([n for n in graph_recommender.graph.nodes if graph_recommender.graph.nodes[n].get('type') == 'image'])
    }        
        
@app.get("/evaluate/{user_id}")
async def evaluate_user(user_id: int):
    evaluator = RecommenderEvaluator(k=5)
    result = await evaluator.evaluate_single_user(user_id, graph_recommender)
    
    if result:
        return result
    else:
        return {
            "error": "No se pudo evaluar el usuario",
            "user_id": user_id,
            "message": "El usuario no tiene suficientes datos o hubo un error"
        }

@app.get("/debug-user/{user_id}")
async def debug_user(user_id: int):
    """Endpoint para debuggear datos del usuario"""
    try:
        # 1. Verificar si el usuario existe
        user_data = await user.find_one({"user_id": user_id})
        
        # 2. Buscar imágenes que el usuario ha likeado
        liked_images = await simple_evaluator.get_user_likes(user_id)
        
        # 3. Verificar algunas imágenes específicas para ver su estructura
        sample_image = await coleccion.find_one({"liked_by": user_id})
        
        # 4. Contar total de imágenes likeadas
        total_liked = await coleccion.count_documents({"liked_by": user_id})
        
        # 5. Ver estructura de campos de una imagen
        image_fields = []
        if sample_image:
            image_fields = list(sample_image.keys())
        
        return {
            "user_id": user_id,
            "user_exists": user_data is not None,
            "total_liked_images": total_liked,
            "liked_images_sample": liked_images[:5],
            "image_fields": image_fields,
            "sample_image_structure": {
                "has_liked_by": "liked_by" in sample_image if sample_image else False,
                "liked_by_sample": sample_image.get("liked_by", [])[:3] if sample_image else [],
                "has_image_id": "image_id" in sample_image if sample_image else False
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/evaluate-all")
async def evaluate_all():
    evaluator = RecommenderEvaluator(k=10)
    results = await evaluator.evaluate_all_users(graph_recommender)
    metrics = await evaluator.calculate_metrics(results)
    
    return {
        "metrics": metrics,
        "detailed_results": results[:10]  # Mostrar solo primeros 10 para no saturar
    }
    
@app.get("/debug/recommendations/{user_id}")
async def debug_recommendations(user_id: int):
    """Endpoint para diagnóstico del sistema de recomendación"""
    viewed_images = await get_user_viewed_images(user_id)
    popular_images = await get_popular_images(5)
    
    return {
        "user_id": user_id,
        "viewed_images_count": len(viewed_images),
        "viewed_images_sample": viewed_images[:5],
        "popular_images_available": popular_images,
        "database_stats": {
            "total_images": await coleccion.count_documents({}),
            "total_users": await user.count_documents({}),
            "images_with_likes": await coleccion.count_documents({"interactions.likes": {"$gt": 0}})
        }
    }
    
@app.get("/simple-evaluate/{user_id}")
async def simple_evaluate(user_id: int):
    evaluator = SimpleEvaluator(k=5)
    result = await evaluator.evaluate_user(user_id)
    
    if result:
        return result
    else:
        return {"error": "No se pudo evaluar", "user_id": user_id}

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

@app.get("/debug-image/{image_id}")
async def debug_image(image_id: int):
    """Debug de una imagen específica"""
    try:
        image_data = await coleccion.find_one({"image_id": image_id})
        if not image_data:
            return {"error": "Imagen no encontrada"}
        
        return {
            "image_id": image_id,
            "fields": list(image_data.keys()),
            "liked_by": image_data.get("liked_by", []),
            "liked_by_count": len(image_data.get("liked_by", [])),
            "has_interactions": "interactions" in image_data,
            "interactions": image_data.get("interactions", {})
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/simple-evaluate/{user_id}")
async def simple_evaluate(user_id: int):
    result = await simple_evaluator.evaluate_user(user_id)
    return result

@app.get("/evaluate-model/{user_id}")
async def evaluate_model(user_id: int):
    """Evaluar el modelo completo con graph recommender"""
    try:
        evaluator = RecommenderEvaluator(k=5)
        result = await evaluator.evaluate_single_user(user_id, graph_recommender)
        
        if result:
            return result
        else:
            return {
                "error": "No se pudo evaluar el modelo",
                "user_id": user_id
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/evaluate-all-users")
async def evaluate_all_users():
    """Evaluar el modelo para todos los usuarios"""
    try:
        evaluator = RecommenderEvaluator(k=5)
        results = await evaluator.evaluate_all_users(graph_recommender)
        metrics = await evaluator.calculate_metrics(results)
        
        return {
            "metrics": metrics,
            "users_evaluated": len(results),
            "sample_results": results[:3]  
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)