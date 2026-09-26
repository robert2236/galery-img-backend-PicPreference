# routers/recommendations.py
from fastapi import APIRouter, HTTPException
from services.graph import InteractionGraph
from services.genetic import optimize_weights
from database.databases import coleccion, user
from bson import ObjectId
from typing import Optional
import asyncio
import logging
from models.Pagination import PaginationParams
from services.evaluation import RecommenderEvaluator
from services.recommendation_engine import RecommendationEngine
from services.visual_search import (
    ImageNotFoundError,
    find_similar_images,
)
from vector_store import (
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_RELATIVE,
    DEFAULT_MIN_SCORE,
    VectorStoreError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["Recommendations"])
graph_recommender = InteractionGraph()

@router.get("/similar/{image_id}")
async def get_similar_images(
    image_id: int,
    limit: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
    min_relative: Optional[float] = DEFAULT_MIN_RELATIVE,
    max_gap: Optional[float] = DEFAULT_MAX_GAP
):
    """
    Imágenes visualmente similares - usa el mismo motor de búsqueda que
    /api/v1/recommendations/visual-similar (ChromaDB + filtro de relevancia),
    con el score coseno real en cada resultado.
    """
    try:
        outcome = await find_similar_images(
            image_id=image_id,
            limit=limit,
            min_score=min_score,
            min_relative=min_relative,
            max_gap=max_gap
        )

        return {
            "original_image_id": image_id,
            "similar_images": outcome["results"],
            "total_similar": len(outcome["results"]),
            "closest_match": outcome["closest"],
            "below_threshold": outcome["below_threshold"],
            "thresholds": outcome["thresholds"],
            "has_embedding": outcome["has_embedding"]
        }

    except ImageNotFoundError:
        raise HTTPException(404, "Imagen no encontrada")
    except VectorStoreError as e:
        logger.error(f"❌ Servicio de vectores no disponible: {e}")
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.error(f"❌ Error en /similar/{image_id}: {e}")
        raise HTTPException(500, f"Error: {str(e)}")

from fastapi import HTTPException, Depends, Query
from typing import Optional


@router.get("/user/{user_id}")
async def get_user_recommendations(
    user_id: int, 
    pagination: PaginationParams = Depends()
):
    """Recomendaciones para un usuario específico con paginación"""
    try:
        user_id_str = str(user_id)
        print(f"🔍 Buscando usuario: {user_id} (como string: {user_id_str})")
        
        # Verificar que el usuario existe
        user_exists = await user.find_one({"user_id": user_id})
        if not user_exists:
            print(f"❌ Usuario {user_id} no encontrado en la base de datos")
            raise HTTPException(404, f"Usuario {user_id} no encontrado")
        
        # Obtener TODAS las recomendaciones del grafo (sin límite para paginar después)
        graph_recs = await graph_recommender.recommend_for_user(user_id_str, k=1000)  # Número alto para obtener todas
        
        # Si no hay recomendaciones, usar fallback (sin llamar al método de la clase)
        if not graph_recs:
            print("⚠️ No hay recomendaciones del grafo, usando fallback...")
            # Lógica de fallback - obtener todas las populares
            popular_images = await coleccion.find(
                {"interactions.views": {"$gt": 0}}
            ).sort("interactions.likes", -1).to_list(None)
            
            graph_recs = []
            for image in popular_images:
                image_id = image.get("image_id")
                if image_id:
                    graph_recs.append((str(image_id), 1.0))
        
        # Procesar TODAS las recomendaciones primero
        all_recommendations = []
        for img_id, score in graph_recs:
            print(f"🔍 Buscando imagen con ID: {img_id} (tipo: {type(img_id)})")
            
            # Buscar la imagen - intentar múltiples formatos
            image = None
            
            # Intentar como número (si es posible)
            try:
                numeric_id = int(img_id)
                image = await coleccion.find_one({"image_id": numeric_id})
                if image:
                    print(f"   ✅ Encontrada con ID numérico: {numeric_id}")
            except (ValueError, TypeError):
                pass
            
            # Si no se encontró, intentar como string
            if not image:
                image = await coleccion.find_one({"image_id": str(img_id)})
                if image:
                    print(f"   ✅ Encontrada con ID string: {img_id}")
            
            if image:
                all_recommendations.append({
                    "image_id": image.get("image_id"),
                    "title": image.get("title", "Sin título"),
                    "url": image.get("image_url", ""),
                    "score": float(score),
                    "type": "behavioral" if score > 1.0 else "fallback"
                })
            else:
                print(f"   ❌ Imagen no encontrada en BD para ID: {img_id}")
        
        # Aplicar paginación
        total_count = len(all_recommendations)
        total_pages = (total_count + pagination.limit - 1) // pagination.limit if pagination.limit > 0 else 1
        has_next = (pagination.skip + pagination.limit) < total_count
        has_prev = pagination.skip > 0
        
        # Obtener solo la página solicitada
        paginated_recommendations = all_recommendations[
            pagination.skip:pagination.skip + pagination.limit
        ]
        
        print(f"🎯 {len(paginated_recommendations)} recomendaciones finales para usuario {user_id} (página {pagination.page})")
        
        return {
            "user_id": user_id,
            "recommendations": paginated_recommendations,
            "pagination": {
                "total": total_count,
                "page": pagination.page,
                "limit": pagination.limit,
                "skip": pagination.skip,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
                "next_page": pagination.page + 1 if has_next else None,
                "prev_page": pagination.page - 1 if has_prev else None
            },
            "total_recommendations": total_count
        }
        
    except Exception as e:
        print(f"💥 Error en recomendaciones para usuario {user_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)}")
    
@router.get("/optimize-weights")
async def get_optimized_weights():
    """Optimiza y devuelve los pesos para el sistema de recomendación"""
    try:
        weights = await optimize_weights()
        return weights
    except Exception as e:
        raise HTTPException(500, f"Error optimizando pesos: {str(e)}")

@router.post("/interaction/{image_id}")
async def log_interaction(image_id: str, action: str = "view", user_id: Optional[int] = None):
    """Registra una interacción con una imagen"""
    try:
        # Verificar que la imagen existe
        image_exists = await coleccion.find_one({"_id": ObjectId(image_id)})
        if not image_exists:
            raise HTTPException(404, "Imagen no encontrada")
        
        # Registrar la interacción directamente en la base de datos
        update_field = f"interactions.{action}s"
        await coleccion.update_one(
            {"_id": ObjectId(image_id)},
            {"$inc": {update_field: 1}}
        )
        
        # Si hay usuario, actualizar el grafo (esto necesita implementación)
        if user_id:
            # Actualizar el grafo de interacciones
            user_exists = await user.find_one({"user_id": user_id})
            if user_exists:
                # Aquí deberías agregar la lógica para actualizar el grafo
                pass
            
        return {"message": f"Interacción '{action}' registrada para imagen {image_id}"}
    except Exception as e:
        raise HTTPException(500, f"Error registrando interacción: {str(e)}")

@router.get("/popular")
async def get_popular_recommendations(limit: int = 10):
    """Obtiene las imágenes más populares"""
    try:
        popular_images = await coleccion.find(
            {"interactions.views": {"$gt": 0}}
        ).sort("interactions.likes", -1).limit(limit).to_list(None)
        
        recommendations = []
        for image in popular_images:
            recommendations.append({
                "id": str(image["_id"]),
                "title": image.get("title", "Sin título"),
                "url": image.get("image_url", ""),
                "likes": image.get("interactions", {}).get("likes", 0),
                "views": image.get("interactions", {}).get("views", 0),
                "type": "popular"
            })
        
        return {
            "recommendations": recommendations,
            "total": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
    
# En tu archivo de rutas de recomendaciones
@router.get("/debug/graph")
async def debug_database():
    """Verifica los datos en la base de datos"""
    try:
        # Verificar imágenes con likes
        images_with_likes = await coleccion.find({
            "liked_by": {"$exists": True, "$ne": []}
        }).to_list(length=None)
        
        # Verificar todas las imágenes
        all_images = await coleccion.find().to_list(length=5)
        
        # Verificar estructura de algunas imágenes
        sample_images = []
        for img in all_images[:3]:
            sample_images.append({
                "image_id": img.get("image_id"),
                "title": img.get("title"),
                "has_liked_by": "liked_by" in img,
                "liked_by_count": len(img.get("liked_by", [])),
                "liked_by_sample": img.get("liked_by", [])[:3] if "liked_by" in img else []
            })
        
        return {
            "total_images": await coleccion.count_documents({}),
            "images_with_likes": len(images_with_likes),
            "sample_images": sample_images,
            "images_with_likes_sample": [
                {
                    "image_id": img.get("image_id"),
                    "liked_by": img.get("liked_by", [])[:3],
                    "liked_by_count": len(img.get("liked_by", []))
                } for img in images_with_likes[:3]
            ] if images_with_likes else []
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
    
#Reconstruir grafo

@router.post("/rebuild-graph")
async def rebuild_graph():
    """Forzar reconstrucción del grafo"""
    try:
        await graph_recommender.build_from_db()
        return {
            "message": "Grafo reconstruido exitosamente", 
            "nodes": graph_recommender.graph.number_of_nodes(),
            "edges": graph_recommender.graph.number_of_edges(),
            "user_nodes": len([n for n in graph_recommender.graph.nodes() if n.startswith('user_')]),
            "image_nodes": len([n for n in graph_recommender.graph.nodes() if n.startswith('image_')])
        }
    except Exception as e:
        raise HTTPException(500, f"Error reconstruyendo grafo: {str(e)}")
    
# routers/recommendations.py - Agrega estos endpoints
from services.evaluation import RecommenderEvaluator

# Crear evaluador global
evaluator = RecommenderEvaluator(k=10)

@router.get("/evaluate/metrics")
async def evaluate_metrics():
    """Endpoint para evaluar las métricas del sistema"""
    try:
        results = await evaluator.evaluate_all_users(graph_recommender)
        metrics = await evaluator.calculate_metrics(results)
        
        return {
            "status": "success",
            "metrics": metrics,
            "detailed_results": results[:10]  # Primeros 10 resultados para debug
        }
    except Exception as e:
        raise HTTPException(500, f"Error en evaluación: {str(e)}")

@router.get("/evaluate/user/{user_id}")
async def evaluate_user(user_id: int):
    """Evalúa las recomendaciones para un usuario específico"""
    try:
        # Obtener recomendaciones
        recommendations = await graph_recommender.recommend_for_user(str(user_id), k=10)
        rec_ids = [img_id for img_id, score in recommendations]
        
        # Obtener ground truth (lo que realmente le gustó al usuario)
        user_likes = []
        cursor = coleccion.find({"liked_by": user_id})
        async for image in cursor:
            user_likes.append(image.get("image_id"))
        
        # Evaluar
        precision, recall, f1, hits = await evaluator.evaluate_recommendations(
            user_id, rec_ids, user_likes
        )
        
        return {
            "user_id": user_id,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "hits": hits,
            "total_recommendations": len(rec_ids),
            "total_positives": len(user_likes),
            "recommendations": rec_ids,
            "actual_likes": user_likes
        }
    except Exception as e:
        raise HTTPException(500, f"Error evaluando usuario: {str(e)}")