# routers/recommendations.py
from fastapi import APIRouter, HTTPException
from services.recommender import VisualRecommender
from services.graph import InteractionGraph
from services.genetic import optimize_weights
from database.databases import coleccion, user
from bson import ObjectId
from typing import Optional
import asyncio

router = APIRouter(prefix="/recommend", tags=["Recommendations"])
visual_recommender = VisualRecommender()
graph_recommender = InteractionGraph()

@router.get("/similar/{image_id}")
async def get_similar_images(image_id: str, limit: int = 5):
    """Obtiene imágenes visualmente similares - CORREGIDO para IDs numéricos"""
    try:
        # Verificar que la imagen existe (por image_id numérico)
        image_exists = await coleccion.find_one({"image_id": int(image_id)})
        if not image_exists:
            raise HTTPException(404, "Imagen no encontrada")
        
        similar_ids = await visual_recommender.find_similar(image_id, k=limit)
        
        # Obtener información completa de las imágenes similares
        similar_images = []
        for img_id in similar_ids:
            image = await coleccion.find_one({"image_id": img_id})
            if image:
                similar_images.append({
                    "id": image["image_id"],  # Usar image_id numérico
                    "title": image.get("title", "Sin título"),
                    "url": image.get("image_url", ""),
                    "likes": image.get("interactions", {}).get("likes", 0),
                    "views": image.get("interactions", {}).get("views", 0)
                })
        
        return {
            "original_image_id": image_id,
            "similar_images": similar_images,
            "total_similar": len(similar_images)
        }
    except ValueError:
        raise HTTPException(400, "ID de imagen debe ser numérico")
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@router.get("/user/{user_id}")
async def get_user_recommendations(user_id: int, limit: int = 20):
    """Recomendaciones para un usuario específico"""
    try:
        print(f"🔍 Buscando usuario: {user_id}")
        
        # Verificar que el usuario existe
        user_exists = await user.find_one({"user_id": user_id})
        if not user_exists:
            print(f"❌ Usuario {user_id} no encontrado")
            raise HTTPException(404, f"Usuario {user_id} no encontrado")
        
        print(f"✅ Usuario {user_id} encontrado, generando recomendaciones...")
        
        # Obtener recomendaciones del grafo de interacciones
        graph_recs = graph_recommender.recommend_for_user(user_id, k=limit)
        print(f"📊 Recomendaciones del grafo: {len(graph_recs)}")
        
        # Obtener información de las imágenes recomendadas
        recommendations = []
        for img_id, score in graph_recs:
            image = await coleccion.find_one({"image_id": img_id})
            if image:
                recommendations.append({
                    "id": str(image["image_id"]),
                    "title": image.get("title", "Sin título"),
                    "url": image.get("image_url", ""),
                    "score": float(score),
                    "type": "behavioral"
                })
        
        print(f"🎯 Recomendaciones finales: {len(recommendations)}")
        
        return {
            "user_id": user_id,
            "recommendations": recommendations[:limit],
            "total_recommendations": len(recommendations[:limit])
        }
    except Exception as e:
        print(f"💥 Error en recomendaciones para usuario {user_id}: {str(e)}")
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
        
        # Registrar la interacción
        await visual_recommender.log_interaction(image_id, action)
        
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