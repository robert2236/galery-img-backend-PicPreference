from fastapi import APIRouter, Depends, HTTPException
from services.recommender import ALSRecommender
from database.databases import coleccion
import asyncio

router = APIRouter()
recommender = ALSRecommender()

# Variable para almacenar la matriz entrenada
user_item_matrix = None

@router.on_event("startup")
async def startup_event():
    """Carga y entrena el modelo al iniciar la aplicación"""
    global user_item_matrix
    user_item_matrix, _ = await recommender.load_data(coleccion)
    recommender.train(user_item_matrix)
    print("✅ Modelo ALS entrenado")

@router.get("/als/recommend/{user_id}")
async def get_als_recommendations(user_id: int, limit: int = 10):
    if user_item_matrix is None:
        raise HTTPException(503, "Modelo no entrenado aún")
    
    return {
        "recommendations": recommender.recommend_for_user(
            user_id, 
            user_item_matrix, 
            N=limit
        )
    }

@router.post("/als/retrain")
async def retrain_model():
    """Reentrena el modelo con nuevos datos (ejecutar periódicamente)"""
    global user_item_matrix
    user_item_matrix, _ = await recommender.load_data(coleccion)
    recommender.train(user_item_matrix)
    return {"status": "Modelo reentrenado"}