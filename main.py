from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from decouple import config
from fastapi import FastAPI,BackgroundTasks
from routers.users import users
from routers.galery import galery
from routers.category import category
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import FastAPI, Depends, HTTPException, status
from contextlib import asynccontextmanager
from services.genetic import optimize_weights
from services.graph import InteractionGraph
from services.recommender import VisualRecommender
from models.galery import Image
from database.databases import coleccion, user
from bson import ObjectId
from services.auxiliary import get_popular_images,get_user_viewed_images
from utils.feature_extractor import FeatureExtractor

app = FastAPI()
# Instancias globales
graph_recommender = InteractionGraph()
visual_recommender = VisualRecommender()
recommender = VisualRecommender()

@app.on_event("startup")
async def startup():
    await graph_recommender.build_from_db()
    await visual_recommender.build_index()

@app.post("/process-image")
async def process_image(image: Image, background_tasks: BackgroundTasks):
    """Endpoint para procesar nuevas imágenes"""
    img_dict = image.dict(by_alias=True)
    
    # Extraer características en segundo plano
    background_tasks.add_task(
        process_image_features,
        img_dict["image_url"],
        img_dict["_id"]
    )
    
    await coleccion.insert_one(img_dict)
    return {"message": "Imagen en procesamiento"}

async def process_image_features(image_url: str, image_id: str):
    """Tarea en segundo plano para extraer características"""
    features = FeatureExtractor().extract(image_url)
    await coleccion.update_one(
        {"_id": ObjectId(image_id)},
        {"$set": {"features": features}}
    )

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int, image_id: Optional[str] = None):
    """
    Recomendaciones PERSONALIZADAS para un usuario.
    Si se proporciona image_id, recomienda basado en esa imagen específica.
    Si no, recomienda basado en el historial del usuario.
    """
    # Verificar que el usuario existe
    user_exists = await user.find_one({"user_id": user_id})
    if not user_exists:
        raise HTTPException(404, "Usuario no encontrado")
    
    if image_id:
        # Recomendación basada en una imagen específica
        return await get_image_based_recommendations(user_id, image_id)
    else:
        # Recomendación basada en el historial del usuario
        return await get_user_based_recommendations(user_id)

async def get_image_based_recommendations(user_id: int, image_id: str):
    """Recomendaciones basadas en una imagen específica (tu código original)"""
    similar_images = await recommender.find_similar(image_id)
    
    # Filtrar para excluir imágenes que el usuario ya ha visto
    user_viewed = await get_user_viewed_images(user_id)
    filtered_recommendations = [
        img for img in similar_images 
        if img["_id"] not in user_viewed
    ]
    
    return {
        "user_id": user_id,
        "based_on_image": image_id,
        "recommendations": filtered_recommendations[:10]
    }

async def get_user_based_recommendations(user_id: int):
    """Recomendaciones basadas en el historial completo del usuario"""
    # 1. Obtener imágenes con las que el usuario ha interactuado
    user_interactions = await coleccion.find({
        "interactions.user_id": user_id
    }).sort("interactions.last_interaction", -1).limit(5).to_list(None)
    
    if not user_interactions:
        # Cold start: imágenes populares
        return await get_popular_images()
    
    # 2. Generar recomendaciones basadas en su historial
    all_recommendations = []
    
    for interacted_image in user_interactions:
        # Recomendaciones por contenido similar
        content_recs = await recommender.find_similar(interacted_image["_id"])
        
        # Recomendaciones por comportamiento similar (grafos)
        graph_recs = await graph_recommender.recommend_for_user(user_id)
        
        all_recommendations.extend(content_recs + graph_recs)
    
    # 3. Filtrar y ordenar
    user_viewed = await get_user_viewed_images(user_id)
    unique_recommendations = [
        img for img in all_recommendations 
        if img["_id"] not in user_viewed
    ]
    
    # Ordenar por score (puedes implementar tu scoring system aquí)
    unique_recommendations.sort(
        key=lambda x: x.get("interactions", {}).get("likes", 0), 
        reverse=True
    )
    
    return {
        "user_id": user_id,
        "based_on_history": True,
        "recommendations": unique_recommendations[:15]
    }

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
security = HTTPBasic()


origins = [
    "*",
]


app.add_middleware(
    CORSMiddleware,
    
    
    
    allow_origins=[
        "https://5j6k3nm7-5173.use2.devtunnels.ms",  # Tu frontend
        "https://5j6k3nm7-5000.use2.devtunnels.ms",  # Tu backend
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

app.include_router(users)
app.include_router(galery)
app.include_router(category)

