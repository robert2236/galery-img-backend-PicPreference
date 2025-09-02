# services/auxiliary.py
from bson import ObjectId
from database.databases import user, coleccion
from datetime import datetime

# Importar con manejo de errores
try:
    from services.recommender import visual_recommender
    VISUAL_RECOMMENDER_AVAILABLE = True
except ImportError:
    print("⚠️ visual_recommender no disponible en services.recommender")
    VISUAL_RECOMMENDER_AVAILABLE = False
    # Crear una clase dummy para evitar errores
    class DummyVisualRecommender:
        async def find_similar(self, *args, **kwargs):
            print("⚠️ visual_recommender no disponible, retornando lista vacía")
            return []
    visual_recommender = DummyVisualRecommender()

async def get_popular_images():
    """Obtiene imágenes populares"""
    try:
        popular_images = await coleccion.find(
            {"interactions.views": {"$gt": 0}}
        ).sort("interactions.likes", -1).limit(15).to_list(None)
        
        serializable_images = []
        for image in popular_images:
            if "image_id" in image:
                serializable_images.append({
                    "image_id": image["image_id"],
                    "title": image.get("title", "Sin título"),
                    "image_url": image.get("image_url", ""),
                    "interactions": image.get("interactions", {})
                })
            
        return serializable_images
    except Exception as e:
        print(f"Error obteniendo imágenes populares: {e}")
        return []

async def get_user_viewed_images(user_id: int):
    """Obtiene las IDs de las imágenes que el usuario ya ha visto"""
    try:
        user_data = await user.find_one({"user_id": user_id})
        if not user_data:
            return []
        
        return user_data.get("images", [])
    except Exception as e:
        print(f"Error obteniendo imágenes vistas: {e}")
        return []
    
async def get_content_based_recommendations(viewed_images: list, limit: int = 5):
    """Obtiene recomendaciones basadas en contenido similar"""
    recommendations = []
    
    if not VISUAL_RECOMMENDER_AVAILABLE:
        print("⚠️ Saltando recomendaciones de contenido - visual_recommender no disponible")
        return recommendations
    
    for img_id in viewed_images[-2:]:
        try:
            similar_ids = await visual_recommender.find_similar(str(img_id), k=limit)
            for similar_id in similar_ids:
                if similar_id not in viewed_images:
                    image = await coleccion.find_one({"image_id": similar_id})
                    if image:
                        recommendations.append({
                            "id": image["image_id"],
                            "title": image.get("title", "Sin título"),
                            "image_url": image.get("image_url", ""),
                            "score": 0.8,
                            "type": "content_based",
                            "source": f"similar_to_{img_id}"
                        })
        except Exception as e:
            print(f"⚠️ Error en contenido similar para {img_id}: {e}")
    
    return recommendations

async def get_graph_based_recommendations(user_id: int, viewed_images: list, graph_recommender, limit: int = 8):
    """Obtiene recomendaciones del grafo de interacciones"""
    recommendations = []
    
    try:
        if graph_recommender is None or not hasattr(graph_recommender, 'recommend_for_user'):
            print("❌ graph_recommender no disponible o no tiene el método recommend_for_user")
            return recommendations
            
        graph_recs = graph_recommender.recommend_for_user(user_id, k=limit * 2)
        
        for img_id, score in graph_recs:
            try:
                img_id_int = int(img_id)
                if img_id_int not in viewed_images:
                    image = await coleccion.find_one({"image_id": img_id_int})
                    if image:
                        recommendations.append({
                            "id": image["image_id"],
                            "title": image.get("title", "Sin título"),
                            "image_url": image.get("image_url", ""),
                            "score": float(score) * 0.7,
                            "type": "behavioral",
                            "source": "user_similarity"
                        })
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error procesando imagen ID {img_id}: {e}")
                
    except Exception as e:
        print(f"⚠️ Error en recomendaciones del grafo: {e}")
    
    return recommendations

async def get_popular_recommendations(viewed_images: list, limit: int = 5):
    """Obtiene recomendaciones populares excluyendo las ya vistas"""
    recommendations = []
    
    try:
        popular_images = await get_popular_images()
        
        for pop_img in popular_images[:limit * 2]:
            if pop_img["image_id"] not in viewed_images:
                recommendations.append({
                    "id": pop_img["image_id"],
                    "title": pop_img.get("title", "Sin título"),
                    "image_url": pop_img.get("image_url", ""),
                    "score": 0.5,
                    "type": "popular",
                    "source": "trending"
                })
    except Exception as e:
        print(f"⚠️ Error en recomendaciones populares: {e}")
    
    return recommendations

def remove_duplicates(recommendations: list):
    """Elimina recomendaciones duplicadas"""
    unique_recs = []
    seen_ids = set()
    
    for rec in recommendations:
        if rec["id"] not in seen_ids:
            seen_ids.add(rec["id"])
            unique_recs.append(rec)
    
    return unique_recs