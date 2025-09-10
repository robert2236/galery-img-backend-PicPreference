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

async def get_popular_images(limit=15, min_likes=1):
    """Obtiene imágenes populares con mejor filtrado"""
    try:
        pipeline = [
            {"$match": {"interactions.likes": {"$gte": min_likes}}},
            {"$addFields": {
                "popularity_score": {
                    "$add": [
                        {"$multiply": ["$interactions.likes", 2]},
                        "$interactions.views",
                        {"$cond": [{"$gt": ["$interactions.saves", 0]}, 5, 0]}
                    ]
                }
            }},
            {"$sort": {"popularity_score": -1}},
            {"$limit": limit},
            {"$project": {
                "image_id": 1,
                "title": 1,
                "image_url": 1,
                "interactions": 1,
                "popularity_score": 1,
                "comments": 1  # Incluir comentarios en la proyección
            }}
        ]
        
        popular_images = await coleccion.aggregate(pipeline).to_list(None)
        
        serializable_images = []
        for image in popular_images:
            # Procesar comentarios para asegurar formato consistente
            comments = image.get("comments", [])
            processed_comments = []
            
            for comment in comments:
                processed_comments.append({
                    "comment_id": comment.get("comment_id"),
                    "user_id": comment.get("user_id"),
                    "comment": comment.get("comment"),
                    "created_at": comment.get("created_at"),
                    "parent_comment_id": comment.get("parent_comment_id"),
                    "likes": comment.get("likes", 0)
                })
            
            serializable_images.append({
                "image_id": image["image_id"],
                "title": image.get("title", "Sin título"),
                "image_url": image.get("image_url", ""),
                "interactions": image.get("interactions", {}),
                "popularity_score": image.get("popularity_score", 0),
                "comments": processed_comments  # Comentarios procesados
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
                        # Procesar comentarios de la imagen
                        comments = image.get("comments", [])
                        processed_comments = []
                        
                        for comment in comments:
                            processed_comments.append({
                                "comment_id": comment.get("comment_id"),
                                "user_id": comment.get("user_id"),
                                "comment": comment.get("comment"),
                                "created_at": comment.get("created_at"),
                                "parent_comment_id": comment.get("parent_comment_id"),
                                "likes": comment.get("likes", 0)
                            })
                        
                        recommendations.append({
                            "image_id": image["image_id"],
                            "title": image.get("title", "Sin título"),
                            "image_url": image.get("image_url", ""),
                            "score": 0.8,
                            "type": "content_based",
                            "source": f"similar_to_{img_id}",
                            "comments": processed_comments  # Comentarios procesados
                        })
        except Exception as e:
            print(f"⚠️ Error en contenido similar para {img_id}: {e}")
    
    return recommendations

async def get_graph_based_recommendations(user_id: int, viewed_images: list, graph_recommender, limit: int = 8):
    """Obtiene recomendaciones del grafo de interacciones"""
    recommendations = []
    
    try:
        if graph_recommender is None:
            print("❌ Graph recommender no disponible")
            return recommendations
            
        graph_recs = graph_recommender.recommend_for_user(user_id, k=limit * 2)
        
        for img_id, score in graph_recs:
            try:
                img_id_int = int(img_id)
                if img_id_int not in viewed_images:
                    image = await coleccion.find_one({"image_id": img_id_int})
                    if image:
                        # Procesar comentarios de la imagen
                        comments = image.get("comments", [])
                        processed_comments = []
                        
                        for comment in comments:
                            processed_comments.append({
                                "comment_id": comment.get("comment_id"),
                                "user_id": comment.get("user_id"),
                                "comment": comment.get("comment"),
                                "created_at": comment.get("created_at"),
                                "parent_comment_id": comment.get("parent_comment_id"),
                                "likes": comment.get("likes", 0)
                            })
                        
                        recommendations.append({
                            "image_id": image["image_id"],
                            "title": image.get("title", "Sin título"),
                            "image_url": image.get("image_url", ""),
                            "score": float(score) * 0.7,
                            "type": "behavioral",
                            "source": "user_similarity",
                            "comments": processed_comments  # Comentarios procesados
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
                    "image_id": pop_img["image_id"],
                    "title": pop_img.get("title", "Sin título"),
                    "image_url": pop_img.get("image_url", ""),
                    "score": 0.5,
                    "type": "popular",
                    "source": "trending",
                    "comments": pop_img.get("comments", [])  # Comentarios ya procesados
                })
    except Exception as e:
        print(f"⚠️ Error en recomendaciones populares: {e}")
    
    return recommendations

def remove_duplicates(recommendations: list):
    """Elimina recomendaciones duplicadas"""
    unique_recs = []
    seen_ids = set()
    
    for rec in recommendations:
        if rec["image_id"] not in seen_ids:
            seen_ids.add(rec["image_id"])
            unique_recs.append(rec)
    
    return unique_recs

# =============================================================================
# FUNCIONES ADICIONALES PARA MANEJO DE COMENTARIOS
# =============================================================================

async def add_comment_to_image(image_id: int, user_id: str, comment_text: str, parent_comment_id=None):
    """Añade un comentario a una imagen"""
    try:
        # Generar un ID único para el comentario
        comment_id = int(datetime.now().timestamp())
        
        new_comment = {
            "comment_id": comment_id,
            "user_id": user_id,
            "comment": comment_text,
            "created_at": datetime.now(),
            "parent_comment_id": parent_comment_id,
            "likes": 0
        }
        
        # Actualizar la imagen con el nuevo comentario
        result = await coleccion.update_one(
            {"image_id": image_id},
            {"$push": {"comments": new_comment}}
        )
        
        if result.modified_count > 0:
            return new_comment
        else:
            print(f"⚠️ No se pudo agregar comentario a la imagen {image_id}")
            return None
            
    except Exception as e:
        print(f"Error añadiendo comentario: {e}")
        return None

async def get_image_comments(image_id: int, limit: int = 50):
    """Obtiene los comentarios de una imagen específica"""
    try:
        image = await coleccion.find_one(
            {"image_id": image_id},
            {"comments": 1, "_id": 0}
        )
        
        if image and "comments" in image:
            # Ordenar comentarios por fecha (más recientes primero)
            comments = sorted(
                image["comments"], 
                key=lambda x: x.get("created_at", datetime.min), 
                reverse=True
            )
            return comments[:limit]
        
        return []
        
    except Exception as e:
        print(f"Error obteniendo comentarios: {e}")
        return []

async def like_comment(image_id: int, comment_id: int):
    """Incrementa el contador de likes de un comentario"""
    try:
        result = await coleccion.update_one(
            {"image_id": image_id, "comments.comment_id": comment_id},
            {"$inc": {"comments.$.likes": 1}}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error dando like al comentario: {e}")
        return False