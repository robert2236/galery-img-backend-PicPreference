# services/simple_evaluator.py
from database.databases import user, coleccion

class SimpleEvaluator:
    def __init__(self, k=5):
        self.k = k
    
    async def get_user_likes(self, user_id: int):
        """Obtiene los likes del usuario desde la colección de imágenes"""
        try:
            # Buscar todas las imágenes que este usuario ha likeado
            liked_images_cursor = coleccion.find({
                "liked_by": user_id
            })
            
            liked_images_docs = await liked_images_cursor.to_list(length=None)
            liked_images = [img["image_id"] for img in liked_images_docs if "image_id" in img]
            
            print(f"📊 Usuario {user_id} tiene {len(liked_images)} likes (encontrados en liked_by)")
            return liked_images
            
        except Exception as e:
            print(f"❌ Error obteniendo likes desde liked_by: {e}")
            return []
    
    async def get_user_likes_alternative(self, user_id: int):
        """Método alternativo para buscar likes en diferentes campos"""
        try:
            # Intentar diferentes campos donde podrían estar los likes
            possible_fields = ["liked_by", "interactions.likes", "likes", "liked_users"]
            
            for field in possible_fields:
                try:
                    if "." in field:
                        # Para campos anidados como "interactions.likes"
                        parent, child = field.split(".")
                        query = {parent: {child: user_id}}
                    else:
                        # Para campos directos
                        query = {field: user_id}
                    
                    liked_images_cursor = coleccion.find(query)
                    liked_images_docs = await liked_images_cursor.to_list(length=None)
                    
                    if liked_images_docs:
                        liked_images = [img["image_id"] for img in liked_images_docs if "image_id" in img]
                        print(f"✅ Likes encontrados en campo '{field}': {len(liked_images)}")
                        return liked_images
                        
                except Exception as e:
                    print(f"⚠️ Error buscando en campo {field}: {e}")
                    continue
            
            print("❌ No se encontraron likes en ningún campo")
            return []
            
        except Exception as e:
            print(f"❌ Error en búsqueda alternativa: {e}")
            return []
    
    async def evaluate_user(self, user_id: int):
        """Evaluación simple basada en liked_by"""
        try:
            # Obtener likes del usuario desde liked_by
            true_positives = await self.get_user_likes(user_id)
            
            # Si no encuentra en liked_by, intentar método alternativo
            if not true_positives:
                true_positives = await self.get_user_likes_alternative(user_id)
            
            if not true_positives:
                return {
                    "user_id": user_id,
                    "status": "no_data",
                    "message": "El usuario no tiene likes registrados en ningún campo",
                    "recommendations": await self.get_fallback_recommendations(),
                    "advice": "El usuario necesita interactuar con más imágenes"
                }
            
            print(f"🎯 Likes encontrados para evaluación: {true_positives}")
            
            # Obtener recomendaciones (imágenes populares excluyendo las ya likes)
            popular_images = await coleccion.find({
                "image_id": {"$nin": true_positives}
            }).sort("interactions.likes", -1).limit(self.k * 3).to_list(None)
            
            recommendations = [img["image_id"] for img in popular_images if "image_id" in img][:self.k]
            
            # Calcular métricas
            hits = len(set(recommendations).intersection(set(true_positives)))
            precision = hits / len(recommendations) if recommendations else 0
            recall = hits / len(true_positives) if true_positives else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "user_id": user_id,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "hits": hits,
                "total_recommendations": len(recommendations),
                "total_positives": len(true_positives),
                "recommendations": recommendations,
                "actual_likes": true_positives,
                "status": "success",
                "data_source": "liked_by"
            }
            
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            import traceback
            traceback.print_exc()
            return {
                "user_id": user_id,
                "status": "error",
                "error": str(e)
            }
    
    async def get_fallback_recommendations(self):
        """Recomendaciones de fallback cuando no hay datos"""
        popular_images = await coleccion.find().sort("interactions.likes", -1).limit(self.k).to_list(None)
        return [img["image_id"] for img in popular_images if "image_id" in img]