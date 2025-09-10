from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from database.databases import coleccion, user
import networkx as nx

class RecommenderEvaluator:
    def __init__(self, k=10):
        self.k = k  # Top-K recomendaciones a evaluar
    
    async def get_all_interactions(self):
        """Obtiene todas las interacciones de la base de datos"""
        try:
            # Obtener todas las imágenes con sus interacciones
            cursor = coleccion.find({})
            images = await cursor.to_list(length=None)
            
            interactions = []
            for image in images:
                image_id = image.get("image_id")
                if not image_id:
                    continue
                    
                # Obtener usuarios que dieron like (de las interacciones)
                likes = []
                interactions_data = image.get("interactions", {})
                
                # Si tienes un campo específico para likes
                if "likes" in interactions_data and isinstance(interactions_data["likes"], list):
                    likes = interactions_data["likes"]
                # O si tienes un campo liked_by
                elif "liked_by" in interactions_data and isinstance(interactions_data["liked_by"], list):
                    likes = interactions_data["liked_by"]
                
                if likes:
                    interactions.append({
                        "image_id": image_id,
                        "liked_by": likes
                    })
            
            return interactions
            
        except Exception as e:
            print(f"❌ Error obteniendo interacciones: {e}")
            return []
    
    async def train_test_split(self, test_size=0.2):
        """Divide las interacciones en train y test por usuario"""
        try:
            interactions = await self.get_all_interactions()
            
            # Reorganizar datos por usuario
            user_interactions = {}
            for interaction in interactions:
                image_id = interaction["image_id"]
                for user_id in interaction["liked_by"]:
                    if user_id not in user_interactions:
                        user_interactions[user_id] = []
                    user_interactions[user_id].append(image_id)
            
            # Dividir por usuario
            train_data = []
            test_data = []
            
            for user_id, image_ids in user_interactions.items():
                if len(image_ids) > 1:  # Solo usuarios con múltiples interacciones
                    split_idx = int(len(image_ids) * (1 - test_size))
                    train_images = image_ids[:split_idx]
                    test_images = image_ids[split_idx:]
                    
                    # Agregar a train data
                    for img_id in train_images:
                        train_data.append({"user_id": user_id, "image_id": img_id})
                    
                    # Agregar a test data
                    for img_id in test_images:
                        test_data.append({"user_id": user_id, "image_id": img_id})
            
            return train_data, test_data
            
        except Exception as e:
            print(f"❌ Error en train_test_split: {e}")
            return [], []
    
    async def evaluate_recommendations(self, user_id, recommendations, true_positives):
        """Evalúa las recomendaciones para un usuario específico"""
        try:
            if not recommendations or not true_positives:
                return 0, 0, 0, 0
            
            # Convertir a sets para comparación
            rec_set = set(recommendations[:self.k])  # Top-K recomendaciones
            true_set = set(true_positives)  # Imágenes que realmente le gustaron
            
            if not rec_set:
                return 0, 0, 0, 0
            
            # Calcular métricas
            y_true = [1 if img_id in true_set else 0 for img_id in rec_set]
            y_pred = [1] * len(y_true)  # Todas las recomendadas son positivas según el modelo
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calcular hits (recomendaciones correctas)
            hits = len(rec_set.intersection(true_set))
            
            return precision, recall, f1, hits
            
        except Exception as e:
            print(f"❌ Error en evaluate_recommendations para usuario {user_id}: {e}")
            return 0, 0, 0, 0
    
    async def evaluate_user(self, user_id, graph_recommender, test_data):
        """Evalúa las recomendaciones para un usuario específico"""
        try:
            # Obtener imágenes de test para este usuario
            user_test_images = [item["image_id"] for item in test_data if item["user_id"] == user_id]
            
            if not user_test_images:
                return None
            
            # Obtener recomendaciones
            recommendations = graph_recommender.recommend_for_user(user_id, k=self.k)
            rec_ids = [int(img_id) for img_id, score in recommendations] if recommendations else []
            
            # Evaluar
            precision, recall, f1, hits = await self.evaluate_recommendations(
                user_id, rec_ids, user_test_images
            )
            
            return {
                "user_id": user_id,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "hits": hits,
                "total_recommendations": len(rec_ids),
                "total_positives": len(user_test_images),
                "recommendations": rec_ids,
                "actual_likes": user_test_images
            }
            
        except Exception as e:
            print(f"❌ Error evaluando usuario {user_id}: {e}")
            return None
    
    async def evaluate_all_users(self, graph_recommender):
        """Evalúa el sistema para todos los usuarios"""
        try:
            # Dividir datos
            train_data, test_data = await self.train_test_split()
            
            if not train_data or not test_data:
                print("⚠️ No hay suficientes datos para evaluación")
                return []
            
            # Reconstruir grafo solo con datos de entrenamiento
            original_graph = graph_recommender.graph.copy()
            
            # Construir grafo de entrenamiento
            graph_recommender.graph = nx.Graph()
            
            # Agregar nodos y aristas de entrenamiento
            for interaction in train_data:
                user_id = interaction["user_id"]
                image_id = interaction["image_id"]
                
                user_node = f"user_{user_id}"
                image_node = f"image_{image_id}"
                
                graph_recommender.graph.add_node(user_node, type="user")
                graph_recommender.graph.add_node(image_node, type="image")
                graph_recommender.graph.add_edge(user_node, image_node, weight=1)
            
            # Obtener usuarios únicos en test
            test_users = list(set([item["user_id"] for item in test_data]))
            
            # Evaluar para cada usuario en test
            results = []
            for user_id in test_users:
                result = await self.evaluate_user(user_id, graph_recommender, test_data)
                if result:
                    results.append(result)
            
            # Restaurar grafo original
            graph_recommender.graph = original_graph
            
            return results
            
        except Exception as e:
            print(f"❌ Error en evaluate_all_users: {e}")
            return []
    
    async def calculate_metrics(self, results):
        """Calcula métricas agregadas"""
        if not results:
            return {
                "avg_precision": 0,
                "avg_recall": 0,
                "avg_f1_score": 0,
                "total_hits": 0,
                "total_recommendations": 0,
                "hit_rate": 0,
                "num_users_evaluated": 0
            }
        
        # Filtrar usuarios con recomendaciones
        valid_results = [r for r in results if r["total_recommendations"] > 0]
        
        if not valid_results:
            return {
                "avg_precision": 0,
                "avg_recall": 0,
                "avg_f1_score": 0,
                "total_hits": 0,
                "total_recommendations": 0,
                "hit_rate": 0,
                "num_users_evaluated": 0
            }
        
        avg_precision = np.mean([r["precision"] for r in valid_results])
        avg_recall = np.mean([r["recall"] for r in valid_results])
        avg_f1 = np.mean([r["f1_score"] for r in valid_results])
        total_hits = sum([r["hits"] for r in valid_results])
        total_recommendations = sum([r["total_recommendations"] for r in valid_results])
        
        return {
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1_score": round(avg_f1, 4),
            "total_hits": total_hits,
            "total_recommendations": total_recommendations,
            "hit_rate": round(total_hits / total_recommendations, 4) if total_recommendations > 0 else 0,
            "num_users_evaluated": len(valid_results)
        }

    async def evaluate_single_user(self, user_id, graph_recommender):
        """Evalúa las recomendaciones para un solo usuario"""
        try:
            # Obtener todas las imágenes que le gustaron al usuario
            user_data = await user.find_one({"user_id": user_id})
            if not user_data:
                print(f"❌ Usuario {user_id} no encontrado")
                return None
            
            # Obtener los likes del usuario - depende de tu estructura de datos
            true_positives = []
            if "liked_images" in user_data:
                true_positives = user_data.get("liked_images", [])
            elif "interactions" in user_data:
                # Si los likes están en interactions
                interactions = user_data.get("interactions", {})
                true_positives = interactions.get("likes", [])
            
            print(f"📊 Usuario {user_id} tiene {len(true_positives)} likes")
            
            # Obtener recomendaciones - ¡AGREGAR AWAIT!
            recommendations = await graph_recommender.recommend_for_user(user_id, k=self.k)
            rec_ids = [int(img_id) for img_id, score in recommendations] if recommendations else []
            
            print(f"🎯 Recomendaciones generadas: {rec_ids}")
            
            # Evaluar
            precision, recall, f1, hits = await self.evaluate_recommendations(
                user_id, rec_ids, true_positives
            )
            
            return {
                "user_id": user_id,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "hits": hits,
                "total_recommendations": len(rec_ids),
                "total_positives": len(true_positives),
                "recommendations": rec_ids,
                "actual_likes": true_positives
            }
            
        except Exception as e:
            print(f"❌ Error evaluando usuario individual {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return None