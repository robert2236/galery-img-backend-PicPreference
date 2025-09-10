from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import numpy as np
from database.databases import coleccion, user

class RecommenderEvaluator:
    def __init__(self, k=10):
        self.k = k  # Top-K recomendaciones a evaluar
    
    async def train_test_split(self, test_size=0.2):
        """Divide las interacciones en train y test"""
        try:
            # Obtener todas las interacciones
            cursor = coleccion.find({})
            images = await cursor.to_list(length=None)
            
            train_data = []
            test_data = []
            
            for image in images:
                image_id = image.get("image_id")
                liked_by = image.get("liked_by", [])
                
                # Dividir los likes de cada imagen
                if len(liked_by) > 1:
                    split_idx = int(len(liked_by) * (1 - test_size))
                    train_likes = liked_by[:split_idx]
                    test_likes = liked_by[split_idx:]
                    
                    train_data.append({"image_id": image_id, "liked_by": train_likes})
                    test_data.append({"image_id": image_id, "liked_by": test_likes})
            
            return train_data, test_data
            
        except Exception as e:
            print(f"❌ Error en train_test_split: {e}")
            return [], []
    
    async def evaluate_recommendations(self, user_id, recommendations, true_positives):
        """Evalúa las recomendaciones para un usuario específico"""
        try:
            # Convertir a sets para comparación
            rec_set = set(recommendations[:self.k])  # Top-K recomendaciones
            true_set = set(true_positives)  # Imágenes que realmente le gustaron
            
            # Calcular métricas
            y_true = [1 if img_id in true_set else 0 for img_id in rec_set]
            y_pred = [1] * len(y_true)  # Todas las recomendadas son positivas según el modelo
            
            if not y_true:  # Evitar división por cero
                return 0, 0, 0, 0
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calcular hits (recomendaciones correctas)
            hits = len(rec_set.intersection(true_set))
            
            return precision, recall, f1, hits
            
        except Exception as e:
            print(f"❌ Error en evaluate_recommendations: {e}")
            return 0, 0, 0, 0
    
    async def evaluate_all_users(self, graph_recommender):
        """Evalúa el sistema para todos los usuarios"""
        try:
            # Dividir datos
            train_data, test_data = await self.train_test_split()
            
            # Reconstruir grafo solo con datos de entrenamiento
            original_graph = graph_recommender.graph.copy()
            
            # Construir grafo de entrenamiento
            graph_recommender.graph.clear()
            for image in train_data:
                image_id = image.get("image_id")
                liked_by = image.get("liked_by", [])
                
                image_node = f"image_{image_id}"
                graph_recommender.graph.add_node(image_node, type="image")
                
                for user_id in liked_by:
                    user_node = f"user_{user_id}"
                    graph_recommender.graph.add_node(user_node, type="user")
                    graph_recommender.graph.add_edge(user_node, image_node, weight=1)
            
            # Evaluar para cada usuario en test
            results = []
            for image in test_data:
                for user_id in image["liked_by"]:
                    try:
                        # Obtener recomendaciones
                        recommendations = await graph_recommender.recommend_for_user(user_id, k=self.k)
                        rec_ids = [img_id for img_id, score in recommendations]
                        
                        # Obtener ground truth (lo que realmente le gustó en test)
                        true_positives = []
                        for test_img in test_data:
                            if user_id in test_img["liked_by"]:
                                true_positives.append(test_img["image_id"])
                        
                        # Evaluar
                        precision, recall, f1, hits = await self.evaluate_recommendations(
                            user_id, rec_ids, true_positives
                        )
                        
                        results.append({
                            "user_id": user_id,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "hits": hits,
                            "total_recommendations": len(rec_ids),
                            "total_positives": len(true_positives)
                        })
                        
                    except Exception as e:
                        print(f"❌ Error evaluando usuario {user_id}: {e}")
                        continue
            
            # Restaurar grafo original
            graph_recommender.graph = original_graph
            
            return results
            
        except Exception as e:
            print(f"❌ Error en evaluate_all_users: {e}")
            return []
    
    async def calculate_metrics(self, results):
        """Calcula métricas agregadas"""
        if not results:
            return {}
        
        avg_precision = np.mean([r["precision"] for r in results])
        avg_recall = np.mean([r["recall"] for r in results])
        avg_f1 = np.mean([r["f1_score"] for r in results])
        total_hits = sum([r["hits"] for r in results])
        total_recommendations = sum([r["total_recommendations"] for r in results])
        
        return {
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1_score": round(avg_f1, 4),
            "total_hits": total_hits,
            "total_recommendations": total_recommendations,
            "hit_rate": round(total_hits / total_recommendations, 4) if total_recommendations > 0 else 0,
            "num_users_evaluated": len(results)
        }