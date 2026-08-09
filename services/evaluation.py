from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from database.databases import coleccion, user
from collections import defaultdict
import random

class RecommenderEvaluator:
    def __init__(self, k=10):
        self.k = k

    async def train_test_split(self, test_size=0.2, seed=42):
        """Divide las interacciones por-usuario: 80% train, 20% test"""
        try:
            rng = random.Random(seed)

            cursor = coleccion.find({})
            images = await cursor.to_list(length=None)

            # Colectar por usuario: {user_id: [image_id, ...]}
            user_likes = defaultdict(list)
            for image in images:
                image_id = image.get("image_id")

                liked_by = []
                if "liked_by" in image and isinstance(image["liked_by"], list) and image["liked_by"]:
                    liked_by = image["liked_by"]

                for uid in liked_by:
                    user_likes[uid].append(image_id)

            train_data = []
            test_data = []

            for uid, liked_images in user_likes.items():
                if len(liked_images) < 2:
                    train_data.append({"user_id": uid, "image_ids": liked_images})
                    continue

                rng.shuffle(liked_images)
                split_idx = max(1, int(len(liked_images) * (1 - test_size)))
                train_ids = liked_images[:split_idx]
                test_ids = liked_images[split_idx:]

                train_data.append({"user_id": uid, "image_ids": train_ids})
                test_data.append({"user_id": uid, "image_ids": test_ids})

            return train_data, test_data

        except Exception as e:
            print(f"[EVAL] Error en train_test_split: {e}")
            return [], []

    async def evaluate_recommendations(self, user_id, recommendations, true_positives):
        try:
            rec_set = set(recommendations[:self.k])
            true_set = set(true_positives)

            if not rec_set:
                return 0, 0, 0, 0

            y_true = [1 if img_id in true_set else 0 for img_id in rec_set]
            y_pred = [1] * len(y_true)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            hits = len(rec_set.intersection(true_set))

            return precision, recall, f1, hits

        except Exception as e:
            print(f"[EVAL] Error en evaluate_recommendations: {e}")
            return 0, 0, 0, 0

    async def evaluate_all_users(self, graph_recommender):
        try:
            train_data, test_data = await self.train_test_split()

            original_graph = graph_recommender.graph.copy()

            # Construir grafo de entrenamiento con TODOS los usuarios
            # (cada usuario aporta sus likes de train, asi siempre tienen aristas)
            graph_recommender.graph.clear()

            # Mapa rapido: image_id -> set de user_ids en train
            image_train_users = defaultdict(set)
            for user_entry in train_data:
                uid = user_entry["user_id"]
                for img_id in user_entry["image_ids"]:
                    image_train_users[img_id].add(uid)

            # Construir grafo desde el mapa
            for img_id, uids in image_train_users.items():
                image_node = f"image_{img_id}"
                graph_recommender.graph.add_node(image_node, type="image")
                for uid in uids:
                    user_node = f"user_{uid}"
                    graph_recommender.graph.add_node(user_node, type="user")
                    graph_recommender.graph.add_edge(user_node, image_node, weight=1)

            # Mapa de ground truth por usuario (test)
            test_ground_truth = defaultdict(list)
            for user_entry in test_data:
                uid = user_entry["user_id"]
                test_ground_truth[uid].extend(user_entry["image_ids"])

            # Evaluar solo usuarios que tienen likes en test
            results = []
            for uid, true_positives in test_ground_truth.items():
                try:
                    recommendations = await graph_recommender.recommend_for_user(uid, k=self.k)
                    rec_ids = [img_id for img_id, score in recommendations]

                    precision, recall, f1, hits = await self.evaluate_recommendations(
                        uid, rec_ids, true_positives
                    )

                    results.append({
                        "user_id": uid,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "hits": hits,
                        "total_recommendations": len(rec_ids),
                        "total_positives": len(true_positives)
                    })

                except Exception as e:
                    print(f"[EVAL] Error evaluando usuario {uid}: {e}")
                    continue

            # Restaurar grafo original
            graph_recommender.graph = original_graph

            return results

        except Exception as e:
            print(f"[EVAL] Error en evaluate_all_users: {e}")
            return []

    async def calculate_metrics(self, results):
        if not results:
            return {}

        avg_precision = np.mean([r["precision"] for r in results])
        avg_recall = np.mean([r["recall"] for r in results])
        avg_f1 = np.mean([r["f1_score"] for r in results])
        total_hits = sum([r["hits"] for r in results])
        total_recommendations = sum([r["total_recommendations"] for r in results])
        total_positives = sum([r["total_positives"] for r in results])

        return {
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1_score": round(avg_f1, 4),
            "total_hits": total_hits,
            "total_recommendations": total_recommendations,
            "total_positives": total_positives,
            "hit_rate": round(total_hits / total_recommendations, 4) if total_recommendations > 0 else 0,
            "recall_rate": round(total_hits / total_positives, 4) if total_positives > 0 else 0,
            "num_users_evaluated": len(results)
        }
