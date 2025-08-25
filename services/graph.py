# services/graph.py
import networkx as nx
from database.databases import coleccion

class InteractionGraph:
    def __init__(self):
        self.graph = nx.Graph()
    
    async def build_from_db(self):
        """Grafo usuario-imagen basado en likes reales"""
        # Obtener imágenes con likes
        images_with_likes = await coleccion.find({
            "interactions.likes": {"$gt": 0}
        }).to_list(None)
        
        for img in images_with_likes:
            # Agregar nodos y aristas para cada like
            for user_id in img.get("liked_by", []):
                self.graph.add_edge(
                    f"user_{user_id}",
                    f"image_{img['image_id']}",
                    weight=img["interactions"]["likes"]  # Peso basado en likes
                )
    
    def recommend_for_user(self, user_id, k=5):
        """Recomienda imágenes que han gustado a usuarios con tastes similares"""
        try:
            user_node = f"user_{user_id}"
            
            # Encontrar usuarios similares (que hayan likeado las mismas imágenes)
            similar_users = []
            for neighbor in self.graph.neighbors(user_node):
                if neighbor.startswith("user_"):
                    similar_users.append(neighbor)
            
            # Recomendar imágenes de usuarios similares
            recommendations = []
            for sim_user in similar_users:
                for image_node in self.graph.neighbors(sim_user):
                    if image_node.startswith("image_"):
                        image_id = image_node.replace("image_", "")
                        score = self.graph[sim_user][image_node]["weight"]
                        recommendations.append((image_id, score))
            
            return sorted(set(recommendations), key=lambda x: x[1], reverse=True)[:k]
            
        except nx.NetworkXError:
            return []