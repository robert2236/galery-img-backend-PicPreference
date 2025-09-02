# services/graph.py
import networkx as nx
from database.databases import coleccion

class InteractionGraph:
    def __init__(self):
        self.graph = nx.Graph()
    
    async def build_from_db(self):
        """Grafo usuario-imagen basado en likes reales"""
        try:
            print("🔄 Construyendo grafo de interacciones desde la base de datos...")
            
            # Obtener imágenes con likes
            images_with_likes = await coleccion.find({
                "liked_by": {"$exists": True, "$ne": []}
            }).to_list(None)
            
            print(f"📊 Encontradas {len(images_with_likes)} imágenes con likes")
            
            for img in images_with_likes:
                image_id = img.get("image_id")
                liked_by = img.get("liked_by", [])
                
                # Agregar nodos y aristas para cada like
                for user_id in liked_by:
                    self.graph.add_edge(
                        f"user_{user_id}",
                        f"image_{image_id}",
                        weight=img.get("interactions", {}).get("likes", 1)
                    )
            
            print(f"✅ Grafo construido: {self.graph.number_of_nodes()} nodos, {self.graph.number_of_edges()} aristas")
            
        except Exception as e:
            print(f"❌ Error construyendo grafo desde BD: {e}")
    
    def recommend_for_user(self, user_id, k=5):
        """Recomienda imágenes que han gustado a usuarios con tastes similares"""
        try:
            user_node = f"user_{user_id}"
            
            # Verificar si el usuario existe en el grafo
            if user_node not in self.graph:
                print(f"⚠️ Usuario {user_id} no encontrado en el grafo")
                return []
            
            # Encontrar usuarios similares (que hayan likeado las mismas imágenes)
            similar_users = []
            for neighbor in self.graph.neighbors(user_node):
                if neighbor.startswith("user_"):
                    similar_users.append(neighbor)
            
            if not similar_users:
                print(f"⚠️ No se encontraron usuarios similares para {user_id}")
                return []
            
            # Recomendar imágenes de usuarios similares
            recommendations = []
            for sim_user in similar_users:
                for image_node in self.graph.neighbors(sim_user):
                    if image_node.startswith("image_"):
                        image_id = image_node.replace("image_", "")
                        score = self.graph[sim_user][image_node].get("weight", 1)
                        recommendations.append((image_id, score))
            
            # Ordenar por score y eliminar duplicados
            unique_recommendations = []
            seen_ids = set()
            
            for img_id, score in recommendations:
                if img_id not in seen_ids:
                    seen_ids.add(img_id)
                    unique_recommendations.append((img_id, score))
            
            return sorted(unique_recommendations, key=lambda x: x[1], reverse=True)[:k]
            
        except Exception as e:
            print(f"❌ Error en recommend_for_user: {e}")
            return []

# Crear la instancia global
graph_recommender = InteractionGraph()