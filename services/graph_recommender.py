# services/graph_recommender.py
import networkx as nx
from database.databases import coleccion

class GraphRecommender:
    def __init__(self):
        self.graph = nx.Graph()
    
    async def build_graph(self):
        """Construye el grafo de interacciones desde la base de datos"""
        try:
            self.graph.clear()
            
            # Obtener todas las imágenes con interacciones
            cursor = coleccion.find({})
            images = await cursor.to_list(length=None)
            
            for image in images:
                image_id = image.get("image_id")
                if not image_id:
                    continue
                
                # Nodo de imagen
                image_node = f"image_{image_id}"
                self.graph.add_node(image_node, type="image", image_id=image_id)
                
                # Agregar usuarios que interactuaron con la imagen
                liked_by = image.get("liked_by", [])
                for user_id in liked_by:
                    user_node = f"user_{user_id}"
                    self.graph.add_node(user_node, type="user", user_id=user_id)
                    self.graph.add_edge(user_node, image_node, weight=1, interaction="like")
            
            print(f"✅ Grafo construido: {self.graph.number_of_nodes()} nodos, {self.graph.number_of_edges()} aristas")
            
        except Exception as e:
            print(f"❌ Error construyendo grafo: {e}")
    
    def recommend_for_user(self, user_id, k=10):
        """Genera recomendaciones basadas en el grafo"""
        try:
            user_node = f"user_{user_id}"
            
            if user_node not in self.graph:
                print(f"⚠️ Usuario {user_id} no encontrado en el grafo")
                return []
            
            # Encontrar imágenes likeadas por usuarios similares
            recommendations = {}
            
            # Buscar usuarios similares (que hayan likeado las mismas imágenes)
            user_neighbors = list(self.graph.neighbors(user_node))
            user_image_nodes = [n for n in user_neighbors if n.startswith("image_")]
            
            for image_node in user_image_nodes:
                # Otros usuarios que likearon la misma imagen
                other_users = [n for n in self.graph.neighbors(image_node) if n.startswith("user_") and n != user_node]
                
                for other_user in other_users:
                    # Imágenes likeadas por estos usuarios similares
                    other_user_images = [n for n in self.graph.neighbors(other_user) if n.startswith("image_")]
                    
                    for other_image in other_user_images:
                        image_id = int(other_image.split("_")[1])
                        if image_id not in recommendations:
                            recommendations[image_id] = 0
                        recommendations[image_id] += 1  # Peso basado en coincidencias
            
            # Ordenar por peso y devolver top-K
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:k]
            
        except Exception as e:
            print(f"❌ Error en recommend_for_user: {e}")
            return []

# Instancia global del graph recommender
graph_recommender = GraphRecommender()