# services/graph.py
import networkx as nx
from database.databases import coleccion, user

class InteractionGraph:
    def __init__(self):
        self.graph = nx.Graph()
    
    async def build_from_db(self):
        """Construye el grafo desde la base de datos (async)"""
        try:
            # Limpiar grafo existente
            self.graph.clear()
            
            # Obtener todas las imágenes con sus interacciones
            cursor = coleccion.find({})
            images = await cursor.to_list(length=None)
            
            print(f"📊 Procesando {len(images)} imágenes para construir grafo...")
            
            for image in images:
                image_id = image.get("image_id")
                liked_by = image.get("liked_by", [])
                
                # Añadir nodo de imagen
                image_node = f"image_{image_id}"
                self.graph.add_node(image_node, type="image")
                
                # Añadir conexiones entre usuarios e imágenes
                for user_id in liked_by:
                    user_node = f"user_{user_id}"
                    self.graph.add_node(user_node, type="user")
                    self.graph.add_edge(user_node, image_node, weight=1)
                    print(f"   ➕ Añadida arista: {user_node} -> {image_node}")
            
            print(f"✅ Grafo construido: {self.graph.number_of_nodes()} nodos, {self.graph.number_of_edges()} aristas")
            
            # Debug: mostrar estadísticas del grafo
            user_nodes = [n for n in self.graph.nodes() if isinstance(n, str) and n.startswith('user_')]
            image_nodes = [n for n in self.graph.nodes() if isinstance(n, str) and n.startswith('image_')]
            
            print(f"   👥 Nodos de usuario: {len(user_nodes)}")
            print(f"   📷 Nodos de imagen: {len(image_nodes)}")
            print(f"   🔗 Total de aristas: {self.graph.number_of_edges()}")
            
        except Exception as e:
            print(f"❌ Error construyendo grafo: {e}")
            raise
    
    async def recommend_for_user(self, user_id, k=5):
        """Recomienda imágenes que han gustado a usuarios con tastes similares"""
        try:
            user_node = f"user_{user_id}"
            print(f"🎯 Iniciando recomendaciones para usuario {user_id} (nodo: {user_node})")
            
            # Verificar si el usuario existe en el grafo
            if user_node not in self.graph:
                available_users = [n for n in self.graph.nodes() if isinstance(n, str) and n.startswith('user_')]
                print(f"⚠️ Usuario {user_id} no encontrado en el grafo.")
                print(f"   Nodos de usuario disponibles: {available_users[:10]}")
                return []
            
            print(f"✅ Usuario {user_node} encontrado en grafo")
            
            # Obtener vecinos del usuario (imágenes que ha likeado)
            user_neighbors = list(self.graph.neighbors(user_node))
            print(f"   📌 Imágenes likeadas por el usuario: {user_neighbors}")
            
            # Encontrar usuarios similares (que hayan likeado las mismas imágenes)
            similar_users = []
            for image_node in user_neighbors:
                if isinstance(image_node, str) and image_node.startswith("image_"):
                    # Otros usuarios que likearon esta misma imagen
                    for other_user in self.graph.neighbors(image_node):
                        if isinstance(other_user, str) and other_user.startswith("user_") and other_user != user_node:
                            similar_users.append(other_user)
            
            similar_users = list(set(similar_users))  # Eliminar duplicados
            
            print(f"👥 Usuarios similares encontrados: {len(similar_users)} - {similar_users}")
            
            if not similar_users:
                print(f"⚠️ No se encontraron usuarios similares para {user_id}")
                # Debug: ver por qué no hay usuarios similares
                for neighbor in self.graph.neighbors(user_node):
                    print(f"   Vecino: {neighbor} (tipo: {type(neighbor)})")
                return []
            
            # Obtener imágenes que el usuario YA ha likeado (usando el grafo en lugar de la BD)
            # Esto es crucial para la evaluación, donde el grafo solo contiene datos de entrenamiento
            user_liked_images = set()
            for neighbor in self.graph.neighbors(user_node):
                if isinstance(neighbor, str) and neighbor.startswith("image_"):
                    image_id = neighbor.replace("image_", "")
                    user_liked_images.add(str(image_id))
            
            print(f"❤️  Imágenes likeadas por usuario {user_id}: {list(user_liked_images)}")
            
            # Recomendar imágenes de usuarios similares
            recommendations = {}
            for sim_user in similar_users:
                print(f"🔍 Revisando usuario similar: {sim_user}")
                sim_user_neighbors = list(self.graph.neighbors(sim_user))
                print(f"   📷 Imágenes likeadas por {sim_user}: {sim_user_neighbors}")
                
                for image_node in sim_user_neighbors:
                    if isinstance(image_node, str) and image_node.startswith("image_"):
                        image_id = image_node.replace("image_", "")
                        
                        # Verificar si el usuario YA likeó esta imagen
                        if image_id not in user_liked_images:
                            score = self.graph[sim_user][image_node].get("weight", 1)
                            if image_id not in recommendations:
                                recommendations[image_id] = 0
                            recommendations[image_id] += score
                            print(f"   ✅ Recomendación añadida/actualizada: {image_id} (score acumulado: {recommendations[image_id]})")
            
            print(f"📋 Total de recomendaciones encontradas: {len(recommendations)}")
            print(f"📝 Lista de recomendaciones: {recommendations}")
            
            # Ordenar por score (mayor primero) y tomar top k
            result = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]
            print(f"🎯 {len(result)} recomendaciones finales para usuario {user_id}: {result}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en recommend_for_user: {e}")
            import traceback
            traceback.print_exc()
            return []
    async def add_test_likes(self):
        """Método para agregar likes de prueba"""
        try:
            print("➕ Agregando likes de prueba...")
            
            # Obtener algunas imágenes
            images = await coleccion.find().to_list(length=5)
            test_user_ids = [1, 2, 3]  # IDs de prueba
            
            for i, img in enumerate(images):
                image_id = img.get("image_id")
                if image_id:
                    await coleccion.update_one(
                        {"image_id": image_id},
                        {"$set": {"liked_by": test_user_ids[:i+1]}}
                    )
                    print(f"✅ Imagen {image_id}: likes de usuarios {test_user_ids[:i+1]}")
            
            print("🎉 Likes de prueba agregados exitosamente!")
            
        except Exception as e:
            print(f"❌ Error agregando likes de prueba: {e}")
            import traceback
            traceback.print_exc()
    async def get_fallback_recommendations(self, user_id: str, limit: int):
        """Recomendaciones de fallback cuando el grafo no tiene datos"""
        try:
            # Obtener imágenes populares como fallback
            popular_images = await coleccion.find(
                {"interactions.views": {"$gt": 0}}
            ).sort("interactions.likes", -1).limit(limit).to_list(None)
            
            recommendations = []
            for image in popular_images:
                image_id = image.get("image_id")
                if image_id:
                    recommendations.append((str(image_id), 1.0))  # Score básico
            
            print(f"🔄 Usando {len(recommendations)} recomendaciones de fallback")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error en fallback: {e}")
            return []
# Crear la instancia global
graph_recommender = InteractionGraph()