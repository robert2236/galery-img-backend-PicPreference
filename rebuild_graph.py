# rebuild_graph.py
import asyncio
import networkx as nx
from database.databases import coleccion, user

class InteractionGraph:
    def __init__(self):
        self.graph = nx.Graph()
    
    async def build_from_db(self):
        """Reconstruye el grafo de interacciones desde cero"""
        try:
            print("🔄 Reconstruyendo grafo de interacciones desde la base de datos...")
            
            # Limpiar grafo existente
            self.graph.clear()
            
            # Obtener todas las imágenes con likes
            images_with_likes = await coleccion.find({
                "liked_by": {"$exists": True, "$ne": []}
            }).to_list(None)
            
            print(f"📊 Encontradas {len(images_with_likes)} imágenes con likes")
            
            total_connections = 0
            for img in images_with_likes:
                image_id = img.get("image_id")
                liked_by = img.get("liked_by", [])
                
                # Agregar nodo de imagen
                image_node = f"image_{image_id}"
                self.graph.add_node(image_node, type="image", image_id=image_id)
                
                # Agregar conexiones con usuarios
                for user_id in liked_by:
                    user_node = f"user_{user_id}"
                    
                    # Agregar nodo de usuario si no existe
                    if user_node not in self.graph:
                        self.graph.add_node(user_node, type="user", user_id=user_id)
                    
                    # Agregar arista con peso basado en likes
                    weight = img.get("interactions", {}).get("likes", 1)
                    self.graph.add_edge(user_node, image_node, weight=weight)
                    total_connections += 1
            
            print(f"✅ Grafo reconstruido exitosamente!")
            print(f"📈 Estadísticas:")
            print(f"   - Nodos: {self.graph.number_of_nodes()}")
            print(f"   - Aristas: {self.graph.number_of_edges()}")
            print(f"   - Conexiones totales: {total_connections}")
            
            # Información adicional
            user_nodes = [n for n in self.graph.nodes if n.startswith("user_")]
            image_nodes = [n for n in self.graph.nodes if n.startswith("image_")]
            
            print(f"   - Usuarios: {len(user_nodes)}")
            print(f"   - Imágenes: {len(image_nodes)}")
            
            # Mostrar usuarios con más interacciones
            if user_nodes:
                user_degrees = [(n, self.graph.degree[n]) for n in user_nodes]
                user_degrees.sort(key=lambda x: x[1], reverse=True)
                print(f"   - Usuario más activo: {user_degrees[0][0]} ({user_degrees[0][1]} interacciones)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reconstruyendo grafo: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Función principal para reconstruir el grafo"""
    print("🔧 RECONSTRUCTOR DE GRAFO DE INTERACCIONES")
    print("=" * 50)
    
    graph = InteractionGraph()
    success = await graph.build_from_db()
    
    if success:
        print("\n🎉 ¡Grafo reconstruido exitosamente!")
        print("\n📋 Para usar el grafo en tu aplicación, puedes:")
        print("   1. Copiar esta instancia a tu main.py")
        print("   2. O ejecutar este script periódicamente")
        print("   3. O integrar esta función en tu startup")
    else:
        print("\n❌ Error reconstruyendo el grafo")
    
    return graph

if __name__ == "__main__":
    # Ejecutar la reconstrucción
    graph = asyncio.run(main())
    
    # Opcional: guardar el grafo para debugging
    try:
        import pickle
        with open('interaction_graph.pkl', 'wb') as f:
            pickle.dump(graph.graph, f)
        print("💾 Grafo guardado en 'interaction_graph.pkl' para debugging")
    except Exception as e:
        print(f"⚠️ No se pudo guardar el grafo: {e}")