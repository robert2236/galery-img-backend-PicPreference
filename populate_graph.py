# populate_graph.py
import asyncio
from services.graph import InteractionGraph
from database.databases import coleccion

async def populate_graph_with_interactions():
    """Poblar el grafo con interacciones existentes"""
    graph = InteractionGraph()
    
    # Obtener todas las imágenes con likes
    images = await coleccion.find({"liked_by": {"$exists": True, "$ne": []}}).to_list(None)
    
    print(f"📊 Encontradas {len(images)} imágenes con likes")
    
    for image in images:
        image_id = image.get("image_id")
        liked_by = image.get("liked_by", [])
        
        for user_id in liked_by:
            graph.graph.add_edge(
                f"user_{user_id}",
                f"image_{image_id}",
                weight=image.get("interactions", {}).get("likes", 1)
            )
    
    print(f"✅ Grafo poblado con {graph.graph.number_of_edges()} conexiones")
    return graph

if __name__ == "__main__":
    asyncio.run(populate_graph_with_interactions())