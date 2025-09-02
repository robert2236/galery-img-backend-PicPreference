import asyncio
from services.graph import graph_recommender

async def test_graph_instance():
    """Prueba que la instancia global funcione"""
    print("🧪 Probando instancia global graph_recommender...")
    
    # 1. Verificar que la instancia existe
    print(f"✅ Instancia graph_recommender: {type(graph_recommender)}")
    
    # 2. Verificar que tiene los métodos
    print(f"✅ Tiene build_from_db: {hasattr(graph_recommender, 'build_from_db')}")
    print(f"✅ Tiene recommend_for_user: {hasattr(graph_recommender, 'recommend_for_user')}")
    
    # 3. Construir el grafo
    await graph_recommender.build_from_db()
    
    # 4. Probar el método
    if hasattr(graph_recommender, 'recommend_for_user'):
        # Probar con un usuario de prueba
        recommendations = graph_recommender.recommend_for_user(1, 3)
        print(f"✅ recommend_for_user funciona: {recommendations}")

if __name__ == "__main__":
    asyncio.run(test_graph_instance())