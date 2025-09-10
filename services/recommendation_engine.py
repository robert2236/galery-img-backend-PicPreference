# services/recommendation_engine.py
from datetime import datetime, timedelta
from .auxiliary import (
    get_popular_images, 
    get_user_viewed_images,
    get_content_based_recommendations,
    get_graph_based_recommendations,
    get_popular_recommendations,
    remove_duplicates
)
from .graph_recommender import graph_recommender  # Importar la instancia

class RecommendationEngine:
    def __init__(self):
        self.graph_recommender = graph_recommender
    
    async def initialize(self):
        """Inicializar el motor (construir el grafo)"""
        await self.graph_recommender.build_graph()
        print("✅ Recommendation Engine inicializado")
    
    async def get_recommendations(self, user_id: int, limit: int = 10):
        """Sistema principal de recomendación"""
        try:
            viewed_images = await get_user_viewed_images(user_id)
            print(f"📊 Usuario {user_id} ha interactuado con {len(viewed_images)} imágenes")
            
            all_recommendations = []
            
            # 1. Recomendaciones del grafo
            try:
                graph_recs = await get_graph_based_recommendations(
                    user_id, viewed_images, self.graph_recommender, limit
                )
                all_recommendations.extend(graph_recs)
                print(f"📈 Recomendaciones del grafo: {len(graph_recs)}")
            except Exception as e:
                print(f"⚠️ Error en recomendaciones del grafo: {e}")
            
            # 2. Recomendaciones de contenido
            if viewed_images:
                try:
                    content_recs = await get_content_based_recommendations(viewed_images, limit)
                    all_recommendations.extend(content_recs)
                    print(f"🎨 Recomendaciones de contenido: {len(content_recs)}")
                except Exception as e:
                    print(f"⚠️ Error en recomendaciones de contenido: {e}")
            
            # 3. Recomendaciones populares
            try:
                popular_recs = await get_popular_recommendations(viewed_images, limit)
                all_recommendations.extend(popular_recs)
                print(f"🔥 Recomendaciones populares: {len(popular_recs)}")
            except Exception as e:
                print(f"⚠️ Error en recomendaciones populares: {e}")
            
            # 4. Procesar resultados finales
            unique_recommendations = remove_duplicates(all_recommendations)
            unique_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            final_recommendations = [
                rec for rec in unique_recommendations 
                if rec['id'] not in viewed_images
            ][:limit]
            
            print(f"✅ Recomendaciones finales: {len(final_recommendations)}")
            return final_recommendations
            
        except Exception as e:
            print(f"❌ Error crítico: {e}")
            # Fallback a populares
            popular = await get_popular_recommendations([], limit)
            return popular[:limit]

# Instancia global
recommendation_engine = RecommendationEngine()