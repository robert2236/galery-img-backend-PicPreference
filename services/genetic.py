# services/genetic.py (versión corregida)
from deap import base, creator, tools, algorithms
import random
import numpy as np
from database.databases import coleccion

async def optimize_weights():
    """Optimiza pesos basado en interacciones reales de la BD - VERSIÓN CORREGIDA"""
    try:
        # 1. Obtener datos reales de la base de datos
        feedback_data = await coleccion.aggregate([
            {"$match": {"interactions.views": {"$gt": 0}}},
            {"$project": {
                "likes": {"$ifNull": ["$interactions.likes", 0]},
                "downloads": {"$ifNull": ["$interactions.downloads", 0]}, 
                "views": {"$ifNull": ["$interactions.views", 0]},
                "recency_score": {
                    "$cond": [
                        {"$gt": ["$interactions.last_interaction", None]},
                        1.0,  # Reciente
                        0.0   # Antiguo
                    ]
                }
            }}
        ]).to_list(None)

        if not feedback_data:
            return {"likes": 0.4, "downloads": 0.3, "views": 0.2, "recency": 0.1}

        # Limpiar tipos de creator existentes
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
            
        # 2. Configurar algoritmo genético CORREGIDO
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        
        # Asegurar que los pesos sean positivos y sumen ~1
        def create_individual():
            weights = [random.uniform(0.1, 0.8) for _ in range(4)]
            total = sum(weights)
            return [w/total for w in weights]
        
        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=50)

        def evaluate(individual):
            """Función de evaluación MEJORADA"""
            likes_weight, downloads_weight, views_weight, recency_weight = individual
            
            # Calcular score para cada imagen
            total_score = 0
            for data in feedback_data:
                # Score basado en engagement
                engagement_score = (
                    data['likes'] * likes_weight +
                    data['downloads'] * downloads_weight +
                    data['views'] * views_weight
                )
                
                # Score de recencia
                recency_score = data['recency_score'] * recency_weight
                
                # Score total para esta imagen
                total_score += engagement_score + recency_score
            
            return (total_score,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 3. Ejecutar algoritmo genético
        population = toolbox.population()
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, verbose=False)
        
        # 4. Obtener el mejor individuo
        best_individual = tools.selBest(population, k=1)[0]
        
        # Asegurar que todos los pesos sean positivos
        best_individual = [max(0.1, w) for w in best_individual]
        
        # Normalizar para que sumen 1
        total = sum(best_individual)
        normalized_weights = [w/total for w in best_individual]
        
        return {
            "likes": normalized_weights[0],
            "downloads": normalized_weights[1],
            "views": normalized_weights[2],
            "recency": normalized_weights[3]
        }
        
    except Exception as e:
        print(f"Error en optimización genética: {e}")
        return {"likes": 0.4, "downloads": 0.3, "views": 0.2, "recency": 0.1}