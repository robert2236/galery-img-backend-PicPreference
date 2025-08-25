from deap import base, creator, tools, algorithms
import random
import numpy as np
from database.databases import coleccion

async def optimize_weights():
    """Optimiza pesos basado en interacciones reales de la BD"""
    # 1. Obtener datos reales de la base de datos
    feedback_data = await coleccion.aggregate([
        {"$match": {"interactions.views": {"$gt": 0}}},
        {"$project": {
            "likes": "$interactions.likes",
            "downloads": "$interactions.downloads", 
            "views": "$interactions.views",
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
        return [0.4, 0.3, 0.2, 0.1]  # Valores por defecto

    # 2. Configurar algoritmo genético
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=30)

    def evaluate(individual):
        weights = {
            'likes': individual[0],
            'downloads': individual[1],
            'views': individual[2],
            'recency': individual[3]
        }
        return (calculate_fitness(weights, feedback_data),)

def calculate_fitness(weights, feedback):
    """Calcula qué tan buenos son los pesos según feedback histórico"""
    # Implementa tu métrica de evaluación aquí
    return np.corrcoef(weights, feedback)[0, 1]