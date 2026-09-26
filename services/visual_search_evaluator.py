"""
services/visual_search_evaluator.py - Evaluador de precisión de búsqueda visual

Evalúa la calidad del sistema de búsqueda visual usando:
- Leave-One-Out Cross-Validation
- Precision@K y Recall@K
- Mean Average Precision (MAP)
- Category-based accuracy
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualSearchEvaluator:
    """
    Evaluador del sistema de búsqueda visual por vectores.
    
    Mide la precisión usando la estrategia leave-one-out:
    1. Para cada imagen, se retira del índice
    2. Se busca imágenes similares
    3. Se verifica si las imágenes de la misma categoría aparecen en los resultados
    """
    
    def __init__(self, vector_store=None):
        """
        Args:
            vector_store: Instancia de VectorStore (None = auto-init)
        """
        self.vector_store = vector_store
    
    def _get_vector_store(self):
        """Obtiene la instancia de VectorStore."""
        if self.vector_store is None:
            from vector_store import get_vector_store
            self.vector_store = get_vector_store()
        return self.vector_store
    
    async def _get_images_with_categories(self, coleccion) -> List[Dict[str, Any]]:
        """Obtiene imágenes con sus categorías desde MongoDB."""
        try:
            cursor = coleccion.find(
                {},
                {"image_id": 1, "category": 1, "title": 1}
            )
            images = await cursor.to_list(length=None)
            return [
                {
                    "image_id": img.get("image_id"),
                    "category": img.get("category", "unknown"),
                    "title": img.get("title", "")
                }
                for img in images if img.get("image_id") is not None
            ]
        except Exception as e:
            logger.error(f"Error obteniendo imágenes: {e}")
            return []
    
    def _precision_at_k(
        self, 
        retrieved_ids: List[int], 
        relevant_ids: set, 
        k: int
    ) -> float:
        """Calcula Precision@K."""
        if k == 0:
            return 0.0
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for img_id in retrieved_at_k if img_id in relevant_ids)
        return relevant_retrieved / k
    
    def _recall_at_k(
        self, 
        retrieved_ids: List[int], 
        relevant_ids: set, 
        k: int
    ) -> float:
        """Calcula Recall@K."""
        if not relevant_ids:
            return 0.0
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for img_id in retrieved_at_k if img_id in relevant_ids)
        return relevant_retrieved / len(relevant_ids)
    
    def _average_precision(
        self, 
        retrieved_ids: List[int], 
        relevant_ids: set
    ) -> float:
        """Calcula Average Precision (AP) para una query."""
        if not relevant_ids:
            return 0.0
        
        hits = 0
        sum_precision = 0.0
        
        for i, img_id in enumerate(retrieved_ids):
            if img_id in relevant_ids:
                hits += 1
                sum_precision += hits / (i + 1)
        
        return sum_precision / len(relevant_ids) if relevant_ids else 0.0
    
    async def evaluate_leave_one_out(
        self,
        coleccion,
        k_values: List[int] = [5, 10, 20],
        sample_size: Optional[int] = None,
        min_images_per_category: int = 2
    ) -> Dict[str, Any]:
        """
        Evalúa el sistema usando leave-one-out cross-validation.
        
        Para cada imagen:
        1. Se busca en ChromaDB por su vector
        2. Se excluye a sí misma
        3. Se verifican los resultados por categoría
        
        Args:
            coleccion: Colección de MongoDB
            k_values: Valores de K para evaluar
            sample_size: Limitar número de imágenes a evaluar (None = todas)
            min_images_per_category: Mínimo de imágenes por categoría para evaluar
        
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            vector_store = self._get_vector_store()
            
            if not vector_store.health_check():
                return {"error": "VectorStore no disponible"}
            
            # Obtener imágenes con categorías
            images = await self._get_images_with_categories(coleccion)
            
            if not images:
                return {"error": "No hay imágenes para evaluar"}
            
            # Agrupar por categoría
            category_images = defaultdict(list)
            for img in images:
                category_images[img["category"]].append(img["image_id"])
            
            # Filtrar categorías con suficientes imágenes
            valid_categories = {
                cat: ids for cat, ids in category_images.items()
                if len(ids) >= min_images_per_category
            }
            
            # Seleccionar imágenes para evaluar
            eval_images = []
            for cat, ids in valid_categories.items():
                for img_id in ids:
                    eval_images.append({"image_id": img_id, "category": cat})
            
            if sample_size and sample_size < len(eval_images):
                eval_images = eval_images[:sample_size]
            
            logger.info(f"📊 Evaluando {len(eval_images)} imágenes en {len(valid_categories)} categorías")
            
            # Inicializar métricas
            metrics = {k: {"precisions": [], "recalls": [], "aps": []} for k in k_values}
            category_metrics = defaultdict(lambda: {k: {"precisions": [], "recalls": []} for k in k_values})
            
            # Evaluar cada imagen
            evaluated = 0
            errors = 0
            
            for eval_item in eval_images:
                image_id = eval_item["image_id"]
                category = eval_item["category"]
                
                try:
                    # Obtener vector de la imagen
                    embedding = vector_store.get_embedding(image_id)
                    if embedding is None:
                        continue
                    
                    # Buscar imágenes similares (excluyendo la misma)
                    max_k = max(k_values)
                    results = vector_store.search_similar(
                        query_vector=embedding["vector"],
                        k=max_k + 1,  # +1 para compensar la exclusión
                        exclude_image_id=image_id,
                        min_score=0.0
                    )
                    
                    # IDs recuperados
                    retrieved_ids = [r["image_id"] for r in results]
                    
                    # IDs relevantes (misma categoría, excluyendo la consulta)
                    relevant_ids = set(category_images[category]) - {image_id}
                    
                    if not relevant_ids:
                        continue
                    
                    # Calcular métricas para cada K
                    for k in k_values:
                        p_at_k = self._precision_at_k(retrieved_ids, relevant_ids, k)
                        r_at_k = self._recall_at_k(retrieved_ids, relevant_ids, k)
                        ap = self._average_precision(retrieved_ids, relevant_ids)
                        
                        metrics[k]["precisions"].append(p_at_k)
                        metrics[k]["recalls"].append(r_at_k)
                        metrics[k]["aps"].append(ap)
                        
                        category_metrics[category][k]["precisions"].append(p_at_k)
                        category_metrics[category][k]["recalls"].append(r_at_k)
                    
                    evaluated += 1
                    
                except Exception as e:
                    errors += 1
                    logger.debug(f"Error evaluando imagen {image_id}: {e}")
                    continue
            
            # Calcular métricas agregadas
            result = {
                "total_images_evaluated": evaluated,
                "total_errors": errors,
                "total_images_available": len(images),
                "valid_categories": len(valid_categories),
                "category_distribution": {cat: len(ids) for cat, ids in valid_categories.items()},
                "metrics_by_k": {},
                "category_breakdown": {}
            }
            
            for k in k_values:
                if metrics[k]["precisions"]:
                    result["metrics_by_k"][f"P@{k}"] = round(
                        np.mean(metrics[k]["precisions"]), 4
                    )
                    result["metrics_by_k"][f"R@{k}"] = round(
                        np.mean(metrics[k]["recalls"]), 4
                    )
                    result["metrics_by_k"][f"MAP@{k}"] = round(
                        np.mean(metrics[k]["aps"]), 4
                    )
            
            # Breakdown por categoría (top 10)
            for cat in list(valid_categories.keys())[:10]:
                if category_metrics[cat][k_values[0]]["precisions"]:
                    result["category_breakdown"][cat] = {
                        f"P@{k_values[0]}": round(
                            np.mean(category_metrics[cat][k_values[0]]["precisions"]), 4
                        ),
                        "count": len(valid_categories[cat])
                    }
            
            # Calcular accuracy general (>80% target)
            if metrics[k_values[0]]["precisions"]:
                avg_precision = np.mean(metrics[k_values[0]]["precisions"])
                result["overall_accuracy"] = round(avg_precision, 4)
                result["accuracy_target_met"] = avg_precision >= 0.80
            
            logger.info(f"✅ Evaluación completada: {evaluated} imágenes procesadas")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en evaluación: {e}")
            return {"error": str(e)}
    
    async def evaluate_single_image(
        self,
        coleccion,
        image_id: int,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Evalúa la búsqueda visual para una imagen específica.
        
        Returns:
            Diccionario con resultados detallados
        """
        try:
            vector_store = self._get_vector_store()
            
            # Obtener imagen de referencia
            image_doc = await coleccion.find_one({"image_id": image_id})
            if not image_doc:
                return {"error": f"Imagen {image_id} no encontrada"}
            
            category = image_doc.get("category", "unknown")
            
            # Obtener vector
            embedding = vector_store.get_embedding(image_id)
            if embedding is None:
                return {"error": f"Imagen {image_id} sin embedding"}
            
            # Buscar similares
            results = vector_store.search_similar(
                query_vector=embedding["vector"],
                k=k + 1,
                exclude_image_id=image_id,
                min_score=0.0
            )
            
            # Verificar categorías
            retrieved_ids = [r["image_id"] for r in results[:k]]
            
            # Obtener categorías de resultados
            result_categories = []
            same_category_count = 0
            
            for r in results[:k]:
                r_doc = await coleccion.find_one({"image_id": r["image_id"]})
                r_cat = r_doc.get("category", "unknown") if r_doc else "unknown"
                result_categories.append({
                    "image_id": r["image_id"],
                    "category": r_cat,
                    "similarity": r["similarity_score"],
                    "is_same_category": r_cat == category
                })
                if r_cat == category:
                    same_category_count += 1
            
            precision = same_category_count / k if k > 0 else 0
            
            return {
                "image_id": image_id,
                "query_category": category,
                "k": k,
                "precision": round(precision, 4),
                "same_category_count": same_category_count,
                "total_results": len(result_categories),
                "results": result_categories,
                "accuracy_target_met": precision >= 0.80
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluando imagen {image_id}: {e}")
            return {"error": str(e)}
