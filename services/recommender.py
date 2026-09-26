"""
services/recommender.py - Compatibilidad del recomendador visual.

Este módulo antes implementaba un KDTree propio sobre `features` de MongoDB.
Se eliminó por dos problemas reales:

1. Distancia euclidiana sobre vectores ResNet50 sin normalizar (normas de 26-69):
   el resultado dependía de la magnitud del vector, no de su dirección, así que
   los "similares" eran prácticamente aleatorios.
2.KDTree no devolvía score, por lo que era imposible aplicar un umbral.

Ahora delega en el VectorStore (ChromaDB, coseno real) y mantiene la misma API
pública para no romper los llamadores existentes.
"""

import logging
from typing import Any, Dict, List, Optional

from vector_store import (
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_RELATIVE,
    DEFAULT_MIN_SCORE,
    VectorStoreError,
    get_vector_store,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualRecommender:
    """
    Fachada del recomendador visual basado en ChromaDB.

    Mantiene los nombres de método históricos (build_index / find_similar) para
    no romper los imports, pero toda la lógica vive en vector_store.py, que es
    el único punto de verdad del sistema de similitud.
    """

    def __init__(self):
        self._store = None

    # --- Estado (propiedades Historically usadas por /health y /system-status) ---
    @property
    def store(self):
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    @property
    def ready(self) -> bool:
        """True si el índice vectorial está disponible."""
        try:
            return self.store.collection.count() > 0
        except VectorStoreError:
            return False
        except Exception as e:
            logger.debug(f"No se pudo verificar el índice vectorial: {e}")
            return False

    @property
    def image_ids(self) -> List[int]:
        """IDs indexados (compatibilidad con el reporte de estado)."""
        try:
            return self.store.get_all_ids()
        except Exception as e:
            logger.debug(f"No se pudieron obtener los ids indexados: {e}")
            return []

    @property
    def feature_size(self) -> int:
        return self.store.feature_size

    async def build_index(self) -> bool:
        """
        Ya no hay índice en memoria que construir: ChromaDB mantiene el índice
        HNSW en disco. Se verifica que esté sano y con vectores.
        """
        try:
            store = self.store
            count = store.collection.count()
            logger.info(
                f"✅ Índice vectorial listo: {count} embeddings "
                f"(model={store.model_name}, dim={store.feature_size})"
            )
            return count > 0
        except Exception as e:
            logger.error(f"❌ Índice vectorial no disponible: {e}")
            return False

    async def find_similar(
        self,
        image_id,
        k: int = 5,
        min_score: float = DEFAULT_MIN_SCORE,
        min_relative: Optional[float] = DEFAULT_MIN_RELATIVE,
        max_gap: Optional[float] = DEFAULT_MAX_GAP
    ) -> List[int]:
        """
        Devuelve los IDs de las imágenes similares (contrato histórico).

        Para obtener el score de similitud usar
        `services.visual_search.find_similar_images`.
        """
        try:
            embedding = self.store.get_embedding(int(image_id))
            if embedding is None:
                logger.warning(f"⚠️ Imagen {image_id} sin embedding vectorial")
                return []

            results = self.store.search_similar(
                query_vector=embedding["vector"],
                k=k,
                exclude_image_id=int(image_id),
                min_score=min_score,
                min_relative=min_relative,
                max_gap=max_gap
            )
            return [r["image_id"] for r in results]

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"❌ Error buscando similares para {image_id}: {e}")
            raise


visual_recommender = VisualRecommender()
