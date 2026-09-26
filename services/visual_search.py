"""
services/visual_search.py - Servicio único de búsqueda visual por similitud.

Fuente de verdad para TODOS los endpoints de "imágenes similares":
  - GET /api/v1/recommendations/visual-similar/{image_id}
  - GET /api/v1/recommendations/visual-similar-hybrid/{image_id}
  - GET /recommend/similar/{image_id}
  - GET /recommend/{user_id}?image_id=...

Antes cada endpoint tenía su propia implementación (ChromaDB, KDTree, KDTree sin
índice) con resultados y scores distintos. Ahora todos devuelven la misma
similitud coseno real, con el mismo filtro de relevancia.

Score de similitud
------------------
El score es el coseno real entre embeddings: `1 - distance` de ChromaDB.
Con ResNet50 (2048d) el corpus entero vive en 0.39-0.76, así que un umbral
absoluto de 0.8 no se cumple nunca. Por eso el filtro combina:

    score >= min_score                              (piso absoluto)
    score >= best_score * min_relative              (ratio al mejor vecino)
    score >= best_score - max_gap                   (distancia al mejor vecino)

Si nada supera el umbral se devuelve una lista vacía con `below_threshold=True`
y el mejor candidato en `closest_match`, para que el frontend pueda informar
"no hay coincidencias similares" sin mostrar recommendations irrelevantes.
"""

import logging
from typing import Any, Dict, List, Optional

from database.databases import coleccion
from vector_store import (
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_RELATIVE,
    DEFAULT_MIN_SCORE,
    DimensionMismatchError,
    ModelMismatchError,
    VectorStoreError,
    get_vector_store,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageNotFoundError(Exception):
    """La imagen no existe en MongoDB."""


async def find_similar_images(
    image_id: int,
    limit: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
    min_relative: Optional[float] = DEFAULT_MIN_RELATIVE,
    max_gap: Optional[float] = DEFAULT_MAX_GAP,
    category: Optional[str] = None,
    vector_store=None,
) -> Dict[str, Any]:
    """
    Busca imágenes visualmente similares y las enriquece con MongoDB.

    Args:
        image_id: ID numérico de la imagen de referencia
        limit: Máximo de resultados a devolver
        min_score: Piso absoluto de similitud coseno
        min_relative: Ratio mínimo respecto al mejor vecino (None = desactivado)
        max_gap: Diferencia máxima respecto al mejor vecino (None = desactivado)
        category: Si se indica, solo se devuelven imágenes de esa categoría
        vector_store: Instancia a usar (por defecto, la global)

    Returns:
        Dict con la imagen original, los resultados, el mejor candidato y los
        umbrales aplicados.

    Raises:
        ImageNotFoundError: la imagen no existe en MongoDB
        DimensionMismatchError / ModelMismatchError: el índice está corrupto
            o mezcla modelos de embeddings
        VectorStoreError: ChromaDB no disponible
    """
    store = vector_store or get_vector_store()

    original_image = await coleccion.find_one({"image_id": int(image_id)})
    if not original_image:
        raise ImageNotFoundError(f"Imagen con ID {image_id} no encontrada")

    if not store.health_check():
        raise VectorStoreError("Servicio de búsqueda vectorial no disponible")

    embedding = store.get_embedding(int(image_id))
    if embedding is None:
        return {
            "image": _format_image(original_image),
            "results": [],
            "closest": None,
            "below_threshold": True,
            "has_embedding": False,
            "total_candidates": 0,
            "thresholds": _thresholds(None, min_score, min_relative, max_gap),
            "total_embeddings": store.collection.count(),
            "embedding_model": store.model_name
        }

    outcome = store.search_similar_detailed(
        query_vector=embedding["vector"],
        k=limit,
        exclude_image_id=int(image_id),
        min_score=min_score,
        min_relative=min_relative,
        max_gap=max_gap
    )

    results = await _hydrate(
        outcome["results"],
        category=category,
        limit=limit
    )
    closest = await _hydrate_single(outcome["closest"])

    return {
        "image": _format_image(original_image),
        "results": results,
        "closest": closest,
        "below_threshold": outcome["below_threshold"],
        "has_embedding": True,
        "total_candidates": outcome["candidates_evaluated"],
        "thresholds": outcome["thresholds"],
        "total_embeddings": store.collection.count(),
        "embedding_model": store.model_name
    }


async def find_similar_images_hybrid(
    image_id: int,
    limit: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
    min_relative: Optional[float] = DEFAULT_MIN_RELATIVE,
    max_gap: Optional[float] = DEFAULT_MAX_GAP,
    category_boost: float = 0.15,
    popularity_weight: float = 0.10,
    vector_store=None
) -> Dict[str, Any]:
    """
    Búsqueda híbrida: similitud visual + boost de categoría + popularidad.

    El filtro de relevancia se aplica sobre la similitud visual (antes del
    re-ranking) porque los boosts no son comparables con un coseno.
    """
    store = vector_store or get_vector_store()

    original_image = await coleccion.find_one({"image_id": int(image_id)})
    if not original_image:
        raise ImageNotFoundError(f"Imagen con ID {image_id} no encontrada")

    if not store.health_check():
        raise VectorStoreError("Servicio de búsqueda vectorial no disponible")

    embedding = store.get_embedding(int(image_id))
    if embedding is None:
        return {
            "image": _format_image(original_image),
            "results": [],
            "closest": None,
            "below_threshold": True,
            "has_embedding": False,
            "total_candidates": 0,
            "thresholds": _thresholds(None, min_score, min_relative, max_gap),
            "total_embeddings": store.collection.count(),
            "embedding_model": store.model_name
        }

    popularity_scores = await _calculate_popularity_scores()

    outcome = store.search_hybrid_detailed(
        query_vector=embedding["vector"],
        k=limit,
        exclude_image_id=int(image_id),
        min_score=min_score,
        category=original_image.get("category", "unknown"),
        category_boost=category_boost,
        popularity_scores=popularity_scores,
        popularity_weight=popularity_weight,
        min_relative=min_relative,
        max_gap=max_gap
    )

    results = await _hydrate(outcome["results"], category=None, limit=limit)

    return {
        "image": _format_image(original_image),
        "results": results,
        "closest": await _hydrate_single(outcome["closest"]),
        "below_threshold": outcome["below_threshold"],
        "has_embedding": True,
        "total_candidates": outcome["candidates_evaluated"],
        "thresholds": outcome["thresholds"],
        "total_embeddings": store.collection.count(),
        "embedding_model": store.model_name
    }


async def _hydrate(
    vector_results: List[Dict[str, Any]],
    category: Optional[str],
    limit: int
) -> List[Dict[str, Any]]:
    """ Enriquece resultados de ChromaDB con los datos de MongoDB. """
    if not vector_results:
        return []

    # Una sola consulta en vez de N (evita el patrón N+1)
    image_ids = [r["image_id"] for r in vector_results]
    cursor = coleccion.find({"image_id": {"$in": image_ids}})
    docs = {doc["image_id"]: doc for doc in await cursor.to_list(length=None)}

    hydrated = []
    for result in vector_results:
        doc = docs.get(result["image_id"])
        if not doc:
            logger.debug(f"Embedding sin documento en MongoDB: {result['image_id']}")
            continue

        if category and doc.get("category") != category:
            continue

        item = _format_image(doc)
        item.update({
            "similarity_score": result["similarity_score"],
            "relative_score": result.get("relative_score"),
            "gap_to_best": result.get("gap_to_best"),
            "rank": result.get("rank"),
            "final_score": result.get("final_score"),
            "category_boost": result.get("category_boost"),
            "popularity_boost": result.get("popularity_boost")
        })
        item = {k: v for k, v in item.items() if v is not None}
        hydrated.append(item)

        if len(hydrated) >= limit:
            break

    return hydrated


async def _hydrate_single(vector_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """ Enriquece un único resultado (el mejor candidato, aunque no pase el filtro). """
    if not vector_result:
        return None

    doc = await coleccion.find_one({"image_id": vector_result["image_id"]})
    if not doc:
        return {
            "image_id": vector_result["image_id"],
            "similarity_score": vector_result["similarity_score"]
        }

    item = _format_image(doc)
    item["similarity_score"] = vector_result["similarity_score"]
    return item


def _thresholds(best, min_score, min_relative, max_gap) -> Dict[str, Any]:
    return {
        "min_score": min_score,
        "min_relative": min_relative,
        "max_gap": max_gap,
        "best_score": round(best, 4) if best is not None else None,
        "effective_threshold": None
    }


def _format_image(image_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Formatea un documento de MongoDB para la respuesta de la API."""
    return {
        "image_id": image_doc.get("image_id"),
        "title": image_doc.get("title", "Sin título"),
        "image_url": image_doc.get("image_url", ""),
        "category": image_doc.get("category", "unknown"),
        "username": image_doc.get("username"),
        "likes": image_doc.get("interactions", {}).get("likes", 0),
        "views": image_doc.get("interactions", {}).get("views", 0),
        "downloads": image_doc.get("interactions", {}).get("downloads", 0),
        "ai_features": {
            "auto_tags": image_doc.get("ai_features", {}).get("auto_tags", []),
            "scene_type": image_doc.get("ai_features", {}).get("scene_type")
        }
    }


async def _calculate_popularity_scores() -> Dict[int, float]:
    """Calcula scores de popularidad normalizados a [0, 1] para todas las imágenes."""
    try:
        cursor = coleccion.find(
            {},
            {"image_id": 1, "interactions.likes": 1, "interactions.views": 1}
        )
        images = await cursor.to_list(length=None)

        if not images:
            return {}

        pop_data = []
        for img in images:
            interactions = img.get("interactions", {})
            score = interactions.get("likes", 0) * 2 + interactions.get("views", 0) * 0.1
            pop_data.append((img["image_id"], score))

        max_score = max((s for _, s in pop_data), default=0) or 1.0
        return {img_id: score / max_score for img_id, score in pop_data}

    except Exception as e:
        logger.debug(f"Error calculando popularidad: {e}")
        return {}
