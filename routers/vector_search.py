"""
routers/vector_search.py - Router para búsqueda visual por vectores

Endpoint para buscar imágenes visualmente similares usando ChromaDB.

Arquitectura no-destructiva: No modifica endpoints existentes de MongoDB.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos del proyecto
try:
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
    from services.visual_search import (
        ImageNotFoundError,
        find_similar_images,
        find_similar_images_hybrid,
    )
except ImportError as e:
    logger.error(f"❌ Error importando módulos: {e}")
    raise

# Crear router
router = APIRouter(
    prefix="/api/v1/recommendations",
    tags=["Vector Search"],
    responses={
        404: {"description": "Imagen no encontrada"},
        503: {"description": "Servicio de vectores no disponible"}
    }
)

# Instancia global de VectorStore (lazy initialization)
_vector_store = None


def get_vector_store_instance():
    """Obtiene la instancia de VectorStore (lazy init)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
    return _vector_store


def _handle_visual_error(e: Exception, image_id: int) -> HTTPException:
    """
    Traduce los errores del vector store a respuestas HTTP explicitas.

    Antes estos errores se convertían en `similar_images: []` con HTTP 200,
    lo que hacía imposible distinguir "no hay coincidencias" de "el índice
    está roto".
    """
    if isinstance(e, ImageNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, (DimensionMismatchError, ModelMismatchError)):
        logger.error(f"❌ Índice vectorial inconsistente: {e}")
        return HTTPException(
            status_code=503,
            detail=f"Índice vectorial inconsistente: {e}"
        )
    if isinstance(e, VectorStoreError):
        logger.error(f"❌ Servicio de vectores no disponible: {e}")
        return HTTPException(status_code=503, detail=str(e))
    logger.error(f"❌ Error en búsqueda visual para image_id={image_id}: {e}")
    return HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get(
    "/visual-similar/{image_id}",
    response_model=Dict[str, Any],
    summary="Buscar imágenes visualmente similares",
    description="""
    Busca imágenes visualmente similares a una imagen dada usando embeddings
    vectoriales almacenados en ChromaDB.

    **Características:**
    - Búsqueda por similitud coseno (score real, sin recortar)
    - Filtro de relevancia absoluto + relativo (ratio y gap contra el mejor vecino)
    - `below_threshold` indica que ningún vecino superó el umbral
    - `closest_match` informa el mejor candidato aunque no sea una coincidencia
    - Filtrado por categoría opcional

    **Scores:** con ResNet50 el corpus completo vive entre 0.39 y 0.76 de
    similitud coseno, por lo que los defaults son 0.48 / 0.90 / 0.08 y no 0.80.
    """
)
async def get_visual_similar_images(
    image_id: int,
    limit: int = Query(
        default=5,
        ge=1,
        le=50,
        description="Número de imágenes similares a retornar"
    ),
    min_score: float = Query(
        default=DEFAULT_MIN_SCORE,
        ge=0.0,
        le=1.0,
        description="Similitud coseno mínima (piso absoluto)"
    ),
    min_relative: Optional[float] = Query(
        default=DEFAULT_MIN_RELATIVE,
        ge=0.0,
        le=1.0,
        description="Ratio mínimo respecto al mejor vecino (1.0 = solo el mejor)"
    ),
    max_gap: Optional[float] = Query(
        default=DEFAULT_MAX_GAP,
        ge=0.0,
        le=1.0,
        description="Diferencia máxima respecto al mejor vecino"
    ),
    category: Optional[str] = Query(
        default=None,
        description="Filtrar resultados por categoría"
    )
):
    """
    Obtiene imágenes visualmente similares usando búsqueda vectorial.

    Returns:
        Imagen original, similares con su score real, el mejor candidato y los
        umbrales aplicados.
    """
    try:
        outcome = await find_similar_images(
            image_id=image_id,
            limit=limit,
            min_score=min_score,
            min_relative=min_relative,
            max_gap=max_gap,
            category=category,
            vector_store=get_vector_store_instance()
        )

        if not outcome["has_embedding"]:
            return {
                "status": "warning",
                "message": (
                    f"Imagen {image_id} sin embedding vectorial. "
                    "Ejecuta migrate_vectors.py --execute --complete para reindexar."
                ),
                "original_image": outcome["image"],
                "similar_images": [],
                "total_similar": 0,
                "closest_match": None,
                "below_threshold": True,
                "thresholds": outcome["thresholds"],
                "vector_backend": "chromadb",
                "has_embedding": False,
                "total_embeddings": outcome["total_embeddings"],
                "query_params": {
                    "image_id": image_id,
                    "limit": limit,
                    "min_score": min_score,
                    "min_relative": min_relative,
                    "max_gap": max_gap,
                    "category": category
                }
            }

        logger.info(
            f"🔍 Búsqueda visual: image_id={image_id}, "
            f"{len(outcome['results'])} similares, mejor={outcome['thresholds']['best_score']}, "
            f"umbral={outcome['thresholds']['effective_threshold']}"
        )

        return {
            "status": "success",
            "original_image": outcome["image"],
            "similar_images": outcome["results"],
            "total_similar": len(outcome["results"]),
            "closest_match": outcome["closest"],
            "below_threshold": outcome["below_threshold"],
            "thresholds": outcome["thresholds"],
            "vector_backend": "chromadb",
            "embedding_model": outcome["embedding_model"],
            "has_embedding": True,
            "total_embeddings": outcome["total_embeddings"],
            "total_candidates": outcome["total_candidates"],
            "query_params": {
                "image_id": image_id,
                "limit": limit,
                "min_score": min_score,
                "min_relative": min_relative,
                "max_gap": max_gap,
                "category": category
            }
        }

    except HTTPException:
        raise
    except (ImageNotFoundError, VectorStoreError) as e:
        raise _handle_visual_error(e, image_id)
    except Exception as e:
        raise _handle_visual_error(e, image_id)


@router.get(
    "/visual-similar-hybrid/{image_id}",
    response_model=Dict[str, Any],
    summary="Búsqueda visual híbrida (visual + categoría + popularidad)",
    description="""
    Búsqueda híbrida que combina:
    - Similitud visual (peso principal, es la que se filtra)
    - Coincidencia de categoría (boost)
    - Popularidad de la imagen (boost menor)
    """
)
async def get_visual_similar_hybrid(
    image_id: int,
    limit: int = Query(default=5, ge=1, le=50),
    min_score: float = Query(default=DEFAULT_MIN_SCORE, ge=0.0, le=1.0),
    min_relative: Optional[float] = Query(default=DEFAULT_MIN_RELATIVE, ge=0.0, le=1.0),
    max_gap: Optional[float] = Query(default=DEFAULT_MAX_GAP, ge=0.0, le=1.0),
    category_boost: float = Query(default=0.15, ge=0.0, le=0.3, description="Peso del boost de categoría"),
    popularity_weight: float = Query(default=0.10, ge=0.0, le=0.2, description="Peso de popularidad")
):
    """
    Búsqueda visual híbrida con re-ranking por categoría y popularidad.

    El filtro de relevancia se aplica sobre la similitud visual; los boosts
    solo reordenan los que ya pasaron el filtro.
    """
    try:
        outcome = await find_similar_images_hybrid(
            image_id=image_id,
            limit=limit,
            min_score=min_score,
            min_relative=min_relative,
            max_gap=max_gap,
            category_boost=category_boost,
            popularity_weight=popularity_weight,
            vector_store=get_vector_store_instance()
        )

        return {
            "status": "success" if outcome["has_embedding"] else "warning",
            "search_type": "hybrid",
            "original_image": outcome["image"],
            "similar_images": outcome["results"],
            "total_similar": len(outcome["results"]),
            "closest_match": outcome["closest"],
            "below_threshold": outcome["below_threshold"],
            "thresholds": outcome["thresholds"],
            "vector_backend": "chromadb",
            "embedding_model": outcome["embedding_model"],
            "has_embedding": outcome["has_embedding"],
            "total_embeddings": outcome["total_embeddings"],
            "query_params": {
                "image_id": image_id,
                "limit": limit,
                "min_score": min_score,
                "min_relative": min_relative,
                "max_gap": max_gap,
                "category_boost": category_boost,
                "popularity_weight": popularity_weight
            }
        }

    except HTTPException:
        raise
    except (ImageNotFoundError, VectorStoreError) as e:
        raise _handle_visual_error(e, image_id)
    except Exception as e:
        raise _handle_visual_error(e, image_id)


@router.get(
    "/visual-evaluate",
    response_model=Dict[str, Any],
    summary="Evaluar precisión de búsqueda visual",
    description="""
    Evalúa la precisión del sistema de búsqueda visual usando leave-one-out.
    Retorna Precision@K, Recall@K y MAP.
    """
)
async def evaluate_visual_search(
    sample_size: int = Query(
        default=100,
        ge=10,
        le=1000,
        description="Número de imágenes a evaluar"
    ),
    k: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Valor de K para métricas"
    )
):
    """
    Evalúa la precisión del sistema de búsqueda visual.
    
    Usa leave-one-out cross-validation para medir:
    - Precision@K: ¿las imágenes recuperadas son de la misma categoría?
    - Recall@K: ¿se recuperaron las imágenes relevantes?
    - MAP: Mean Average Precision
    """
    try:
        from services.visual_search_evaluator import VisualSearchEvaluator
        
        evaluator = VisualSearchEvaluator(get_vector_store_instance())
        
        results = await evaluator.evaluate_leave_one_out(
            coleccion=coleccion,
            k_values=[k],
            sample_size=sample_size,
            min_images_per_category=2
        )
        
        return {
            "status": "success",
            "evaluation": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error en evaluación: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get(
    "/vector-stats",
    response_model=Dict[str, Any],
    summary="Estadísticas del sistema vectorial",
    description="Retorna estadísticas del sistema de búsqueda vectorial ChromaDB"
)
async def get_vector_stats():
    """
    Obtiene estadísticas del sistema vectorial.
    
    Returns:
        Diccionario con estadísticas de ChromaDB
    """
    try:
        vector_store = get_vector_store_instance()
        
        if not vector_store.health_check():
            return {
                "status": "error",
                "message": "ChromaDB no está disponible",
                "available": False
            }
        
        stats = vector_store.get_collection_stats()
        
        return {
            "status": "success",
            "available": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        return {
            "status": "error",
            "message": str(e),
            "available": False
        }


@router.get(
    "/check-embedding/{image_id}",
    response_model=Dict[str, Any],
    summary="Verificar si una imagen tiene embedding",
    description="Verifica si una imagen específica tiene embedding vectorial en ChromaDB"
)
async def check_image_embedding(image_id: int):
    """
    Verifica si una imagen tiene embedding en ChromaDB.
    
    Args:
        image_id: ID numérico de la imagen
    
    Returns:
        Diccionario con estado del embedding
    """
    try:
        # Verificar en MongoDB
        image = await coleccion.find_one({"image_id": image_id})
        if not image:
            raise HTTPException(
                status_code=404,
                detail=f"Imagen {image_id} no encontrada en MongoDB"
            )
        
        # Verificar en ChromaDB
        vector_store = get_vector_store_instance()
        embedding = vector_store.get_embedding(image_id)
        
        return {
            "status": "success",
            "image_id": image_id,
            "has_mongodb": True,
            "has_embedding": embedding is not None,
            "embedding_dimension": len(embedding["vector"]) if embedding else None,
            "image_url": image.get("image_url"),
            "category": image.get("category")
        }
        
    except HTTPException:
        raise
    except (VectorStoreError, DimensionMismatchError, ModelMismatchError) as e:
        raise _handle_visual_error(e, image_id)
    except Exception as e:
        logger.error(f"❌ Error verificando embedding: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )
