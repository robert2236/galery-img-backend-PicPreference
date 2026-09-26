"""
vector_store.py - Módulo de almacenamiento vectorial con ChromaDB

Encapsula la conexión y operaciones CRUD con ChromaDB para almacenar
embeddings de imágenes y realizar búsquedas por similitud visual.

Arquitectura no-destructiva: No modifica modelos o endpoints existentes de MongoDB.
"""

import chromadb
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelo de embeddings usado por el pipeline actual.
# IMPORTANTE: no mezclar modelos en la misma colección. ResNet50=2048, CLIP=512.
DEFAULT_MODEL = "resnet50"
DEFAULT_FEATURE_SIZE = 2048

# Umbrales por defecto para el filtro de relevancia.
# Calibrados sobre las 142 imagenes reales del corpus (leave-one-out, cosine):
#   - score absoluto: el corpus vive en 0.39-0.76, asi que 0.48 descarta el ruido de fondo
#   - ratio con el mejor vecino: descarta vecinos claramente mas lejanos
#   - gap con el mejor vecino: descarta la "cola" cuando el mejor match es fuerte
DEFAULT_MIN_SCORE = 0.48
DEFAULT_MIN_RELATIVE = 0.90
DEFAULT_MAX_GAP = 0.08


class VectorStoreError(RuntimeError):
    """Error base del VectorStore."""


class DimensionMismatchError(VectorStoreError):
    """El vector no corresponde al espacio vectorial de la coleccion."""


class ModelMismatchError(VectorStoreError):
    """El vector pertenece a otro modelo de embeddings."""


class VectorStore:
    """
    Clase para gestionar embeddings vectoriales en ChromaDB.
    
    Características:
    - Persistencia local en ./chroma_db
    - Colección 'visual_embeddings' para vectores CLIP (512d) o ResNet50 (2048d)
    - Soporte para metadata enriquecida
    - Búsqueda por similitud coseno
    - Búsqueda híbrida (visual + metadata + popularidad)
    - Filtro de relevancia absoluto + relativo (ratio y gap contra el mejor vecino)
    
    Nota: la dimensión y el modelo se resuelven SIEMPRE desde la colección real.
    Nunca se rellena ni se recorta un vector para "hacerlo caber": eso mezcla
    espacios vectoriales distintos y produce similitudes sin significado.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        feature_size: int = DEFAULT_FEATURE_SIZE,
        model_name: str = DEFAULT_MODEL
    ):
        """
        Inicializa la conexión con ChromaDB.
        
        Args:
            persist_directory: Directorio de persistencia local
            feature_size: Dimensión esperada de los embeddings (2048 ResNet50, 512 CLIP)
            model_name: Modelo que genera los embeddings ("resnet50" | "clip")
        """
        try:
            # Crear directorio si no existe
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Conexión con persistencia local
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            self.requested_feature_size = feature_size
            self.requested_model = model_name
            self.persist_directory = persist_directory
            self.collection_name = "visual_embeddings"
            
            # Crear o obtener colección con métrica coseno.
            # get_or_create_collection ignora el metadata si la colección ya existe,
            # por eso la dimensión real se resuelve despues (nunca se asume).
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.feature_size = self._resolve_dimension(feature_size)
            self.model_name = self._resolve_model(model_name)
            
            logger.info(f"✅ ChromaDB inicializado en: {persist_directory}")
            logger.info(
                f"📊 Colección '{self.collection_name}' lista "
                f"(model={self.model_name}, dim={self.feature_size}, "
                f"vectores={self.collection.count()})"
            )
            
        except Exception as e:
            logger.error(f"❌ Error inicializando ChromaDB: {e}")
            raise
    
    def _resolve_dimension(self, fallback: int) -> int:
        """
        Resuelve la dimensión real de la colección.
        
        La colección se considera la fuente de verdad. Si está vacía se usa la
        dimensión solicitada. Si no se puede determinar, se lanza un error en
        lugar de adivinar (adivinar rompe las búsquedas silenciosamente).
        """
        count = self.collection.count()
        if count == 0:
            return fallback
        
        # 1) Dimensión declarada por la colección
        try:
            declared = getattr(self.collection, "dimension", None)
            if isinstance(declared, int) and declared > 0:
                if declared != fallback:
                    logger.warning(
                        f"⚠️ La colección '{self.collection_name}' tiene dimensión {declared} "
                        f"y se solicitó {fallback}. Se usa la dimensión real de la colección."
                    )
                return declared
        except Exception as e:
            logger.debug(f"collection.dimension no disponible: {e}")
        
        # 2) Muestreo de un vector real
        # OJO: con chromadb>=1.5 + numpy 2.x los embeddings llegan como ndarray,
        # y `if sample["embeddings"]` lanza ValueError (truth value ambiguous).
        # Hay que comprobar la longitud explícitamente.
        try:
            sample = self.collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings") if sample else None
            if embeddings is not None and len(embeddings) > 0:
                detected = len(embeddings[0])
                if detected != fallback:
                    logger.warning(
                        f"⚠️ Dimensión detectada ({detected}) difiere de la solicitada "
                        f"({fallback}). Se usa la dimensión real de la colección."
                    )
                return detected
        except Exception as e:
            logger.warning(f"No se pudo muestrear un vector para detectar la dimensión: {e}")
        
        raise VectorStoreError(
            f"No se pudo determinar la dimensión de la colección '{self.collection_name}' "
            f"({count} vectores). Revisar el índice de ChromaDB."
        )
    
    def _resolve_model(self, requested: str) -> str:
        """Resuelve el modelo de embeddings registrado en la colección."""
        try:
            stored = (self.collection.metadata or {}).get("embedding_model")
        except Exception as e:
            logger.debug(f"No se pudo leer metadata de la colección: {e}")
            stored = None
        
        if not stored:
            return requested
        
        if stored != requested:
            logger.warning(
                f"⚠️ La colección fue creada con el modelo '{stored}' pero se solicito "
                f"'{requested}'. Se usará el modelo de la colección."
            )
        return stored
    
    def _validate_vector(self, vector: List[float], context: str) -> None:
        """
        Valida que un vector pertenezca al espacio de la colección.
        
        Lanza DimensionMismatchError / ModelMismatchError. Nunca rellena con ceros
        ni recorta: un vector recortado/padeado rompe la similitud coseno.
        """
        if vector is None or len(vector) == 0:
            raise VectorStoreError(f"{context}: vector vacío")
        
        if len(vector) != self.feature_size:
            raise DimensionMismatchError(
                f"{context}: vector de dimensión {len(vector)} pero la colección "
                f"'{self.collection_name}' (modelo={self.model_name}) espera "
                f"{self.feature_size}. No se rellena ni se recorta: regenerá el "
                f"embedding con el modelo correcto o reconstruí la colección."
            )
    
    def _validate_query(self, query_vector: List[float]) -> None:
        self._validate_vector(query_vector, "query")
    
    def add_embedding(
        self, 
        image_id: int, 
        vector: List[float], 
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Agrega un embedding a la colección.
        
        Args:
            image_id: ID numérico de la imagen (de MongoDB)
            vector: Vector de características (2048d para ResNet50, 512d para CLIP)
            metadata: Metadatos opcionales (categoría, usuario, etc.)
            model_name: Modelo que generó el vector. Si no coincide con el de la
                colección se rechaza (mezclar modelos invalida las similitudes).
        
        Returns:
            True si se insertó correctamente
        
        Raises:
            DimensionMismatchError, ModelMismatchError, VectorStoreError
        """
        try:
            if model_name and model_name != self.model_name:
                raise ModelMismatchError(
                    f"image_id={image_id}: vector generado con '{model_name}' pero la "
                    f"colección usa '{self.model_name}'. No se puede mezclar."
                )
            
            # Valida dimensión (lanza excepción, nunca rellena ni recorta)
            self._validate_vector(vector, f"add_embedding(image_id={image_id})")
            
            # Verificar que no sea vector de ceros
            if all(v == 0.0 for v in vector):
                raise VectorStoreError(f"add_embedding(image_id={image_id}): vector de ceros")
            
            # Preparar metadata
            if metadata is None:
                metadata = {}
            
            # Añadir campos obligatorios
            metadata["image_id"] = image_id
            metadata["dimension"] = self.feature_size
            metadata["model"] = self.model_name
            
            # Convertir vector a lista de floats (Requiere ChromaDB)
            vector_float = [float(v) for v in vector]
            
            # Insertar/actualizar en ChromaDB
            # upsert = insert si no existe, update si existe
            self.collection.upsert(
                ids=[str(image_id)],
                embeddings=[vector_float],
                metadatas=[metadata],
                documents=[f"image_{image_id}"]
            )
            
            logger.debug(f"✅ Embedding insertado: image_id={image_id}")
            return True
            
        except (VectorStoreError, chromadb.errors.ChromaError) as e:
            logger.error(f"❌ Error insertando embedding image_id={image_id}: {e}")
            raise
    
    def get_embedding(self, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtiene un embedding por ID de imagen.
        
        Args:
            image_id: ID numérico de la imagen
        
        Returns:
            Diccionario con vector y metadata, o None si la imagen no tiene embedding
        """
        try:
            result = self.collection.get(
                ids=[str(image_id)],
                include=["embeddings", "metadatas"]
            )
            
            if not result["ids"] or len(result["ids"]) == 0:
                logger.debug(f"🔍 Embedding no encontrado: image_id={image_id}")
                return None
            
            vector = result["embeddings"][0]
            
            # Verificación de integridad: un vector truncado en disco es peor
            # que un embedding ausente.
            if len(vector) != self.feature_size:
                raise DimensionMismatchError(
                    f"El embedding almacenado de image_id={image_id} tiene dimensión "
                    f"{len(vector)} y la colección espera {self.feature_size}. "
                    f"Reindexá la colección (migrate_vectors.py --execute --complete)."
                )
            
            return {
                "image_id": image_id,
                "vector": vector,
                "metadata": result["metadatas"][0] if result.get("metadatas") else {}
            }
            
        except (VectorStoreError, chromadb.errors.ChromaError) as e:
            logger.error(f"❌ Error obteniendo embedding image_id={image_id}: {e}")
            raise
    
    def _query_candidates(
        self,
        query_vector: List[float],
        k: int,
        exclude_image_id: Optional[int] = None,
        candidate_multiplier: int = 4,
        min_candidates: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta la consulta en ChromaDB y devuelve candidatos SIN filtrar,
        ordenados por similitud descendente.
        
        No aplica ningún umbral: el filtrado es responsabilidad de
        `_apply_relevance_filter` para que el llamador pueda reportar
        `closest_match` y `below_threshold`.
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("Colección de vectores vacía")
            return []
        
        # Valida dimensión (lanza excepción, nunca rellena ni recorta)
        self._validate_query(query_vector)
        
        # Suficientes candidatos para que el filtro relativo tenga margen
        n_results = min(count, max(k * candidate_multiplier, min_candidates))
        n_results = max(n_results, k + 1)
        n_results = min(n_results, count)
        
        try:
            results = self.collection.query(
                query_embeddings=[[float(v) for v in query_vector]],
                n_results=n_results,
                include=["metadatas", "distances"]
            )
        except chromadb.errors.ChromaError as e:
            raise VectorStoreError(f"Error consultando ChromaDB: {e}") from e
        
        ids = (results.get("ids") or [[]])[0]
        if not ids:
            return []
        
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        
        candidates = []
        for i, doc_id in enumerate(ids):
            try:
                image_id = int(doc_id)
            except (TypeError, ValueError):
                logger.warning(f"⚠️ id de embedding inválido en ChromaDB: {doc_id!r}")
                continue
            
            if exclude_image_id is not None and image_id == exclude_image_id:
                continue
            
            # ChromaDB con hnsw:space=cosine devuelve distancia = 1 - similitud,
            # con rango [0, 2]. La similitud NO se recorta a 0 para no hidear
            # el comportamiento real del modelo.
            distance = float(distances[i]) if i < len(distances) else 1.0
            similarity = 1.0 - distance
            
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            
            candidates.append({
                "image_id": image_id,
                "similarity_score": round(similarity, 4),
                "distance": round(distance, 4),
                "metadata": metadata
            })
        
        candidates.sort(key=lambda c: c["similarity_score"], reverse=True)
        return candidates
    
    @staticmethod
    def _apply_relevance_filter(
        candidates: List[Dict[str, Any]],
        k: int,
        min_score: float = 0.0,
        min_relative: Optional[float] = None,
        max_gap: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Aplica el filtro de relevancia absoluto + relativo.
        
        Se conserva un candidato si:
            score >= min_score
          AND score >= best_score * min_relative      (si se define)
          AND score >= best_score - max_gap           (si se define)
        
        El umbral efectivo es max(min_score, best*min_relative, best-max_gap).
        """
        best = candidates[0]["similarity_score"] if candidates else 0.0
        
        effective = min_score
        if min_relative is not None:
            effective = max(effective, best * min_relative)
        if max_gap is not None:
            effective = max(effective, best - max_gap)
        
        kept = [c for c in candidates if c["similarity_score"] >= effective][:k]
        
        for rank, item in enumerate(kept, start=1):
            item["rank"] = rank
            item["relative_score"] = round(item["similarity_score"] / best, 4) if best > 0 else 0.0
            item["gap_to_best"] = round(best - item["similarity_score"], 4)
        
        return {
            "results": kept,
            "closest": candidates[0] if candidates else None,
            "best_score": round(best, 4),
            "effective_threshold": round(effective, 4),
            "below_threshold": len(kept) == 0,
            "candidates_evaluated": len(candidates)
        }
    
    def search_similar_detailed(
        self, 
        query_vector: List[float], 
        k: int = 5,
        exclude_image_id: Optional[int] = None,
        min_score: float = 0.0,
        min_relative: Optional[float] = None,
        max_gap: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Busca imágenes similares y devuelve el detalle del filtrado.
        
        Args:
            query_vector: Vector de consulta (2048d ResNet50 o 512d CLIP)
            k: Número máximo de resultados a retornar
            exclude_image_id: ID de imagen a excluir (la original)
            min_score: Score mínimo absoluto de similitud coseno
            min_relative: Mínimo ratio respecto al mejor vecino (ej. 0.90)
            max_gap: Diferencia máxima respecto al mejor vecino (ej. 0.08)
        
        Returns:
            Dict con results, closest, umbrales aplicados y below_threshold
        """
        candidates = self._query_candidates(
            query_vector=query_vector,
            k=k,
            exclude_image_id=exclude_image_id
        )
        
        outcome = self._apply_relevance_filter(
            candidates,
            k=k,
            min_score=min_score,
            min_relative=min_relative,
            max_gap=max_gap
        )
        
        outcome["thresholds"] = {
            "min_score": min_score,
            "min_relative": min_relative,
            "max_gap": max_gap,
            "best_score": outcome["best_score"],
            "effective_threshold": outcome["effective_threshold"]
        }
        
        logger.debug(
            f"🔍 Búsqueda similar: {len(outcome['results'])} resultados "
            f"(mejor={outcome['best_score']}, umbral={outcome['effective_threshold']})"
        )
        
        return outcome
    
    def search_similar(
        self, 
        query_vector: List[float], 
        k: int = 5,
        exclude_image_id: Optional[int] = None,
        min_score: float = 0.0,
        min_relative: Optional[float] = None,
        max_gap: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca imágenes similares por vector de consulta.
        
        Versión simplificada de `search_similar_detailed` para callers que
        solo necesitan la lista (ej. el evaluador de métricas, que no debe
        aplicar el filtro relativo para medir el ranking crudo).
        
        Returns:
            Lista de diccionarios con image_id, similarity_score, rank y metadata
        """
        return self.search_similar_detailed(
            query_vector=query_vector,
            k=k,
            exclude_image_id=exclude_image_id,
            min_score=min_score,
            min_relative=min_relative,
            max_gap=max_gap
        )["results"]
    
    def search_hybrid(
        self,
        query_vector: List[float],
        k: int = 5,
        exclude_image_id: Optional[int] = None,
        min_score: float = 0.0,
        category: Optional[str] = None,
        category_boost: float = 0.15,
        popularity_scores: Optional[Dict[int, float]] = None,
        popularity_weight: float = 0.10,
        min_relative: Optional[float] = None,
        max_gap: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Búsqueda híbrida (ver `search_hybrid_detailed`)."""
        return self.search_hybrid_detailed(
            query_vector=query_vector,
            k=k,
            exclude_image_id=exclude_image_id,
            min_score=min_score,
            category=category,
            category_boost=category_boost,
            popularity_scores=popularity_scores,
            popularity_weight=popularity_weight,
            min_relative=min_relative,
            max_gap=max_gap
        )["results"]
    
    def search_hybrid_detailed(
        self,
        query_vector: List[float],
        k: int = 5,
        exclude_image_id: Optional[int] = None,
        min_score: float = 0.0,
        category: Optional[str] = None,
        category_boost: float = 0.15,
        popularity_scores: Optional[Dict[int, float]] = None,
        popularity_weight: float = 0.10,
        min_relative: Optional[float] = None,
        max_gap: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Búsqueda híbrida: similitud visual + coincidencia de categoría + popularidad.
        
        El filtro de relevancia se aplica SOBRE EL SCORE VISUAL (antes del re-ranking),
        porque los boosts de categoría/popularidad no son comparables con una
        similitud coseno.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados
            exclude_image_id: ID a excluir
            min_score: Score mínimo de similitud visual
            category: Categoría para boost (opcional)
            category_boost: Peso del boost de categoría (0.0 - 0.3)
            popularity_scores: Dict[image_id, score_de_popularidad] (0.0-1.0)
            popularity_weight: Peso de la popularidad (0.0 - 0.2)
            min_relative: Mínimo ratio respecto al mejor vecino visual
            max_gap: Diferencia máxima respecto al mejor vecino visual
        
        Returns:
            Dict con results, closest, umbrales aplicados y below_threshold
        """
        # 1. Candidatos + filtro de relevancia sobre la similitud visual
        outcome = self.search_similar_detailed(
            query_vector=query_vector,
            k=k,
            exclude_image_id=exclude_image_id,
            min_score=min_score,
            min_relative=min_relative,
            max_gap=max_gap
        )
        
        visual_results = outcome["results"]
        if not visual_results:
            outcome["results"] = []
            return outcome
        
        # 2. Re-ranking híbrido sobre los que ya pasaron el filtro
        reranked = []
        for result in visual_results:
            visual_score = result["similarity_score"]
            metadata = result.get("metadata", {})
            img_id = result["image_id"]
            
            # Boost de categoría
            category_score = 0.0
            if category and metadata.get("category") == category:
                category_score = category_boost
            
            # Boost de popularidad
            pop_score = 0.0
            if popularity_scores and img_id in popularity_scores:
                pop_score = popularity_scores[img_id] * popularity_weight
            
            reranked.append({
                **result,
                "final_score": round(visual_score + category_score + pop_score, 4),
                "visual_score": visual_score,
                "category_boost": category_score,
                "popularity_boost": pop_score
            })
        
        # 3. Ordenar por score final (el rank se recalcula tras el reorden)
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        for rank, item in enumerate(reranked, start=1):
            item["rank"] = rank
        
        outcome["results"] = reranked[:k]
        return outcome
    
    def delete_embedding(self, image_id: int) -> bool:
        """
        Elimina un embedding por ID de imagen.
        
        Args:
            image_id: ID numérico de la imagen
        
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        try:
            # Verificar que existe
            existing = self.collection.get(ids=[str(image_id)])
            if not existing["ids"] or len(existing["ids"]) == 0:
                logger.warning(f"⚠️ Embedding no encontrado para eliminar: image_id={image_id}")
                return False
            
            # Eliminar
            self.collection.delete(ids=[str(image_id)])
            logger.info(f"🗑️ Embedding eliminado: image_id={image_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error eliminando embedding image_id={image_id}: {e}")
            return False
    
    def get_all_ids(self) -> List[int]:
        """
        Obtiene todos los IDs de imágenes almacenados.
        
        Returns:
            Lista de image_ids
        """
        try:
            result = self.collection.get(include=[])
            if not result["ids"]:
                return []
            
            return [int(doc_id) for doc_id in result["ids"]]
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo todos los IDs: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la colección.
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            count = self.collection.count()
            all_ids = self.get_all_ids()
            
            return {
                "total_embeddings": count,
                "collection_name": self.collection.name,
                "persist_directory": self.persist_directory,
                "feature_size": self.feature_size,
                "embedding_model": self.model_name,
                "requested_feature_size": self.requested_feature_size,
                "space": (self.collection.metadata or {}).get("hnsw:space", "unknown"),
                "default_thresholds": {
                    "min_score": DEFAULT_MIN_SCORE,
                    "min_relative": DEFAULT_MIN_RELATIVE,
                    "max_gap": DEFAULT_MAX_GAP
                },
                "sample_ids": all_ids[:10] if all_ids else []
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    def batch_add_embeddings(
        self, 
        embeddings: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Inserta múltiples embeddings en lote.
        
        Args:
            embeddings: Lista de diccionarios con image_id, vector, metadata
        
        Returns:
            Diccionario con conteo de éxitos y fallos
        """
        success_count = 0
        error_count = 0
        
        for item in embeddings:
            image_id = item.get("image_id")
            vector = item.get("vector")
            metadata = item.get("metadata", {})
            model_name = item.get("model")
            
            try:
                self.add_embedding(image_id, vector, metadata, model_name=model_name)
                success_count += 1
            except VectorStoreError as e:
                logger.error(f"❌ image_id={image_id}: {e}")
                error_count += 1
        
        logger.info(f"📊 Lote completado: {success_count} éxitos, {error_count} fallos")
        return {"success": success_count, "errors": error_count}
    
    def health_check(self) -> bool:
        """
        Verifica que ChromaDB esté funcionando correctamente.
        
        Returns:
            True si está saludable, False en caso contrario
        """
        try:
            count = self.collection.count()
            logger.info(
                f"✅ ChromaDB saludable. Embeddings: {count} "
                f"(model={self.model_name}, dim={self.feature_size})"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ ChromaDB no saludable: {e}")
            return False


# Instancia global para uso en la aplicación
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store(
    persist_directory: str = "./chroma_db",
    feature_size: int = DEFAULT_FEATURE_SIZE,
    model_name: str = DEFAULT_MODEL
) -> VectorStore:
    """
    Obtiene la instancia global de VectorStore (singleton).
    
    La dimensión y el modelo se toman siempre de la colección real, por lo que
    los argumentos solo aplican en la primera llamada (colección vacía).
    
    Args:
        persist_directory: Directorio de persistencia (solo en primera llamada)
        feature_size: Dimensión de embeddings (2048 ResNet50, 512 CLIP)
        model_name: Modelo de embeddings ("resnet50" | "clip")
    
    Returns:
        Instancia de VectorStore
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(
            persist_directory,
            feature_size=feature_size,
            model_name=model_name
        )
    
    return _vector_store_instance


# Ejemplo de uso directo
if __name__ == "__main__":
    store = get_vector_store()
    
    if store.health_check():
        print(f"Estadísticas: {store.get_collection_stats()}")
    else:
        print("Error: ChromaDB no está disponible")
