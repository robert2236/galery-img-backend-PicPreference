"""
migrate_vectors.py - Script de migración batch para embeddings vectoriales

Migra imágenes existentes de MongoDB a ChromaDB, generando embeddings
con ResNet50 (2048 dimensiones).

Uso:
    python migrate_vectors.py --dry-run              # Preview sin cambios
    python migrate_vectors.py --execute              # Migrar todas
    python migrate_vectors.py --execute --incremental  # Solo nuevas
    python migrate_vectors.py --execute --limit 50   # Limitar cantidad
    python migrate_vectors.py --stats                # Ver estadísticas
"""

import argparse
import logging
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Configurar logging
log_file = "migrate_vectors.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Verificar dependencias
try:
    from pymongo import MongoClient
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    import requests
    from io import BytesIO
except ImportError as e:
    logger.error(f"❌ Dependencia faltante: {e}")
    logger.error("Instalar con: pip install pymongo tqdm numpy pillow requests")
    sys.exit(1)

# Importar módulos del proyecto
try:
    from vector_store import VectorStore
    from utils.feature_extractor import FeatureExtractor
except ImportError as e:
    logger.error(f"❌ Error importando módulos del proyecto: {e}")
    logger.error("Asegúrate de ejecutar desde el directorio raíz del proyecto")
    sys.exit(1)


class VectorMigrator:
    """
    Clase para migrar embeddings de MongoDB a ChromaDB.
    
    Características:
    - Modo dry-run para preview
    - Manejo de errores por imagen
    - Barra de progreso con tqdm
    - Logs detallados
    - Migración incremental
    - Soporte para CLIP (512d) y ResNet50 (2048d)
    """
    
    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        db_name: str = "galery",
        chroma_dir: str = "./chroma_db",
        model_type: str = "clip"
    ):
        """
        Inicializa el migrador.
        
        Args:
            mongo_url: URL de conexión a MongoDB
            db_name: Nombre de la base de datos
            chroma_dir: Directorio de persistencia ChromaDB
            model_type: Tipo de modelo ('clip' o 'resnet')
        """
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.model_type = model_type.lower()
        
        # Conexión MongoDB (síncrona para script CLI)
        logger.info(f"🔄 Conectando a MongoDB: {mongo_url}")
        self.mongo_client = MongoClient(
            mongo_url,
            serverSelectionTimeoutMS=5000
        )
        self.db = self.mongo_client[db_name]
        self.images_collection = self.db["images"]
        
        # Verificar conexión MongoDB
        try:
            self.mongo_client.admin.command("ping")
            logger.info("✅ MongoDB conectado correctamente")
        except Exception as e:
            logger.error(f"❌ No se pudo conectar a MongoDB: {e}")
            raise
        
        # Determinar dimensión según modelo
        feature_size = 512 if self.model_type == "clip" else 2048
        
        # Inicializar ChromaDB con la dimensión correcta
        self.vector_store = VectorStore(chroma_dir, feature_size=feature_size)
        
        # Inicializar extractor de features según modelo
        if self.model_type == "clip":
            from utils.feature_extractor import CLIPFeatureExtractor
            self.feature_extractor = CLIPFeatureExtractor()
            logger.info("🧠 Usando CLIP (512d) para extracción de features")
        else:
            self.feature_extractor = FeatureExtractor()
            logger.info("🧠 Usando ResNet50 (2048d) para extracción de features")
        
        # Estadísticas
        self.stats = {
            "total_processed": 0,
            "success": 0,
            "errors": 0,
            "skipped": 0,
            "not_found": 0
        }
    
    def get_images_to_migrate(
        self, 
        incremental: bool = True,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene imágenes de MongoDB que necesitan migración.
        
        Args:
            incremental: Si es True, solo imágenes sin embedding en ChromaDB
            limit: Límite máximo de imágenes a procesar
        
        Returns:
            Lista de documentos de imágenes
        """
        try:
            # Obtener todos los image_ids en ChromaDB
            existing_ids = set(self.vector_store.get_all_ids())
            
            # Consultar MongoDB
            query = {}
            if incremental and existing_ids:
                # Excluir imágenes que ya están en ChromaDB
                # IMPORTANTE: $nin con lista vacia retorna 0 resultados en MongoDB
                query["image_id"] = {"$nin": list(existing_ids)}
            
            # Ejecutar consulta
            cursor = self.images_collection.find(query)
            
            if limit:
                cursor = cursor.limit(limit)
            
            images = list(cursor)
            
            logger.info(
                f"📊 Encontradas {len(images)} imágenes para migrar"
                f"{' (incremental)' if incremental else ' (completa)'}"
            )
            
            return images
            
        except Exception as e:
            logger.error(f"❌ Error consultando MongoDB: {e}")
            return []
    
    def download_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Descarga una imagen desde URL o archivo local.
        
        Args:
            image_url: URL de la imagen o ruta local
        
        Returns:
            Objeto PIL.Image o None si falla
        """
        try:
            # Verificar si es base64
            if image_url.startswith("data:image"):
                # Decodificar base64
                import base64
                header, data = image_url.split(",", 1)
                img_data = base64.b64decode(data)
                return Image.open(BytesIO(img_data)).convert("RGB")
            
            # Verificar si es ruta local
            if image_url.startswith("/uploads/"):
                # Ruta local relativa
                local_path = Path(image_url.lstrip("/"))
                if local_path.exists():
                    return Image.open(local_path).convert("RGB")
                else:
                    logger.warning(f"⚠️ Archivo local no encontrado: {local_path}")
                    return None
            
            # URL remota
            if image_url.startswith(("http://", "https://")):
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            
            logger.warning(f"⚠️ URL no reconocida: {image_url}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error descargando imagen {image_url}: {e}")
            return None
    
    def extract_features(self, image: Image.Image) -> Optional[List[float]]:
        """
        Extrae features de una imagen usando el modelo configurado.
        
        Args:
            image: Objeto PIL.Image
        
        Returns:
            Lista de features (512 para CLIP, 2048 para ResNet50) o None si falla
        """
        try:
            # Redimensionar según modelo
            if self.model_type == "clip":
                # CLIP usa 224x224 también
                image = image.resize((224, 224))
            else:
                image = image.resize((224, 224))
            
            # Convertir a base64 para FeatureExtractor
            import base64
            from io import BytesIO
            
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Extraer features con augmentation
            features = self.feature_extractor.extract_with_augmentation(img_base64)
            
            expected_size = 512 if self.model_type == "clip" else 2048
            if not features or len(features) != expected_size:
                logger.warning(f"⚠️ Features inválidas: {len(features) if features else 0}, esperado: {expected_size}")
                return None
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo features: {e}")
            return None
    
    def migrate_single_image(
        self, 
        image_doc: Dict[str, Any],
        dry_run: bool = False
    ) -> bool:
        """
        Migra una sola imagen a ChromaDB.
        
        Args:
            image_doc: Documento de MongoDB
            dry_run: Si es True, no inserta en ChromaDB
        
        Returns:
            True si fue exitosa, False en caso contrario
        """
        image_id = image_doc.get("image_id")
        image_url = image_doc.get("image_url", "")
        category = image_doc.get("category", "unknown")
        user_id = image_doc.get("user_id")
        title = image_doc.get("title", "")
        
        try:
            # Verificar si ya tiene features en MongoDB
            existing_features = image_doc.get("features")
            if existing_features and len(existing_features) == 2048:
                # Usar features existentes de MongoDB
                features = existing_features
                logger.debug(f"ℹ️ Usando features existentes para image_id={image_id}")
            else:
                # Descargar y procesar imagen
                image = self.download_image(image_url)
                if image is None:
                    self.stats["not_found"] += 1
                    return False
                
                # Extraer features
                features = self.extract_features(image)
                if features is None:
                    self.stats["errors"] += 1
                    return False
            
            # Preparar metadata
            metadata = {
                "category": category,
                "user_id": user_id,
                "title": title,
                "image_url": image_url,
                "migrated_at": datetime.now().isoformat()
            }
            
            # Insertar en ChromaDB (si no es dry-run)
            if not dry_run:
                success = self.vector_store.add_embedding(
                    image_id=image_id,
                    vector=features,
                    metadata=metadata
                )
                if not success:
                    self.stats["errors"] += 1
                    return False
            
            self.stats["success"] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrando image_id={image_id}: {e}")
            self.stats["errors"] += 1
            return False
    
    def run_migration(
        self,
        dry_run: bool = False,
        incremental: bool = True,
        limit: Optional[int] = None
    ):
        """
        Ejecuta el proceso de migración.
        
        Args:
            dry_run: Si es True, solo muestra preview
            incremental: Si es True, solo migra imágenes nuevas
            limit: Límite de imágenes a procesar
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("🚀 INICIANDO MIGRACIÓN DE VECTORES")
        logger.info(f"   Modo: {'DRY-RUN (preview)' if dry_run else 'EJECUCIÓN REAL'}")
        logger.info(f"   Tipo: {'Incremental' if incremental else 'Completa'}")
        logger.info(f"   Límite: {limit or 'Sin límite'}")
        logger.info("=" * 60)
        
        # Obtener imágenes a migrar
        images = self.get_images_to_migrate(incremental, limit)
        
        if not images:
            logger.info("ℹ️ No hay imágenes para migrar")
            return
        
        # Barra de progreso
        pbar = tqdm(
            images,
            desc="Migrando imágenes",
            unit="img",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Procesar cada imagen
        for image_doc in pbar:
            image_id = image_doc.get("image_id")
            
            # Actualizar descripción
            pbar.set_postfix({
                "ID": image_id,
                "Éxitos": self.stats["success"],
                "Errores": self.stats["errors"]
            })
            
            # Migrar imagen
            success = self.migrate_single_image(image_doc, dry_run)
            
            self.stats["total_processed"] += 1
            
            # Log detallado cada 10 imágenes
            if self.stats["total_processed"] % 10 == 0:
                logger.info(
                    f"📊 Progreso: {self.stats['total_processed']}/{len(images)} "
                    f"({self.stats['success']} éxitos, {self.stats['errors']} errores)"
                )
        
        pbar.close()
        
        # Resumen final
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE MIGRACIÓN")
        logger.info("=" * 60)
        logger.info(f"   Total procesadas: {self.stats['total_processed']}")
        logger.info(f"   Exitosas: {self.stats['success']}")
        logger.info(f"   Errores: {self.stats['errors']}")
        logger.info(f"   No encontradas: {self.stats['not_found']}")
        logger.info(f"   Tiempo total: {elapsed_time:.2f} segundos")
        logger.info(f"   Velocidad: {self.stats['total_processed']/elapsed_time:.2f} img/seg")
        logger.info("=" * 60)
        
        if dry_run:
            logger.info("ℹ️ Modo DRY-RUN: No se insertaron vectores en ChromaDB")
            logger.info("   Ejecuta con --execute para realizar la migración real")
    
    def show_stats(self):
        """Muestra estadísticas de la migración."""
        stats = self.vector_store.get_collection_stats()
        
        logger.info("=" * 60)
        logger.info("📊 ESTADÍSTICAS DE CHROMADB")
        logger.info("=" * 60)
        logger.info(f"   Colección: {stats.get('collection_name', 'N/A')}")
        logger.info(f"   Total embeddings: {stats.get('total_embeddings', 0)}")
        logger.info(f"   Dimensión: {stats.get('feature_size', 'N/A')}")
        logger.info(f"   Directorio: {stats.get('persist_directory', 'N/A')}")
        
        if stats.get('sample_ids'):
            logger.info(f"   IDs de ejemplo: {stats['sample_ids'][:5]}")
        
        logger.info("=" * 60)
    
    def close(self):
        """Cierra las conexiones."""
        self.mongo_client.close()
        logger.info("🔌 Conexiones cerradas")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Migrar embeddings de imágenes de MongoDB a ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python migrate_vectors.py --dry-run              # Preview sin cambios (CLIP por defecto)
  python migrate_vectors.py --execute              # Migrar todas con CLIP
  python migrate_vectors.py --execute --model resnet  # Migrar con ResNet50
  python migrate_vectors.py --execute --incremental  # Solo nuevas
  python migrate_vectors.py --execute --limit 50   # Limitar cantidad
  python migrate_vectors.py --stats                # Ver estadísticas
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar preview sin realizar cambios"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Ejecutar la migración real"
    )
    
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Solo migrar imágenes nuevas (por defecto: True)"
    )
    
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Migrar todas las imágenes (ignora --incremental)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite de imágenes a procesar"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estadísticas de ChromaDB"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip", "resnet"],
        default="clip",
        help="Modelo para extracción de features: clip (512d, recomendado) o resnet (2048d). Default: clip"
    )
    
    parser.add_argument(
        "--mongo-url",
        default=os.getenv("MONGO_URL", "mongodb://localhost:27017"),
        help="URL de conexión a MongoDB"
    )
    
    parser.add_argument(
        "--db-name",
        default=os.getenv("DB_NAME", "galery"),
        help="Nombre de la base de datos"
    )
    
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db",
        help="Directorio de persistencia ChromaDB"
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.dry_run and not args.execute and not args.stats:
        parser.error("Debes especificar --dry-run, --execute o --stats")
    
    if args.execute and args.dry_run:
        parser.error("No puedes usar --execute y --dry-run juntos")
    
    try:
        # Crear migrador
        migrator = VectorMigrator(
            mongo_url=args.mongo_url,
            db_name=args.db_name,
            chroma_dir=args.chroma_dir,
            model_type=args.model
        )
        
        # Ejecutar según comando
        if args.stats:
            migrator.show_stats()
        else:
            incremental = not args.complete
            migrator.run_migration(
                dry_run=args.dry_run,
                incremental=incremental,
                limit=args.limit
            )
        
        # Cerrar conexiones
        migrator.close()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Migración interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
