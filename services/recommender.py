import numpy as np
from sklearn.neighbors import KDTree
import logging
from database.databases import coleccion
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualRecommender:
    def __init__(self):
        self.tree = None
        self.features = []
        self.image_ids = []
        self.feature_size = 2048  # ¡CAMBIADO de 4096 a 2048 para ResNet50!

    async def build_index(self):
        """Construye el índice KDTree con el tamaño correcto"""
        try:
            logger.info("🔄 Construyendo índice visual para ResNet50 (2048 features)...")
            
            # Obtener todas las imágenes con características
            images = await coleccion.find({
                "features": {"$exists": True},
                "features": {"$ne": None}
            }).to_list(None)
            
            if not images:
                logger.warning("⚠️ No hay imágenes con características para construir índice")
                return False
            
            self.features = []
            self.image_ids = []
            
            # Procesar características asegurando forma consistente
            valid_count = 0
            for image in images:
                try:
                    features = image.get("features")
                    if features is None:
                        continue
                    
                    # Convertir a numpy array
                    features = np.array(features)
                    
                    # Asegurar forma correcta (2048 para ResNet50)
                    if features.shape[0] != self.feature_size:
                        logger.warning(f"⚠️ Características con forma incorrecta {features.shape} para imagen {image.get('image_id')}")
                        
                        # Corregir forma
                        if features.shape[0] < self.feature_size:
                            features = np.pad(features, (0, self.feature_size - features.shape[0]))
                        else:
                            features = features[:self.feature_size]
                    
                    # Verificar que no sea todo ceros
                    if np.all(features == 0) or np.isnan(features).any():
                        logger.warning(f"⚠️ Características inválidas para imagen {image.get('image_id')}")
                        continue
                    
                    self.features.append(features)
                    self.image_ids.append(image.get("image_id"))
                    valid_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando imagen {image.get('image_id')}: {e}")
                    continue
            
            if valid_count < 2:
                logger.warning("⚠️ No hay suficientes imágenes válidas para construir índice")
                self.tree = None
                return False
            
            # Construir KDTree
            self.features = np.array(self.features)
            logger.info(f"📊 Características shape: {self.features.shape}")
            
            self.tree = KDTree(self.features, leaf_size=40)
            logger.info(f"✅ Índice ResNet50 construido con {valid_count} imágenes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error construyendo índice: {e}")
            import traceback
            traceback.print_exc()
            self.tree = None
            return False

    async def find_similar(self, image_id, k=5):
        """Encuentra imágenes similares"""
        if self.tree is None:
            logger.warning("⚠️ Índice no disponible")
            return []
        
        try:
            # Buscar la imagen de referencia
            ref_image = await coleccion.find_one({"image_id": int(image_id)})
            if not ref_image or "features" not in ref_image:
                logger.warning(f"⚠️ Imagen {image_id} no encontrada o sin características")
                return []
            
            features = ref_image["features"]
            features = np.array(features)
            
            # Asegurar forma correcta (2048)
            if features.shape[0] != self.feature_size:
                if features.shape[0] < self.feature_size:
                    features = np.pad(features, (0, self.feature_size - features.shape[0]))
                else:
                    features = features[:self.feature_size]
            
            # Buscar similares
            distances, indices = self.tree.query([features], k=k+1)
            
            # Excluir la imagen misma
            similar_ids = []
            for idx in indices[0]:
                if idx < len(self.image_ids) and self.image_ids[idx] != int(image_id):
                    similar_ids.append(self.image_ids[idx])
                if len(similar_ids) >= k:
                    break
            
            return similar_ids[:k]
            
        except Exception as e:
            logger.error(f"❌ Error buscando similares: {e}")
            return []
        
visual_recommender = VisualRecommender()