from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        try:
            self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.feature_size = 2048  # ResNet50 con pooling avg produce 2048 features
            logger.info("✅ Modelo ResNet50 cargado exitosamente")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo ResNet50: {e}")
            self.model = None
            self.feature_size = 2048

    def extract(self, image_base64: str) -> list:
        """Extrae características de una imagen en base64"""
        try:
            if self.model is None:
                logger.error("Modelo no disponible para extracción")
                return []
            
            # Decodificar base64
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
                
            img_data = base64.b64decode(image_base64)
            img = Image.open(BytesIO(img_data)).convert('RGB').resize((224, 224))
            
            # Convertir a array y preprocesar para ResNet50
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocesamiento específico para ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input
            img_array = preprocess_input(img_array)
            
            # Extraer características
            features = self.model.predict(img_array, verbose=0)
            features = features.flatten()
            
            # Verificar que tenga la forma correcta (2048 para ResNet50)
            if features.shape[0] != self.feature_size:
                logger.warning(f"⚠️ Características con forma inesperada: {features.shape}, esperado: {self.feature_size}")
                # Ajustar a la forma correcta
                if features.shape[0] < self.feature_size:
                    features = np.pad(features, (0, self.feature_size - features.shape[0]))
                else:
                    features = features[:self.feature_size]
            
            logger.info(f"✅ Características extraídas: forma {features.shape}")
            return features.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo características: {e}")
            return []

    def extract_from_url(self, image_url: str) -> list:
        """Extrae características desde una URL (para compatibilidad)"""
        try:
            import requests
            from io import BytesIO
            
            response = requests.get(image_url)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content)).convert('RGB').resize((224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            from tensorflow.keras.applications.resnet50 import preprocess_input
            img_array = preprocess_input(img_array)
            
            features = self.model.predict(img_array, verbose=0)
            features = features.flatten()
            
            if features.shape[0] != self.feature_size:
                if features.shape[0] < self.feature_size:
                    features = np.pad(features, (0, self.feature_size - features.shape[0]))
                else:
                    features = features[:self.feature_size]
            
            return features.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo desde URL {image_url}: {e}")
            return []