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
            
            # Normalización L2 para estabilidad numérica y mejor similitud coseno
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            logger.info(f"✅ Características extraídas: forma {features.shape}")
            return features.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo características: {e}")
            return []

    def extract_with_augmentation(self, image_base64: str) -> list:
        """Extrae características con test-time augmentation (original + flip horizontal)"""
        try:
            if self.model is None:
                logger.error("Modelo no disponible para extracción")
                return []
            
            # Decodificar base64
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
                
            img_data = base64.b64decode(image_base64)
            img = Image.open(BytesIO(img_data)).convert('RGB').resize((224, 224))
            
            from tensorflow.keras.applications.resnet50 import preprocess_input
            
            # Extraer de imagen original
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features_orig = self.model.predict(img_array, verbose=0).flatten()
            
            # Extraer de imagen con flip horizontal
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip_array = img_to_array(img_flipped)
            img_flip_array = np.expand_dims(img_flip_array, axis=0)
            img_flip_array = preprocess_input(img_flip_array)
            features_flip = self.model.predict(img_flip_array, verbose=0).flatten()
            
            # Promediar ambas características
            features = (features_orig + features_flip) / 2.0
            
            # Ajustar dimensión si es necesario
            if features.shape[0] != self.feature_size:
                if features.shape[0] < self.feature_size:
                    features = np.pad(features, (0, self.feature_size - features.shape[0]))
                else:
                    features = features[:self.feature_size]
            
            # Normalización L2
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            logger.info(f"✅ Características con augmentation extraídas: forma {features.shape}")
            return features.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo con augmentation: {e}")
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
            
            # Normalización L2
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo desde URL {image_url}: {e}")
            return []


class CLIPFeatureExtractor:
    """
    Extractor de características usando CLIP (Contrastive Language-Image Pre-training).
    
    CLIP produce embeddings de 512 dimensiones que capturan similitud semántica
    entre imágenes, mucho mejor que ResNet50 para búsqueda visual por similitud.
    
    Requiere: transformers, torch/torchvision
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.feature_size = 512  # CLIP ViT-B/32 produces 512-dim embeddings
            self.model.eval()
            
            logger.info(f"✅ CLIP modelo cargado: {model_name} en {self.device}")
            logger.info(f"   Dimensión de embeddings: {self.feature_size}")
            
        except ImportError as e:
            logger.error(f"❌ Dependencia faltante para CLIP: {e}")
            logger.error("   Instalar con: pip install transformers torch torchvision")
            self.model = None
            self.processor = None
            self.feature_size = 512
        except Exception as e:
            logger.error(f"❌ Error cargando CLIP: {e}")
            self.model = None
            self.processor = None
            self.feature_size = 512
    
    def _decode_image(self, image_input: str) -> Image.Image:
        """Decodifica imagen desde base64 o URL."""
        import requests
        
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        
        # Base64
        if "," in image_input:
            image_input = image_input.split(",")[1]
        
        img_data = base64.b64decode(image_input)
        return Image.open(BytesIO(img_data)).convert("RGB")
    
    def _extract_features(self, images: list) -> np.ndarray:
        """Extrae embeddings CLIP de una lista de imágenes PIL."""
        import torch
        
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)
        
        # L2-normalizar
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
    
    def extract(self, image_input: str) -> list:
        """
        Extrae embedding CLIP de una imagen.
        
        Args:
            image_input: base64 string o URL de la imagen
        
        Returns:
            Lista de 512 floats (embedding normalizado)
        """
        try:
            if self.model is None:
                logger.error("Modelo CLIP no disponible")
                return []
            
            img = self._decode_image(image_input)
            features = self._extract_features([img])
            embedding = features[0]
            
            logger.info(f"✅ CLIP embedding extraído: dim={embedding.shape[0]}")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo CLIP embedding: {e}")
            return []
    
    def extract_with_augmentation(self, image_input: str) -> list:
        """
        Extrae embedding CLIP con augmentation (original + flip horizontal).
        
        Promedia los embeddings de la imagen original y su versión horizontal
        para mayor robustez ante simetría.
        """
        try:
            if self.model is None:
                logger.error("Modelo CLIP no disponible")
                return []
            
            img = self._decode_image(image_input)
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            features = self._extract_features([img, img_flipped])
            
            # Promediar original + flip
            embedding = (features[0] + features[1]) / 2.0
            
            # Re-normalizar después del promedio
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            logger.info(f"✅ CLIP embedding con augmentation: dim={embedding.shape[0]}")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo CLIP con augmentation: {e}")
            return []
    
    def extract_from_url(self, image_url: str) -> list:
        """Extrae embedding CLIP desde una URL."""
        return self.extract(image_url)
    
    def extract_batch(self, image_inputs: list) -> list:
        """
        Extrae embeddings CLIP de múltiples imágenes en batch (más eficiente).
        
        Args:
            image_inputs: Lista de base64 strings o URLs
        
        Returns:
            Lista de embeddings (cada uno es lista de 512 floats)
        """
        try:
            if self.model is None:
                logger.error("Modelo CLIP no disponible")
                return [[] for _ in image_inputs]
            
            images = [self._decode_image(inp) for inp in image_inputs]
            all_features = self._extract_features(images)
            
            logger.info(f"✅ CLIP batch extraído: {len(all_features)} imágenes")
            return [feat.tolist() for feat in all_features]
            
        except Exception as e:
            logger.error(f"❌ Error en CLIP batch: {e}")
            return [[] for _ in image_inputs]