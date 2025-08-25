from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64
from io import BytesIO
from PIL import Image


class FeatureExtractor:
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def extract(self, image_base64: str) -> list:
        """Extrae características de una imagen en base64"""
        try:
            # Decodificar base64
            img_data = base64.b64decode(image_base64.split(",")[1])
            img = Image.open(BytesIO(img_data)).resize((224, 224))
            
            # Convertir a array y preprocesar
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Extraer características
            features = self.model.predict(img_array)
            return features.flatten().tolist()
        except Exception as e:
            print(f"Error extrayendo características: {e}")
            return []