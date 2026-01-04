import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import pipeline
import base64
from io import BytesIO

class ImageAIAnalyzer:
    def __init__(self):
        # Modelo para extraer características visuales
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # Modelo para etiquetado automático
        self.tagging_model = pipeline(
            "image-classification", 
            model="microsoft/resnet-50"
        )
        
        # Modelo para detección de objetos
        self.detector = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50"
        )
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def analyze_image(self, image_input) -> dict:
        """
        Procesa una imagen desde diferentes fuentes:
        - Ruta de archivo (str)
        - Objeto PIL Image
        - Base64 string
        """
        # Determinar tipo de entrada
        if isinstance(image_input, str):
            if image_input.startswith("data:image") or len(image_input) > 1000:
                # Es base64
                image = self._base64_to_pil(image_input)
            else:
                # Es ruta de archivo
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            # Ya es PIL Image
            image = image_input
        else:
            raise ValueError("Tipo de imagen no soportado")
        
        return self._analyze_pil_image(image)
    
    def _analyze_pil_image(self, image: Image.Image) -> dict:
        """Procesa un objeto PIL Image"""
        try:
            # 1. Extraer embedding visual
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.model(image_tensor)
            visual_embedding = features.numpy().flatten().tolist()
            
            # 2. Generar etiquetas automáticas
            tags_result = self.tagging_model(image)
            auto_tags = [tag['label'] for tag in tags_result[:5]]
            
            # 3. Detectar objetos
            objects_result = self.detector(image)
            detected_objects = [obj['label'] for obj in objects_result[:3]]
            
            # 4. Extraer paleta de colores
            color_palette = self.extract_color_palette(image)
            
            # 5. Clasificar tipo de escena
            scene_type = self.classify_scene(image)
            
            return {
                "visual_embedding": visual_embedding,
                "auto_tags": auto_tags,
                "detected_objects": detected_objects,
                "color_palette": color_palette,
                "scene_type": scene_type
            }
            
        except Exception as e:
            print(f"Error en análisis de imagen: {str(e)}")
            return self._get_empty_ai_features()
    
    def _base64_to_pil(self, base64_string: str) -> Image.Image:
        """Convierte base64 a PIL Image"""
        try:
            # Limpiar el string base64
            if "base64," in base64_string:
                base64_string = base64_string.split("base64,")[1]
            
            # Decodificar
            image_data = base64.b64decode(base64_string)
            
            # Crear objeto PIL
            image = Image.open(BytesIO(image_data))
            return image.convert("RGB")
            
        except Exception as e:
            raise ValueError(f"Base64 inválido: {str(e)}")
    
    def extract_color_palette(self, image: Image.Image, n_colors: int = 5) -> list:
        """Extrae los colores predominantes"""
        try:
            # Redimensionar para procesamiento más rápido
            image_small = image.resize((100, 100))
            
            # Convertir a array numpy
            img_array = np.array(image_small)
            
            # Aplanar los píxeles
            pixels = img_array.reshape(-1, 3)
            
            # Usar k-means simple (o quantize para aproximación)
            from sklearn.cluster import KMeans
            
            if len(pixels) > n_colors:
                kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_
                
                # Convertir a hex
                color_palette = [
                    f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
                    for c in colors
                ]
                return color_palette
            
            return []
            
        except Exception as e:
            print(f"Error extrayendo paleta: {str(e)}")
            return []
    
    def classify_scene(self, image: Image.Image) -> str:
        """Clasifica tipo de escena básica"""
        try:
            # Convertir a escala de grises
            grayscale = image.convert('L')
            brightness = np.mean(grayscale)
            
            # Clasificación simple basada en brillo y color
            if brightness > 200:
                return "exterior_dia"
            elif brightness < 50:
                return "noche"
            elif brightness < 150:
                # Verificar si es interior por distribución de colores
                img_array = np.array(image)
                std_colors = np.std(img_array, axis=(0, 1))
                if np.mean(std_colors) < 30:
                    return "interior"
                else:
                    return "exterior"
            else:
                return "exterior"
                
        except Exception as e:
            print(f"Error clasificando escena: {str(e)}")
            return "desconocido"
    
    def _get_empty_ai_features(self) -> dict:
        """Devuelve estructura vacía de AI features"""
        return {
            "visual_embedding": [],
            "auto_tags": [],
            "detected_objects": [],
            "color_palette": [],
            "scene_type": None
        }