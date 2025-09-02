# services/recommender.py
import numpy as np
from sklearn.neighbors import KDTree
from utils.feature_extractor import FeatureExtractor
from database.databases import coleccion
from datetime import datetime

class VisualRecommender:
    def __init__(self):
        self.tree = None
        self.image_ids = []  # Almacenará los IDs numéricos
        self.features = []
        self.extractor = FeatureExtractor()

    async def build_index(self):
        """Construye el índice KD-Tree con vectores de características"""
        try:
            images = await coleccion.find({"features": {"$exists": True}}).to_list(None)
            if images:
                self.image_ids = [img.get("image_id") for img in images if img.get("image_id") is not None]
                self.features = [img["features"] for img in images if "features" in img]
                
                if self.features and self.image_ids:
                    self.tree = KDTree(np.array(self.features))
                    print(f"✅ Índice KDTree construido con {len(images)} imágenes")
                else:
                    print("⚠️ No hay características o image_ids para construir el índice")
        except Exception as e:
            print(f"❌ Error construyendo índice: {e}")

    async def find_similar(self, image_id: str, k=5):
        """Encuentra imágenes similares usando KD-Tree"""
        try:
            if not self.tree:
                await self.build_index()
                if not self.tree:
                    return []

            # Convertir a entero y buscar por image_id numérico
            target_id = int(image_id)
            target = await coleccion.find_one({"image_id": target_id})
            
            if not target or "features" not in target:
                print(f"❌ Imagen {target_id} no encontrada o sin características")
                return []

            # Encontrar las k+1 imágenes más similares
            distances, indices = self.tree.query([target["features"]], k=min(k+1, len(self.image_ids)))
            
            # Excluir la imagen original y obtener IDs numéricos
            similar_ids = []
            for i in indices[0]:
                if i < len(self.image_ids):
                    candidate_id = self.image_ids[i]
                    if candidate_id != target_id:
                        similar_ids.append(candidate_id)
            
            print(f"🔍 Similares para {target_id}: {similar_ids}")
            return similar_ids[:k]
            
        except Exception as e:
            print(f"❌ Error buscando imágenes similares: {e}")
            return []

# Crear la instancia global
visual_recommender = VisualRecommender()