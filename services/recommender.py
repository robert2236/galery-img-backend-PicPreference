import numpy as np
from sklearn.neighbors import KDTree
from utils.feature_extractor import FeatureExtractor
from database.databases import coleccion
from bson import ObjectId
from datetime import datetime


class VisualRecommender:
    def __init__(self):
        self.tree = None
        self.image_ids = []
        self.extractor = FeatureExtractor()

    async def build_index(self):
        
        
        
        """Construye el índice KD-Tree con vectores de características"""
        images = list(coleccion.find({"features": {"$exists": True}}))
        if images:
            self.image_ids = [str(img["_id"]) for img in images]
            features = [img["features"] for img in images]
            self.tree = KDTree(np.array(features))

    async def log_interaction(self, image_id: str, action: str = "view"):
        """Registra una interacción con una imagen"""
        await coleccion.update_one(
            {"_id": ObjectId(image_id)},
            {
                "$inc": {f"interactions.{action}": 1},
                "$set": {"interactions.last_interaction": datetime.utcnow()}
            }
        )

    async def find_similar(self, image_id: str, k=5):
        """Encuentra imágenes similares usando KD-Tree"""
        if not self.tree:
            await self.build_index()

        target = await coleccion.find_one({"_id": ObjectId(image_id)})
        if not target or "features" not in target:
            return []

        _, indices = self.tree.query([target["features"]], k=k+1)
        return [self.image_ids[i] for i in indices[0] if self.image_ids[i] != image_id][:k]