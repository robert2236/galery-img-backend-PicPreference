from typing import Optional
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from typing import Optional, List
import re


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, _):
        """Acepta 2 argumentos (valor y campo) para compatibilidad con Pydantic"""
        if isinstance(v, ObjectId):
            return str(v)
        if ObjectId.is_valid(v):
            return str(v)
        raise ValueError("Invalid ObjectId")

class ImageResponse(BaseModel):
    image_id: Optional[int] = None
    image_url: str
    category: str
    username: Optional[str] = None

       
class Image(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    image_id: Optional[int] = None
    user_id: int
    username: Optional[str] = None
    image_url: str
    category: str
    title: Optional[str] = None
    features: Optional[list] = None
    interactions: dict = Field(
        default_factory=lambda: {
            "likes": 0,
            "downloads": 0,
            "views": 0,
            "last_interaction": None
        }
    )
    liked_by: List[str] = Field(default_factory=list) 




    
    @validator('image_url')
    def validate_image_data(cls, v):
        # Opcional: agregar validación para distinguir entre URL y Base64
        if not (v.startswith('http') or v.startswith('data:image')):
            # Aquí podrías agregar validación más estricta del Base64
            if len(v) > 10_000_000:  # Ejemplo: límite de ~10MB
                raise ValueError("Image data too large")
        return v
    

class UpdateImage(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    image_url: Optional[str] = None
    
class InteractionUpdate(BaseModel):
    action: str  # "likes", "downloads", "views"
    increment: int = 1  # Valor a incrementar (por defecto 1)
    

class Config:
        orm_mode = True
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str
        }