from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class CategoryResponse(BaseModel):
    id: str = Field(..., description="ID único de la categoría")
    name: str = Field(..., min_length=2, max_length=50, description="Nombre de la categoría")
    description: Optional[str] = Field(None, max_length=200, description="Descripción de la categoría")
    tags: List[str] = Field(default_factory=list, description="Tags asociados a la categoría")
    is_active: bool = Field(True, description="Indica si la categoría está activa")
    created_at: datetime = Field(..., description="Fecha de creación de la categoría")
    updated_at: datetime = Field(..., description="Fecha de última actualización")

    class Config:
        schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "anime",
                "description": "Animación japonesa",
                "tags": ["japón", "animación", "manga"],
                "is_active": True,
                "created_at": "2023-12-01T10:30:00.000Z",
                "updated_at": "2023-12-01T10:30:00.000Z"
            }
        }

class CategorySimpleResponse(BaseModel):
    id: str = Field(..., description="ID único de la categoría")
    name: str = Field(..., description="Nombre de la categoría")

    class Config:
        schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "anime"
            }
        }

class CategoryNameResponse(BaseModel):
    name: str = Field(..., description="Nombre de la categoría")

    class Config:
        schema_extra = {
            "example": {
                "name": "anime"
            }
        }

class CategoriesResponse(BaseModel):
    categories: List[CategoryResponse] = Field(..., description="Lista de categorías")
    total: int = Field(..., description="Número total de categorías que coinciden con el filtro")
    skip: int = Field(0, description="Número de categorías saltadas (paginación)")
    limit: int = Field(50, description="Límite de resultados por página")

    class Config:
        schema_extra = {
            "example": {
                "categories": [
                    {
                        "id": "507f1f77bcf86cd799439011",
                        "name": "anime",
                        "description": "Animación japonesa",
                        "tags": ["japón", "animación", "manga"],
                        "is_active": True,
                        "created_at": "2023-12-01T10:30:00.000Z",
                        "updated_at": "2023-12-01T10:30:00.000Z"
                    }
                ],
                "total": 1,
                "skip": 0,
                "limit": 50
            }
        }

class CategoryListResponse(BaseModel):
    categories: List[CategorySimpleResponse] = Field(..., description="Lista simplificada de categorías")

class CategoryNamesResponse(BaseModel):
    categories: List[str] = Field(..., description="Lista de nombres de categorías")
    total: int = Field(..., description="Número total de categorías")

    class Config:
        schema_extra = {
            "example": {
                "categories": ["anime", "videojuegos", "informatica"],
                "total": 3
            }
        }