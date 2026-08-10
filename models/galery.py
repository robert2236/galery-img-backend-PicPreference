from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from bson import ObjectId


class ImageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: Optional[int] = None
    title: Optional[str] = None
    image_url: str
    category: str
    username: Optional[str] = None
    liked_by: List[str] = Field(default_factory=list)
    comments: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("liked_by", mode="before")
    @classmethod
    def convert_liked_by(cls, v):
        if isinstance(v, list):
            return [str(item) if isinstance(item, ObjectId) else item for item in v]
        return v


class Image(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(None, alias='_id')
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

    ai_features: Optional[dict] = Field(
        default_factory=lambda: {
            "visual_embedding": [],
            "auto_tags": [],
            "detected_objects": [],
            "color_palette": [],
            "scene_type": None
        }
    )

    social_features: Optional[dict] = Field(
        default_factory=lambda: {
            "comment_sentiment": 0.0,
            "comment_keywords": [],
            "popularity_score": 0.0
        }
    )

    qualification: Optional[int] = Field(None, ge=1, le=5)

    liked_by: List[Union[str, int]] = Field(default_factory=list)
    comments: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def convert_objectid(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        return v


class UpdateImage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(None, alias='_id')
    image_url: Optional[str] = None

    @field_validator("id", mode="before")
    @classmethod
    def convert_objectid(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        return v


class InteractionUpdate(BaseModel):
    action: str
    increment: int = 1


class CommentCreate(BaseModel):
    comment: str = Field(..., min_length=1, max_length=500, description="Comentario de la imagen")
    parent_comment_id: Optional[int] = Field(None, description="ID del comentario padre si es una respuesta")


class QualificationResponse(BaseModel):
    image_id: int
    qualification: int = Field(ge=1, le=5)
    likes: int
    views: int
    ratio: float
