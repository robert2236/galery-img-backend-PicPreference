from pydantic import BaseModel
from typing import Optional, List
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, values, **kwargs):
     if not ObjectId.is_valid(v):
        raise ValueError('Invalid ObjectId')
     return str(v)

class User(BaseModel):
    username: str
    email: str
    name: Optional[str] = None
    surname: Optional[str] = None
    info: Optional[str] = None
    web: Optional[str] = None
    password: Optional[str] = None
    image: Optional[str] = None
    admin: bool = False
    theme: bool = False
    images: List[int] = []
    user_id: Optional[int] = None
    
    class Config:
        # Permitir el uso de alias para la serialización
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    images: List[int] = []
    name: Optional[str] = None
    surname: Optional[str] = None
    image: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    info: Optional[str] = None
    web: Optional[str] = None
    theme: bool = False

        
class Login(BaseModel):
    username: str
    password: str
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: Optional[str] = None