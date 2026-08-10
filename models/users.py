from pydantic import BaseModel, ConfigDict
from typing import Optional, List


class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

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
    profile_public: bool = True
    images: List[int] = []
    saved_images: List[int] = []
    user_id: Optional[int] = None


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class UserUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    images: List[int] = []
    name: Optional[str] = None
    surname: Optional[str] = None
    image: Optional[str] = None
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
