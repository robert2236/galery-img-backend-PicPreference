from fastapi import FastAPI, HTTPException, Depends, Request, status, APIRouter, Header,BackgroundTasks, Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from database.databases import user
from pymongo import MongoClient
from utils.auth.oauth import get_current_user
from utils.auth.jwttoken import create_access_token
from utils.auth.hashing import Hash
from utils.services.email import send_email
from utils.auth.token import generate_short_token, verify_short_token
from fastapi.security import OAuth2PasswordRequestForm,OAuth2PasswordBearer
from fastapi_pagination import Page, add_pagination, paginate
from models.users import User, UserUpdate,ResetPasswordRequest
from jose import JWTError, jwt
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
import base64
from datetime import datetime, timedelta, timezone
import asyncio
import logging
import random
from typing import List, Dict

users = APIRouter()

class PermissionChecker:
    def __init__(self, required_permissions: List[str]) -> None:
        self.required_permissions = required_permissions

    def __call__(self, user: User = Depends(get_current_user)) -> bool:
        # Verifica si el usuario tiene los permisos requeridos
        for r_perm in self.required_permissions:
            if r_perm not in user.permissions:  # Asegúrate de que 'permissions' esté en el modelo User
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='No tienes permisos suficientes para acceder a este recurso'
                )
        return True

logging.basicConfig(level=logging.INFO)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
temporary_tokens = {}
timers = {}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    db_user = await user.find_one({"username": username})
    if db_user is None:
        raise credentials_exception
    return db_user

@users.get("/api/users")
async def read_users_me(current_user: User = Depends(get_current_user)):

    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "image": current_user["image"],
        "user_id": current_user["user_id"],
        "admin": current_user["admin"],
        "name": current_user["name"],
        "surname": current_user["surname"],
        "info": current_user["info"],
        "web": current_user["web"],
        "theme": current_user["theme"]
    }

def get_image_base64(image_path: str) -> str:
    """Convierte una imagen a base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

DEFAULT_IMAGE_PATH = os.path.join("static", "user.png")

@users.post('/api/register')
async def create_user(request: User):
    if not request.username.strip() or not request.email.strip() or not request.password.strip():
        raise HTTPException(status_code=400, detail="fields cannot be empty")
    
    existing_user = await user.find_one({"username": request.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    existing_email = await user.find_one({"email": request.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    hashed_pass = Hash.bcrypt(request.password)
    user_object = dict(request)
    user_object["password"] = hashed_pass

    
    
    # Asignar la imagen por defecto
    user_object["image"] = f"data:image/png;base64,{get_image_base64(DEFAULT_IMAGE_PATH)}"
    
    while True:
        user_id = random.randint(1000, 9999)  # Genera un ID aleatorio de 4 dígitos
        if not await user.find_one({"user_id": user_id}):  # Verifica si el ID ya existe
            break
    
    user_object["user_id"] = user_id  # Asigna el user_id al objeto del usuario
    
    user_id = await user.insert_one(user_object)
    return {"res": "User created successfully"}

@users.post('/api/login')
async def login(request: OAuth2PasswordRequestForm = Depends()):
    userSearch = await user.find_one({"username": request.username})  
    if not userSearch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not Hash.verify(userSearch["password"], request.password):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    access_token = create_access_token(data={"sub": userSearch["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@users.post("/api/request-password-reset")
async def request_password_reset(email: str):
    user = await user.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    token = generate_short_token()

    temporary_tokens[token] = email
    
    send_email(
        to_email=email,
        subject="Password Reset Request",
        body=f"To reset your password, use this token: {token}"
    )

    return {"res": "Password reset token sent to your email."}

@users.put("/api/reset-password")
async def reset_password(request: ResetPasswordRequest):
    # Verificar si el token existe en el almacenamiento temporal
    email = temporary_tokens.get(request.token)
    if not email:
        raise HTTPException(status_code=404, detail="Token not found or invalid.")

    # Verificar si el usuario existe
    user = await user.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or email invalid")

    # Actualizar la contraseña
    hashed_pass = Hash.bcrypt(request.new_password)
    await user.update_one({"email": email}, {"$set": {"password": hashed_pass}})

    # Eliminar el token de la memoria después de usarlo
    del temporary_tokens[request.token]

    return {"res": "Password updated successfully."}

@users.put("/api/change-theme")
async def change_theme(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    update_data = {}
    if user_update.theme is not None:
        update_data["theme"] = user_update.theme
        
    if update_data:
        await user.update_one({"username": current_user["username"]}, {"$set": update_data})

    new_access_token = create_access_token(data={"sub": current_user["username"]})
    
    return {
        "res": "Tema cambiado",
        "new_access_token": new_access_token
    }

@users.put("/api/profile")
async def update_user_data(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    if update_data:
        await user.update_one({"username": current_user["username"]}, {"$set": update_data})

    new_access_token = create_access_token(data={"sub": current_user["username"]})
    
    
    
    # Crear un diccionario para los cambios
    update_data = {}
    if user_update.name is not None:
        update_data["name"] = user_update.name
    if user_update.surname is not None:
        update_data["surname"] = user_update.surname
    if user_update.username is not None:
        update_data["username"] = user_update.username
    if user_update.email is not None:
        update_data["email"] = user_update.email
    if user_update.info is not None:
        update_data["info"] = user_update.info
    if user_update.web is not None:
        update_data["web"] = user_update.web
    if user_update.image is not None:
        update_data["image"] = user_update.image

          
    # Actualizar el documento en la base de datos
    if update_data:
        await user.update_one({"username": current_user["username"]}, {"$set": update_data})

    new_access_token = create_access_token(data={"sub": current_user["username"]})

    return {
        "res": "User data updated successfully",
        "new_access_token": new_access_token
    }

@users.put("/api/update_password")
async def update_password(
    user_update: UserUpdate,
    token: str, 
    current_user: dict = Depends(get_current_user)
):
    if not verify_short_token(token):
        raise HTTPException(status_code=403, detail="Verification token invalid or expired")
    
    update_data = {}
    if user_update.password is not None:
        update_data["password"] = Hash.bcrypt(user_update.password)
    
    if update_data:
        await user.update_one({"username": current_user["username"]}, {"$set": update_data})

    new_access_token = create_access_token(data={"sub": current_user["username"]})

    return {
        "res": "password updated successfully",
        "new_access_token": new_access_token
    }
    
