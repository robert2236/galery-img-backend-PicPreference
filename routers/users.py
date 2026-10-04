from fastapi import FastAPI, HTTPException, Depends, Request, status, APIRouter, Header,BackgroundTasks, Query, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from database.databases import user
from pymongo import MongoClient
from utils.auth.oauth import get_current_user
from utils.auth.jwttoken import create_access_token
from utils.auth.hashing import Hash
from utils.services.email import send_email
from utils.auth.token import generate_short_token, verify_short_token
from fastapi_pagination import Page, add_pagination, paginate
from models.users import User, UserUpdate,ResetPasswordRequest
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
import base64
from datetime import datetime, timedelta, timezone
import asyncio
import logging
import random
import uuid
import shutil
from typing import List, Dict
from pathlib import Path as PathLib

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

temporary_tokens = {}
timers = {}

@users.get("/api/users")
async def read_users_me(current_user: User = Depends(get_current_user)):

    return {
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "image": current_user.get("image"),
        "user_id": current_user.get("user_id"),
        "admin": current_user.get("admin", False),
        "name": current_user.get("name"),
        "surname": current_user.get("surname"),
        "info": current_user.get("info"),
        "web": current_user.get("web"),
        "theme": current_user.get("theme", False)
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
async def login(username: str = Form(...), password: str = Form(...)):
    userSearch = await user.find_one({"username": username})  
    if not userSearch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not Hash.verify(userSearch["password"], password):
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

# ============ PROFILE PICTURE ENDPOINTS ============

ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
UPLOAD_DIR = "uploads"

@users.post("/api/users/profile-picture")
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a profile picture for the authenticated user.
    - Accepts JPEG, PNG, GIF, WebP formats
    - Maximum file size: 5MB
    - Saves to uploads/ directory with UUID filename
    - Updates user's image field in database
    """
    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    # Read file contents
    contents = await file.read()
    
    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )
    
    # Create uploads directory if it doesn't exist
    PathLib(UPLOAD_DIR).mkdir(exist_ok=True)
    
    # Generate unique filename
    ext_map = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp"
    }
    extension = ext_map.get(file.content_type, "jpg")
    filename = f"profile_{current_user['user_id']}_{uuid.uuid4().hex}.{extension}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    # Save file
    with open(filepath, "wb") as f:
        f.write(contents)
    
    # Update user's image field in database
    image_url = f"/uploads/{filename}"
    await user.update_one(
        {"username": current_user["username"]},
        {"$set": {"image": image_url}}
    )
    
    # Generate new access token
    new_access_token = create_access_token(data={"sub": current_user["username"]})
    
    return {
        "res": "Profile picture uploaded successfully",
        "image_url": image_url,
        "new_access_token": new_access_token
    }

@users.get("/api/users/{user_id}/profile-picture")
async def get_profile_picture(user_id: int):
    """
    Get profile picture for a user by user_id.
    - Returns the image URL if user has a custom profile picture
    - Returns default avatar if no custom picture is set
    """
    # Find user by user_id
    user_data = await user.find_one({"user_id": user_id})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    image_url = user_data.get("image")
    
    # If no image or still using base64 default, return default avatar path
    if not image_url or image_url.startswith("data:"):
        # Return default avatar
        default_path = os.path.join("static", "user.png")
        if os.path.exists(default_path):
            return {
                "user_id": user_id,
                "image_url": "/static/user.png",
                "is_default": True
            }
        else:
            return {
                "user_id": user_id,
                "image_url": None,
                "is_default": True,
                "message": "No profile picture available"
            }
    
    return {
        "user_id": user_id,
        "image_url": image_url,
        "is_default": False
    }

@users.delete("/api/users/profile-picture")
async def delete_profile_picture(current_user: dict = Depends(get_current_user)):
    """
    Delete the authenticated user's profile picture and reset to default.
    """
    user_data = await user.find_one({"username": current_user["username"]})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    current_image = user_data.get("image", "")
    
    # Delete the file if it's a custom upload (not base64 or default)
    if current_image and current_image.startswith("/uploads/"):
        filepath = os.path.join(UPLOAD_DIR, current_image.split("/")[-1])
        if os.path.exists(filepath):
            os.remove(filepath)
    
    # Reset to default avatar
    default_path = os.path.join("static", "user.png")
    if os.path.exists(default_path):
        default_image = f"data:image/png;base64,{get_image_base64(default_path)}"
    else:
        default_image = None
    
    await user.update_one(
        {"username": current_user["username"]},
        {"$set": {"image": default_image}}
    )
    
    new_access_token = create_access_token(data={"sub": current_user["username"]})
    
    return {
        "res": "Profile picture deleted successfully",
        "new_access_token": new_access_token
    }

# ============ PUBLIC PROFILE ENDPOINT ============

@users.get("/api/users/{user_id}/profile")
async def get_public_profile(user_id: str):
    """
    Get public profile for a user (no authentication required).
    Accepts either numeric user_id or username string.
    """
    # Try to find by numeric user_id first
    user_data = None
    try:
        numeric_id = int(user_id)
        user_data = await user.find_one({"user_id": numeric_id})
    except (ValueError, TypeError):
        pass
    
    # If not found by numeric id, try by username
    if not user_data:
        user_data = await user.find_one({"username": user_id})
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Count saved images
    saved_images = user_data.get("saved_images", [])
    
    return {
        "user_id": user_data.get("user_id"),
        "username": user_data.get("username"),
        "name": user_data.get("name"),
        "surname": user_data.get("surname"),
        "image": user_data.get("image"),
        "info": user_data.get("info"),
        "web": user_data.get("web"),
        "profile_public": user_data.get("profile_public", True),
        "saved_images_count": len(saved_images)
    }

# ============ PROFILE VISIBILITY ENDPOINT ============

@users.put("/api/users/profile-visibility")
async def update_profile_visibility(
    profile_public: bool,
    current_user: dict = Depends(get_current_user)
):
    """
    Update profile visibility (public/private).
    - Public: Anyone can view the user's gallery
    - Private: Only the user can view their own gallery
    """
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    await user.update_one(
        {"user_id": user_id},
        {"$set": {"profile_public": profile_public}}
    )
    
    new_access_token = create_access_token(data={"sub": current_user["username"]})
    
    return {
        "res": "Profile visibility updated",
        "profile_public": profile_public,
        "new_access_token": new_access_token
    }
