from fastapi import APIRouter, HTTPException, Depends,status, BackgroundTasks
from database.galery import (get_all_images,create_image, get_one_image,delete_image)
from models.galery import Image,UpdateImage,ImageResponse,InteractionUpdate, CommentCreate, QualificationResponse
from models.users import User
from database.databases import user,coleccion
from utils.auth.oauth import get_current_user
import base64
import random
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from utils.feature_extractor import FeatureExtractor 
import numpy as np
from utils.auth.oauth import extract_user_id
import asyncio
from typing import Optional, List
from models.Pagination import PaginationParams
from fastapi import Path
from fastapi import HTTPException, Depends, Query
from typing import Optional
from datetime import datetime
import uuid
from pathlib import Path as PathLib
import os
from bson import ObjectId
try:
    from services.ai_processor import ai_processor
    from services.ai_image_processor import ImageAIAnalyzer
    from services.ai_comment_analyzer import CommentAIAnalyzer
except ImportError:
    ai_processor = None
    print("[WARNING] Modulos de IA no disponibles")

feature_extractor = FeatureExtractor()

galery= APIRouter()

image_cache = None
cache_timestamp = None
CACHE_DURATION = 10  # 5 minutos en segundos

class PaginationParams:
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Número de elementos a saltar"),
        limit: int = Query(20, ge=1, le=100, description="Límite de resultados por página"),
        page: int = Query(1, ge=1, description="Número de página (alternativo a skip)")
    ):
        self.limit = limit
        # Si se proporciona page, calcular skip automáticamente
        if page > 1:
            self.skip = (page - 1) * limit
        else:
            self.skip = skip
        self.page = page if page > 1 else (skip // limit) + 1

# ============ FUNCIONES AUXILIARES ============

async def save_base64_image(base64_str: str) -> tuple:
    """
    Guarda una imagen en base64 y devuelve (filename, filepath)
    """
    try:
        # Extraer datos base64
        if "base64," in base64_str:
            # Formato: data:image/png;base64,iVBORw0KGgo...
            header, data = base64_str.split("base64,")
            mime_type = header.split(":")[1].split(";")[0]
        else:
            # Base64 puro
            data = base64_str
            mime_type = "image/jpeg"
        
        # Determinar extensión
        ext_map = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp"
        }
        extension = ext_map.get(mime_type, "jpg")
        
        # Crear directorio si no existe
        upload_dir = "uploads"
        PathLib(upload_dir).mkdir(exist_ok=True)
        
        # Generar nombre único
        filename = f"{uuid.uuid4().hex}.{extension}"
        filepath = os.path.join(upload_dir, filename)
        
        # Decodificar y guardar
        image_data = base64.b64decode(data)
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        return filename, filepath
        
    except Exception as e:
        raise ValueError(f"Error procesando base64: {str(e)}")

def calculate_qualification(likes: int, views: int) -> int:
    if views == 0:
        return 1
    ratio = likes / views
    if ratio >= 0.5:
        return 5
    elif ratio >= 0.4:
        return 4
    elif ratio >= 0.3:
        return 3
    elif ratio >= 0.2:
        return 2
    return 1

async def process_image_ai_background(image_id: str, image_path: str):
    """
    Procesa una imagen con IA en segundo plano - VERSIÓN CORREGIDA
    """
    if not ai_processor:
        print("⚠️ IA no disponible, omitiendo procesamiento")
        return
    
    try:
        print(f"🚀 Iniciando procesamiento IA para imagen {image_id}")
        
        # 1. Actualizar estado a processing
        update_result = await coleccion.update_one(
            {"_id": ObjectId(image_id)},
            {"$set": {"ai_features.status": "processing"}}
        )
        
        print(f"📊 Update result (processing): matched={update_result.matched_count}, modified={update_result.modified_count}")
        
        # 2. Procesar con IA
        print("🖼️ Llamando a ai_processor.process_image()...")
        ai_features = await ai_processor.process_image(image_path)
        
        print(f"📊 Resultados IA obtenidos:")
        print(f"   - Tags: {ai_features.get('auto_tags', [])}")
        print(f"   - Scene: {ai_features.get('scene_type')}")
        print(f"   - Colors: {ai_features.get('color_palette', [])}")
        
        # 3. Actualizar en base de datos
        update_result = await coleccion.update_one(
            {"_id": ObjectId(image_id)},
            {
                "$set": {
                    "ai_features": {
                        **ai_features,
                        "status": "completed",
                        "processed_at": datetime.utcnow().isoformat(),
                        "model_version": "v1.0"
                    }
                }
            }
        )
        
        print(f"📊 Update result (completed): matched={update_result.matched_count}, modified={update_result.modified_count}")
        
        if update_result.modified_count > 0:
            print(f"✅ IA completada y guardada para {image_id}")
            print(f"   Etiquetas: {ai_features.get('auto_tags', [])}")
        else:
            print(f"⚠️ No se pudo actualizar la imagen {image_id}")
            print(f"   ¿Existe el documento?")
            
    except Exception as e:
        print(f"❌ Error en IA para {image_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        try:
            await coleccion.update_one(
                {"_id": ObjectId(image_id)},
                {
                    "$set": {
                        "ai_features.status": "failed",
                        "ai_features.error": str(e)[:200]
                    }
                }
            )
        except:
            pass

async def update_social_features_background(image_id_str: str):
    """
    Actualiza análisis social de comentarios en background - VERSIÓN CORREGIDA
    """
    if not ai_processor:
        print("⚠️ IA no disponible para análisis social")
        return
    
    try:
        print(f"🔍 Iniciando análisis social para: {image_id_str}")
        
        # CONVERTIR string a ObjectId
        try:
            object_id = ObjectId(image_id_str)
            print(f"   ✅ String convertido a ObjectId: {object_id}")
        except Exception as e:
            print(f"❌ Error convirtiendo a ObjectId: {e}")
            return
        
        # Buscar usando ObjectId
        image = await coleccion.find_one({"_id": object_id})
        
        if not image:
            print(f"❌ Imagen no encontrada con ObjectId: {object_id}")
            return
        
        print(f"✅ Imagen encontrada: {image.get('title', 'Sin título')}")
        
        comments = image.get("comments", [])
        print(f"📝 Total de comentarios: {len(comments)}")
        
        if not comments:
            print("⚠️ No hay comentarios para analizar")
            return
        
        # Mostrar comentarios
        for i, comment in enumerate(comments[:3]):  # Mostrar solo 3
            text = comment.get("comment", "")
            print(f"   {i+1}. {text[:80]}...")
        
        # Formatear comentarios para el analizador
        formatted_comments = []
        for comment in comments:
            text = comment.get("comment", "")
            if text and text.strip():  # Solo comentarios con texto
                formatted_comments.append({"text": text.strip()})
        
        print(f"🤖 Enviando {len(formatted_comments)} comentarios a IA...")
        
        # Procesar comentarios con IA
        social_features = await ai_processor.process_comments(formatted_comments)
        
        print(f"📊 Resultados IA obtenidos:")
        print(f"   - Sentimiento: {social_features.get('sentiment_score', 0):.2f}")
        print(f"   - Keywords: {social_features.get('keywords', [])}")
        print(f"   - Popularidad: {social_features.get('popularity_score', 0):.2f}")
        
        # Añadir timestamp
        social_features["last_updated"] = datetime.utcnow().isoformat()
        
        # Actualizar en base de datos
        update_result = await coleccion.update_one(
            {"_id": object_id},
            {"$set": {"social_features": social_features}}
        )
        
        print(f"📊 Resultado update: matched={update_result.matched_count}, modified={update_result.modified_count}")
        
        if update_result.modified_count > 0:
            print(f"✅ Análisis social guardado exitosamente")
            print(f"   Se actualizó: {update_result.modified_count} documento(s)")
        else:
            print(f"⚠️ No se pudo actualizar (matched: {update_result.matched_count})")
        
    except Exception as e:
        print(f"❌ Error crítico en análisis social: {str(e)}")
        import traceback
        traceback.print_exc()

# ============ ENDPOINT PRINCIPAL ACTUALIZADO ============

@galery.post('/api/create_image', response_model=Image)
async def save_image(
    img: Image,
    background_tasks: BackgroundTasks
):
    """
    Endpoint para crear una nueva imagen con procesamiento de IA
    """
    # 1. Verificar si el usuario existe
    user_exists = await user.find_one({"user_id": img.user_id})
    user_data = await user.find_one({"user_id": img.user_id})
    
    if not user_exists:
        raise HTTPException(status_code=404, detail="Usuario inválido")
    
    # 2. Verificar si la imagen ya existe (si es URL)
    if img.image_url and not img.image_url.startswith("data:image"):
        imgFound = await get_one_image(img.image_url)
        if imgFound:
            raise HTTPException(409, "Image already exists")
    
    # 3. Procesar base64 si aplica
    is_base64 = False
    saved_filepath = None
    final_image_url = img.image_url
    
    if img.image_url and img.image_url.startswith("data:image"):
        is_base64 = True
        try:
            filename, saved_filepath = await save_base64_image(img.image_url)
            # Actualizar image_url para apuntar al archivo guardado
            final_image_url = f"/uploads/{filename}"
        except ValueError as e:
            raise HTTPException(400, f"Base64 inválido: {str(e)}")
    
    # 4. Extraer características visuales (FEATURES - código existente)
    features_list = []
    try:
        # Tu feature_extractor existente
        if img.image_url and not img.image_url.startswith("data:image"):
            features = feature_extractor.extract(img.image_url)
        elif saved_filepath:
            features = feature_extractor.extract(saved_filepath)
        else:
            features = []
        
        features_array = np.array(features)
        features_list = features_array.tolist()
    except Exception as e:
        print(f"⚠️ Error al extraer características: {str(e)}")
        # No lanzamos excepción, continuamos sin features
    
    # 5. Preparar los datos de la imagen para guardar
    img_data = img.dict(by_alias=True, exclude_none=True)
    
    # Actualizar image_url si era base64
    if is_base64:
        img_data["image_url"] = final_image_url
    
    # Añadir campos adicionales
    img_data.update({
        "upload_date": datetime.utcnow(),
        "username": user_data.get("username", "Usuario"),
        "image_id": random.randint(1000, 9999),
        "features": features_list,
        "title": img.title if img.title else "Imagen sin título",
        
        # Campos de IA
        "ai_features": {
            "visual_embedding": [],
            "auto_tags": [],
            "detected_objects": [],
            "color_palette": [],
            "scene_type": None,
            "status": "pending",
            "source": "base64" if is_base64 else "url",
            "processed_at": None
        },
        
        "social_features": {
            "comment_sentiment": 0.0,
            "comment_keywords": [],
            "popularity_score": 0.0,
            "last_updated": None
        },
        
        "interactions": img_data.get("interactions", {
            "likes": 0,
            "downloads": 0,
            "views": 0,
            "last_interaction": None
        }),
        
        "liked_by": img_data.get("liked_by", []),
        "comments": img_data.get("comments", [])
    })
    
    # 6. Guardar en la base de datos
    response = await create_image(img_data)
    
    if not response:
        raise HTTPException(400, "Something went wrong")
    
    # 7. Programar procesamiento IA en background (si hay archivo)
    if saved_filepath and ai_processor:
        background_tasks.add_task(
            process_image_ai_background,
            str(response.get("_id")),
            saved_filepath
        )
        print(f"✅ Procesamiento IA programado para imagen {response.get('_id')}")
    else:
        print("⚠️ Procesamiento IA no disponible para esta imagen")
    
    # 8. Si es una URL externa (no base64), intentar procesar también
    if img.image_url and img.image_url.startswith(("http://", "https://")) and ai_processor:
        # Importar aquí para evitar dependencia si no se usa
        import aiohttp
        import tempfile
        
        async def process_external_url():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(img.image_url, timeout=10) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                                tmp_file.write(image_data)
                                tmp_path = tmp_file.name
                            
                            await process_image_ai_background(str(response.get("_id")), tmp_path)
                            
                            import os
                            os.unlink(tmp_path)
                            
            except Exception as e:
                print(f"❌ Error procesando URL {img.image_url}: {str(e)}")
        
        background_tasks.add_task(process_external_url)
    
    return response

@galery.get('/api/image/{image_id}/ai-analysis')
async def get_ai_analysis(image_id: str):
    """
    Obtener análisis AI de una imagen
    """
    image = await coleccion.find_one({"_id": image_id})
    if not image:
        raise HTTPException(404, "Imagen no encontrada")
    
    ai_features = image.get("ai_features", {})
    social_features = image.get("social_features", {})
    
    return {
        "image_id": image_id,
        "title": image.get("title"),
        "ai_status": ai_features.get("status", "pending"),
        "ai_analysis": {
            "tags": ai_features.get("auto_tags", []),
            "scene": ai_features.get("scene_type"),
            "objects": ai_features.get("detected_objects", []),
            "colors": ai_features.get("color_palette", []),
            "embedding_length": len(ai_features.get("visual_embedding", [])),
            "processed_at": ai_features.get("processed_at")
        } if ai_features.get("status") == "completed" else None,
        "social_analysis": {
            "sentiment": social_features.get("comment_sentiment", 0.0),
            "keywords": social_features.get("comment_keywords", []),
            "popularity": social_features.get("popularity_score", 0.0),
            "last_updated": social_features.get("last_updated")
        }
    }
    
# ============ ENDPOINT PARA VERIFICAR IA ============

@galery.get('/api/ai/status')
async def check_ai_status():
    """
    Verificar estado del sistema de IA
    """
    if ai_processor:
        status = ai_processor.get_system_status()
        return {
            "available": True,
            "initialized": status.get("initialized", False),
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "available": False,
            "message": "Sistema de IA no configurado",
            "timestamp": datetime.utcnow().isoformat()
        }


@galery.get('/api/images')
async def get_images():
    global image_cache, cache_timestamp
    
    # Verificar cache
    if (image_cache is not None and 
        cache_timestamp is not None and 
        (datetime.now() - cache_timestamp).total_seconds() < CACHE_DURATION):
        
        print("✅ Sirviendo desde cache")
        return JSONResponse(content=image_cache)
    
    # Obtener datos frescos
    print("🔄 Obteniendo datos frescos")
    images = await get_all_images()
    
    # Convertir a formato serializable manualmente
    serializable_response = []
    for image in images:
        image_dict = image.dict()
        
        # Convertir datetime en comentarios
        if 'comments' in image_dict and image_dict['comments']:
            for comment in image_dict['comments']:
                if 'created_at' in comment and isinstance(comment['created_at'], datetime):
                    comment['created_at'] = comment['created_at'].isoformat()
        
        serializable_response.append(image_dict)
    
    # Actualizar cache
    image_cache = serializable_response
    cache_timestamp = datetime.now()
    
    return JSONResponse(content=serializable_response)



@galery.get('/api/images/search')
async def search_images(
    q: Optional[str] = None,
    tags: Optional[str] = None,
    min_likes: Optional[int] = None,
    min_views: Optional[int] = None,
    pagination: PaginationParams = Depends()
):
    try:
        # Obtener todas las imágenes usando la función existente
        all_images = await get_all_images()
        
        # Convertir a lista de diccionarios
        images_dicts = []
        for image in all_images:
            # Convertir el modelo Pydantic a diccionario
            image_dict = image.dict()
            
            # Asegurar que los campos existan y no sean None
            image_dict.setdefault('interactions', {})
            image_dict['interactions'].setdefault('likes', 0)
            image_dict['interactions'].setdefault('views', 0)
            image_dict.setdefault('tags', [])
            image_dict.setdefault('title', '')  # Asegurar que title no sea None
            image_dict.setdefault('category', '')
            image_dict.setdefault('username', '')
            
            # Convertir valores None a string vacío
            if image_dict['title'] is None:
                image_dict['title'] = ''
            if image_dict['username'] is None:
                image_dict['username'] = ''
            
            # Convertir datetime en comentarios
            if 'comments' in image_dict and image_dict['comments']:
                for comment in image_dict['comments']:
                    if 'created_at' in comment and isinstance(comment['created_at'], datetime):
                        comment['created_at'] = comment['created_at'].isoformat()
            
            images_dicts.append(image_dict)
        
        # Aplicar filtros paso a paso
        filtered_images = images_dicts
        
        # 1. Filtrar por texto de búsqueda (q)
        if q:
            q_lower = q.lower().strip()
            temp_filtered = []
            
            for img in filtered_images:
                # Asegurar que los campos no sean None antes de usar .lower()
                title = (img.get('title') or '').lower()
                category = (img.get('category') or '').lower()
                username = (img.get('username') or '').lower()
                image_tags = [tag.lower() for tag in (img.get('tags') or []) if tag is not None]
                
                # Buscar coincidencias
                title_match = q_lower in title
                category_match = q_lower in category
                username_match = q_lower in username
                tags_match = any(q_lower in tag for tag in image_tags)
                
                if title_match or category_match or username_match or tags_match:
                    temp_filtered.append(img)
            
            filtered_images = temp_filtered
        
        # 2. Filtrar por tags específicos
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(',') if tag.strip()]
            temp_filtered = []
            for img in filtered_images:
                image_tags = [tag.lower() for tag in (img.get('tags') or []) if tag is not None]
                if any(tag in image_tags for tag in tag_list):
                    temp_filtered.append(img)
            filtered_images = temp_filtered
        
        # 3. Filtrar por mínimo de likes
        if min_likes is not None:
            filtered_images = [
                img for img in filtered_images 
                if img['interactions'].get('likes', 0) >= min_likes
            ]
        
        # 4. Filtrar por mínimo de views
        if min_views is not None:
            filtered_images = [
                img for img in filtered_images 
                if img['interactions'].get('views', 0) >= min_views
            ]
        
        # Aplicar paginación
        total_count = len(filtered_images)
        
        # Calcular información de paginación
        total_pages = (total_count + pagination.limit - 1) // pagination.limit if pagination.limit > 0 else 1
        has_next = (pagination.skip + pagination.limit) < total_count
        has_prev = pagination.skip > 0
        
        paginated_images = filtered_images[pagination.skip:pagination.skip + pagination.limit]
        
        return {
            "results": paginated_images,
            "total": total_count,
            "limit": pagination.limit,
            "skip": pagination.skip,
            "page": pagination.page,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev,
            "next_page": pagination.page + 1 if has_next else None,
            "prev_page": pagination.page - 1 if has_prev else None,
            "search_query": q,
            "filters_applied": {
                "text_search": q is not None,
                "tags_filter": tags is not None,
                "min_likes": min_likes,
                "min_views": min_views
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")

@galery.get('/api/images/{image_id}/qualification', response_model=QualificationResponse)
async def get_image_qualification(image_id: int):
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    interactions = image.get("interactions", {})
    likes = interactions.get("likes", 0)
    views = interactions.get("views", 0)

    ratio = likes / views if views > 0 else 0.0
    qualification = calculate_qualification(likes, views)

    await coleccion.update_one(
        {"image_id": image_id},
        {"$set": {"qualification": qualification}}
    )

    return QualificationResponse(
        image_id=image_id,
        qualification=qualification,
        likes=likes,
        views=views,
        ratio=round(ratio, 4)
    )

""" @galery.post('/api/create_image', response_model=Image)
async def save_image(img: Image):
    # 1. Verificar si el usuario existe
    user_exists = await user.find_one({"user_id": img.user_id})
    user_data = await user.find_one({"user_id": (img.user_id)})
    if not user_exists:
        raise HTTPException(status_code=404, detail="Usuario inválido")
    
    # 2. Verificar si la imagen ya existe
    if img.image_url:
        imgFound = await get_one_image(img.image_url)
        if imgFound:
            raise HTTPException(409, "Image already exists")
    
    # 3. Extraer características visuales (FEATURES)
    try:
        # Ejemplo: img.image_url contiene la ruta o URL de la imagen
        features = feature_extractor.extract(img.image_url)
        features_array = np.array(features)
        features_list = features_array.tolist()
    except Exception as e:
        raise HTTPException(500, f"Error al extraer características: {str(e)}")
    
    # 4. Preparar los datos de la imagen para guardar
    img_data = img.dict(by_alias=True, exclude_none=True)
    img_data.update({
        "upload_date": datetime.utcnow(),  # Fecha de subida
        "username": user_data["username"],
        "image_id": random.randint(1000, 9999),  # ID aleatorio (¿o usar UUID?)
        "features": features_list,  
        "title": img.title
    })
    
    # 5. Guardar en la base de datos
    response = await create_image(img_data)
    
    if response:
        return response
    raise HTTPException(400, "Something went wrong") """

@galery.put('/api/images/{id}', response_model=Image)
async def put_img(id: str, data: UpdateImage):
    data.id = id
    response = UpdateImage(**data.dict())
    if response:
        return response
    raise HTTPException(404, f"There is no task with the id {id}")

@galery.delete('/api/images/{id}')
async def remove_image(id: str):
    response = await delete_image(id)
    if response:
        return "Successfully image delete from the database"
    raise HTTPException(404, f"There is no task with the id {id}")

@galery.post("/api/upload-base64/{user_id}")
async def upload_base64_image(
    image: Image,
    user_id: int,
):
    try:
        header, encoded = image.image_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        if len(image_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="La imagen excede el tamaño máximo de 5MB")
        
        image_data = {
            "user_id": user_id,  # <-- ahora sí existe
            "image_url": image.image_url,
            "category": image.category,
            "upload_date": datetime.utcnow()
        }
        result = await coleccion.insert_one(image_data)
        await user.update_one(
            {"user_id": user_id},
            {"$push": {"images": result.inserted_id}}
        )
        return {"id": str(result.inserted_id), "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@galery.put("/api/images/{image_id}/interactions/{id}", response_model=Image)
async def update_interactions(
    image_id: int,
    id: int, 
    interaction: InteractionUpdate,
    user_data: dict = Depends(extract_user_id)
):
    user_id = user_data["user_id"]
    print(user_id)
    usuario = await user.find_one({"username": user_id})
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # 1. Validar acción permitida (corregí el typo "likes")
    valid_actions = ["likes", "downloads", "views"]
    if interaction.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Acción no válida. Use: {', '.join(valid_actions)}"
        )

    # 2. Verificar si la imagen existe primero
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    # 3. Preparar operación de actualización para la imagen
    update_operation = {
        "$inc": {f"interactions.{interaction.action}": interaction.increment},
        "$set": {"interactions.last_interaction": datetime.utcnow()}
    }
    
    # 4. Si es like, agregar al array liked_by
    if interaction.action == "likes":
        """ if "liked_by" in image and user_id in image["liked_by"]:
            raise HTTPException(
                status_code=400,
                detail="Ya has dado like a esta imagen"
            ) """
        update_operation["$addToSet"] = {"liked_by": id}

    # 5. Actualizar la imagen
    update_result = await coleccion.update_one(
        {"image_id": image_id},
        update_operation
    )

    if update_result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    # 6. Actualizar el usuario - VERSIÓN CORREGIDA
    update_result = await user.update_one(
    {"username": user_id},
    {
        "$addToSet": {"images": image_id},
        "$set": {"last_updated": datetime.utcnow()}
    }
    )

    
    # Verificar resultado
    if update_result.modified_count == 1:
        print("Imagen agregada exitosamente")
    else:
        print("No se realizaron cambios")
        
    user_data = await user.find_one({"user_id": user_id})
    print("Lista actual de imágenes:", user_data)
            

    # 7. Calcular y guardar calificación
    updated_image = await coleccion.find_one({"image_id": image_id})
    interactions = updated_image.get("interactions", {})
    likes = interactions.get("likes", 0)
    views = interactions.get("views", 0)
    qualification = calculate_qualification(likes, views)
    await coleccion.update_one(
        {"image_id": image_id},
        {"$set": {"qualification": qualification}}
    )

    # 8. Devolver la imagen actualizada
    updated_image = await coleccion.find_one({"image_id": image_id})
    return Image(**updated_image)


@galery.put("/api/images/{image_id}/comments", response_model=Image)
async def add_comment(
    background_tasks: BackgroundTasks,
    image_id: int,
    comment_data: CommentCreate,
    user_data: dict = Depends(extract_user_id)
):
    user_id = user_data["user_id"]
    
    print(f"🎯 Iniciando add_comment para imagen {image_id}, usuario {user_id}")
    
    # Verificar que el usuario existe
    usuario = await user.find_one({"username": user_id})
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar que la imagen existe (por image_id numérico)
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    print(f"📊 Imagen encontrada. ObjectId: {image['_id']}, title: {image.get('title', 'Sin título')}")
    
    # Generar ID único para el comentario
    comment_id = datetime.utcnow().timestamp()
    
    # Estructura del comentario
    new_comment = {
        "comment_id": int(comment_id),
        "user_id": usuario["user_id"],
        "username": user_id,
        "user_image": usuario.get("image", ""),
        "comment": comment_data.comment,
        "created_at": datetime.utcnow(),
        "parent_comment_id": comment_data.parent_comment_id,
        "likes": 0,
        "replies": []
    }
    
    print(f"💬 Nuevo comentario: {comment_data.comment[:50]}...")
    
    # Si es un comentario padre, agregarlo al array principal
    if comment_data.parent_comment_id is None:
        update_operation = {
            "$push": {"comments": new_comment},
            "$inc": {"interactions.comment_count": 1},
            "$set": {"interactions.last_interaction": datetime.utcnow()}
        }
    else:
        # Si es una respuesta, agregarlo al comentario padre
        update_operation = {
            "$push": {"comments.$[comment].replies": new_comment},
            "$inc": {"interactions.comment_count": 1},
            "$set": {"interactions.last_interaction": datetime.utcnow()}
        }
    
    # Actualizar la imagen (por image_id numérico)
    if comment_data.parent_comment_id is None:
        update_result = await coleccion.update_one(
            {"image_id": image_id},
            update_operation
        )
    else:
        update_result = await coleccion.update_one(
            {"image_id": image_id, "comments.comment_id": comment_data.parent_comment_id},
            update_operation,
            array_filters=[{"comment.comment_id": comment_data.parent_comment_id}]
        )
    
    if update_result.modified_count == 0:
        if comment_data.parent_comment_id:
            raise HTTPException(
                status_code=404, 
                detail="Comentario padre no encontrado"
            )
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    print(f"✅ Comentario agregado. Modified count: {update_result.modified_count}")
    
    # Actualizar el usuario
    await user.update_one(
        {"username": user_id},
        {
            "$addToSet": {"commented_images": image_id},
            "$set": {"last_updated": datetime.utcnow()}
        }
    )
    
    # 🎯 ¡IMPORTANTE! - Pasar el ObjectId CORRECTO
    if background_tasks and ai_processor:
        # image ya contiene el documento con el ObjectId
        object_id_str = str(image["_id"])
        print(f"🎯 Programando análisis social para ObjectId: {object_id_str}")
        
        # Asegurarnos de que la función reciba el ObjectId como string
        background_tasks.add_task(
            update_social_features_background,
            object_id_str  # Esto ya es el string del ObjectId
        )
    
    # Devolver la imagen actualizada
    updated_image = await coleccion.find_one({"image_id": image_id})
    return updated_image

@galery.delete("/api/images/{image_id}/likes/{id}", response_model=Image)
async def remove_like(
    image_id: int,
    id:int,
    user_data: dict = Depends(extract_user_id)
):
    user_id = user_data["user_id"]
    
    # Verificar si el usuario existe
    usuario = await user.find_one({"username": user_id})
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar si la imagen existe
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Verificar si el usuario tiene like en esta imagen
    liked_by = image.get("liked_by", [])
    if id not in liked_by:
        raise HTTPException(
            status_code=400,
            detail="No tienes like en esta imagen"
        )
    
    # Remover el like
    update_operation = {
        "$pull": {"liked_by": id},
        "$inc": {"interactions.likes": -1},
        "$set": {"interactions.last_interaction": datetime.utcnow()}
    }
    
    # Actualizar la imagen
    update_result = await coleccion.update_one(
        {"image_id": image_id},
        update_operation
    )
    
    if update_result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Recalcular calificación después de remover like
    updated_image = await coleccion.find_one({"image_id": image_id})
    interactions = updated_image.get("interactions", {})
    likes = interactions.get("likes", 0)
    views = interactions.get("views", 0)
    qualification = calculate_qualification(likes, views)
    await coleccion.update_one(
        {"image_id": image_id},
        {"$set": {"qualification": qualification}}
    )

    # Devolver la imagen actualizada
    updated_image = await coleccion.find_one({"image_id": image_id})
    return Image(**updated_image)


# ============ SAVED IMAGES ENDPOINTS ============

@galery.put("/api/images/{image_id}/save")
async def save_image_to_profile(
    image_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Save an image to the user's profile collection"""
    # Verify image exists
    image = await coleccion.find_one({"image_id": image_id})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get user_id from current_user
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    # Check if already saved
    saved_images = current_user.get("saved_images", [])
    if image_id in saved_images:
        raise HTTPException(status_code=400, detail="Image already saved")
    
    # Add to saved_images
    await user.update_one(
        {"user_id": user_id},
        {"$addToSet": {"saved_images": image_id}}
    )
    
    return {"res": "Image saved to profile", "image_id": image_id}


@galery.delete("/api/images/{image_id}/save")
async def unsave_image_from_profile(
    image_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Remove an image from the user's profile collection"""
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    # Check if image is saved
    saved_images = current_user.get("saved_images", [])
    if image_id not in saved_images:
        raise HTTPException(status_code=400, detail="Image not saved")
    
    # Remove from saved_images
    await user.update_one(
        {"user_id": user_id},
        {"$pull": {"saved_images": image_id}}
    )
    
    return {"res": "Image removed from profile", "image_id": image_id}


@galery.get("/api/users/{user_id}/saved-images")
async def get_user_saved_images(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """
    Get all saved images for a user.
    - If viewing own gallery: always allowed
    - If viewing another user's gallery: only if profile is public
    """
    # Find the target user
    target_user = await user.find_one({"user_id": user_id})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if the requesting user is the owner
    is_owner = current_user.get("user_id") == user_id
    
    # If not the owner, check if profile is public
    if not is_owner and not target_user.get("profile_public", True):
        raise HTTPException(status_code=403, detail="This profile is private")
    
    # Get user's saved_images list
    saved_image_ids = target_user.get("saved_images", [])
    
    if not saved_image_ids:
        return {"user_id": user_id, "saved_images": [], "count": 0}
    
    # Fetch all saved images from database
    saved_images = []
    for img_id in saved_image_ids:
        image = await coleccion.find_one({"image_id": img_id})
        if image:
            # Convert ObjectId to string
            if "_id" in image:
                image["_id"] = str(image["_id"])
            # Convert liked_by to strings
            if "liked_by" in image and image["liked_by"]:
                image["liked_by"] = [str(uid) for uid in image["liked_by"]]
            saved_images.append(image)
    
    return {
        "user_id": user_id,
        "saved_images": saved_images,
        "count": len(saved_images)
    }


@galery.get("/api/users/{user_id}/saved-images/public")
async def get_user_saved_images_public(user_id: str):
    """
    Get saved images for a public profile (no authentication required).
    Accepts either numeric user_id or username string.
    Only works if the user's profile is public.
    """
    # Try to find by numeric user_id first
    target_user = None
    try:
        numeric_id = int(user_id)
        target_user = await user.find_one({"user_id": numeric_id})
    except (ValueError, TypeError):
        pass
    
    # If not found by numeric id, try by username
    if not target_user:
        target_user = await user.find_one({"username": user_id})
    
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if profile is public
    if not target_user.get("profile_public", True):
        raise HTTPException(status_code=403, detail="This profile is private")
    
    # Get user's saved_images list
    saved_image_ids = target_user.get("saved_images", [])
    
    if not saved_image_ids:
        return {"user_id": target_user.get("user_id"), "saved_images": [], "count": 0}
    
    # Fetch all saved images from database
    saved_images = []
    for img_id in saved_image_ids:
        image = await coleccion.find_one({"image_id": img_id})
        if image:
            # Convert ObjectId to string
            if "_id" in image:
                image["_id"] = str(image["_id"])
            # Convert liked_by to strings
            if "liked_by" in image and image["liked_by"]:
                image["liked_by"] = [str(uid) for uid in image["liked_by"]]
            saved_images.append(image)
    
    return {
        "user_id": user_id,
        "saved_images": saved_images,
        "count": len(saved_images)
    }
