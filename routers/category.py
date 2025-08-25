from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from models.category import  CategoryResponse, CategoriesResponse
from database.databases import categories

category= APIRouter()

async def get_categories(
    search: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True
) -> List[dict]:
    """
    Obtener categorías con filtros
    """
    query = {}
    
    if active_only:
        query["is_active"] = True
    
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"description": {"$regex": search, "$options": "i"}},
            {"tags": {"$in": [search.lower()]}}
        ]
    
    # Usar un nombre diferente para evitar conflicto
    cursor = categories.find(query).skip(skip).limit(limit)
    categories_list = await cursor.to_list(length=limit)  # Cambié el nombre aquí
    return categories_list  # Y aquí

@category.get("/api/categories/autocomplete", response_model=List[str])
async def autocomplete_categories(
    q: str = Query(..., description="Término para autocompletar"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Autocompletado de nombres de categorías
    """
    try:
        categories_data = await get_categories(q, 0, limit, True)
        return [cat["name"] for cat in categories_data]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en autocompletado: {str(e)}")