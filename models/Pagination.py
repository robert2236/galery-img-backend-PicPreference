from fastapi import HTTPException, Query, Depends



class PaginationParams:
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Número de elementos a saltar"),
        limit: int = Query(20, ge=1, le=100, description="Límite de resultados por página"),
        page: int = Query(1, ge=1, description="Número de página (alternativo a skip)")
    ):
        self.limit = limit
        if page > 1:
            self.skip = (page - 1) * limit
        else:
            self.skip = skip
        self.page = page if page > 1 else (skip // limit) + 1