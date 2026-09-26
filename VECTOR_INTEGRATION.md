# Integración ChromaDB - Guía de Uso

## Archivos Creados

| Archivo | Descripción |
|---------|-------------|
| `vector_store.py` | Módulo de almacenamiento vectorial con ChromaDB |
| `migrate_vectors.py` | Script CLI para migrar embeddings de MongoDB a ChromaDB |
| `routers/vector_search.py` | Router FastAPI con endpoint de búsqueda visual |

## Instalación

```bash
# Instalar nuevas dependencias
pip install -r requirements.txt
```

## Uso del Script de Migración

### Preview (Dry-Run)
```bash
# Ver qué imágenes se migrarían sin hacer cambios
python migrate_vectors.py --dry-run

# Limitar a 10 imágenes
python migrate_vectors.py --dry-run --limit 10
```

### Ejecutar Migración
```bash
# Migrar todas las imágenes nuevas
python migrate_vectors.py --execute

# Migrar solo imágenes nuevas (incremental)
python migrate_vectors.py --execute --incremental

# Migrar todas (re-migrar)
python migrate_vectors.py --execute --complete

# Limitar cantidad
python migrate_vectors.py --execute --limit 50
```

### Ver Estadísticas
```bash
# Ver estadísticas de ChromaDB
python migrate_vectors.py --stats
```

### Parámetros Adicionales
```bash
# Especificar URL de MongoDB
python migrate_vectors.py --execute --mongo-url mongodb://localhost:27017

# Especificar base de datos
python migrate_vectors.py --execute --db-name galery

# Especificar directorio ChromaDB
python migrate_vectors.py --execute --chroma-dir ./chroma_db
```

## Endpoints Disponibles

### 1. Búsqueda Visual por Similitud

```
GET /api/v1/recommendations/visual-similar/{image_id}?limit=5
```

**Parámetros:**
- `image_id` (path): ID numérico de la imagen
- `limit` (query): Número de resultados (1-50, default: 5)

**Ejemplo de respuesta:**
```json
{
  "status": "success",
  "original_image": {
    "image_id": 1234,
    "title": "Paisaje montañoso",
    "image_url": "/uploads/mountain.jpg",
    "category": "nature",
    "likes": 15,
    "views": 120
  },
  "similar_images": [
    {
      "image_id": 5678,
      "title": "Valle verde",
      "image_url": "/uploads/valley.jpg",
      "category": "nature",
      "similarity_score": 0.8923,
      "likes": 8,
      "views": 45
    }
  ],
  "total_similar": 5,
  "vector_backend": "chromadb",
  "has_embedding": true
}
```

### 2. Estadísticas del Sistema Vectorial

```
GET /api/v1/recommendations/vector-stats
```

**Respuesta:**
```json
{
  "status": "success",
  "available": true,
  "stats": {
    "total_embeddings": 150,
    "collection_name": "visual_embeddings",
    "persist_directory": "./chroma_db",
    "feature_size": 2048
  }
}
```

### 3. Verificar Embedding de Imagen

```
GET /api/v1/recommendations/check-embedding/{image_id}
```

**Respuesta:**
```json
{
  "status": "success",
  "image_id": 1234,
  "has_mongodb": true,
  "has_embedding": true,
  "embedding_dimension": 2048,
  "image_url": "/uploads/mountain.jpg",
  "category": "nature"
}
```

## Flujo de Trabajo Recomendado

### 1. Primera Instalación
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar migración completa
python migrate_vectors.py --execute

# Verificar estadísticas
python migrate_vectors.py --stats
```

### 2. Uso Diario
```bash
# Después de agregar nuevas imágenes vía API
# Ejecutar migración incremental
python migrate_vectors.py --execute --incremental
```

### 3. Verificación
```bash
# Verificar que ChromaDB está funcionando
python migrate_vectors.py --stats

# Probar endpoint
curl http://localhost:8000/api/v1/recommendations/visual-similar/1234
```

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                  │
├─────────────────────────────────────────────────────────┤
│  Routers existentes (MongoDB)  │  Nuevo router         │
│  - /api/images                 │  - /api/v1/           │
│  - /recommend/similar          │    recommendations/   │
│                                │    visual-similar/    │
├─────────────────────────────────────────────────────────┤
│  Services existentes           │  Nuevo servicio       │
│  - VisualRecommender (KDTree)  │  - VectorStore        │
│  - FeatureExtractor (ResNet50) │    (ChromaDB)         │
├─────────────────────────────────────────────────────────┤
│  MongoDB (Motor)               │  ChromaDB (local)     │
│  - Images collection           │  - visual_embeddings  │
│  - Features field existente    │  - metadata + vectors │
└─────────────────────────────────────────────────────────┘
```

## Manejo de Errores

### ChromaDB No Disponible
El endpoint retorna error 503:
```json
{
  "detail": "Servicio de búsqueda vectorial no disponible"
}
```

### Imagen Sin Embedding
El endpoint retorna advertencia:
```json
{
  "status": "warning",
  "message": "Imagen 1234 no tiene embedding vectorial...",
  "has_embedding": false
}
```

### Imagen No Encontrada
El endpoint retorna error 404:
```json
{
  "detail": "Imagen con ID 1234 no encontrada"
}
```

## Logs

Los logs se guardan en `migrate_vectors.log`:
```
2024-01-15 10:30:00 - INFO - 🚀 INICIANDO MIGRACIÓN DE VECTORES
2024-01-15 10:30:01 - INFO - 📊 Encontradas 150 imágenes para migrar
2024-01-15 10:30:05 - INFO - 📊 Progreso: 50/150 (45 éxitos, 5 errores)
2024-01-15 10:30:10 - INFO - 📊 RESUMEN DE MIGRACIÓN
```

## Solución de Problemas

### Error: "Dependencia faltante"
```bash
pip install chromadb tqdm
```

### Error: "No se pudo conectar a MongoDB"
Verificar que MongoDB está corriendo:
```bash
# Windows
net start MongoDB

# Linux/Mac
sudo systemctl start mongod
```

### Error: "ChromaDB no está disponible"
Eliminar directorio corrupto y re-migrar:
```bash
rm -rf ./chroma_db
python migrate_vectors.py --execute
```

### Logs Detallados
Para ver logs en tiempo real:
```bash
tail -f migrate_vectors.log
```
