🚀 PicPreference - Backend API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-009688?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python)
![MongoDB](https://img.shields.io/badge/MongoDB-6.0-47A248?style=for-the-badge&logo=mongodb)


**API Inteligente de Recomendación de Imágenes con Machine Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)


</div>

## 🌟 Características Principales

- **🤖 Sistema de Recomendación Híbrido**: Combina 3 estrategias (colaborativo, contenido y popularidad)
- **🖼️ Procesamiento de Imágenes**: Extracción de características visuales con ResNet50
- **⚡ Alto Rendimiento**: Construido con FastAPI para respuestas en menos de 200ms
- **🔐 Autenticación JWT**: Sistema seguro de autenticación con tokens
- **📊 Monitorización en Tiempo Real**: Endpoints de salud y métricas del sistema


## 🛠️ Stack Tecnológico

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **FastAPI** | 0.68.0 | Framework API ASGI |
| **Python** | 3.9 | Lenguaje de programación |
| **MongoDB** | 6.0 | Base de datos NoSQL |
| **Motor** | 3.1.1 | Driver async para MongoDB |
| **TensorFlow** | 2.10.0 | Procesamiento de imágenes |
| **JWT** | 1.7.1 | Autenticación por tokens |
| **Docker** | 20.10 | Containerización |

## 📦 Instalación y Configuración

### Prerrequisitos
```bash
Python 3.9+
MongoDB 6.0+
Docker (opcional)

Pasos de instalación
Clonar el repositorio

bash
git clone <repository-url>
cd <project-directory>
Crear entorno virtual

bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
Instalar dependencias

bash
pip install -r requirements.txt
Configurar variables de entorno

bash
# Crear archivo .env
echo "DATABASE_URL=mongodb://localhost:27017/art_gallery" > .env
echo "SECRET_KEY=tu_clave_secreta_aqui" >> .env
Ejecutar la aplicación

bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
📡 API Endpoints
Salud del Sistema
Método	Endpoint	Descripción
GET	/health	Verificar estado del sistema
Recomendaciones
Método	Endpoint	Descripción
GET	/recommend/{user_id}	Recomendaciones personalizadas
GET	/api/recommend/{user_id}	Recomendaciones con paginación
GET	/recommend/{user_id}?image_id={id}	Recomendaciones basadas en imagen específica
Evaluación
Método	Endpoint	Descripción
GET	/evaluate/{user_id}	Evaluar recomendaciones para usuario
GET	/evaluate-all	Evaluar sistema completo
GET	/simple-evaluate/{user_id}	Evaluación simplificada
Debug y Monitoreo
Método	Endpoint	Descripción
GET	/system-status	Estado detallado del sistema
GET	/graph-stats	Estadísticas del grafo
GET	/debug-user/{user_id}	Debug de datos de usuario
GET	/debug-image/{image_id}	Debug de imagen específica
🏗️ Estructura del Proyecto
text
sistema-recomendacion-galeria/
├── main.py                 # Aplicación principal FastAPI
├── requirements.txt        # Dependencias del proyecto
├── routers/               # Módulos de routers
│   ├── users.py           # Endpoints de usuarios
│   ├── galery.py          # Endpoints de galería
│   ├── category.py        # Endpoints de categorías
│   └── recommendations.py # Endpoints de recomendaciones
├── services/              # Lógica de negocio
│   ├── genetic.py         # Optimización con algoritmos genéticos
│   ├── graph.py           # Manejo de grafos de interacciones
│   ├── recommender.py     # Sistema de recomendación visual
│   └── auxiliary.py       # Funciones auxiliares
├── models/                # Modelos de datos Pydantic
│   └── galery.py          # Modelos de imágenes
├── database/              # Configuración de base de datos
│   └── databases.py       # Conexión a MongoDB
├── utils/                 # Utilidades
│   └── feature_extractor.py # Extracción de características
└── .gitignore            # Archivos ignorados por Git
⚙️ Configuración
Variables de Entorno
Variable	Descripción	Valor por Defecto
DATABASE_URL	URL de conexión a MongoDB	mongodb://localhost:27017/art_gallery
SECRET_KEY	Clave secreta para autenticación	-
ORIGINS	URLs permitidas para CORS	["http://localhost:3000"]
Base de Datos
El sistema utiliza MongoDB con las siguientes colecciones principales:

coleccion: Almacena información de imágenes y características visuales

user: Almacena datos de usuarios e interacciones

🎯 Estrategias de Recomendación
Estrategia	Descripción	Caso de Uso
Cold Start	Imágenes populares	Usuarios nuevos
Híbrida Ligera	Combinación de métodos	Usuarios con pocas interacciones
Personalizada	Grafo + contenido	Usuarios con historial completo
📊 Métricas de Evaluación
El sistema incluye evaluación automática con:

Precisión (@k)

Recall (@k)

F1-Score

Coverage

Novelty

🚦 Desarrollo
Ejecutar en modo desarrollo
bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Documentación API
Swagger UI: http://localhost:8000/docs

Redoc: http://localhost:8000/redoc
