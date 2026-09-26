# syntax=docker/dockerfile:1.4

########## STAGE 1: builder (instala Python deps + pesos ResNet50) ##########
FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# 1) PyTorch CUDA primero (capa cacheable independiente del resto)
COPY requirements-gpu.txt requirements-gpu.txt
RUN pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r requirements-gpu.txt

# 2) Resto de dependencias (PyPI por defecto)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3) Pre-descarga pesos ResNet50 (imagenet) -> /root/.keras
#    Evita descarga en el primer arranque (cold-start) en Render/Koyeb
RUN python -c "from tensorflow.keras.applications import ResNet50; ResNet50(weights='imagenet', include_top=False, pooling='avg')"

########## STAGE 2: runtime (imagen final minima) ##########
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOME=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Librerias del sistema (TF, torch, scikit-learn OpenMP, opencv-compat)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/uploads

# Entorno Python desde builder (sin cache de pip ni apt del builder)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pesos ResNet50 (se copian a HOME=/app/.keras, escribible por appuser)
COPY --from=builder /root/.keras /app/.keras

# Codigo de la aplicacion (COPY explicito: solo lo necesario)
COPY main.py ./
COPY vector_store.py ./
COPY routers/ ./routers/
COPY services/ ./services/
COPY models/ ./models/
COPY utils/ ./utils/
COPY database/ ./database/
COPY static/ ./static/

# Usuario no root (seguridad)
RUN useradd --no-create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4).status==200 else sys.exit(1)"

# PORT lee la variable inyectada por Render/Koyeb; fallback a 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]
