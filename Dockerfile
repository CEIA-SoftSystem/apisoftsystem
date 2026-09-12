# ── build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Dependências do sistema necessárias para OpenCV e compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# PyTorch CPU — troque pela linha de GPU abaixo se o servidor tiver CUDA
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# GPU (descomente e remova a linha CPU acima):
# RUN pip install --no-cache-dir torch torchvision \
#     --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt


# ── runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Libs de runtime para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia pacotes instalados do stage de build
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Código da aplicação
COPY app/       ./app/
COPY predict.py ./predict.py

# Diretório de modelos (montar como volume no EasyPanel)
RUN mkdir -p /app/models

# Variáveis de ambiente padrão
ENV SEG_MODEL_PATH=/app/models/unet_seg.pt \
    DET_MODEL_PATH=/app/models/best_yolo11 \
    DEVICE=auto \
    DEFAULT_THRESHOLD=0.5 \
    DEFAULT_CONF=0.25 \
    MAX_IMAGE_SIZE_MB=10 \
    DEBUG=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
