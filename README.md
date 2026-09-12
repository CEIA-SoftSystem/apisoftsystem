# CowFace API

API REST para **segmentação** e **detecção** de faces bovinas usando modelos de deep learning.

| Tarefa | Modelo | Saída |
|---|---|---|
| Segmentação | U-Net (`unet_seg.pt`) | Máscara binária pixel a pixel |
| Detecção | YOLO11 (`best_yolo11`) | Bounding boxes com confiança |

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Health check da API |
| `GET` | `/api/v1/models` | Info dos modelos carregados |
| `POST` | `/api/v1/segment` | Segmentação de face bovina |
| `POST` | `/api/v1/detect` | Detecção de faces bovinas |
| `GET` | `/docs` | Swagger UI interativo |
| `GET` | `/redoc` | Documentação ReDoc |

---

## Início Rápido (Docker)

### 1. Preparar modelos

Copie os modelos para a pasta `models/`:

```
models/
├── unet_seg.pt      ← modelo de segmentação
└── best_yolo11      ← modelo de detecção YOLO11
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# edite .env conforme necessário
```

### 3. Subir a API

```bash
docker compose up --build
```

A API estará disponível em `http://localhost:8000`.  
Acesse `http://localhost:8000/docs` para o Swagger interativo.

---

## Configuração

Todas as opções são controladas por variáveis de ambiente (ou arquivo `.env`):

| Variável | Padrão | Descrição |
|---|---|---|
| `SEG_MODEL_PATH` | `/app/models/unet_seg.pt` | Caminho do modelo U-Net |
| `DET_MODEL_PATH` | `/app/models/best_yolo11` | Caminho do modelo YOLO11 |
| `DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `DEFAULT_THRESHOLD` | `0.5` | Limiar padrão para segmentação |
| `DEFAULT_CONF` | `0.25` | Confiança mínima padrão para detecção |
| `MAX_IMAGE_SIZE_MB` | `10` | Tamanho máximo de imagem aceito |
| `DEBUG` | `false` | Ativa logs detalhados |

---

## Exemplos de Uso

### Segmentação

```bash
curl -X POST http://localhost:8000/api/v1/segment \
  -F "file=@foto_bovina.jpg" \
  -F "threshold=0.5" \
  | python -c "
import sys, json, base64
body = json.load(sys.stdin)
print(f'Cobertura: {body[\"coverage_pct\"]}%')
with open('mask.png', 'wb') as f:
    f.write(base64.b64decode(body['mask_base64']))
with open('overlay.jpg', 'wb') as f:
    f.write(base64.b64decode(body['overlay_base64']))
"
```

**Resposta:**
```json
{
  "task": "segment",
  "coverage_pct": 32.4,
  "mask_base64": "<base64-png>",
  "overlay_base64": "<base64-jpeg>",
  "image_size": { "width": 1280, "height": 720 },
  "processing_time_ms": 87.3
}
```

### Detecção

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@foto_bovina.jpg" \
  -F "conf=0.25"
```

**Resposta:**
```json
{
  "task": "detect",
  "total_detections": 2,
  "boxes": [
    { "x1": 120.0, "y1": 80.0, "x2": 380.0, "y2": 290.0, "conf": 0.94 },
    { "x1": 500.0, "y1": 60.0, "x2": 750.0, "y2": 310.0, "conf": 0.87 }
  ],
  "overlay_base64": "<base64-jpeg>",
  "image_size": { "width": 1280, "height": 720 },
  "processing_time_ms": 42.1
}
```

### Parâmetros de Query

| Parâmetro | Rota | Tipo | Descrição |
|---|---|---|---|
| `threshold` | `/segment` | `float` 0–1 | Limiar da máscara |
| `conf` | `/detect` | `float` 0–1 | Confiança mínima |
| `overlay` | ambas | `bool` | `true` (padrão) retorna overlay base64 |

**Desativar overlay** (resposta menor, mais rápida):
```bash
curl -X POST "http://localhost:8000/api/v1/segment?overlay=false" -F "file=@foto.jpg"
```

---

## Desenvolvimento Local

### Pré-requisitos

- Python 3.11+
- PyTorch (CPU ou CUDA)

### Setup

```bash
# 1. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Instalar PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Instalar dependências
pip install -r requirements-dev.txt

# 4. Configurar variáveis
cp .env.example .env
# edite SEG_MODEL_PATH e DET_MODEL_PATH para apontar para seus modelos

# 5. Rodar
uvicorn app.main:app --reload --port 8000
```

### Rodar Testes

```bash
# todos os testes
pytest

# com cobertura
pytest --cov=app --cov-report=html

# apenas testes de health
pytest tests/test_health.py -v
```

Os testes **não requerem modelos reais** — usam mocks para garantir execução em CI/CD.

---

## Deploy no EasyPanel

### Método 1 — Git + Dockerfile (recomendado)

1. No EasyPanel, crie um novo **App** do tipo **Dockerfile**.
2. Aponte para o repositório Git.
3. Configure o volume persistente:
   - **Host path:** `/data/cowface-models`
   - **Container path:** `/app/models`
4. Adicione as variáveis de ambiente (copie de `.env.example`).
5. Faça deploy.

### Método 2 — Docker Image

```bash
# Build e push da imagem
docker build -t seu-registro/cowface-api:latest .
docker push seu-registro/cowface-api:latest
```

No EasyPanel, use a imagem publicada e configure o volume para `/app/models`.

### Health Check (EasyPanel)

Configure o health check do EasyPanel com:
- **Path:** `/health`
- **Porta:** `8000`
- **Intervalo:** 30s
- **Timeout:** 10s
- **Start period:** 90s (aguarda carga dos modelos)

---

## Estrutura do Projeto

```
ApiSoftSystem/
├── app/
│   ├── main.py                 # Ponto de entrada FastAPI
│   ├── core/
│   │   ├── config.py           # Configurações via env vars
│   │   └── model_manager.py    # Carregamento e singleton dos modelos
│   ├── routes/
│   │   ├── health.py           # GET /health, GET /api/v1/models
│   │   └── inference.py        # POST /api/v1/segment, POST /api/v1/detect
│   ├── schemas/
│   │   └── responses.py        # Modelos Pydantic de resposta
│   └── services/
│       └── inference.py        # Lógica de negócio (wraps predict.py)
├── tests/
│   ├── conftest.py             # Fixtures pytest
│   ├── test_health.py          # Testes de health e modelos
│   └── test_inference.py       # Testes de segmentação e detecção
├── models/                     # Pasta para modelos (monte como volume)
├── predict.py                  # Script original de inferência CLI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── pytest.ini
```

---

## Formato das Imagens de Saída

As imagens na resposta JSON são codificadas em **base64**:

| Campo | Formato | Descrição |
|---|---|---|
| `mask_base64` | PNG 8-bit grayscale | Pixels brancos = face detectada |
| `overlay_base64` | JPEG | Imagem original com overlay colorido |

**Decodificar em Python:**
```python
import base64, cv2, numpy as np

raw = base64.b64decode(response["mask_base64"])
buf = np.frombuffer(raw, np.uint8)
mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
```

**Decodificar em JavaScript:**
```javascript
const img = document.createElement('img');
img.src = `data:image/jpeg;base64,${response.overlay_base64}`;
document.body.appendChild(img);
```

---

## Códigos de Erro

| Código | Situação |
|---|---|
| `400` | Imagem inválida ou arquivo corrompido |
| `413` | Imagem maior que `MAX_IMAGE_SIZE_MB` |
| `415` | Tipo de arquivo não suportado (use JPG, PNG, BMP ou WEBP) |
| `503` | Modelo não carregado (verifique `SEG_MODEL_PATH` / `DET_MODEL_PATH`) |
| `500` | Erro interno na inferência |
