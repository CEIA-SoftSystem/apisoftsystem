import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.model_manager import model_manager
from app.routes import health, inference

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando CowFace API v%s", settings.app_version)
    model_manager.load_models()
    yield
    model_manager.unload_models()
    logger.info("CowFace API encerrada.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "API REST para **segmentação** e **detecção** de faces bovinas.\n\n"
        "- `/api/v1/segment` — segmentação com U-Net (máscara pixel a pixel)\n"
        "- `/api/v1/detect` — detecção com YOLO11 (bounding boxes)\n\n"
        "Resultados de imagem são retornados como strings **base64** dentro do JSON."
    ),
    contact={
        "name": "SoftSystem",
        "email": "duvictorsc@gmail.com",
    },
    license_info={"name": "Proprietário"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(inference.router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
