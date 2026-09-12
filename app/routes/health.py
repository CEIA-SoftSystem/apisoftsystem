from fastapi import APIRouter

from app.core.config import settings
from app.core.model_manager import model_manager
from app.schemas.responses import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificação de saúde da API",
    tags=["health"],
)
async def health() -> HealthResponse:
    s = model_manager.status
    return HealthResponse(
        status="ok",
        seg_model_loaded=s["seg_model_loaded"],
        det_model_loaded=s["det_model_loaded"],
        device=s["device"],
        version=settings.app_version,
    )


@router.get(
    "/api/v1/models",
    response_model=ModelInfoResponse,
    summary="Informações dos modelos carregados",
    tags=["models"],
)
async def models_info() -> ModelInfoResponse:
    s = model_manager.status
    return ModelInfoResponse(
        seg_model_loaded=s["seg_model_loaded"],
        seg_image_size=model_manager.seg_image_size,
        det_model_loaded=s["det_model_loaded"],
        device=s["device"],
    )
