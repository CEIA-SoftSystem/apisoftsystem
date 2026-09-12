from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.schemas.responses import DetectResponse, SegmentResponse
from app.services.inference import run_detection, run_segmentation

router = APIRouter(prefix="/api/v1")

_MAX_BYTES = settings.max_image_size_mb * 1024 * 1024
_ACCEPTED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


async def _read_validated_image(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in _ACCEPTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. Use JPEG, PNG, BMP ou WEBP.",
        )
    data = await file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem muito grande. Máximo permitido: {settings.max_image_size_mb} MB.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")
    return data


@router.post(
    "/segment",
    response_model=SegmentResponse,
    summary="Segmentação de face bovina (U-Net)",
    description=(
        "Recebe uma imagem e retorna a máscara de segmentação binária da face bovina, "
        "além de uma imagem com overlay colorido opcional. "
        "Os resultados de imagem são retornados codificados em base64."
    ),
    tags=["inference"],
)
async def segment(
    file: UploadFile = File(..., description="Imagem da face bovina (JPG, PNG, BMP, WEBP)"),
    threshold: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Limiar da máscara de segmentação (padrão: configuração do servidor)",
    ),
    overlay: bool = Query(default=True, description="Incluir imagem com overlay na resposta"),
) -> SegmentResponse:
    data = await _read_validated_image(file)
    t = threshold if threshold is not None else settings.default_threshold
    return run_segmentation(data, t, overlay)


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detecção de faces bovinas (YOLO11)",
    description=(
        "Recebe uma imagem e retorna as caixas delimitadoras (bounding boxes) das faces bovinas "
        "detectadas com suas coordenadas e confiança. "
        "Também retorna uma imagem com overlay opcional codificada em base64."
    ),
    tags=["inference"],
)
async def detect(
    file: UploadFile = File(..., description="Imagem da face bovina (JPG, PNG, BMP, WEBP)"),
    conf: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confiança mínima para detecção (padrão: configuração do servidor)",
    ),
    overlay: bool = Query(default=True, description="Incluir imagem com overlay na resposta"),
) -> DetectResponse:
    data = await _read_validated_image(file)
    c = conf if conf is not None else settings.default_conf
    return run_detection(data, c, overlay)
