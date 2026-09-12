from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Coordenada X do canto superior esquerdo")
    y1: float = Field(..., description="Coordenada Y do canto superior esquerdo")
    x2: float = Field(..., description="Coordenada X do canto inferior direito")
    y2: float = Field(..., description="Coordenada Y do canto inferior direito")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confiança da detecção (0–1)")


class ImageSize(BaseModel):
    width: int
    height: int


class SegmentResponse(BaseModel):
    task: str = Field("segment", description="Tipo da tarefa executada")
    coverage_pct: float = Field(..., description="Percentual da imagem coberta pela máscara")
    mask_base64: str = Field(..., description="Máscara binária em PNG codificada em base64")
    overlay_base64: Optional[str] = Field(
        None, description="Imagem original com overlay em JPEG codificada em base64"
    )
    image_size: ImageSize
    processing_time_ms: float = Field(..., description="Tempo de inferência em milissegundos")

    model_config = {"json_schema_extra": {
        "example": {
            "task": "segment",
            "coverage_pct": 32.4,
            "mask_base64": "<base64-png>",
            "overlay_base64": "<base64-jpeg>",
            "image_size": {"width": 1280, "height": 720},
            "processing_time_ms": 87.3,
        }
    }}


class DetectResponse(BaseModel):
    task: str = Field("detect", description="Tipo da tarefa executada")
    total_detections: int = Field(..., description="Número de faces bovinas detectadas")
    boxes: List[BoundingBox] = Field(..., description="Lista de caixas delimitadoras")
    overlay_base64: Optional[str] = Field(
        None, description="Imagem original com overlay em JPEG codificada em base64"
    )
    image_size: ImageSize
    processing_time_ms: float = Field(..., description="Tempo de inferência em milissegundos")

    model_config = {"json_schema_extra": {
        "example": {
            "task": "detect",
            "total_detections": 2,
            "boxes": [
                {"x1": 120.0, "y1": 80.0, "x2": 380.0, "y2": 290.0, "conf": 0.94},
                {"x1": 500.0, "y1": 60.0, "x2": 750.0, "y2": 310.0, "conf": 0.87},
            ],
            "overlay_base64": "<base64-jpeg>",
            "image_size": {"width": 1280, "height": 720},
            "processing_time_ms": 42.1,
        }
    }}


class HealthResponse(BaseModel):
    status: str
    seg_model_loaded: bool
    det_model_loaded: bool
    device: str
    version: str


class ModelInfoResponse(BaseModel):
    seg_model_loaded: bool
    seg_image_size: Optional[int]
    det_model_loaded: bool
    device: str
