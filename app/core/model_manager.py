import logging
from pathlib import Path
from typing import Any, Optional

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self) -> None:
        self.seg_model: Optional[Any] = None
        self.seg_image_size: Optional[int] = None
        self.det_model: Optional[Any] = None
        self.device: Optional[torch.device] = None
        self.yolo_device: Any = "cpu"  # int (0) para GPU, "cpu" para CPU

    def _resolve_device(self) -> torch.device:
        if settings.device == "cuda":
            return torch.device("cuda")
        if settings.device == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self) -> None:
        self.device = self._resolve_device()
        self.yolo_device = 0 if self.device.type == "cuda" else "cpu"
        logger.info("Dispositivo: %s", self.device)

        if settings.seg_model_path:
            path = Path(settings.seg_model_path)
            if path.exists():
                try:
                    from predict import load_seg_model
                    self.seg_model, self.seg_image_size = load_seg_model(path, self.device)
                    logger.info("Modelo de segmentação carregado: %s", path)
                except Exception:
                    logger.exception("Falha ao carregar modelo de segmentação: %s", path)
            else:
                logger.warning("Modelo de segmentação não encontrado: %s", path)

        if settings.det_model_path:
            path = Path(settings.det_model_path)
            if path.exists():
                try:
                    from predict import load_det_model
                    self.det_model = load_det_model(path, self.yolo_device)
                    logger.info("Modelo de detecção carregado: %s", path)
                except Exception:
                    logger.exception("Falha ao carregar modelo de detecção: %s", path)
            else:
                logger.warning("Modelo de detecção não encontrado: %s", path)

    def unload_models(self) -> None:
        self.seg_model = None
        self.det_model = None
        logger.info("Modelos descarregados.")

    @property
    def status(self) -> dict:
        return {
            "seg_model_loaded": self.seg_model is not None,
            "det_model_loaded": self.det_model is not None,
            "device": str(self.device) if self.device else "não inicializado",
        }


model_manager = ModelManager()
