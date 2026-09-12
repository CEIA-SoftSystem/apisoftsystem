"""Fixtures compartilhadas para todos os testes.

Os testes não dependem de modelos reais — usamos mocks para evitar
o carregamento de arquivos .pt durante o CI/CD.
"""

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True, scope="session")
def disable_model_loading():
    """Impede que load_models e unload_models toquem o disco durante os testes."""
    with patch("app.core.model_manager.ModelManager.load_models"), \
         patch("app.core.model_manager.ModelManager.unload_models"):
        yield


@pytest.fixture(scope="session")
def app():
    from app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def loaded_models():
    """Injeta modelos mock no model_manager para testes de inferência."""
    from app.core.model_manager import model_manager

    prev = {
        "seg_model": model_manager.seg_model,
        "seg_image_size": model_manager.seg_image_size,
        "det_model": model_manager.det_model,
        "device": model_manager.device,
        "yolo_device": model_manager.yolo_device,
    }

    model_manager.seg_model = MagicMock()
    model_manager.seg_image_size = 512
    model_manager.det_model = MagicMock()
    model_manager.device = torch.device("cpu")
    model_manager.yolo_device = "cpu"

    yield model_manager

    for k, v in prev.items():
        setattr(model_manager, k, v)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Cria uma imagem JPEG sintética de 640x480."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:380, 160:480] = [180, 140, 90]   # região simulando face bovina
    cv2.circle(img, (320, 240), 80, (210, 170, 120), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
