"""Testes dos endpoints de inferência (segmentação e detecção)."""

import base64

import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_fake_mask(h=480, w=640) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:380, 160:480] = 1
    return mask


def _make_fake_overlay(h=480, w=640) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_fake_yolo_result(boxes_data: list[dict]):
    """Cria um resultado YOLO mock compatível com o formato ultralytics."""
    result = MagicMock()
    if boxes_data:
        xyxy = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes_data], dtype=np.float32)
        confs = np.array([b["conf"] for b in boxes_data], dtype=np.float32)

        import torch
        result.boxes.xyxy = torch.from_numpy(xyxy)
        result.boxes.conf = torch.from_numpy(confs)
    else:
        result.boxes = None
    return result


def _upload(client, endpoint: str, image_bytes: bytes, **params):
    return client.post(
        endpoint,
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        params=params,
    )


# ---------------------------------------------------------------------------
# testes de segmentação
# ---------------------------------------------------------------------------

class TestSegment:
    def test_success_returns_200(self, client, loaded_models, sample_image_bytes):
        with patch("app.services.inference.predict_mask", return_value=_make_fake_mask()), \
             patch("app.services.inference.overlay_mask", return_value=_make_fake_overlay()):
            r = _upload(client, "/api/v1/segment", sample_image_bytes)

        assert r.status_code == 200

    def test_response_schema(self, client, loaded_models, sample_image_bytes):
        with patch("app.services.inference.predict_mask", return_value=_make_fake_mask()), \
             patch("app.services.inference.overlay_mask", return_value=_make_fake_overlay()):
            body = _upload(client, "/api/v1/segment", sample_image_bytes).json()

        assert body["task"] == "segment"
        assert isinstance(body["coverage_pct"], float)
        assert body["mask_base64"]
        assert body["overlay_base64"]
        assert body["image_size"]["width"] == 640
        assert body["image_size"]["height"] == 480
        assert body["processing_time_ms"] >= 0

    def test_mask_is_valid_base64_image(self, client, loaded_models, sample_image_bytes):
        with patch("app.services.inference.predict_mask", return_value=_make_fake_mask()), \
             patch("app.services.inference.overlay_mask", return_value=_make_fake_overlay()):
            body = _upload(client, "/api/v1/segment", sample_image_bytes).json()

        raw = base64.b64decode(body["mask_base64"])
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        assert img.shape == (480, 640)

    def test_coverage_pct_reflects_mask(self, client, loaded_models, sample_image_bytes):
        full_mask = np.ones((480, 640), dtype=np.uint8)
        with patch("app.services.inference.predict_mask", return_value=full_mask), \
             patch("app.services.inference.overlay_mask", return_value=_make_fake_overlay()):
            body = _upload(client, "/api/v1/segment", sample_image_bytes).json()

        assert body["coverage_pct"] == pytest.approx(100.0, abs=0.01)

    def test_overlay_false_omits_overlay(self, client, loaded_models, sample_image_bytes):
        with patch("app.services.inference.predict_mask", return_value=_make_fake_mask()):
            body = _upload(client, "/api/v1/segment", sample_image_bytes, overlay=False).json()

        assert body["overlay_base64"] is None

    def test_custom_threshold_accepted(self, client, loaded_models, sample_image_bytes):
        with patch("app.services.inference.predict_mask", return_value=_make_fake_mask()) as mock_pm, \
             patch("app.services.inference.overlay_mask", return_value=_make_fake_overlay()):
            _upload(client, "/api/v1/segment", sample_image_bytes, threshold=0.8)

        args = mock_pm.call_args[0]  # positional args: (model, img, image_size, device, threshold)
        assert args[4] == pytest.approx(0.8)

    def test_503_when_model_not_loaded(self, client, sample_image_bytes):
        r = _upload(client, "/api/v1/segment", sample_image_bytes)
        assert r.status_code == 503

    def test_400_on_invalid_image(self, client, loaded_models):
        r = client.post(
            "/api/v1/segment",
            files={"file": ("bad.jpg", b"isso-nao-e-uma-imagem", "image/jpeg")},
        )
        assert r.status_code == 400

    def test_413_on_image_too_large(self, client, loaded_models):
        big_data = b"\xff" * (11 * 1024 * 1024)  # 11 MB
        r = client.post(
            "/api/v1/segment",
            files={"file": ("big.jpg", big_data, "image/jpeg")},
        )
        assert r.status_code == 413

    def test_415_on_unsupported_type(self, client, loaded_models, sample_image_bytes):
        r = client.post(
            "/api/v1/segment",
            files={"file": ("doc.pdf", sample_image_bytes, "application/pdf")},
        )
        assert r.status_code == 415


# ---------------------------------------------------------------------------
# testes de detecção
# ---------------------------------------------------------------------------

class TestDetect:
    _BOXES = [
        {"x1": 100.0, "y1": 80.0, "x2": 350.0, "y2": 290.0, "conf": 0.92},
        {"x1": 400.0, "y1": 60.0, "x2": 620.0, "y2": 300.0, "conf": 0.78},
    ]

    def _patch_yolo(self, boxes=None):
        if boxes is None:
            boxes = self._BOXES
        mock_result = _make_fake_yolo_result(boxes)
        model_mock = MagicMock()
        model_mock.predict.return_value = [mock_result]
        return model_mock

    def test_success_returns_200(self, client, loaded_models, sample_image_bytes):
        loaded_models.det_model = self._patch_yolo()
        with patch("app.services.inference.overlay_boxes", return_value=_make_fake_overlay()):
            r = _upload(client, "/api/v1/detect", sample_image_bytes)
        assert r.status_code == 200

    def test_response_schema(self, client, loaded_models, sample_image_bytes):
        loaded_models.det_model = self._patch_yolo()
        with patch("app.services.inference.overlay_boxes", return_value=_make_fake_overlay()):
            body = _upload(client, "/api/v1/detect", sample_image_bytes).json()

        assert body["task"] == "detect"
        assert body["total_detections"] == 2
        assert len(body["boxes"]) == 2
        assert body["image_size"]["width"] == 640

    def test_boxes_fields(self, client, loaded_models, sample_image_bytes):
        loaded_models.det_model = self._patch_yolo()
        with patch("app.services.inference.overlay_boxes", return_value=_make_fake_overlay()):
            boxes = _upload(client, "/api/v1/detect", sample_image_bytes).json()["boxes"]

        first = boxes[0]
        for field in ("x1", "y1", "x2", "y2", "conf"):
            assert field in first
        assert 0.0 <= first["conf"] <= 1.0

    def test_empty_detections(self, client, loaded_models, sample_image_bytes):
        loaded_models.det_model = self._patch_yolo(boxes=[])
        with patch("app.services.inference.overlay_boxes", return_value=_make_fake_overlay()):
            body = _upload(client, "/api/v1/detect", sample_image_bytes).json()

        assert body["total_detections"] == 0
        assert body["boxes"] == []

    def test_overlay_false(self, client, loaded_models, sample_image_bytes):
        loaded_models.det_model = self._patch_yolo()
        body = _upload(client, "/api/v1/detect", sample_image_bytes, overlay=False).json()
        assert body["overlay_base64"] is None

    def test_503_when_model_not_loaded(self, client, sample_image_bytes):
        r = _upload(client, "/api/v1/detect", sample_image_bytes)
        assert r.status_code == 503

    def test_400_on_invalid_image(self, client, loaded_models):
        r = client.post(
            "/api/v1/detect",
            files={"file": ("bad.jpg", b"dados-invalidos", "image/jpeg")},
        )
        assert r.status_code == 400
