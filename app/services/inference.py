import base64
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.model_manager import model_manager
from predict import overlay_boxes, overlay_mask, predict_mask


def _decode_image(image_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem enviada.")
    return img


def _encode_image(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem de saída.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def run_segmentation(image_bytes: bytes, threshold: float, with_overlay: bool = True) -> dict:
    if model_manager.seg_model is None:
        raise HTTPException(status_code=503, detail="Modelo de segmentação não está carregado.")

    img = _decode_image(image_bytes)
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    mask = predict_mask(
        model_manager.seg_model,
        img,
        model_manager.seg_image_size,
        model_manager.device,
        threshold,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    mask_img = (mask * 255).astype(np.uint8)

    return {
        "task": "segment",
        "coverage_pct": round(float(mask.mean() * 100), 2),
        "mask_base64": _encode_image(mask_img, ".png"),
        "overlay_base64": _encode_image(overlay_mask(img, mask)) if with_overlay else None,
        "image_size": {"width": w, "height": h},
        "processing_time_ms": round(elapsed_ms, 2),
    }


def run_detection(image_bytes: bytes, conf: float, with_overlay: bool = True) -> dict:
    if model_manager.det_model is None:
        raise HTTPException(status_code=503, detail="Modelo de detecção não está carregado.")

    img = _decode_image(image_bytes)
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    # YOLO aceita numpy array diretamente, evitando gravação em arquivo temporário
    res = model_manager.det_model.predict(
        img,
        imgsz=512,
        conf=conf,
        verbose=False,
        device=model_manager.yolo_device,
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    boxes: list[dict] = []
    if res.boxes is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append({"x1": float(x1), "y1": float(y1),
                          "x2": float(x2), "y2": float(y2), "conf": float(c)})

    return {
        "task": "detect",
        "total_detections": len(boxes),
        "boxes": boxes,
        "overlay_base64": _encode_image(overlay_boxes(img, boxes)) if with_overlay else None,
        "image_size": {"width": w, "height": h},
        "processing_time_ms": round(elapsed_ms, 2),
    }
