#!/usr/bin/env python
"""Inferência de bovinos — segmentação (U-Net) OU detecção de caixa (YOLO).

Script autocontido: detecta sozinho se o modelo é de segmentação ou de
detecção e gera a saída correspondente para cada imagem.

  * U-Net  (models/unet_seg.pt)   -> máscara + overlay
  * YOLO11 (models/yolo11_bbox.pt) -> caixa (bbox) + overlay + coordenadas

Uso:
    # segmentação (máscara)
    python predict.py --model models/unet_seg.pt --input foto.jpg

    # detecção (caixa)
    python predict.py --model models/yolo11_bbox.pt --input foto.jpg

    # pasta inteira, escolhendo a pasta de saída, forçando CPU
    python predict.py --model models/unet_seg.pt --input fotos/ --output saida/ --device cpu

Requisitos: pip install -r requirements.txt
Funciona em CPU (segundos/imagem) ou GPU NVIDIA (milissegundos), automático.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # normalização ImageNet
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# cores BGR
AZUL = (214, 120, 42)
VERDE = (60, 220, 60)


# ---------------------------------------------------------------------------
# detecção do tipo de modelo
# ---------------------------------------------------------------------------
def detect_task(model_path: Path) -> str:
    """Retorna 'segment' (U-Net nosso) ou 'detect' (YOLO)."""
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            # Formato antigo: chaves no topo | Formato novo: config aninhada
            if "arch" in ckpt or "arch" in ckpt.get("config", {}):
                return "segment"
    except Exception:
        pass
    return "detect"


# ---------------------------------------------------------------------------
# SEGMENTAÇÃO (U-Net)
# ---------------------------------------------------------------------------
def load_seg_model(model_path: Path, device: torch.device):
    import segmentation_models_pytorch as smp

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Suporta dois formatos de checkpoint:
    #   - Antigo: chaves arch/encoder/image_size no topo do dict
    #   - Novo: mesmas chaves aninhadas em ckpt['config']
    cfg = ckpt.get("config", ckpt)
    arch       = cfg.get("arch")       or ckpt.get("arch", "unet")
    encoder    = cfg.get("encoder")    or ckpt.get("encoder", "resnet34") or "resnet34"
    image_size = cfg.get("image_size") or ckpt.get("image_size", 512)
    iou_ref    = ckpt.get("best_val_iou") or ckpt.get("reference_test_iou", "n/d")

    factories = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus,
                 "fpn": smp.FPN, "segformer": smp.Segformer}
    model = factories[arch](
        encoder_name=encoder, encoder_weights=None,
        in_channels=3, classes=1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[segmentação] {arch} ({encoder}) | entrada {image_size}px | IoU ref. {iou_ref}")
    return model, image_size


@torch.no_grad()
def predict_mask(model, image_bgr, image_size, device, threshold=0.5):
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    x = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    probs = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
    mask = (probs > threshold).astype(np.uint8)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay_mask(image_bgr, mask):
    out = image_bgr.copy()
    out[mask > 0] = (0.55 * out[mask > 0] + 0.45 * np.array(AZUL)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, VERDE, 2)
    return out


# ---------------------------------------------------------------------------
# DETECÇÃO (YOLO)
# ---------------------------------------------------------------------------
def load_det_model(model_path: Path, device):
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # Ultralytics pode não carregar os pesos quando o arquivo não tem extensão .pt.
    # Nesse caso model.model é uma string com o caminho — fazemos o fallback manual.
    if isinstance(model.model, str):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "parameters"):
            model.model = ckpt["model"].float().fuse()

    n = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"[detecção] YOLO ({n:.1f}M params)")
    return model


def predict_boxes(model, image_path, image_size, device, conf=0.25):
    res = model.predict(str(image_path), imgsz=image_size, conf=conf,
                        verbose=False, device=device)[0]
    boxes = []
    if res.boxes is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append({"x1": float(x1), "y1": float(y1), "x2": float(x2),
                          "y2": float(y2), "conf": float(c)})
    return boxes


def overlay_boxes(image_bgr, boxes):
    out = image_bgr.copy()
    for b in boxes:
        p1 = (int(b["x1"]), int(b["y1"]))
        p2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(out, p1, p2, AZUL, 2)
        label = f"animal {b['conf']:.2f}"
        cv2.putText(out, label, (p1[0], max(0, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERDE, 2)
    return out


# ---------------------------------------------------------------------------
# comum
# ---------------------------------------------------------------------------
def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            sys.exit(f"ERRO: nenhuma imagem em {input_path}")
        return files
    sys.exit(f"ERRO: entrada não existe: {input_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True,
                   help="models/unet_seg.pt (segmentação) ou models/yolo11_bbox.pt (detecção)")
    p.add_argument("--input", type=Path, required=True, help="imagem ou pasta")
    p.add_argument("--output", type=Path, default=Path("predicoes"))
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--threshold", type=float, default=0.5, help="segmentação: limiar da máscara")
    p.add_argument("--conf", type=float, default=0.25, help="detecção: confiança mínima")
    p.add_argument("--no-overlay", action="store_true")
    args = p.parse_args()

    if not args.model.exists():
        sys.exit(f"ERRO: modelo não encontrado: {args.model}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Dispositivo: {device}")
    task = detect_task(args.model)

    files = collect_inputs(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    if task == "segment":
        model, image_size = load_seg_model(args.model, device)
    else:
        model = load_det_model(args.model, 0 if device.type == "cuda" else "cpu")
        image_size = 512

    ok, failed = 0, 0
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  AVISO: não consegui ler {path.name}; pulando")
            failed += 1
            continue

        if task == "segment":
            mask = predict_mask(model, image, image_size, device, args.threshold)
            cv2.imwrite(str(args.output / f"{path.stem}_mask.png"), mask * 255)
            if not args.no_overlay:
                cv2.imwrite(str(args.output / f"{path.stem}_overlay.jpg"),
                            overlay_mask(image, mask))
            print(f"  {path.name}: {100 * mask.mean():.1f}% da imagem = animal")
        else:
            boxes = predict_boxes(model, path, image_size,
                                  0 if device.type == "cuda" else "cpu", args.conf)
            (args.output / f"{path.stem}_boxes.json").write_text(
                json.dumps(boxes, indent=2))
            if not args.no_overlay:
                cv2.imwrite(str(args.output / f"{path.stem}_overlay.jpg"),
                            overlay_boxes(image, boxes))
            if boxes:
                c = max(b["conf"] for b in boxes)
                print(f"  {path.name}: {len(boxes)} caixa(s), maior conf {c:.2f}")
            else:
                print(f"  {path.name}: nenhum animal detectado")
        ok += 1

    print(f"\nConcluído: {ok} imagem(ns), {failed} falha(s). Resultados em {args.output}/")


if __name__ == "__main__":
    main()
