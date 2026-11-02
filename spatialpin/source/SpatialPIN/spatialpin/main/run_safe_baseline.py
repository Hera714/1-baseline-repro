#!/usr/bin/env python3
"""Offline, non-destructive single-image SpatialPIN 2D baseline runner."""

import argparse
import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import Compose

from Depth_Anything.depth_anything.dpt import DepthAnything
from Depth_Anything.depth_anything.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from lang_segment_anything.lang_sam import LangSAM


DEFAULT_SPIN_SRC = Path(
    os.environ.get(
        "SPIN_SRC",
        "/root/autodl-fs/baselines/spatialpin/SpatialPIN/spatialpin",
    )
)
DEFAULT_DEPTH_CKPT = Path(
    os.environ.get(
        "SPIN_DEPTH_CKPT",
        "/root/autodl-fs/baselines/models/Depth-Anything/checkpoints/depth_anything_vitl14.pth",
    )
)
DEFAULT_LAMA_ENV = Path(os.environ.get("LAMA_ENV", "/root/spatialpin-lama-env"))
DEFAULT_LAMA_SRC = Path(
    os.environ.get(
        "LAMA_SRC",
        "/root/autodl-fs/baselines/spatialpin/SpatialPIN/lama",
    )
)
DEFAULT_LAMA_MODEL = Path(
    os.environ.get(
        "LAMA_MODEL",
        "/root/autodl-fs/baselines/models/lama/big-lama",
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpatialPIN's local 2D components on one image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inpaint-target", default="")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--mask-dilate", type=int, default=9)
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-checkpoint", type=Path, default=DEFAULT_DEPTH_CKPT)
    parser.add_argument("--lama-env", type=Path, default=DEFAULT_LAMA_ENV)
    parser.add_argument("--lama-source", type=Path, default=DEFAULT_LAMA_SRC)
    parser.add_argument("--lama-model", type=Path, default=DEFAULT_LAMA_MODEL)
    return parser.parse_args()


def require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def normalized_phrase(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def safe_name(value):
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("_.")
    return value or "object"


def box_iou(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    return intersection / (box_area + boxes_area - intersection).clamp(min=1e-6)


def class_aware_nms(boxes, scores, phrases, threshold):
    keep = []
    groups = {}
    for index, phrase in enumerate(phrases):
        groups.setdefault(normalized_phrase(phrase), []).append(index)
    for indices in groups.values():
        order = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        while order:
            current = order.pop(0)
            keep.append(current)
            if not order:
                continue
            remaining = torch.tensor(order, dtype=torch.long)
            overlaps = box_iou(boxes[current], boxes[remaining])
            order = [
                idx
                for idx, overlap in zip(order, overlaps.tolist())
                if overlap <= threshold
            ]
    return sorted(keep)


def draw_overlay(image, boxes, masks, phrases, scores):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    palette = np.asarray(
        [
            [36, 214, 123],
            [255, 99, 132],
            [54, 162, 235],
            [255, 206, 86],
            [153, 102, 255],
            [255, 159, 64],
        ],
        dtype=np.float32,
    )
    for index, mask in enumerate(masks):
        selected = mask.astype(bool)
        color = palette[index % len(palette)]
        base[selected] = base[selected] * 0.55 + color * 0.45
    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for index, (box, phrase, score) in enumerate(zip(boxes, phrases, scores)):
        color = tuple(palette[index % len(palette)].astype(int).tolist())
        coords = tuple(float(value) for value in box)
        draw.rectangle(coords, outline=color, width=3)
        label = f"{phrase} {float(score):.3f}"
        left, top = coords[0], max(0, coords[1] - 13)
        draw.rectangle((left, top, left + 7 * len(label) + 4, top + 13), fill=color)
        draw.text((left + 2, top), label, fill=(0, 0, 0))
    return result


def run_langsam(args, image, output_dir):
    print("[1/3] Loading GroundingDINO + SAM (offline)...", flush=True)
    started = time.time()
    model = LangSAM()
    masks, boxes, phrases, logits = model.predict(
        image,
        args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    boxes = torch.as_tensor(boxes).detach().cpu().float()
    scores = torch.as_tensor(logits).detach().cpu().float().reshape(-1)
    masks = torch.as_tensor(masks).detach().cpu()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = masks.numpy().astype(bool)
    phrases = [str(item) for item in phrases]

    keep = class_aware_nms(boxes, scores, phrases, args.nms_iou)
    boxes = boxes[keep]
    scores = scores[keep]
    masks = masks[keep]
    phrases = [phrases[index] for index in keep]

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    detections = []
    for index, (box, mask, phrase, score) in enumerate(
        zip(boxes.tolist(), masks, phrases, scores.tolist())
    ):
        mask_name = f"{index:03d}_{safe_name(phrase)}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / mask_name)
        detections.append(
            {
                "index": index,
                "phrase": phrase,
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
                "mask": f"masks/{mask_name}",
                "mask_area_pixels": int(mask.sum()),
            }
        )

    overlay = draw_overlay(image, boxes.tolist(), masks, phrases, scores.tolist())
    overlay.save(output_dir / "overlay.png")
    (output_dir / "detections.json").write_text(
        json.dumps(detections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    elapsed = time.time() - started
    print(f"      Kept {len(detections)} detections after class-aware NMS.", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detections, masks, elapsed, peak_gb


def run_depth(args, image, output_dir):
    print("[2/3] Running Depth Anything ViT-L from local files...", flush=True)
    require_file(args.depth_checkpoint, "Depth Anything checkpoint")
    started = time.time()
    config = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "localhub": True,
    }
    model = DepthAnything(config)
    state = torch.load(args.depth_checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(args.device).eval()

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    tensor = torch.from_numpy(transform({"image": rgb})["image"]).unsqueeze(0)
    tensor = tensor.to(args.device)
    with torch.inference_mode():
        depth = model(tensor)
        depth = F.interpolate(
            depth.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
    depth = depth.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / "depth.npy", depth)
    low, high = float(depth.min()), float(depth.max())
    normalized = ((depth - low) / max(high - low, 1e-8) * 255).astype(np.uint8)
    Image.fromarray(normalized).save(output_dir / "depth_gray.png")
    color_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_rgb).save(output_dir / "depth_color.png")
    comparison = Image.new("RGB", (width * 2, height))
    comparison.paste(image.convert("RGB"), (0, 0))
    comparison.paste(Image.fromarray(color_rgb), (width, 0))
    comparison.save(output_dir / "depth_comparison.png")
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    elapsed = time.time() - started

    del tensor, depth, state, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"min": low, "max": high, "shape": [height, width]}, elapsed, peak_gb


def select_inpaint_detection(detections, target):
    target = normalized_phrase(target)
    candidates = [
        item
        for item in detections
        if target in normalized_phrase(item["phrase"])
        or normalized_phrase(item["phrase"]) in target
    ]
    if not candidates:
        available = ", ".join(item["phrase"] for item in detections) or "none"
        raise ValueError(f"Inpaint target '{target}' not detected. Available: {available}")
    return max(candidates, key=lambda item: item["score"])


def run_lama(args, image, detections, masks, output_dir):
    if not args.inpaint_target:
        print("[3/3] LaMa skipped (no --inpaint-target).", flush=True)
        return None, 0.0
    print(f"[3/3] Inpainting '{args.inpaint_target}' with local Big-LaMa...", flush=True)
    python = args.lama_env / "bin/python"
    predictor = args.lama_source / "bin/predict.py"
    config = args.lama_source / "configs/prediction/default.yaml"
    checkpoint = args.lama_model / "models/best.ckpt"
    for path, label in [
        (python, "LaMa Python"),
        (predictor, "LaMa predictor"),
        (config, "LaMa prediction config"),
        (checkpoint, "Big-LaMa checkpoint"),
    ]:
        require_file(path, label)

    selected = select_inpaint_detection(detections, args.inpaint_target)
    mask = masks[selected["index"]].astype(np.uint8) * 255
    if args.mask_dilate > 1:
        size = args.mask_dilate if args.mask_dilate % 2 else args.mask_dilate + 1
        kernel = np.ones((size, size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    input_dir = output_dir / "lama_input"
    lama_output = output_dir / "lama_output"
    input_dir.mkdir(exist_ok=True)
    lama_output.mkdir(exist_ok=True)
    source = input_dir / "source.png"
    mask_path = input_dir / "source_mask001.png"
    image.save(source)
    Image.fromarray(mask).save(mask_path)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.lama_source)
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    started = time.time()
    command = [
        str(python),
        str(predictor),
        f"model.path={args.lama_model}",
        f"indir={input_dir}",
        f"outdir={lama_output}",
        f"device={args.device}",
        "refine=false",
    ]
    subprocess.run(command, cwd=args.lama_source, env=environment, check=True)
    expected = lama_output / mask_path.name
    if not expected.is_file():
        outputs = sorted(lama_output.glob("*.png"))
        if not outputs:
            raise RuntimeError(f"LaMa completed but produced no PNG in {lama_output}")
        expected = outputs[0]
    elapsed = time.time() - started
    return {
        "target": args.inpaint_target,
        "selected_detection": selected,
        "dilated_mask": str(mask_path.relative_to(output_dir)),
        "output": str(expected.relative_to(output_dir)),
    }, elapsed


def main():
    args = parse_args()
    require_file(args.image, "input image")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    image.save(output_dir / "input.png")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    detections, masks, langsam_seconds, langsam_peak = run_langsam(
        args, image, output_dir
    )
    depth_info = None
    depth_seconds = 0.0
    depth_peak = 0.0
    if args.skip_depth:
        print("[2/3] Depth skipped (--skip-depth).", flush=True)
    else:
        depth_info, depth_seconds, depth_peak = run_depth(args, image, output_dir)
    lama_info, lama_seconds = run_lama(args, image, detections, masks, output_dir)

    metadata = {
        "input_image": str(args.image.resolve()),
        "input_size": list(image.size),
        "prompt": args.prompt,
        "offline": True,
        "external_api_used": False,
        "thresholds": {
            "box": args.box_threshold,
            "text": args.text_threshold,
            "class_aware_nms_iou": args.nms_iou,
        },
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "detections": detections,
        "depth": depth_info,
        "inpainting": lama_info,
        "runtime_seconds": {
            "langsam": round(langsam_seconds, 3),
            "depth": round(depth_seconds, 3),
            "lama": round(lama_seconds, 3),
            "total": round(langsam_seconds + depth_seconds + lama_seconds, 3),
        },
        "peak_gpu_memory_gb": {
            "after_langsam": round(langsam_peak, 3),
            "after_depth": round(depth_peak, 3),
        },
        "outputs": {
            "input": "input.png",
            "detections": "detections.json",
            "overlay": "overlay.png",
            "depth_array": None if args.skip_depth else "depth.npy",
            "depth_color": None if args.skip_depth else "depth_color.png",
            "depth_comparison": None if args.skip_depth else "depth_comparison.png",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"SpatialPIN safe baseline PASSED: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
