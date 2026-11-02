#!/usr/bin/env python3
"""SpatialPIN-style matched-backbone adaptation for SpatialLab-MCQ.

This adapter preserves the released SpatialPIN perception configuration:
640x480 working images, GroundingDINO-SwinB, SAM-ViT-H, Depth Anything ViT-L,
and LangSAM's official 0.30/0.25 thresholds.  It never reads gold answers or
SpatialLab category metadata before prediction.

The released SpatialPIN repository does not include a SpatialLab MCQ runner and
its language stages call GPT-4V.  For a reproducible local comparison, the
``discover`` and ``evaluate`` stages use the same local Ovis2.5-9B backbone as
3D-KAR.  Outputs therefore identify themselves as a matched-backbone adapter,
not as an official SpatialPIN benchmark result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


ADAPTER_NAME = "SpatialPIN-style matched-backbone SpatialLab adapter"
WORKING_SIZE = (640, 480)
OFFICIAL_BOX_THRESHOLD = 0.30
OFFICIAL_TEXT_THRESHOLD = 0.25


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def format_options(item: dict[str, Any]) -> str:
    return "\n".join(f"{i}. {item.get(f'Option_{i}', '')}" for i in range(1, 5))


def read_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    required = {"id", "Image_Filename", "Question", "Option_1", "Option_2", "Option_3", "Option_4"}
    for index, item in enumerate(data):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Dataset row {index} is missing {sorted(missing)}")
    return data


def select_items(
    data: list[dict[str, Any]], start: int, end: Optional[int], ids: str
) -> list[dict[str, Any]]:
    if ids:
        requested = [value.strip() for value in ids.split(",") if value.strip()]
        if len(requested) != len(set(requested)):
            raise ValueError("--ids contains duplicates")
        indexed = {str(item["id"]): item for item in data}
        unknown = sorted(set(requested) - set(indexed))
        if unknown:
            raise ValueError(f"Unknown dataset IDs: {unknown}")
        return [indexed[value] for value in requested]
    return data[start:end]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def read_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        result[str(row["id"])] = row
    return result


def append_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def prepare_output(path: Path, overwrite: bool) -> set[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path.unlink(missing_ok=True)
    return {str(row["id"]) for row in read_jsonl(path)}


def first_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found in model response")


ENTITY_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "what", "which", "where",
    "who", "how", "if", "then", "than", "and", "or", "with", "without", "from",
    "into", "onto", "between", "among", "around", "behind", "beside", "inside",
    "outside", "above", "below", "under", "over", "near", "toward", "towards", "of",
    "to", "for", "in", "on", "at", "by", "up", "down", "out", "off", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should", "may", "might", "must", "all", "both",
    "each", "every", "any", "some", "most", "many", "much", "more", "less", "only",
    "same", "similar", "different", "other", "another", "first", "second", "third",
    "last", "middle", "closest", "nearest", "farthest", "foremost", "top", "bottom",
    "left", "right", "front", "back", "horizontal", "vertical", "diagonal", "north",
    "south", "east", "west", "yes", "no", "true", "false", "not", "given", "picture",
    "image", "scene", "view", "direction", "position", "relationship", "distance", "size",
    "shape", "line", "point", "object", "objects", "thing", "things", "answer", "option",
}


def fallback_entities(item: dict[str, Any]) -> list[dict[str, str]]:
    text = item["Question"] + " " + " ".join(str(item.get(f"Option_{i}", "")) for i in range(1, 5))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    candidates = []
    for word in words:
        word = word[:-2] if word.endswith("'s") else word
        if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
            word = word[:-1]
        if len(word) < 3 or word in ENTITY_STOPWORDS:
            continue
        if word not in candidates:
            candidates.append(word)
    if not candidates:
        candidates = ["person", "animal", "vehicle", "furniture"]
    return [{"name": value, "description": value} for value in candidates[:12]]


def clean_discovered_objects(parsed: dict[str, Any], item: dict[str, Any]) -> list[dict[str, str]]:
    raw_objects = parsed.get("objects", [])
    cleaned = []
    seen = set()
    if isinstance(raw_objects, list):
        for obj in raw_objects:
            if isinstance(obj, str):
                name = description = normalize_text(obj)
            elif isinstance(obj, dict):
                name = normalize_text(obj.get("name") or obj.get("description"))
                description = normalize_text(obj.get("description") or name)
            else:
                continue
            key = description.casefold()
            if not description or key in seen:
                continue
            seen.add(key)
            cleaned.append({"name": name or description, "description": description})
    return cleaned or fallback_entities(item)


def import_ovis(kar3d_root: Path):
    src = kar3d_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"3D-KAR src directory not found: {src}")
    sys.path.insert(0, str(src))
    from kar3d.perception import Ovis3DPerceptionModule
    return Ovis3DPerceptionModule


def run_discover(args: argparse.Namespace) -> None:
    from PIL import Image
    from tqdm import tqdm

    data = read_dataset(args.data)
    subset = select_items(data, args.start_index, args.end_index, args.ids)
    completed = prepare_output(args.output, args.overwrite)
    Ovis = import_ovis(args.kar3d_root)
    model = Ovis(model_path=str(args.model_path))
    prompt_template = (
        "This is the object-finding stage of SpatialPIN, adapted to a multiple-choice spatial question.\n"
        "Analyze the 640 by 480 RGB image. Identify every distinct physical object explicitly mentioned "
        "or visually needed to answer the question. Describe each object using precise color, texture, "
        "identity, and 2D location words. Do not use a vague phrase such as multi-colored. Do not answer "
        "the question. Do not use dataset categories or hidden labels.\n"
        "Return only JSON in this form: "
        '{"objects":[{"name":"short object name","description":"detector-ready visual description"}]}\n'
    )
    with args.output.open("a", encoding="utf-8") as output:
        for item in tqdm(subset, desc="SpatialPIN object discovery"):
            item_id = str(item["id"])
            if item_id in completed:
                continue
            record = {
                "id": item_id,
                "image_filename": item["Image_Filename"],
                "adapter": ADAPTER_NAME,
                "uses_category_metadata": False,
                "uses_gold_answer": False,
            }
            started = time.perf_counter()
            try:
                image_path = args.image_dir / item["Image_Filename"]
                image = Image.open(image_path).convert("RGB").resize(WORKING_SIZE, Image.Resampling.LANCZOS)
                prompt = (
                    prompt_template
                    + f"\nQuestion:\n{item['Question']}\nOptions:\n{format_options(item)}\nJSON:"
                )
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                raw = model.generate_text(messages, max_tokens=args.max_new_tokens, enable_thinking=False)
                try:
                    objects = clean_discovered_objects(first_json_object(raw), item)
                    parse_status = "ok"
                except Exception:
                    objects = fallback_entities(item)
                    parse_status = "heuristic_fallback"
                if args.max_objects > 0:
                    objects = objects[: args.max_objects]
                record.update({
                    "status": "ok",
                    "objects": objects,
                    "raw_response": raw,
                    "parse_status": parse_status,
                })
            except Exception as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            append_jsonl(output, record)


def load_spatialpin_models(args: argparse.Namespace):
    import cv2
    import torch
    from torchvision.transforms import Compose
    from Depth_Anything.depth_anything.dpt import DepthAnything
    from Depth_Anything.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
    from lang_segment_anything.lang_sam import LangSAM

    langsam = LangSAM()
    config = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "localhub": True,
    }
    depth_model = DepthAnything(config)
    depth_model.load_state_dict(torch.load(args.depth_checkpoint, map_location="cpu"))
    depth_model = depth_model.to(args.device).eval()
    transform = Compose([
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
    ])
    return langsam, depth_model, transform


def estimate_inverse_depth(image, model, transform, device: str):
    import numpy as np
    import torch
    import torch.nn.functional as F

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    tensor = torch.from_numpy(transform({"image": rgb})["image"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        depth = model(tensor)
        depth = F.interpolate(depth.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    values = depth.detach().cpu().numpy().astype(np.float32)
    low, high = float(values.min()), float(values.max())
    normalized = (values - low) / max(high - low, 1e-8)
    return normalized, {"raw_min": low, "raw_max": high, "semantic": "normalized relative inverse-depth; larger is nearer"}


def mask_geometry(mask, inverse_depth) -> dict[str, Any]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    height, width = mask.shape
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    values = inverse_depth[mask]
    return {
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "center_xy": [round(center_x, 3), round(center_y, 3)],
        "center_normalized": [round(center_x / width, 6), round(center_y / height, 6)],
        "mask_area_pixels": int(mask.sum()),
        "area_ratio": round(float(mask.mean()), 8),
        "inverse_depth_median": round(float(np.median(values)), 6),
        "inverse_depth_mean": round(float(np.mean(values)), 6),
    }


def relation_cards(objects: list[dict[str, Any]]) -> list[str]:
    cards = []
    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            ax, ay = first["center_normalized"]
            bx, by = second["center_normalized"]
            relations = []
            first_name = first["name"]
            second_name = second["name"]
            if abs(ax - bx) >= 0.04:
                relations.append(f"{first_name} is " + ("left of" if ax < bx else "right of") + f" {second_name}")
            if abs(ay - by) >= 0.04:
                relations.append(f"{first_name} is " + ("above" if ay < by else "below") + f" {second_name}")
            depth_gap = first["inverse_depth_median"] - second["inverse_depth_median"]
            if abs(depth_gap) >= 0.04:
                relations.append(f"{first_name} is " + ("in front of" if depth_gap > 0 else "behind") + f" {second_name}")
            area_ratio = first["area_ratio"] / max(second["area_ratio"], 1e-8)
            if area_ratio >= 1.2:
                relations.append(f"{first_name} appears larger than {second_name} in the image")
            elif area_ratio <= (1 / 1.2):
                relations.append(f"{first_name} appears smaller than {second_name} in the image")
            if relations:
                cards.append("; ".join(relations) + ".")
    return cards


def save_prepare_artifact(path: Path, image, objects: list[dict[str, Any]], inverse_depth) -> None:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for index, obj in enumerate(objects):
        box = tuple(obj["bbox_xyxy"])
        draw.rectangle(box, outline=(0, 255, 80), width=3)
        draw.text((box[0] + 2, box[1] + 2), f"{index}: {obj['name']}", fill=(255, 40, 40))
    depth_u8 = np.clip(inverse_depth * 255, 0, 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    panel = Image.new("RGB", (WORKING_SIZE[0] * 3, WORKING_SIZE[1]), "white")
    panel.paste(image, (0, 0))
    panel.paste(overlay, (WORKING_SIZE[0], 0))
    panel.paste(Image.fromarray(depth_rgb), (WORKING_SIZE[0] * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, quality=90)


def run_prepare(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = read_dataset(args.data)
    subset = select_items(data, args.start_index, args.end_index, args.ids)
    discoveries = read_jsonl_map(args.discoveries)
    completed = prepare_output(args.output, args.overwrite)
    if not args.depth_checkpoint.is_file():
        raise FileNotFoundError(args.depth_checkpoint)
    langsam, depth_model, depth_transform = load_spatialpin_models(args)

    with args.output.open("a", encoding="utf-8") as output:
        for item in tqdm(subset, desc="SpatialPIN perception"):
            item_id = str(item["id"])
            if item_id in completed:
                continue
            started = time.perf_counter()
            record = {
                "id": item_id,
                "image_filename": item["Image_Filename"],
                "adapter": ADAPTER_NAME,
                "official_spatialpin_settings": {
                    "working_size": list(WORKING_SIZE),
                    "grounding_dino": "SwinB",
                    "sam": "ViT-H",
                    "depth_anything": "ViT-L/14",
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                },
                "uses_category_metadata": False,
                "uses_gold_answer": False,
            }
            try:
                discovery = discoveries.get(item_id)
                if not discovery or discovery.get("status") != "ok":
                    raise ValueError(f"Missing valid discovery record for id={item_id}")
                image_path = args.image_dir / item["Image_Filename"]
                image = Image.open(image_path).convert("RGB").resize(WORKING_SIZE, Image.Resampling.LANCZOS)
                inverse_depth, depth_info = estimate_inverse_depth(image, depth_model, depth_transform, args.device)
                objects = []
                for candidate in discovery.get("objects", []):
                    description = normalize_text(candidate.get("description") or candidate.get("name"))
                    if not description:
                        continue
                    masks, boxes, phrases, logits = langsam.predict(
                        image,
                        description,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                    )
                    fallback_used = False
                    if int(masks.shape[0]) == 0:
                        fallback_used = True
                        head = description.split(" of ", 1)[0]
                        masks, boxes, phrases, logits = langsam.predict(
                            image,
                            "object most similar to " + head,
                            box_threshold=args.box_threshold,
                            text_threshold=args.text_threshold,
                        )
                    if int(masks.shape[0]) == 0:
                        continue
                    scores = torch.as_tensor(logits).detach().cpu().reshape(-1)
                    selected = int(torch.argmax(scores).item())
                    mask = torch.as_tensor(masks[selected]).detach().cpu().numpy().astype(bool)
                    geometry = mask_geometry(mask, inverse_depth)
                    objects.append({
                        "name": normalize_text(candidate.get("name")) or description,
                        "description": description,
                        "detector_phrase": str(phrases[selected]),
                        "detector_score": round(float(scores[selected]), 6),
                        "fallback_query_used": fallback_used,
                        **geometry,
                    })
                relations = relation_cards(objects)
                record.update({
                    "status": "ok",
                    "objects": objects,
                    "relations": relations,
                    "depth": depth_info,
                    "warnings": [] if objects else ["No object mask survived official thresholds; final VLM still receives the image."],
                })
                if args.artifact_dir:
                    save_prepare_artifact(args.artifact_dir / f"{item_id}.jpg", image, objects, inverse_depth)
            except Exception as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            append_jsonl(output, record)


def build_evidence_panel(image, objects: list[dict[str, Any]]):
    from PIL import Image, ImageDraw

    source = image.resize(WORKING_SIZE, Image.Resampling.LANCZOS)
    overlay = source.copy()
    draw = ImageDraw.Draw(overlay)
    for index, obj in enumerate(objects):
        box = tuple(float(value) for value in obj["bbox_xyxy"])
        draw.rectangle(box, outline=(0, 255, 80), width=3)
        draw.text((box[0] + 2, box[1] + 2), f"{index}: {obj['name']}", fill=(255, 40, 40))
    panel = Image.new("RGB", (WORKING_SIZE[0] * 2, WORKING_SIZE[1]), "white")
    panel.paste(source, (0, 0))
    panel.paste(overlay, (WORKING_SIZE[0], 0))
    return panel


def spatial_evidence_text(prepared: dict[str, Any]) -> str:
    lines = [
        "SpatialPIN perception evidence (relative, not metric):",
        "- The left panel is the resized source image; the right panel shows detected object indices.",
        "- Image coordinates use origin at top-left. Relative inverse-depth is larger for nearer pixels.",
    ]
    for index, obj in enumerate(prepared.get("objects", [])):
        lines.append(
            f"- Object {index}: {obj['name']}; center_norm={obj['center_normalized']}; "
            f"bbox={obj['bbox_xyxy']}; area_ratio={obj['area_ratio']}; "
            f"relative_inverse_depth={obj['inverse_depth_median']}."
        )
    relations = prepared.get("relations", [])
    if relations:
        lines.append("Derived pairwise relations (verify against the image):")
        lines.extend(f"- {value}" for value in relations)
    if not prepared.get("objects"):
        lines.append("- No reliable object mask was found; rely on the image.")
    return "\n".join(lines)


def parse_mcq_answer(text: str, options: list[str]) -> tuple[Optional[str], str]:
    patterns = [
        (r"<answer>\s*([1-4])\s*</answer>", "answer_tag"),
        (r"^\s*\(?([1-4])\)?[\s.,;:!]*$", "bare_number"),
        (r"(?:final\s+)?answer\s*(?:is|:)?\s*(?:option\s*)?([1-4])\b", "answer_number"),
        (r"\boption\s*([1-4])\b", "option_number"),
        (r"(?:final\s+)?answer\s*(?:is|:)?\s*\(?([A-D])\)?\b", "answer_letter"),
    ]
    for pattern, method in patterns:
        matches = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if matches:
            value = matches[-1].upper()
            return (str(ord(value) - ord("A") + 1) if value in "ABCD" else value), method
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    matches = []
    for index, option in enumerate(options, start=1):
        option_norm = re.sub(r"[^a-z0-9]+", " ", option.lower()).strip()
        if option_norm and option_norm in normalized:
            matches.append(str(index))
    if len(set(matches)) == 1:
        return matches[0], "option_text"
    return None, "unparsed"


def summarize_predictions(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    status = Counter(row.get("status", "unknown") for row in rows)
    evaluated = [row for row in rows if row.get("status") == "ok"]
    correct = sum(bool(row.get("is_correct")) for row in evaluated)
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in evaluated:
        key = str(row.get("category", "Unknown"))
        by_category[key][0] += int(bool(row.get("is_correct")))
        by_category[key][1] += 1
    return {
        "adapter": ADAPTER_NAME,
        "official_spatialpin_score": False,
        "evaluated": len(evaluated),
        "correct": correct,
        "accuracy": round(100.0 * correct / len(evaluated), 4) if evaluated else 0.0,
        "status_counts": dict(status),
        "by_category": {
            key: {"correct": value[0], "total": value[1], "accuracy": round(100.0 * value[0] / value[1], 4)}
            for key, value in sorted(by_category.items())
        },
    }


def run_evaluate(args: argparse.Namespace) -> None:
    from PIL import Image
    from tqdm import tqdm

    data = read_dataset(args.data)
    subset = select_items(data, args.start_index, args.end_index, args.ids)
    prepared_map = read_jsonl_map(args.prepared)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    completed = prepare_output(predictions, args.overwrite)
    Ovis = import_ovis(args.kar3d_root)
    model = Ovis(model_path=str(args.model_path))

    with predictions.open("a", encoding="utf-8") as output:
        for item in tqdm(subset, desc="SpatialPIN adaptation on SpatialLab"):
            item_id = str(item["id"])
            if item_id in completed:
                continue
            options = [str(item.get(f"Option_{i}", "")) for i in range(1, 5)]
            record = {
                "id": item_id,
                "image_filename": item["Image_Filename"],
                "category": item.get("Category", ""),
                "sub_category": item.get("Sub_Category", ""),
                "answer": str(item.get("Answer", "")),
                "adapter": ADAPTER_NAME,
                "official_spatialpin_score": False,
                "matched_backbone": str(args.model_path),
                "uses_category_metadata_for_inference": False,
                "uses_gold_answer_for_inference": False,
            }
            started = time.perf_counter()
            try:
                prepared = prepared_map.get(item_id)
                if not prepared or prepared.get("status") != "ok":
                    raise ValueError(f"Missing valid prepared record for id={item_id}")
                image = Image.open(args.image_dir / item["Image_Filename"]).convert("RGB")
                panel = build_evidence_panel(image, prepared.get("objects", []))
                evidence = spatial_evidence_text(prepared)
                prompt = (
                    "Answer this SpatialLab multiple-choice question using the image and SpatialPIN perception evidence. "
                    "Treat derived geometry as noisy auxiliary evidence and verify it against the visual content. "
                    "Do not use dataset category metadata. Output a <think> block, then exactly one choice as "
                    "<answer>N</answer>, where N is 1, 2, 3, or 4.\n\n"
                    + evidence
                    + f"\n\nQuestion:\n{item['Question']}\nOptions:\n{format_options(item)}\nFinal answer:"
                )
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": panel},
                    {"type": "text", "text": prompt},
                ]}]
                raw = model.generate_text(messages, max_tokens=args.max_new_tokens, enable_thinking=True)
                prediction, parse_method = parse_mcq_answer(raw, options)
                record.update({
                    "status": "ok",
                    "prediction": prediction,
                    "parse_method": parse_method,
                    "is_correct": prediction == record["answer"],
                    "raw_response": raw,
                    "objects": prepared.get("objects", []),
                    "relations": prepared.get("relations", []),
                })
                if args.save_artifacts:
                    artifact = args.output_dir / "artifacts" / f"{item_id}.jpg"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    panel.save(artifact, quality=90)
            except Exception as exc:
                record.update({"status": "error", "prediction": "Error", "is_correct": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            append_jsonl(output, record)

    summary = summarize_predictions(predictions)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    def common(stage):
        stage.add_argument("--data", required=True, type=Path)
        stage.add_argument("--image-dir", required=True, type=Path)
        stage.add_argument("--start-index", type=int, default=0)
        stage.add_argument("--end-index", type=int)
        stage.add_argument("--ids", default="")
        stage.add_argument("--overwrite", action="store_true")

    discover = subparsers.add_parser("discover", help="Ovis replacement for SpatialPIN's GPT-4V object-finding stage")
    common(discover)
    discover.add_argument("--output", required=True, type=Path)
    discover.add_argument("--model-path", required=True, type=Path)
    discover.add_argument("--kar3d-root", required=True, type=Path)
    discover.add_argument("--max-new-tokens", type=int, default=1024)
    discover.add_argument("--max-objects", type=int, default=0, help="0 keeps all discovered objects")
    discover.set_defaults(func=run_discover)

    prepare = subparsers.add_parser("prepare", help="Run official SpatialPIN perception models and build spatial evidence")
    common(prepare)
    prepare.add_argument("--discoveries", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--depth-checkpoint", required=True, type=Path)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--box-threshold", type=float, default=OFFICIAL_BOX_THRESHOLD)
    prepare.add_argument("--text-threshold", type=float, default=OFFICIAL_TEXT_THRESHOLD)
    prepare.add_argument("--artifact-dir", type=Path)
    prepare.set_defaults(func=run_prepare)

    evaluate = subparsers.add_parser("evaluate", help="Run matched-backbone MCQ reasoning and score predictions")
    common(evaluate)
    evaluate.add_argument("--prepared", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--model-path", required=True, type=Path)
    evaluate.add_argument("--kar3d-root", required=True, type=Path)
    evaluate.add_argument("--max-new-tokens", type=int, default=2048)
    evaluate.add_argument("--save-artifacts", action="store_true")
    evaluate.set_defaults(func=run_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
