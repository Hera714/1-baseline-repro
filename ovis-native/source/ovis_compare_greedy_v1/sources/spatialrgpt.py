#!/usr/bin/env python3
"""Run the official SpatialRGPT checkpoint on SpatialLab multiple-choice QA."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--depth-anything-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--conv-mode", default="llama_3")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["id"])] = record
    return records


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def pad_to_square(array: np.ndarray) -> np.ndarray:
    height, width = array.shape
    side = max(height, width)
    padded = np.zeros((side, side), dtype=array.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    padded[top : top + height, left : left + width] = array
    return padded


def make_region_masks(
    regions: list[dict[str, Any]],
    width: int,
    height: int,
    image_processor: Any,
    model_config: Any,
) -> torch.Tensor:
    mask_processor = copy.deepcopy(image_processor)
    mask_processor.do_normalize = False
    mask_processor.do_convert_rgb = False
    mask_processor.rescale_factor = 1.0
    processed: list[torch.Tensor] = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        x1 = max(0, min(width, int(round(x1))))
        x2 = max(0, min(width, int(round(x2))))
        y1 = max(0, min(height, int(round(y1))))
        y2 = max(0, min(height, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        if getattr(model_config, "image_aspect_ratio", None) == "pad":
            mask = pad_to_square(mask)
        tensor = mask_processor.preprocess(mask[None, ...], return_tensors="pt")["pixel_values"][0]
        processed.append(tensor)
    return torch.vstack(processed).float()


def build_prompt(item: dict[str, Any], regions: list[dict[str, Any]]) -> str:
    region_lines = []
    for index, region in enumerate(regions):
        label = re.sub(r"\s+", " ", str(region.get("label", "object"))).strip()
        region_lines.append(f"Region [{index}]: <mask> <depth> (automatic detector phrase: {label})")
    options = "\n".join(f"{i}. {item.get(f'Option_{i}', '')}" for i in range(1, 5))
    return (
        "<image>\n"
        "The following regions were generated automatically from the question and image; detector phrases may be noisy.\n"
        + "\n".join(region_lines)
        + "\nUse the original image, regional appearance, and regional depth evidence to answer the multiple-choice question. "
        "Choose exactly one option. End your response with <answer>N</answer>, where N is 1, 2, 3, or 4.\n"
        f"Question: {item['Question']}\nOptions:\n{options}"
    )


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def parse_answer(text, options):
    from protocol import parse_options
    return parse_options(text, options)


def write_summary(predictions_path: Path, summary_path: Path) -> dict[str, Any]:
    total = correct = 0
    status_counts: Counter[str] = Counter()
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            status_counts[record.get("status", "unknown")] += 1
            if record.get("status") != "ok":
                continue
            total += 1
            is_correct = int(bool(record.get("correct")))
            correct += is_correct
            category = record.get("category", "Unknown")
            by_category[category][0] += is_correct
            by_category[category][1] += 1
    summary = {
        "evaluated": total,
        "correct": correct,
        "accuracy": round(100.0 * correct / total, 4) if total else 0.0,
        "status_counts": dict(status_counts),
        "by_category": {
            key: {"correct": values[0], "total": values[1], "accuracy": round(100.0 * values[0] / values[1], 4)}
            for key, values in sorted(by_category.items())
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def save_artifacts(
    output_dir: Path,
    item_id: str,
    image: Image.Image,
    regions: list[dict[str, Any]],
    depth_image: np.ndarray,
) -> None:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for index, region in enumerate(regions):
        box = tuple(region["bbox"])
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), f"Region {index}: {region.get('label', '')}", fill="red")
    overlay.save(artifact_dir / f"{item_id}_regions.jpg", quality=90)
    Image.fromarray(depth_image).save(artifact_dir / f"{item_id}_depth.png")


def main() -> None:
    args = parse_args()
    os.environ["DEPTH_ANYTHING_PATH"] = str(args.depth_anything_path)

    # Imports below use the configured Depth Anything path and the patched
    # SpatialRGPT/VILA installation.
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.eval.eval_spatial import get_depth_map, get_depth_predictor
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite:
        predictions_path.unlink(missing_ok=True)
    completed = load_completed(predictions_path)
    region_map = load_jsonl_map(args.regions)
    with args.data.open("r", encoding="utf-8") as handle:
        data = json.load(handle)[args.start_index : args.end_index]

    disable_torch_init()
    depth_model, depth_transform = get_depth_predictor()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        str(args.model_path), args.model_path.name, None
    )
    model.to(dtype=torch.bfloat16).eval()

    with predictions_path.open("a", encoding="utf-8") as output_handle:
        for item in tqdm(data, desc="SpatialRGPT on SpatialLab"):
            item_id = str(item["id"])
            if item_id in completed:
                continue
            started = time.perf_counter()
            options = [str(item.get(f"Option_{i}", "")) for i in range(1, 5)]
            record: dict[str, Any] = {
                "id": item_id,
                "image": item["Image_Filename"],
                "category": item.get("Category"),
                "sub_category": item.get("Sub_Category"),
                "question": item["Question"],
                "options": options,
                "gt_answer": str(item.get("Answer")),
                "model": args.model_path.name,
            }
            try:
                region_record = region_map.get(item_id)
                if not region_record or region_record.get("status") != "ok":
                    raise ValueError(f"missing valid automatic regions for id={item_id}")
                regions = region_record["regions"]
                image = Image.open(args.image_dir / item["Image_Filename"]).convert("RGB")
                raw_image = np.asarray(image)
                depth_image = get_depth_map(raw_image, depth_model, depth_transform)
                masks = make_region_masks(regions, image.width, image.height, image_processor, model.config)
                image_tensor = process_images([image], image_processor, model.config).to(
                    model.device, dtype=torch.bfloat16
                )
                depth_tensor = process_images([Image.fromarray(depth_image)], image_processor, model.config).to(
                    model.device, dtype=torch.bfloat16
                )
                prompt_text = build_prompt(item, regions)
                conversation = conv_templates[args.conv_mode].copy()
                conversation.append_message(conversation.roles[0], prompt_text)
                conversation.append_message(conversation.roles[1], None)
                prompt = conversation.get_prompt()
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to("cuda", non_blocking=True)
                stop_string = (
                    conv_templates[args.conv_mode].sep
                    if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
                    else conv_templates[args.conv_mode].sep2
                )
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        depths=depth_tensor,
                        masks=[masks.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)],
                        do_sample=False,
                        temperature=0.0,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if stop_string and raw_output.endswith(stop_string):
                    raw_output = raw_output[: -len(stop_string)].strip()
                pred_answer, parse_method = parse_answer(raw_output, options)
                record.update(
                    {
                        "status": "ok",
                        "pred_answer": pred_answer,
                        "parse_method": parse_method,
                        "correct": pred_answer == str(item.get("Answer")),
                        "raw_output": raw_output,
                        "prompt": prompt_text,
                        "regions": regions,
                        "region_fallback": bool(region_record.get("full_image_fallback")),
                        "context_len": context_len,
                    }
                )
                if args.save_artifacts:
                    save_artifacts(args.output_dir, item_id, image, regions, depth_image)
            except Exception as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()

    summary = write_summary(predictions_path, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

