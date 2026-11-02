#!/usr/bin/env python3
"""Generate automatic SpatialRGPT region proposals for SpatialLab.

This stage deliberately uses only the image, question, and answer options. It
does not read category metadata, ground-truth answers, or human annotations.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


NON_VISUAL_OPTIONS = {
    "yes",
    "no",
    "true",
    "false",
    "maybe",
    "mostly",
    "north",
    "south",
    "east",
    "west",
    "left",
    "right",
    "front",
    "back",
    "backward",
    "forward",
    "horizontal",
    "vertical",
    "diagonal",
    "not given",
    "not applicable",
    "nothing",
    "none of them",
    "none of these",
    "can't answer",
    "cannot determine",
    "not measurable",
    "possible",
    "not possible",
    "unable to determine",
}

# These words describe relations, measurements, answers, or question mechanics
# rather than entities that an open-vocabulary detector should localize.  The
# list is deliberately task-agnostic: it is applied only to question/option
# text and never uses SpatialLab category or answer metadata.
BREAK_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "there", "it", "its",
    "they", "them", "their", "he", "she", "his", "her", "we", "you", "your",
    "i", "someone", "something", "anything", "which", "what", "where", "when",
    "who", "whose", "whom", "how", "if", "then", "than", "as", "and", "or",
    "but", "with", "without", "from", "into", "onto", "upon", "through", "across",
    "between", "among", "around", "behind", "beside", "inside", "outside", "above",
    "below", "under", "over", "near", "toward", "towards", "of", "to", "for",
    "in", "on", "at", "by", "up", "down", "out", "off", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should", "may", "might", "must", "all",
    "both", "each", "every", "any", "some", "most", "many", "much", "more",
    "less", "few", "only", "same", "similar", "different", "other", "another",
    "first", "second", "third", "last", "middle", "closest", "nearest", "farthest",
    "frontmost", "foremost", "top", "bottom", "upper", "lower", "left", "right",
    "total", "approximately", "about", "fully", "directly", "together", "also",
    "draw", "drawing", "connect", "connecting", "compare", "comparing", "go", "going",
    "move", "moving", "pass", "passing", "stand", "stands", "standing", "sit", "sits",
    "sitting", "lay", "lays", "lying", "look", "looks", "looking", "appear", "appears",
    "appearing", "happen", "happens", "hitting", "hit", "facing", "face", "flying",
    "hold", "holds", "holding", "placed", "stacked", "visible", "shown", "given",
    "needed", "need", "take", "taking", "melt", "melted", "tilted", "reflected",
    "reflecting", "casting", "cast", "compare", "conclusion", "answer", "option",
    "object", "objects", "thing", "things", "picture", "image", "scene", "view",
    "viewpoint", "background", "foreground", "line", "lines", "point", "points",
    "angle", "angles", "direction", "directions", "position", "positions", "location",
    "relationship", "relationships", "arrangement", "pattern", "patterns", "size",
    "sizes", "scale", "distance", "height", "volume", "shape", "row", "rows", "side",
    "sides", "corner", "order", "turn", "turns", "way", "part", "parts", "pair",
}

ATTRIBUTE_ONLY_WORDS = {
    "black", "blue", "brown", "gray", "green", "grey", "orange", "pink", "purple",
    "red", "white", "yellow", "large", "larger", "largest", "small", "smaller",
    "smallest", "big", "bigger", "biggest", "long", "longer", "longest", "short",
    "shorter", "shortest", "round", "square", "rectangular", "horizontal", "vertical",
    "diagonal", "upward", "downward", "forward", "backward", "north", "south", "east",
    "west", "front", "back", "true", "false", "yes", "no", "maybe",
}

OPTION_REJECTION_WORDS = {
    "only", "same", "similar", "different", "larger", "largest", "smaller", "smallest",
    "more", "less", "none", "nothing", "not", "unable", "cannot", "possible",
    "impossible", "would", "will", "there", "they", "their", "about", "together",
}

IRREGULAR_ENTITY_ALIASES = {
    "people": "person",
    "persons": "person",
    "humans": "person",
    "human": "person",
    "characters": "person",
    "character": "person",
    "audience": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "cattle": "cow",
}

PLURAL_EXCEPTIONS = {
    "bus", "class", "dress", "glass", "glasses", "grass", "pants", "pliers",
    "scissors", "shorts", "stairs", "sunglasses",
}

MULTI_ENTITY_CUES = re.compile(
    r"\b(how many|all|both|each|every|between|among|compare|relative|same|similar|"
    r"total|tallest points|shortest points|more than one)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visualize-dir", type=Path)
    parser.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face Grounding DINO checkpoint.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--retry-box-threshold", type=float, default=0.18)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--nms-threshold", type=float, default=0.55)
    parser.add_argument("--global-nms-threshold", type=float, default=0.92)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    parser.add_argument("--max-per-label", type=int, default=5)
    parser.add_argument("--max-regions", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tokenize_words(value: Any) -> list[str]:
    text = normalize_text(value).lower().replace("’", "'")
    tokens = re.findall(r"[a-z][a-z'-]*", text)
    return [token[:-2] if token.endswith("'s") else token for token in tokens]


def singularize_word(word: str) -> str:
    if word in IRREGULAR_ENTITY_ALIASES:
        return IRREGULAR_ENTITY_ALIASES[word]
    if word in PLURAL_EXCEPTIONS or len(word) <= 3:
        return word
    if re.search(r"(?:ches|shes|sses|xes|zes)$", word):
        return word[:-2]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def canonicalize_phrase(words: list[str]) -> str:
    if not words:
        return ""
    canonical = list(words)
    canonical[-1] = singularize_word(canonical[-1])
    phrase = " ".join(canonical).strip()
    return IRREGULAR_ENTITY_ALIASES.get(phrase, phrase)


def extract_entity_phrases(text: Any, *, option: bool = False) -> list[str]:
    """Extract short, detector-friendly visible-entity phrases using rules only."""
    normalized = normalize_text(text).lower().strip(" .?!,:;\"'")
    if not normalized or normalized in NON_VISUAL_OPTIONS:
        return []
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:°)?", normalized):
        return []

    words = tokenize_words(normalized)
    if not words:
        return []
    if option and (len(words) > 6 or OPTION_REJECTION_WORDS.intersection(words)):
        return []

    runs: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            runs.append(current)
            current = []

    for word in words:
        if word in BREAK_WORDS or word.isdigit():
            flush()
            continue
        current.append(word)
        if len(current) == 4:
            flush()
    flush()

    phrases: list[str] = []
    for run in runs:
        if all(word in ATTRIBUTE_ONLY_WORDS for word in run):
            continue
        phrase = canonicalize_phrase(run)
        if phrase and phrase not in ATTRIBUTE_ONLY_WORDS:
            phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def build_grounding_terms(item: dict[str, Any]) -> dict[str, Any]:
    """Build entity-only detector terms without metadata or gold answers."""
    question = normalize_text(item.get("Question"))
    options = [normalize_text(item.get(f"Option_{i}")) for i in range(1, 5)]
    question_terms = extract_entity_phrases(question)
    option_terms = [
        phrase
        for option in options
        for phrase in extract_entity_phrases(option, option=True)
    ]
    terms = list(dict.fromkeys(question_terms + option_terms))
    used_generic_fallback = False
    if not terms:
        used_generic_fallback = True
        terms = ["person", "animal", "vehicle", "furniture"]
    query = ". ".join(terms)[:900].rstrip(" .") + "."
    question_words = tokenize_words(question)
    requires_multiple = bool(MULTI_ENTITY_CUES.search(question)) or any(
        word.endswith("s") and word not in PLURAL_EXCEPTIONS and word not in BREAK_WORDS
        for word in question_words
    )
    return {
        "question_terms": question_terms,
        "option_terms": option_terms,
        "terms": terms,
        "query": query,
        "requires_multiple": requires_multiple,
        "generic_query_fallback": used_generic_fallback,
    }


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                completed.add(str(record.get("id")))
    return completed


def label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def run_detection(
    model: Any,
    processor: Any,
    image: Image.Image,
    query: str,
    device: torch.device,
    dtype: torch.dtype,
    box_threshold: float,
    text_threshold: float,
    pass_name: str,
) -> list[dict[str, Any]]:
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            inputs[key] = value.to(dtype=dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    labels = [str(label) for label in results.get("text_labels", results.get("labels", []))]
    candidates: list[dict[str, Any]] = []
    for box, score, label in zip(results["boxes"].cpu(), results["scores"].cpu(), labels):
        candidates.append(
            {
                "bbox": [float(value) for value in box.tolist()],
                "score": float(score),
                "label": normalize_text(label),
                "source_pass": pass_name,
            }
        )
    return candidates


def select_regions(
    raw_candidates: list[dict[str, Any]],
    width: int,
    height: int,
    nms_threshold: float,
    global_nms_threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_per_label: int,
    max_regions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_area = float(width * height)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        x1, y1, x2, y2 = candidate["bbox"]
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(image_area, 1.0)
        cleaned = {
            "bbox": [x1, y1, x2, y2],
            "score": float(candidate["score"]),
            "label": normalize_text(candidate["label"]),
            "source_pass": candidate["source_pass"],
            "area_ratio": area_ratio,
        }
        if not cleaned["label"]:
            rejected.append({**cleaned, "reason": "empty_label"})
            continue
        if area_ratio < min_area_ratio:
            rejected.append({**cleaned, "reason": "area_too_small"})
            continue
        if area_ratio > max_area_ratio:
            rejected.append({**cleaned, "reason": "area_too_large"})
            continue
        candidates.append(cleaned)

    # First suppress duplicates within each grounded phrase and cap any one
    # phrase so repeated detections cannot crowd out all other entities.
    retained: list[dict[str, Any]] = []
    grouped: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(label_key(candidate["label"]), []).append(index)
    for indices in grouped.values():
        group_boxes = torch.tensor([candidates[i]["bbox"] for i in indices], dtype=torch.float32)
        group_scores = torch.tensor([candidates[i]["score"] for i in indices], dtype=torch.float32)
        kept_local = nms(group_boxes, group_scores, nms_threshold).tolist()
        for local_index in kept_local[:max_per_label]:
            retained.append(candidates[indices[local_index]])
        for local_index in kept_local[max_per_label:]:
            rejected.append({**candidates[indices[local_index]], "reason": "per_label_limit"})

    # Remove only near-identical cross-label boxes.  A high threshold preserves
    # useful overlaps such as person/shirt while removing duplicate aliases.
    retained.sort(key=lambda entry: entry["score"], reverse=True)
    if retained:
        all_boxes = torch.tensor([entry["bbox"] for entry in retained], dtype=torch.float32)
        all_scores = torch.tensor([entry["score"] for entry in retained], dtype=torch.float32)
        global_kept = set(nms(all_boxes, all_scores, global_nms_threshold).tolist())
        for index, entry in enumerate(retained):
            if index not in global_kept:
                rejected.append({**entry, "reason": "near_duplicate"})
        retained = [entry for index, entry in enumerate(retained) if index in global_kept]

    # Reserve the strongest box for every label before filling remaining slots
    # by score, improving entity coverage for relation/comparison questions.
    best_per_label: dict[str, dict[str, Any]] = {}
    for entry in retained:
        best_per_label.setdefault(label_key(entry["label"]), entry)
    diverse = sorted(best_per_label.values(), key=lambda entry: entry["score"], reverse=True)
    selected_ids = {id(entry) for entry in diverse[:max_regions]}
    selected = diverse[:max_regions]
    for entry in retained:
        if len(selected) >= max_regions:
            break
        if id(entry) not in selected_ids:
            selected.append(entry)
            selected_ids.add(id(entry))
    selected.sort(key=lambda entry: entry["score"], reverse=True)

    regions = []
    for index, entry in enumerate(selected):
        regions.append(
            {
                "index": index,
                "bbox": [round(value, 2) for value in entry["bbox"]],
                "score": round(entry["score"], 5),
                "label": entry["label"],
                "source_pass": entry["source_pass"],
                "area_ratio": round(entry["area_ratio"], 6),
            }
        )
    compact_rejected = [
        {
            "bbox": [round(value, 2) for value in entry["bbox"]],
            "score": round(entry["score"], 5),
            "label": entry["label"],
            "source_pass": entry["source_pass"],
            "area_ratio": round(entry["area_ratio"], 6),
            "reason": entry["reason"],
        }
        for entry in rejected[:50]
    ]
    return regions, compact_rejected


def draw_regions(image: Image.Image, regions: list[dict[str, Any]], path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for region in regions:
        box = tuple(region["bbox"])
        caption = f"{region['index']}: {region['label']} {region['score']:.2f}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), caption, fill="red", stroke_width=2, stroke_fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else load_completed(args.output)
    mode = "w" if args.overwrite else "a"

    with args.data.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data = data[args.start_index : args.end_index]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Keep Grounding DINO in fp32.  The Transformers implementation contains
    # text/position-projection paths that remain fp32, so converting the whole
    # checkpoint to fp16 causes dtype mismatches inside linear layers.
    dtype = torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    with args.output.open(mode, encoding="utf-8") as output_handle:
        for item in tqdm(data, desc="Grounding SpatialLab regions"):
            item_id = str(item["id"])
            if item_id in completed:
                continue
            started = time.perf_counter()
            image_path = args.image_dir / item["Image_Filename"]
            record: dict[str, Any] = {
                "id": item_id,
                "image": item["Image_Filename"],
                "detector": args.model,
                "query_builder": "entity_terms_v2",
                "box_threshold": args.box_threshold,
                "retry_box_threshold": args.retry_box_threshold,
                "text_threshold": args.text_threshold,
                "min_area_ratio": args.min_area_ratio,
                "max_area_ratio": args.max_area_ratio,
            }
            try:
                image = Image.open(image_path).convert("RGB")
                query_info = build_grounding_terms(item)
                primary_candidates = run_detection(
                    model,
                    processor,
                    image,
                    query_info["query"],
                    device,
                    dtype,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    pass_name="primary",
                )
                primary_regions, _ = select_regions(
                    primary_candidates,
                    image.width,
                    image.height,
                    args.nms_threshold,
                    args.global_nms_threshold,
                    args.min_area_ratio,
                    args.max_area_ratio,
                    args.max_per_label,
                    args.max_regions,
                )
                minimum_regions = 2 if query_info["requires_multiple"] else 1
                retry_needed = len(primary_regions) < minimum_regions
                retry_candidates: list[dict[str, Any]] = []
                retry_attempted = retry_needed and args.retry_box_threshold < args.box_threshold
                if retry_attempted:
                    retry_candidates = run_detection(
                        model,
                        processor,
                        image,
                        query_info["query"],
                        device,
                        dtype,
                        box_threshold=args.retry_box_threshold,
                        text_threshold=args.text_threshold,
                        pass_name="low_threshold_retry",
                    )
                regions, rejected_regions = select_regions(
                    primary_candidates + retry_candidates,
                    image.width,
                    image.height,
                    args.nms_threshold,
                    args.global_nms_threshold,
                    args.min_area_ratio,
                    args.max_area_ratio,
                    args.max_per_label,
                    args.max_regions,
                )
                fallback = False
                if not regions:
                    fallback = True
                    regions = [
                        {
                            "index": 0,
                            "bbox": [0.0, 0.0, float(image.width), float(image.height)],
                            "score": 0.0,
                            "label": "entire scene",
                            "source_pass": "full_image_fallback",
                            "area_ratio": 1.0,
                        }
                    ]
                record.update(
                    {
                        "status": "ok",
                        "question_terms": query_info["question_terms"],
                        "option_terms": query_info["option_terms"],
                        "grounding_terms": query_info["terms"],
                        "grounding_query": query_info["query"],
                        "requires_multiple_regions": query_info["requires_multiple"],
                        "generic_query_fallback": query_info["generic_query_fallback"],
                        "retry_triggered": retry_attempted,
                        "detection_passes": [
                            {
                                "name": "primary",
                                "box_threshold": args.box_threshold,
                                "raw_detections": len(primary_candidates),
                            },
                            *(
                                [
                                    {
                                        "name": "low_threshold_retry",
                                        "box_threshold": args.retry_box_threshold,
                                        "raw_detections": len(retry_candidates),
                                    }
                                ]
                                if retry_attempted
                                else []
                            ),
                        ],
                        "image_size": [image.width, image.height],
                        "regions": regions,
                        "rejected_regions": rejected_regions,
                        "full_image_fallback": fallback,
                    }
                )
                if args.visualize_dir:
                    draw_regions(image, regions, args.visualize_dir / item["Image_Filename"])
            except Exception as exc:  # keep long jobs resumable and auditable
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()


if __name__ == "__main__":
    main()
