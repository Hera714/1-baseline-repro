"""MJev-Spatial foreground inference with unified cost/accuracy records."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_jev.kev import KevBuilder, build_model, make_processor, predict_batch


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else value.get("items", value.get("data", []))


def normalize(row: dict, request_dir: Path, index: int) -> dict:
    value = json.loads(json.dumps(row.get("input", row), ensure_ascii=False))
    for media in value.get("media", []):
        uri = Path(media["uri"])
        if not uri.is_absolute():
            media["uri"] = str((request_dir / uri).resolve())
    return {"id": str(row.get("id", row.get("sample_id", f"request-{index:06d}"))), "input": value}


def load_truth(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = load_rows(path)
    truth = {}
    for row in rows:
        truth[str(row.get("id"))] = {
            "answer": str(row.get("Answer", row.get("answer", ""))),
            "category": row.get("Category", "unknown"),
        }
    return truth


class NvidiaSmiSampler:
    def __init__(self):
        self.peak_bytes = 0
        self.available = False
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["stdbuf", "-oL", "nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            return
        self.available = True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        if self.proc is None or self.proc.stdout is None:
            return
        while not self.stop_event.is_set():
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                mib = float(line.strip().split(",")[0])
                self.peak_bytes = max(self.peak_bytes, int(mib * 1024 * 1024))
            except (TypeError, ValueError):
                continue

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=2)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def gpu_meta(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"gpu": None, "gpu_total_bytes": None}
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": int(props.total_memory),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("为保证逐题耗时、调用次数和进度一致，必须使用 --batch-size 1")
    for path in (args.output, args.cost_output, args.summary, args.meta):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            parser.error(f"文件已存在，若确认覆盖请加 --overwrite: {path}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请先确认 nvidia-smi 能看到 RTX 4090")
        torch.cuda.set_device(device)

    requests = [normalize(row, args.input.parent, i) for i, row in enumerate(load_rows(args.input))]
    if len({row["id"] for row in requests}) != len(requests):
        raise ValueError("输入请求 ID 不唯一")
    truth = load_truth(args.data)
    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    config.update(
        preserve_option_order=True,
        max_image_tokens=1024,
        max_sequence_length=max(8192, int(config.get("max_sequence_length", 4096))),
        inference_language_dtype="bf16",
        local_files_only=True,
    )

    nvml = NvidiaSmiSampler()
    nvml.start()
    process_start = time.perf_counter()
    load_events = []
    t0 = time.perf_counter()
    processor = make_processor(config)
    load_events.append({"name": "make_processor", "kind": "load", "status": "ok", "seconds": time.perf_counter() - t0})
    t0 = time.perf_counter()
    builder = KevBuilder(processor, config)
    load_events.append({"name": "build_builder", "kind": "load", "status": "ok", "seconds": time.perf_counter() - t0})
    t0 = time.perf_counter()
    model = build_model(config, args.checkpoint, trainable=False).to(device).eval()
    sync(device)
    load_events.append({"name": "build_model", "kind": "load", "status": "ok", "seconds": time.perf_counter() - t0})
    first_sample_start = time.perf_counter()

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    capture = {"stages": []}
    original_encode = builder.encode_specs

    def capture_encode(specs, training=True):
        packed = original_encode(specs, training=training)
        if not training:
            ids = packed["inputs"]["input_ids"]
            lengths = [int(v) for v in packed["lengths"].tolist()]
            image_placeholders = [int((ids[i, :lengths[i]] == image_pad_id).sum().item()) for i in range(len(lengths))]
            grid = packed["inputs"].get("image_grid_thw")
            merge = getattr(getattr(processor, "image_processor", None), "merge_size", 1)
            if isinstance(merge, (tuple, list)):
                merge = int(merge[0])
            vision_tokens = 0
            if grid is not None:
                vision_tokens = int(sum(int(t) * int(h) * int(w) // max(1, int(merge) ** 2)
                                        for t, h, w in grid.tolist()))
            capture["stages"].append({
                "input_sequence_tokens": int(sum(lengths)),
                "input_text_tokens": int(sum(lengths) - sum(image_placeholders)),
                "image_placeholder_tokens": int(sum(image_placeholders)),
                "vision_tokens": vision_tokens,
                "padded_input_tokens": int(packed.get("padded_tokens", 0)),
            })
        return packed

    builder.encode_specs = capture_encode
    predictions_f = args.output.open("w", encoding="utf-8")
    costs_f = args.cost_output.open("w", encoding="utf-8")
    durations = []
    statuses = {}
    total_model_calls = 0
    total_retries = 0
    global_peak_allocated = 0
    global_peak_reserved = 0
    total_input_text_tokens = 0
    total_image_placeholder_tokens = 0
    total_vision_tokens = 0
    correct = 0
    by_category = {}

    try:
        for index, request in enumerate(requests, 1):
            sample_start = time.perf_counter()
            status = "ok"
            error = None
            value = None
            capture["stages"] = []
            attempts = 0
            while True:
                attempts += 1
                try:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    capture["stages"] = []
                    sync(device)
                    values = predict_batch(model, builder, [request], device)
                    sync(device)
                    value = values[0]
                    break
                except Exception as exc:
                    error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    if attempts > args.retries:
                        status = "error"
                        break
                    print(f"MJev retry: {index}/{len(requests)} attempt={attempts + 1}", flush=True)

            elapsed = time.perf_counter() - sample_start
            durations.append(elapsed)
            stages = list(capture["stages"])
            model_calls = len(stages) * attempts
            total_model_calls += model_calls
            total_retries += max(0, attempts - 1)
            input_text_tokens = sum(int(s["input_text_tokens"]) for s in stages)
            image_placeholders = sum(int(s["image_placeholder_tokens"]) for s in stages)
            vision_tokens = sum(int(s["vision_tokens"]) for s in stages)
            total_input_text_tokens += input_text_tokens
            total_image_placeholder_tokens += image_placeholders
            total_vision_tokens += vision_tokens
            prediction = value.get("prediction") if value else None
            target = truth.get(request["id"], {})
            target_answer = str(target.get("answer", "")) if target else ""
            is_correct = bool(status == "ok" and target_answer and str(prediction) == target_answer)
            if is_correct:
                correct += 1
            category = target.get("category", "unknown")
            bucket = by_category.setdefault(category, {"correct": 0, "total": 0})
            bucket["total"] += 1
            if is_correct:
                bucket["correct"] += 1
            statuses[status] = statuses.get(status, 0) + 1

            if device.type == "cuda":
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            else:
                peak_allocated = peak_reserved = 0
            global_peak_allocated = max(global_peak_allocated, peak_allocated)
            global_peak_reserved = max(global_peak_reserved, peak_reserved)
            prediction_row = {"id": request["id"]}
            if value:
                prediction_row.update(value)
            if error:
                prediction_row["error"] = error
            predictions_f.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")
            predictions_f.flush()
            cost_row = {
                "id": request["id"],
                "events": [{"name": "mjev_predict", "kind": "inference", "status": status, "seconds": elapsed}],
                "sample_seconds": elapsed,
                "model_calls": model_calls,
                "retry_count": max(0, attempts - 1),
                "input_text_tokens": input_text_tokens,
                "image_placeholder_tokens": image_placeholders,
                "vision_tokens": vision_tokens,
                "output_text_tokens": 0,
                "stage_token_stats": stages,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_nvml_bytes_so_far": nvml.peak_bytes or None,
                "method": "mjev_spatial",
                "stage": "predict",
                "status": status,
                "prediction": prediction,
                "target": target_answer or None,
                "correct": is_correct,
                "error": error,
            }
            costs_f.write(json.dumps(cost_row, ensure_ascii=False) + "\n")
            costs_f.flush()
            print(
                f"MJev-Spatial inference: {index}/{len(requests)} "
                f"status={status} id={request['id']} seconds={elapsed:.3f} "
                f"calls={model_calls}",
                flush=True,
            )
    finally:
        predictions_f.close()
        costs_f.close()
        nvml.stop()

    process_seconds = time.perf_counter() - process_start
    total = len(requests)
    accuracy = (100.0 * correct / total) if total else 0.0
    for bucket in by_category.values():
        bucket["accuracy"] = 100.0 * bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    gpu = gpu_meta(device)
    summary = {
        "method": "MJev-Spatial",
        "official_spatiallab_score": True,
        "evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "status_counts": statuses,
        "by_category": by_category,
        "wall": {
            "n": len(durations),
            "total": sum(durations),
            "mean": statistics.mean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p95": percentile(durations, 0.95),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "runtime_seconds": {
            "process_including_load": process_seconds,
            "load": sum(event["seconds"] for event in load_events),
            "until_first_sample": first_sample_start - process_start,
        },
        "calls_and_tokens": {
            "model_calls": total_model_calls,
            "retries": total_retries,
            "input_text_tokens": total_input_text_tokens,
            "image_placeholder_tokens": total_image_placeholder_tokens,
            "vision_tokens": total_vision_tokens,
            "output_text_tokens": 0,
        },
        "peak_memory": {
            "nvml_used_bytes": nvml.peak_bytes or None,
            "nvml_used_gib": (nvml.peak_bytes / 1024**3) if nvml.peak_bytes else None,
            "torch_peak_allocated_bytes": global_peak_allocated or None,
            "torch_peak_reserved_bytes": global_peak_reserved or None,
        },
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {
        "method": "mjev_spatial",
        "stage": "predict",
        "batch_size": 1,
        "host": socket.gethostname(),
        **gpu,
        "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch_num_threads": torch.get_num_threads(),
        "synchronized_component_profiling": True,
        "environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        )},
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "data": str(args.data) if args.data else None,
        "load_events": load_events,
        "models": [{"label": "MJev-Spatial/Qwen3.5-4B", "parameters": int(sum(p.numel() for p in model.parameters()))}],
        "argv": sys.argv,
        "nvml_sampler_available": nvml.available,
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
