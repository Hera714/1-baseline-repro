#!/usr/bin/env python3
"""Rescore saved SpatialRGPT SpatialLab outputs without rerunning the model.

The script never edits predictions.jsonl or the original summary.json. It keeps
already parsed answers and recovers only previously unparsed outputs that begin
with one unambiguous multiple-choice index (1-4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEADING_CHOICE = re.compile(r"^\s*\(?([1-4])\)?(?:\s|[.,;:!<]|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--audit-sample-size", type=int, default=80)
    parser.add_argument("--audit-seed", type=int, default=20261014)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_leading_choice(raw_output: Any) -> str | None:
    match = LEADING_CHOICE.match(str(raw_output or ""))
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    status_counts: Counter[str] = Counter()
    parser_counts_before: Counter[str] = Counter()
    parser_counts_after: Counter[str] = Counter()
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    recovered: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    evaluated = correct = originally_parsed = 0

    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] += 1
        if status != "ok":
            continue
        evaluated += 1
        parser_counts_before[str(row.get("parse_method", "unknown"))] += 1
        prediction = row.get("pred_answer")
        method = str(row.get("parse_method", "unknown"))
        if prediction is not None:
            prediction = str(prediction)
            originally_parsed += 1
        else:
            prediction = recover_leading_choice(row.get("raw_output"))
            if prediction is not None:
                method = "leading_choice_recovery"
                recovered.append(
                    {
                        "id": str(row.get("id")),
                        "raw_output": str(row.get("raw_output", "")),
                        "recovered_answer": prediction,
                        "gt_answer": str(row.get("gt_answer", "")),
                        "category": row.get("category"),
                        "question": row.get("question"),
                        "options": row.get("options"),
                    }
                )
            else:
                method = "unresolved"
                unresolved.append(
                    {
                        "id": str(row.get("id")),
                        "raw_output": str(row.get("raw_output", "")),
                    }
                )
        parser_counts_after[method] += 1
        is_correct = prediction is not None and prediction == str(row.get("gt_answer"))
        correct += int(is_correct)
        category = str(row.get("category", "Unknown"))
        by_category[category][0] += int(is_correct)
        by_category[category][1] += 1

    summary = {
        "source_predictions": str(args.predictions),
        "source_predictions_sha256": sha256(args.predictions),
        "rescoring_policy": (
            "Preserve existing pred_answer; only when pred_answer is null, "
            "recover a leading standalone choice index 1-4."
        ),
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": round(100.0 * correct / evaluated, 4) if evaluated else 0.0,
        "status_counts": dict(status_counts),
        "originally_parsed": originally_parsed,
        "recovered_leading_choices": len(recovered),
        "unresolved": len(unresolved),
        "parser_counts_before": dict(parser_counts_before),
        "parser_counts_after": dict(parser_counts_after),
        "by_category": {
            category: {
                "correct": values[0],
                "total": values[1],
                "accuracy": round(100.0 * values[0] / values[1], 4),
            }
            for category, values in sorted(by_category.items())
        },
    }

    sample_size = min(max(args.audit_sample_size, 0), len(recovered))
    rng = random.Random(args.audit_seed)
    sample = rng.sample(recovered, sample_size)
    audit = {
        "source_predictions": str(args.predictions),
        "source_predictions_sha256": summary["source_predictions_sha256"],
        "population": len(recovered),
        "sample_method": "simple random sample without replacement",
        "sample_seed": args.audit_seed,
        "sample_size": sample_size,
        "records": sample,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
