from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def split_thinking_and_answer(response: str) -> tuple[str, str]:
    """Keep the model's visible thinking separate from its final answer text."""
    if not response:
        return "", ""
    if "</think>" not in response:
        return "", response.strip()
    before, answer = response.rsplit("</think>", 1)
    thinking = before.rsplit("<think>", 1)[-1]
    return thinking.strip(), answer.strip()


def extract_answer(response: str) -> str:
    if not response:
        return "Unknown"
    tagged = re.findall(r"<answer>\s*([1-4])\s*</answer>", response, flags=re.IGNORECASE)
    if tagged:
        return tagged[-1]

    # Ovis may emit the final choice as a bare number after its thinking block.
    # Only inspect the answer portion so numbers used in reasoning are ignored.
    _, answer_text = split_thinking_and_answer(response)
    if answer_text.startswith("<think>"):
        return "Unknown"
    explicit = re.search(
        r"(?:final\s+)?(?:the\s+)?answer\s*"
        r"(?:\(\s*1\s*[-–—]\s*4\s*\))?\s*"
        r"(?:is|[:：])\s*(?:option\s*)?([1-4])(?=\D|$)",
        answer_text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1)

    answer = re.match(r"^\s*(?:option\s*)?([1-4])(?=\D|$)", answer_text, flags=re.IGNORECASE)
    if answer:
        return answer.group(1)
    return "Unknown"


def parse_first_json_object(text: str) -> tuple[Dict[str, Any], str]:
    """Decode the first top-level JSON object and preserve any trailing text.

    ``json.JSONDecoder.raw_decode`` stops after one complete object, unlike a
    greedy regular expression which can accidentally join two JSON objects.
    """
    raw = (text or "").strip()
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found.")
    parsed, end = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(parsed, dict):
        raise ValueError("The first JSON value is not an object.")
    return parsed, raw[start + end :].strip()


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    correct = sum(bool(row.get("is_correct")) for row in rows)
    route_counts = Counter(row.get("routing", {}).get("route", "unknown") for row in rows)
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        category = row.get("category", "Unknown")
        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += int(bool(row.get("is_correct")))
    by_category = {}
    for category, stats in sorted(category_stats.items()):
        total = stats["total"]
        by_category[category] = {
            **stats,
            "accuracy": round(100.0 * stats["correct"] / total, 4) if total else 0.0,
        }
    return {
        "evaluated": len(rows),
        "correct": correct,
        "accuracy": round(100.0 * correct / len(rows), 4) if rows else 0.0,
        "route_counts": dict(sorted(route_counts.items())),
        "by_category": by_category,
    }
