#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-fs/baselines/cost_bench/ovis_compare_greedy_v1
MODE=${1:-check}

if [[ "$MODE" == "check" ]]; then
    exec bash "$ROOT/run_all.sh" \
        check \
        --methods spatialrgpt
fi

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "用法: bash run_spatialrgpt.sh check|smoke|full [全新输出目录]" >&2
    exit 2
fi

OUT=${2:?请提供全新的输出目录}

exec bash "$ROOT/run_all.sh" \
    "$MODE" \
    --methods spatialrgpt \
    --output "$OUT"