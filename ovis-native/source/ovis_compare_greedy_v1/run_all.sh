#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-fs/baselines/cost_bench/ovis_compare_greedy_v1
MODE=${1:-check}
shift || true
exec /root/miniconda3/bin/python -u "$ROOT/suite.py" "$MODE" "$@"
