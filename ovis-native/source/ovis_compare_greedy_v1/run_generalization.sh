#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-fs/baselines/cost_bench/ovis_compare_greedy_v1
PY=/root/router-training-local/envs/ovis-infer/bin/python
MODE=${1:-check}
if [[ $(hostname) != autodl-pro-7919e34c8632 ]]; then echo 'Run generalization on pro-7919e34c8632'; exit 1; fi
unset LD_PRELOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
if [[ $MODE == check ]]; then
  "$PY" "$ROOT/test_protocol.py"
  "$PY" "$ROOT/generalization/eval_ovis_native.py" --benchmark omnispatial --check-only
  "$PY" "$ROOT/generalization/eval_blink_native.py" --check-only
  exit
fi
if [[ $MODE != smoke && $MODE != full ]]; then echo 'Usage: bash run_generalization.sh check|smoke|full [NEW_OUTPUT]'; exit 1; fi
OUT=${2:?Provide a NEW output directory}
if [[ -e $OUT ]]; then echo 'Output exists; choose a new directory'; exit 1; fi
exec 9>"$ROOT/suite.lock"
flock -n 9 || { echo 'Another protocol suite is running'; exit 1; }
/root/miniconda3/bin/python "$ROOT/assets.py"
mkdir -p "$OUT"
OMNI_EXTRA=(); BLINK_EXTRA=()
if [[ $MODE == smoke ]]; then OMNI_EXTRA=(--limit 6); BLINK_EXTRA=(--limit-per-task 2); fi
"$PY" -u "$ROOT/generalization/eval_ovis_native.py" --benchmark omnispatial --output "$OUT/omnispatial" --max-new-tokens 2048 --thinking-budget 1536 --seed 42 "${OMNI_EXTRA[@]}" 2>&1 | tee "$OUT/omnispatial.log"
"$PY" -u "$ROOT/generalization/eval_blink_native.py" --output "$OUT/blink" --max-new-tokens 2048 --thinking-budget 1536 --seed 42 "${BLINK_EXTRA[@]}" 2>&1 | tee "$OUT/blink.log"
