#!/usr/bin/env bash
set -euo pipefail
BENCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN=${COST_RUN:?Set COST_RUN to a new directory}
DATA=${COST_DATA:?Set COST_DATA}
IMAGES=/root/autodl-fs/img
SPIN=/root/autodl-fs/baselines/spatialpin/SpatialPIN/spatialpin
SRGPT=/root/autodl-fs/baselines/spatialrgpt/SpatialRGPT
MODEL=/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B
KAR="$BENCH/frozen_kar"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/autodl-fs/baselines/cache/huggingface
mkdir -p "$RUN"
MODE=${1:?Usage: bash run.sh spatialrgpt-online|spatialpin-online|ovis|bydeway|summary}
if [[ "$MODE" == manifest ]]; then
  /root/autodl-fs/baselines/envs/spatialpin/bin/python "$BENCH/run_cost.py" --method manifest --output "$BENCH/manifest60"
  exit
fi
if [[ "$MODE" == manifest-full ]]; then
  /root/miniconda3/bin/python "$BENCH/prepare_full.py"
  exit
fi
unset LD_PRELOAD
if [[ "$MODE" == *-online || "$MODE" == summary ]]; then
  PY=/root/miniconda3/bin/python
elif [[ "$MODE" == spatialrgpt-regions ]]; then
  PY=/root/cost-detector-env/bin/python
  export PYTHONPATH="$SRGPT"
elif [[ "$MODE" == spatialrgpt-evaluate ]]; then
  PY=/root/autodl-fs/baselines/envs/spatialrgpt/bin/python
  export PYTHONPATH="$SRGPT"
elif [[ "$MODE" == spatialpin-prepare ]]; then
  PY=/root/autodl-fs/baselines/envs/spatialpin/bin/python
  export PYTHONPATH="$SPIN:$SPIN/main:$SPIN/export"
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
else
  PY=/root/ovis-compare-greedy-v1-env/bin/python
  export PYTHONPATH="$KAR/src"
fi
if [[ ! -x "$PY" ]]; then
  echo "Missing environment $PY; run this stage on its designated instance" >&2
  exit 1
fi
cd "$BENCH"
case "$MODE" in
  spatialrgpt-online)
    CMD=("$PY" online_cost.py --method spatialrgpt --data "$DATA" --output "$RUN") ;;
  spatialpin-online)
    CMD=("$PY" online_cost.py --method spatialpin --data "$DATA" --output "$RUN") ;;
  spatialrgpt-regions)
    CMD=("$PY" run_cost.py --method spatialrgpt --stage regions --output "$RUN" --
      --data "$DATA" --image-dir "$IMAGES" --model /root/autodl-fs/baselines/models/grounding-dino-tiny
      --output "$RUN/regions.jsonl") ;;
  spatialrgpt-evaluate)
    CMD=("$PY" run_cost.py --method spatialrgpt --stage evaluate --output "$RUN" --
      --data "$DATA" --image-dir "$IMAGES" --regions "$RUN/regions.jsonl"
      --model-path /root/autodl-fs/baselines/models/SpatialRGPT-VILA1.5-8B
      --depth-anything-path /root/autodl-fs/baselines/models/Depth-Anything
      --save-artifacts --output-dir "$RUN/spatialrgpt") ;;
  spatialpin-discover)
    CMD=("$PY" run_cost.py --method spatialpin --stage discover --output "$RUN" --
      --data "$DATA" --image-dir "$IMAGES" --model-path "$MODEL" --kar3d-root "$KAR"
      --output "$RUN/discoveries.jsonl") ;;
  spatialpin-prepare)
    CMD=("$PY" run_cost.py --method spatialpin --stage prepare --output "$RUN" --
      --data "$DATA" --image-dir "$IMAGES" --discoveries "$RUN/discoveries.jsonl"
      --depth-checkpoint /root/autodl-fs/baselines/models/Depth-Anything/checkpoints/depth_anything_vitl14.pth
      --output "$RUN/prepared.jsonl") ;;
  spatialpin-evaluate)
    CMD=("$PY" run_cost.py --method spatialpin --stage evaluate --output "$RUN" --
      --data "$DATA" --image-dir "$IMAGES" --prepared "$RUN/prepared.jsonl"
      --model-path "$MODEL" --kar3d-root "$KAR" --output-dir "$RUN/spatialpin") ;;
  ovis)
    CMD=("$PY" run_cost.py --method ovis_native --data "$DATA" --model "$MODEL" --output "$RUN") ;;
  bydeway)
    export BYDEWAY_MODEL_DIR=/root/autodl-fs/baselines/models/bydeway
    "$PY" -c 'import cv2; from pathlib import Path; p=Path("/root/autodl-fs/baselines/models/bydeway"); assert (p/"depth_anything_v2_vitl.pth").is_file() and (p/"kosmos-2-patch14-224/config.json").is_file(), "Run setup_bydeway.sh first"'
    CMD=("$PY" run_cost.py --method bydeway --data "$DATA" --model "$MODEL" --output "$RUN") ;;
  summary)
    "$PY" summarize.py "$RUN"
    exit ;;
  *) echo "Unknown stage: $MODE" >&2; exit 1 ;;
esac
# GNU time reports whole-process wall time and CPU RSS, including imports/model loading.
# Recorder reports each synchronized sample and model loading separately.
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -v -o "$RUN/$MODE.process_time.txt" "${CMD[@]}" 2>&1 | tee "$RUN/$MODE.log"
else
  "${CMD[@]}" 2>&1 | tee "$RUN/$MODE.log"
fi
