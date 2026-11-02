#!/usr/bin/env bash
set -Eeuo pipefail

export SPIN_ENV="${SPIN_ENV:-/root/autodl-fs/baselines/envs/spatialpin}"
export SPIN_SRC="${SPIN_SRC:-/root/autodl-fs/baselines/spatialpin/SpatialPIN/spatialpin}"
export SPIN_DEPTH_CKPT="${SPIN_DEPTH_CKPT:-/root/autodl-fs/baselines/models/Depth-Anything/checkpoints/depth_anything_vitl14.pth}"
export LAMA_ENV="${LAMA_ENV:-/root/spatialpin-lama-env}"
export LAMA_SRC="${LAMA_SRC:-/root/autodl-fs/baselines/spatialpin/SpatialPIN/lama}"
export LAMA_MODEL="${LAMA_MODEL:-/root/autodl-fs/baselines/models/lama/big-lama}"

export PYTHONPATH="$SPIN_SRC:$SPIN_SRC/main:$SPIN_SRC/export${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME="${HF_HOME:-/root/autodl-fs/baselines/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/.cache/torch}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SPIN_SRC"
exec "$SPIN_ENV/bin/python" "$SPIN_SRC/main/run_safe_baseline.py" "$@"
