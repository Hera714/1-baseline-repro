# ByDeWay baseline bundle for SpatialLab

This directory contains the original ByDeWay source plus the local adapter used for the unified SpatialLab evaluation protocol on one RTX 4090 (single GPU).

## Included

- `source/ByDeWay-main/`: ByDeWay-Depth-Captioning source snapshot, including Depth Anything V2-Large code and Kosmos depth-captioning code.
- `source/bydeway_adapter.py`: adapter used by the unified Ovis/ByDeWay evaluation runner.
- `source/protocol.py`, `source/canonical_evaluation.py`, `source/run_cost.sh`: protocol and evaluation helpers copied from the locked evaluation bundle.
- `results/cost_run/`: full 1400-question predictions, protocol summary and cost records.

## Protocol and result

- Dataset: SpatialLab, 1400 planned questions.
- Depth caption stage: Depth Anything V2-Large plus Kosmos-2 caption generation.
- Final answer model: local Ovis2.5-9B, greedy decoding, `max_new_tokens=2048`, thinking budget 1536.
- Warm-up requests are excluded from reported timing; invalid/unparsed answers count as wrong in the denominator.
- Result in `results/cost_run/protocol_summary.json`: 728/1400 = 52.0000 percent; 10 answers were unparsed or invalid.
- `results/cost_run/cost_summary.json` records Ovis calls, Kosmos calls, depth inference time, token counts, timing distribution and sampled memory.

## Weights and environments (not included)

- Kosmos-2 patch14-224: `model.safetensors`, from https://huggingface.co/microsoft/kosmos-2-patch14-224
- Depth Anything V2-Large: `depth_anything_v2_vitl.pth`, from https://huggingface.co/depth-anything/Depth-Anything-V2-Large
- Ovis2.5-9B: shared external model, from https://www.modelscope.cn/models/AIDC-AI/Ovis2.5-9B

The Ovis conda environment, Hugging Face caches, raw SpatialLab images and all large checkpoints are intentionally excluded. The included source README and scripts show the expected local paths and generation settings. The Kosmos/Depth model files are several gigabytes and must be downloaded separately.

## Provenance

The original project is https://github.com/Rajarshi12321/ByDeWay-Depth-Captioning. The SpatialLab score is a project-specific adaptation under the locked `ovis_compare_greedy_v1` protocol, not an official paper benchmark score.
