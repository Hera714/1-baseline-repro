# SpatialPIN baseline bundle for SpatialLab

This directory contains the reproducible code snapshot used for the SpatialPIN-style SpatialLab evaluation on one RTX 4090 (single GPU). It is an adaptation of the original SpatialPIN pipeline, not the official SpatialPIN benchmark score.

## Included

- `source/SpatialPIN/spatialpin/`: original SpatialPIN code with the local Depth Anything path and offline/local checkpoint handling used in the run.
- `source/SpatialPIN/spatialpin/spatiallab_adapter/run_spatiallab.py`: SpatialLab adapter used for discovery, perception preparation and final Ovis answer generation.
- `source/spatialpin_spatiallab.py`: same adapter snapshot kept as a standalone reference.
- `source/lama/`: only the LaMa Python source/configuration and prediction entrypoint; the embedded conda environment is intentionally excluded.
- `results/cost_run/`: full 1400-question prediction, protocol and cost records from `ovis_compare_greedy_v1`.

## Protocol and result

- Dataset: SpatialLab, 1400 planned questions.
- Image input: original 640x480 SpatialPIN path.
- Perception: GroundingDINO Swin-B + SAM ViT-H + Depth Anything V1-L.
- Final answer model: local Ovis2.5-9B, greedy decoding, `max_new_tokens=2048`, thinking budget 1536.
- Warm-up requests are excluded from reported timing; invalid/unparsed answers count as wrong in the denominator.
- Result in `results/cost_run/protocol_summary.json`: 668/1400 = 47.7143 percent; 8 answers were unparsed or invalid.
- Runtime records are in `cost_summary.json`, including stage/component timing, token counts, call counts and sampled NVML/Torch memory when available.

## Weights and environments (not included)

The following large files are deliberately not committed:

- SAM ViT-H: `sam_vit_h_4b8939.pth`, from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- GroundingDINO Swin-B: `groundingdino_swinb_cogcoor.pth` and `GroundingDINO_SwinB.cfg.py`, from https://huggingface.co/ShilongLiu/GroundingDINO
- Depth Anything V1-L checkpoint: `depth_anything_vitl14.pth`, from the Depth Anything release https://github.com/LiheYoung/Depth-Anything
- Big-LaMa: `big-lama/models/best.ckpt`, from https://huggingface.co/smartywu/big-lama
- Ovis2.5-9B: shared external model, from https://www.modelscope.cn/models/AIDC-AI/Ovis2.5-9B

The original SpatialPIN conda environment, the LaMa conda environment, Hugging Face caches, SpatialLab images and raw checkpoints are also excluded. Install dependencies from the included `requirements.txt`/conda YAML files and place checkpoints at the paths expected by the adapter, or change the paths in the local configuration.

## Important reproducibility note

The reported adapter is explicitly named `SpatialPIN-style matched-backbone SpatialLab adapter` in the source. It keeps the SpatialPIN perception backbone and image size, while using the local Ovis model for SpatialLab multiple-choice answers. It should not be described as an official SpatialPIN benchmark number.
