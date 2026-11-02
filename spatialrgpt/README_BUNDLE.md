# SpatialRGPT SpatialLab bundle

Contents:
- source/: current server working tree, including spatiallab_adapter/.
- results/summary_original.json: original parser result, 453/1400.
- results/summary_rescored.json: deterministic leading-choice recovery result, 640/1400 (45.71 percent).
- results/predictions.jsonl and results/regions.jsonl: per-question records.
- results/cost_summary.json: full online cost summary.

Evaluation boundary:
Grounding DINO-Tiny creates automatic open-vocabulary regions from the RGB image, question and options. SpatialLab categories, gold answers, manual boxes and manual masks are not supplied to the model. This is not the GT-region version of SpatialRGPT.

The current answer generation is greedy: do_sample=False and num_beams=1. The source adapter README contains the environment and run commands.

Weights are excluded:
- SpatialRGPT-VILA1.5-8B: https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B
- Depth Anything V1: https://huggingface.co/spaces/LiheYoung/Depth-Anything
- Grounding DINO-Tiny: use the official/Hugging Face Grounding DINO Tiny checkpoint.

The current source tree was copied from the server working tree after cleanup; old deleted files and generated visualization artifacts are not included.
