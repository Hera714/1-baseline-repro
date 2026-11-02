# MJev-Spatial SpatialLab bundle

Contents:
- source/: current MJev-Spatial source and scripts.
- source/scripts/predict_profile.py: project profiling entrypoint.
- results/summary.json and results/meta.json: full-run summary and environment metadata.
- results/predictions.jsonl and results/cost.jsonl: per-question predictions and cost events.

Full SpatialLab result:
- 597/1400 correct, 42.64 percent.
- 1400 records are ok; retries 0.
- RTX 4090 48 GB, batch 1.
- Online total 11384.05 seconds; mean 8.131 seconds; P95 8.331 seconds.
- NVML peak 16.65 GiB.
- One candidate-scoring forward call per question; output text tokens 0.

MJev-Spatial uses candidate scoring rather than free-form text generation. This is a project-specific SpatialLab MCQ adaptation, not the official MJev paper benchmark score.

Weights are excluded:
- MJev-Spatial-4B: https://huggingface.co/mindlab-research/MJev-Spatial-4B
- Base model: https://huggingface.co/Qwen/Qwen3.5-4B

Use source/scripts/predict.py for normal inference and source/scripts/predict_profile.py for profiling. Set the local checkpoint, image directory and SpatialLab JSONL paths at runtime.
