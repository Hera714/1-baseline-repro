# Native Ovis baseline bundle for SpatialLab

This directory contains the native Ovis SpatialLab evaluator, the locked decoding/profiling protocol and the complete 1400-question result records.

## Included

- `source/ovis_compare_greedy_v1/run_cost.py` with `--method ovis_native`: the SpatialLab native Ovis evaluation entrypoint used for the bundled result.
- `source/ovis_compare_greedy_v1/generalization/eval_ovis_native.py`: the related native-Ovis generalization evaluator kept for reference.
- `source/ovis_compare_greedy_v1/protocol.py`: deterministic parsing, timing and generation protocol.
- `source/ovis_compare_greedy_v1/run.sh`, `run_all.sh`, `run_generalization.sh` and helper modules: launch/profiling utilities used by the experiment.
- `results/cost_run/`: predictions, protocol summary, component traces and cost summary.

## Protocol and result

- Dataset: SpatialLab, 1400 planned questions.
- No external detector, depth model or reconstruction module is used for this native baseline.
- Model: local Ovis2.5-9B, greedy decoding (`do_sample=false`), `max_new_tokens=2048`, thinking budget 1536.
- Warm-up requests are excluded from reported timing; invalid/unparsed answers count as wrong in the denominator.
- Result in `results/cost_run/protocol_summary.json`: 729/1400 = 52.0714 percent; 1 answer was unparsed or invalid.
- `results/cost_run/cost_summary.json` records model calls, input/output token counts, timing distribution, and sampled NVML/Torch memory.

## Model and environment (not included)

The Ovis2.5-9B checkpoint and its conda environment are intentionally excluded because they are shared with the main project and other baselines. Download/use the model from https://www.modelscope.cn/models/AIDC-AI/Ovis2.5-9B and set the local model path in the launch command. Raw SpatialLab images and caches are also excluded.

## Reproducibility note

This is the native Ovis SpatialLab adaptation used in the project. It is not an official Ovis paper benchmark score. The exact generation settings, seed and parser are preserved in the source and result metadata.
