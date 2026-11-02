# Baseline reproduction bundles

This repository contains two independent baseline bundles:
- spatialrgpt/: current SpatialRGPT source, SpatialLab adapter, result summaries and cost records.
- mjev-spatial/: current MJev-Spatial source, profiling entrypoint, result summaries and cost records.

Large model weights, conda environments, Hugging Face caches, raw SpatialLab images/data, and large visualization folders are intentionally excluded. Each bundle documents the official download sources.

Recorded SpatialLab results:
- SpatialRGPT: 640/1400 = 45.71 percent after the documented deterministic rescoring rule.
- MJev-Spatial: 597/1400 = 42.64 percent.
These are project-specific SpatialLab adaptations, not official paper benchmark scores.
