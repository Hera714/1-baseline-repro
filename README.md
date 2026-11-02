# Baseline reproduction bundles

This repository contains five separately packaged baseline bundles used in the project-specific SpatialLab comparison:

- `spatialrgpt/`: SpatialRGPT source, SpatialLab adapter, summaries and cost records.
- `mjev-spatial/`: MJev-Spatial source, profiling entrypoint, summaries and cost records.
- `spatialpin/`: SpatialPIN-style matched-backbone adapter, source snapshot and 1400-question cost records.
- `bydeway/`: ByDeWay-Depth-Captioning source, local Ovis adapter and 1400-question cost records.
- `ovis-native/`: native Ovis SpatialLab evaluator, locked protocol and 1400-question cost records.

Large model weights, conda environments, Hugging Face/ModelScope caches, raw SpatialLab images and large visualization folders are intentionally excluded. Each bundle documents official download sources, expected paths, decoding settings and the exact result files.

All five results are project-specific SpatialLab adaptations, not official paper benchmark scores. The three new bundles use the same `ovis_compare_greedy_v1` reporting convention: single RTX 4090, 1400 planned questions, warm-ups excluded from timing, and invalid/unparsed answers counted as wrong.

Recorded results in the bundled protocol summaries:

- Ovis native: 729/1400 = 52.0714 percent.
- ByDeWay: 728/1400 = 52.0000 percent.
- SpatialPIN-style adapter: 668/1400 = 47.7143 percent.
- SpatialRGPT: see `spatialrgpt/README_BUNDLE.md`.
- MJev-Spatial: see `mjev-spatial/README_BUNDLE.md`.
