# SpatialRGPT → SpatialLab 自动适配

## 设计边界

适配不修改 SpatialRGPT 官方模型、区域提取器或深度注入机制。新增两个外围阶段：

1. `prepare_regions.py`：使用 Grounding DINO，仅根据原图、问题和四个选项自动产生候选区域框。
2. `run_spatiallab.py`：将区域框转换为 SpatialRGPT 所需的区域掩码，为每个区域注入 RGB 与 Depth Anything 特征，然后完成四选一推理。

明确不用于路由或区域生成的信息：

- `Category`
- `Sub_Category`
- `Answer`
- `Answer_Text`
- 人工边界框、人工掩码或其他 GT 空间标注

若检测器没有产生候选框，使用整幅图像作为唯一自动回退区域，并在输出中记录 `region_fallback=true`。

## 目录与环境

SpatialRGPT 推理继续使用：

```bash
conda activate /root/autodl-fs/baselines/envs/spatialrgpt
```

Grounding DINO 使用单独环境，避免升级 SpatialRGPT 固定的 Transformers 4.37.2：

```bash
/root/autodl-fs/baselines/envs/spatialrgpt/bin/python \
  -m venv --clear --system-site-packages /root/spatialrgpt-regions
source /root/spatialrgpt-regions/bin/activate

python -m pip install -U pip==24.3.1
python -m pip install transformers==4.47.1 timm==1.0.11
```

## 一、区域生成冒烟测试

```bash
source /root/spatialrgpt-regions/bin/activate
export HF_HOME=/root/autodl-fs/baselines/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120

cd /root/autodl-fs/baselines/spatialrgpt/SpatialRGPT

/root/spatialrgpt-regions/bin/python spatiallab_adapter/prepare_regions.py \
  --model /root/autodl-fs/baselines/models/grounding-dino-tiny \
  --data /root/autodl-fs/data.json \
  --image-dir /root/autodl-fs/img \
  --output /root/autodl-fs/baselines/spatialrgpt/outputs/regions_smoke.jsonl \
  --visualize-dir /root/autodl-fs/baselines/spatialrgpt/outputs/regions_smoke_vis \
  --end-index 3 \
  --overwrite
```

## 二、SpatialRGPT 推理冒烟测试

```bash
conda activate /root/autodl-fs/baselines/envs/spatialrgpt

export DEPTH_ANYTHING_PATH=/root/autodl-fs/baselines/models/Depth-Anything
export PYTHONPATH=/root/autodl-fs/baselines/spatialrgpt/SpatialRGPT:$PYTHONPATH

cd /root/autodl-fs/baselines/spatialrgpt/SpatialRGPT

python spatiallab_adapter/run_spatiallab.py \
  --data /root/autodl-fs/data.json \
  --image-dir /root/autodl-fs/img \
  --regions /root/autodl-fs/baselines/spatialrgpt/outputs/regions_smoke.jsonl \
  --model-path /root/autodl-fs/baselines/models/SpatialRGPT-VILA1.5-8B \
  --depth-anything-path /root/autodl-fs/baselines/models/Depth-Anything \
  --output-dir /root/autodl-fs/baselines/spatialrgpt/outputs/spatiallab_smoke \
  --end-index 3 \
  --save-artifacts \
  --overwrite
```

## 三、完整实验

先生成全部区域：

```bash
source /root/spatialrgpt-regions/bin/activate
export HF_HOME=/root/autodl-fs/baselines/cache/huggingface
export HF_HUB_DISABLE_XET=1

cd /root/autodl-fs/baselines/spatialrgpt/SpatialRGPT

/root/spatialrgpt-regions/bin/python spatiallab_adapter/prepare_regions.py \
  --model /root/autodl-fs/baselines/models/grounding-dino-tiny \
  --data /root/autodl-fs/data.json \
  --image-dir /root/autodl-fs/img \
  --output /root/autodl-fs/baselines/spatialrgpt/outputs/regions_full.jsonl \
  --visualize-dir /root/autodl-fs/baselines/spatialrgpt/outputs/regions_full_vis
```

再执行 SpatialRGPT：

```bash
conda activate /root/autodl-fs/baselines/envs/spatialrgpt
export DEPTH_ANYTHING_PATH=/root/autodl-fs/baselines/models/Depth-Anything
export PYTHONPATH=/root/autodl-fs/baselines/spatialrgpt/SpatialRGPT:$PYTHONPATH

cd /root/autodl-fs/baselines/spatialrgpt/SpatialRGPT

python spatiallab_adapter/run_spatiallab.py \
  --data /root/autodl-fs/data.json \
  --image-dir /root/autodl-fs/img \
  --regions /root/autodl-fs/baselines/spatialrgpt/outputs/regions_full.jsonl \
  --model-path /root/autodl-fs/baselines/models/SpatialRGPT-VILA1.5-8B \
  --depth-anything-path /root/autodl-fs/baselines/models/Depth-Anything \
  --output-dir /root/autodl-fs/baselines/spatialrgpt/outputs/spatiallab_full \
  --save-artifacts
```

两个阶段均支持断点续跑。完整结果位于：

- `predictions.jsonl`：逐题原始输出、答案解析、区域框、耗时与正确性。
- `summary.json`：总体准确率、状态计数和按类别准确率。
- `artifacts/`：区域框可视化与深度图，用于案例分析。

## 论文中应说明

该结果属于 **SpatialRGPT official checkpoint + automatic open-vocabulary region proposals**。区域由 Grounding DINO-Tiny 自动生成，未使用 SpatialLab 的任务类别、正确答案或人工空间标注。它与直接使用官方 benchmark GT 区域的结果不可混为一谈。

## 实现变更记录

- 2026-10-13：自动区域检测器固定使用 FP32。Hugging Face Grounding DINO 的部分文本与位置投影保持 FP32，整模型 FP16 会出现卷积或矩阵乘法 dtype 不一致。该修改只影响数值精度兼容性，不改变检测提示、阈值、NMS 或后续 SpatialRGPT pipeline。
- 2026-10-13：答案解析新增裸数字格式，例如模型仅输出 `1` 或 `1.` 时记为选项 1。该修改只修正结果解析，不改变模型推理。
- 2026-10-13：Grounding DINO 与 SpatialRGPT 分别运行在独立环境中；前者使用 Transformers 4.47.1，后者继续保持官方要求的 Transformers 4.37.2。
