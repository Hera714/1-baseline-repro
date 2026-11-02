# ByDeWay: Depth-based Captioning for Multimodal Large Language Models

## Overview

**ByDeWay** is a training-free framework designed to improve the spatial reasoning and grounding abilities of Multimodal Large Language Models (MLLMs) through **Layered-Depth-based Prompting (LDP)**. This Python package uses monocular depth estimation along with automatic region-aware captioning to enrich model prompts with structured spatial context, boosting performance on tasks prone to hallucination and those needing detailed visual reasoning.

## Features

- **Training-Free Enhancement:** No fine-tuning or model parameter updates required.
- **Plug-and-Play:** Compatible with any black-box MLLM that accepts image and text inputs.
- **Layered Spatial Context:** Utilizes monocular depth estimation to segment images into closest, mid-range, and farthest regions.
- **Region-Specific Captioning:** Automatically generates captions for each depth layer.
- **Improved Performance:** Consistently enhances hallucination resistance and spatial reasoning—as shown on POPE and GQA tasks.
- **Modular & Scalable:** Add LDP as an input prompt component for diverse use cases.

## Datasets used for Evaluation:

1. POPE Dataset (150 Samples) : [POPE 150 Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/lmms-lab-POPE)
2. GQA Dataset (150 Samples) : [GQA 150 Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/GQA-dataset-150)

## Workflow

1. **Image Input:** Provide an RGB image for analysis.
2. **Depth Estimation:** Segment the image into three depth-based layers (closest, mid-range, farthest) using an off-the-shelf monocular depth estimator (e.g., Depth Anything V2).
3. **Region Masking & Captioning:** Apply spatial masks to extract regions and generate captions using a grounded vision-language model (e.g., KOSMOS-2).
4. **Prompt Construction:** Concatenate the layer-wise region captions into a structured, spatially-aware prompt.
5. **MLLM Inference:** Feed the image, question/instruction, and enhanced prompt into any supported MLLM for improved, hallucination-resistant outputs.

## Repository Structure

```
ByDeWay-Depth-Captioning/
│
├── data/                   # Sample images and data (optional)
├── notebooks/              # Example and demo Jupyter notebooks
├── src/
│   └── depth_captioning/   # Core depth and captioning modules
├── test/                   # Unit and functional tests
├── install_depth_anything.sh # Script to install Depth Anything model
├── requirements.txt        # List of Python dependencies
├── .gitignore
└── README.md               # (This file)
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rajarshi12321/ByDeWay-Depth-Captioning.git
cd ByDeWay-Depth-Captioning
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Depth Anything

Install the dependencies and download weights for the Depth Anything V2 monocular depth estimator:

```bash
bash install_depth_anything.sh
```

**Note:** Instructions above assume a Python 3.8+ environment.

## Quick Start Example

Open a demo notebook from the `notebooks/` directory or run a Python script from the `src/depth_captioning/` module to process an image:

```python
import requests
from src.depth_captioning.depth_kosmos import DepthKosmosCaptioner, Image as Image_PIL

# Initialize the depth-aware captioning pipeline
depth_kosmos_captioner = DepthKosmosCaptioner()

# Load the image from a URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTp5jMLHnfiO56w8iVWAwI4VvOu4B_5c2C1ww&s"
response = requests.get(url, stream=True)
image = Image_PIL.open(response.raw)

# (Optional) Visualize depth segmentation
# depth_kosmos_captioner.display_depth_images(image)

# Generate and print the depth-based structured caption
full_caption_string = depth_kosmos_captioner.get_caption_with_depth(image)
print(full_caption_string)
```

## Usage Details

- **Depth Estimation:** Uses `Depth Anything V2` for zero-shot monocular depth prediction.
- **Region Segmentation:** Depth values split into top 30% (closest), middle 40% (mid-range), bottom 30% (farthest).
- **Captioning:** Each masked region is described using a vision-language model such as KOSMOS-2.
- **Prompting:** Region captions are concatenated with the original question/task to instruct the MLLM.

## Benchmark Results

| Model        | Task | Baseline Accuracy | LDP Accuracy | Δ |
|--------------|------|-------------------|--------------|----|
| GPT-4o       | POPE | 0.860             | 0.873        | +0.013 |
| Qwen2.5-VL   | POPE | 0.7267            | 0.9000       | +0.1733 |
| ViLT         | POPE | 0.8533            | 0.9267       | +0.0734 |
| BLIP         | POPE | 0.8733            | 0.9533       | +0.08   |
| Qwen2.5-VL   | GQA  | 0.5007            | 0.6592       | +0.1585 |
| ViLT         | GQA  | 0.527             | 0.627        | +0.1    |
| BLIP         | GQA  | 0.5552            | 0.6704       | +0.1152 |

*LDP = With Layered-Depth-based Prompting. See the project paper for F1, Precision, Recall, and qualitative examples.*

## License

This project is released for research and academic use. See the repository for details.

## Citation

If you use **ByDeWay** or its depth captioning workflow in your research, please cite:

> @misc{roy2025bydewayboostmultimodalllm,
      title={ByDeWay: Boost Your multimodal LLM with DEpth prompting in a Training-Free Way}, 
      author={Rajarshi Roy and Devleena Das and Ankesh Banerjee and Arjya Bhattacharjee and Kousik Dasgupta and Subarna Tripathi},
      year={2025},
      eprint={2507.08679},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.08679 }, 
}

## Links
- [Full Project Paper (PDF)](https://arxiv.org/pdf/2507.08679)
- [GitHub Repository](https://github.com/Rajarshi12321/ByDeWay-Depth-Captioning)
