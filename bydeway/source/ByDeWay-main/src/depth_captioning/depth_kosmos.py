#coidng=utf-8
import os

# 固定本地模型目录
MODEL_DIR = os.environ.get("BYDEWAY_MODEL_DIR", "/root/autodl-fs/baselines/models/bydeway")

# Verify the current directory
current_dir = os.getcwd()

# Change to the desired directory
# Deployment uses absolute imports; do not change the caller working directory.

current_dir = os.getcwd()
print(f"Current directory: {current_dir}")
print(f"MODEL_DIR: {MODEL_DIR}")

import cv2
import numpy as np
import torch
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
from transformers import AutoProcessor, AutoModelForVision2Seq
import matplotlib.pyplot as plt
import requests


def _resolve_local_kosmos_dir(ckpt: str):
    """
    本地优先解析 Kosmos 路径，支持：
    1) 直接传入绝对/相对目录
    2) 传入 'microsoft/kosmos-2-patch14-224'
    3) 传入 'kosmos-2-patch14-224'
    """
    if os.path.isdir(ckpt):
        return ckpt

    basename = os.path.basename(ckpt.rstrip("/"))
    candidate = os.path.join(MODEL_DIR, basename)
    if os.path.isdir(candidate):
        return candidate

    candidate = os.path.join(MODEL_DIR, ckpt)
    if os.path.isdir(candidate):
        return candidate

    return None


class DepthContextCreator:
    def __init__(self, encoder="vitl"):
        # Set device
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        self.encoder2name = {
            "vits": "Small",
            "vitb": "Base",
            "vitl": "Large",
            "vitg": "Giant",
        }
        self.encoder = encoder
        self.model_name = self.encoder2name[self.encoder]
        self.model = DepthAnythingV2(**self.model_configs[self.encoder])

        # 本地优先加载 Depth-Anything-V2 权重
        local_path = os.path.join(MODEL_DIR, f"depth_anything_v2_{self.encoder}.pth")
        if os.path.exists(local_path):
            filepath = local_path
            print(f"Using local Depth-Anything-V2 checkpoint: {filepath}")
        else:
            filepath = hf_hub_download(
                repo_id=f"depth-anything/Depth-Anything-V2-{self.model_name}",
                filename=f"depth_anything_v2_{self.encoder}.pth",
                repo_type="model",
            )
            print(f"Using remote Depth-Anything-V2 checkpoint: {filepath}")

        state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.DEVICE).eval()
        self.cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    def predict_depth(self, image):
        return self.model.infer_image(image)

    def on_submit(self, image):
        original_image = image.copy()
        depth = self.predict_depth(image[:, :, ::-1])
        raw_depth = Image.fromarray(depth.astype("uint16"))
        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        raw_depth.save(tmp_raw_depth.name)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        colored_depth = (self.cmap(depth)[:, :, :3] * 255).astype(np.uint8)
        gray_depth = Image.fromarray(depth)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        gray_depth.save(tmp_gray_depth.name)
        return [
            (original_image, colored_depth),
            tmp_gray_depth.name,
            tmp_raw_depth.name,
        ]

    def make_depth_context_img(self, image, top_threshold=70, bottom_threshold=30):
        image_array = np.array(image)
        all_outputs = self.on_submit(image_array)
        intensity_image = all_outputs[0][1]
        height, width, channels = image_array.shape
        flattened_intensity_image = intensity_image.reshape(-1, 3)
        intensities = np.mean(flattened_intensity_image, axis=1)
        top30_threshold = np.percentile(intensities, top_threshold)
        bottom30_threshold = np.percentile(intensities, bottom_threshold)
        top30_mask_flat = intensities > top30_threshold
        bottom30_mask_flat = intensities < bottom30_threshold
        mid40_mask_flat = (intensities >= bottom30_threshold) & (
            intensities <= top30_threshold
        )
        top30_mask = top30_mask_flat.reshape(height, width, 1)
        bottom30_mask = bottom30_mask_flat.reshape(height, width, 1)
        mid40_mask = mid40_mask_flat.reshape(height, width, 1)
        top30_image = np.where(top30_mask, image_array, 0)
        bottom30_image = np.where(bottom30_mask, image_array, 0)
        mid40_image = np.where(mid40_mask, image_array, 0)
        return [top30_image, bottom30_image, mid40_image]


class Kosmos2Captioner:
    def __init__(self, ckpt="microsoft/kosmos-2-patch14-224", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # 本地优先加载 Kosmos
        local_ckpt_dir = _resolve_local_kosmos_dir(ckpt)
        if local_ckpt_dir is not None:
            print(f"Using local Kosmos checkpoint at {local_ckpt_dir}")
            self.processor = AutoProcessor.from_pretrained(local_ckpt_dir)
            self.model = AutoModelForVision2Seq.from_pretrained(local_ckpt_dir).to(self.device)
        else:
            print(f"Using remote Kosmos checkpoint: {ckpt}")
            self.processor = AutoProcessor.from_pretrained(ckpt)
            self.model = AutoModelForVision2Seq.from_pretrained(ckpt).to(self.device)

    def get_caption_and_entities(self, image_array):
        image = Image.fromarray(image_array.astype("uint8"))
        prompt = "<grounding> An image of"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        caption, entities = self.processor.post_process_generation(generated_text)
        return caption, entities


class DepthKosmosCaptioner:
    def __init__(self, ckpt="microsoft/kosmos-2-patch14-224", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # 不再单独重复加载 processor/model，统一交给 Kosmos2Captioner 本地优先处理
        self.depth_context = DepthContextCreator()
        self.captioner = Kosmos2Captioner(ckpt=ckpt, device=self.device)
        self.location = ["Closest", "Farthest", "Mid Range"]

    def get_caption_with_depth(self, image, top_threshold=70, bottom_threshold=30):
        images = self.depth_context.make_depth_context_img(
            image, top_threshold=top_threshold, bottom_threshold=bottom_threshold
        )
        full_string = ""
        for i in range(3):
            caption, entities = self.captioner.get_caption_and_entities(images[i])
            entity_text = entities[0][0] if entities and len(entities) > 0 and len(entities[0]) > 0 else caption
            full_string += f"{self.location[i]}: {entity_text}\n----\n"
        return full_string

    def display_depth_images(self, image):
        images = self.depth_context.make_depth_context_img(image)
        plt.figure(figsize=(15, 5))
        titles = ["Top 30%", "Bottom 30%", "Mid 40%"]
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 3, i + 1)
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

