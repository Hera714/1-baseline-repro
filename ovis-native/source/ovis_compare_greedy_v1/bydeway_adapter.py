# coding=utf-8
"""
test_benchmark_bydeway_ovis.py

完整 benchmark 文件：
- 使用 ByDeWay 的 DepthKosmosCaptioner 生成 depth-aware captions
- 使用 Ovis2.5-9B 做最终多选推理
- 保留 benchmark 输入 / 推理 / 输出结果保存 这条主链路
- 不包含原始 3D scene graph / routing / knowledge injection / depth projection 等模块

使用前请确认：
1) 本文件位于 ByDeWay-Depth-Captioning 项目根目录，或至少能 import:
      from src.depth_captioning.depth_kosmos import DepthKosmosCaptioner
2) JSON_PATH / IMAGE_DIR 路径正确
3) 显存足够同时加载 DepthKosmosCaptioner + Ovis2.5-9B
"""

import os
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-fs/modelscope_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
# 把Depth-Anything-V2目录加入Python搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Depth-Anything-V2"))
import gc
import json
import re
import traceback
from typing import Dict, Any, Optional

import torch
from PIL import Image
from tqdm import tqdm
from modelscope import AutoModelForCausalLM

from src.depth_captioning.depth_kosmos import DepthKosmosCaptioner


# =========================
# 1. 基础配置
# =========================
MODEL_PATH = os.environ.get("BYDEWAY_OVIS_MODEL", "/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B")
JSON_PATH = "/root/autodl-fs/baselines/spatialpin/data/spatiallab_mcq.json"
IMAGE_DIR = "/root/autodl-fs/img"

OUTPUT_EVAL_JSON = "bydeway_ovis_eval.json"
OUTPUT_META_JSON = "bydeway_ovis_meta.json"

START_IDX = 0
END_IDX = None

# 不筛选则设为空列表 []
TARGET_TASK_ID = []
TARGET_CATEGORY = []
TARGET_SUB_CATEGORY = []

ENABLE_THINKING = True
ENABLE_THINKING_BUDGET = True
MAX_NEW_TOKENS = 2048
THINKING_BUDGET = 1536
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# =========================
# 2. 答案提取
# =========================
def extract_answer(response: str) -> str:
    if not response:
        return "Unknown"

    match = re.search(r"<answer>\s*([1-4])\s*</answer>", response)
    if match:
        return match.group(1)

    match = re.search(r"\b([1-4])\b", response)
    if match:
        return match.group(1)

    return "Unknown"


# =========================
# 3. ByDeWay + Ovis Runner
# =========================
class ByDeWayOvisRunner:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: Optional[torch.device] = None,
        depth_ckpt: str = "microsoft/kosmos-2-patch14-224",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device for inference: {self.device}")

        print("Loading ByDeWay depth captioner...")
        self.depth_captioner = DepthKosmosCaptioner(
            ckpt=depth_ckpt,
            device=self.device
        )

        print(f"Loading Ovis model: {model_path}")
        self.ovis_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        ).to(self.device)
        self.ovis_model.eval()

    def build_mcq_prompt(
        self,
        depth_captions: str,
        question: str,
        options_text: str,
        spatial_relations_text: str = "",
    ) -> str:
        prompt = "Below are region descriptions of the image according to depth layers (Closest, Farthest, Mid Range):\n"
        prompt += depth_captions.strip() + "\n\n"

        if spatial_relations_text and spatial_relations_text.strip():
            prompt += "--- Spatial / Derived Relations ---\n"
            prompt += spatial_relations_text.strip() + "\n\n"

        prompt += f"Question:\n{question}\n\n"
        prompt += f"Options:\n{options_text}\n\n"
        prompt += (
            "Instruction: Provide step-by-step reasoning in a <think>...</think> block, "
            "then give the final answer as a single number (1-4) wrapped in <answer>...</answer> tags. "
            "Output only these blocks and nothing else."
        )
        return prompt

    def _generate_with_ovis(self, image: Image.Image, prompt_text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        input_ids, pixel_values, grid_thws = self.ovis_model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING
        )

        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device) if pixel_values is not None else None
        grid_thws = grid_thws.to(self.device) if grid_thws is not None else None

        generate_kwargs = {
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

        if ENABLE_THINKING:
            generate_kwargs["enable_thinking"] = True
            generate_kwargs["enable_thinking_budget"] = ENABLE_THINKING_BUDGET
            generate_kwargs["thinking_budget"] = THINKING_BUDGET

        with torch.no_grad():
            outputs = self.ovis_model.generate(**generate_kwargs)

        # 只解码新生成部分，避免把 prompt 一起 decode 出来
        input_len = input_ids.shape[1]
        if len(outputs[0]) > input_len and torch.equal(outputs[0][:input_len], input_ids[0]):
            generated_tokens = outputs[0][input_len:]
        else:
            generated_tokens = outputs[0]

        response = self.ovis_model.text_tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        del input_ids, pixel_values, grid_thws, outputs, generated_tokens
        torch.cuda.empty_cache()
        gc.collect()
        return response

    def infer(
        self,
        image: Image.Image,
        question: str,
        options_text: str,
        spatial_relations_text: str = "",
    ) -> Dict[str, Any]:
        # 1) ByDeWay depth captions
        depth_captions = self.depth_captioner.get_caption_with_depth(image)

        # 2) build prompt
        prompt_text = self.build_mcq_prompt(
            depth_captions=depth_captions,
            question=question,
            options_text=options_text,
            spatial_relations_text=spatial_relations_text,
        )

        # 3) Ovis reasoning
        raw_response = self._generate_with_ovis(image=image, prompt_text=prompt_text)
        prediction = extract_answer(raw_response)

        return {
            "depth_captions": depth_captions,
            "prompt_text": prompt_text,
            "raw_response": raw_response,
            "prediction": prediction,
        }


# =========================
# 4. Benchmark 主流程
# =========================
def main():
    runner = ByDeWayOvisRunner(model_path=MODEL_PATH)

    if os.path.exists(OUTPUT_EVAL_JSON):
        print(f"发现已有进度文件 {OUTPUT_EVAL_JSON}，正在恢复...")
        with open(OUTPUT_EVAL_JSON, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
    else:
        print(f"加载全新数据集 {JSON_PATH} ...")
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)

    if os.path.exists(OUTPUT_META_JSON):
        with open(OUTPUT_META_JSON, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
    else:
        meta_dict = {}

    correct_count = 0
    total_run_count = 0

    target_task_ids = [str(x) for x in TARGET_TASK_ID] if TARGET_TASK_ID else []
    target_cats = TARGET_CATEGORY if TARGET_CATEGORY else []
    target_subcats = TARGET_SUB_CATEGORY if TARGET_SUB_CATEGORY else []

    end_val = END_IDX if END_IDX is not None else len(full_dataset)
    test_subset = full_dataset[START_IDX:end_val]

    print(f"开始 ByDeWay + Ovis2.5-9B 评测，测试区间: {START_IDX} -> {end_val}")

    for item in tqdm(test_subset):
        task_id = str(item.get("id"))
        category = item.get("Category", "").strip()
        sub_category = item.get("Sub_Category", "").strip()

        if target_task_ids and task_id not in target_task_ids:
            continue
        if target_cats and category not in target_cats:
            continue
        if target_subcats and sub_category not in target_subcats:
            continue

        total_run_count += 1

        try:
            image_path = os.path.join(IMAGE_DIR, item["Image_Filename"])
            image = Image.open(image_path).convert("RGB")

            options_text = (
                f"1. {item.get('Option_1', '')}\n"
                f"2. {item.get('Option_2', '')}\n"
                f"3. {item.get('Option_3', '')}\n"
                f"4. {item.get('Option_4', '')}"
            )

            result = runner.infer(
                image=image,
                question=item["Question"],
                options_text=options_text,
                spatial_relations_text=""  # 目前为空；如需额外关系文本，可在这里接入
            )

            item["Model_Raw_Response"] = result["raw_response"]
            item["Model_Prediction"] = result["prediction"]
            item["Is_Correct"] = (result["prediction"] == str(item["Answer"]))

            meta_dict[task_id] = {
                "framework": "ByDeWay+Ovis2.5-9B",
                "category": category,
                "sub_category": sub_category,
                "depth_captions": result["depth_captions"],
                "prompt_text": result["prompt_text"],
            }

            if item["Is_Correct"]:
                correct_count += 1

        except Exception as e:
            print(f"\n[ERROR] 任务 {task_id} 运行失败: {e}")
            item["Model_Prediction"] = "Error"
            item["Model_Raw_Response"] = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
            item["Is_Correct"] = False

            meta_dict[task_id] = {
                "framework": "ByDeWay+Ovis2.5-9B",
                "category": category,
                "sub_category": sub_category,
                "error": str(e),
            }

            torch.cuda.empty_cache()
            gc.collect()

        finally:
            with open(OUTPUT_EVAL_JSON, "w", encoding="utf-8") as f:
                json.dump(full_dataset, f, ensure_ascii=False, indent=4)

            with open(OUTPUT_META_JSON, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, ensure_ascii=False, indent=4)

    accuracy = (correct_count / total_run_count * 100) if total_run_count > 0 else 0.0
    print("\n=== 测试结束 ===")
    print(f"有效任务数: {total_run_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"结果文件: {OUTPUT_EVAL_JSON}")
    print(f"附加信息文件: {OUTPUT_META_JSON}")


if __name__ == "__main__":
    main()

