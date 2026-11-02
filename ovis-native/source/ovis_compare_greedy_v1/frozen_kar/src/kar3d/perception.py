# coding=utf-8
# perception_ovis_3d.py
import json
import torch
from PIL import Image
from modelscope import AutoModelForCausalLM

from .evaluation import parse_first_json_object

class Ovis3DPerceptionModule:
    def __init__(self, model_path="AIDC-AI/Ovis2.5-9B", attn_implementation="flash_attention_2"):
        print(f"Loading {model_path} 3D Module on vGPU (Full Precision bf16)...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation
        ).cuda()
        self.model.eval()

        # ================= Prompts (强约束，穷尽提取) =================
        self.prompt_line1_base = (
            "Analyze the image and the question. You MUST EXHAUSTIVELY extract ALL key objects that are explicitly mentioned or implicitly needed to answer the question. Missing any relevant reference object will cause downstream failures.\n"
            "Use the exact wording of an answer option when it names the object. Always include every named reference object, even when it is not itself an answer option.\n"
            "OUTPUT FORMAT: Return a JSON object with strictly ONE key:\n"
            "1. 'objects': A JSON array containing EVERY relevant object. Do not omit any. Each object MUST have:\n"
            "   - 'name': Object name.\n"
            "   - 'point_2d': [x, y] normalized coords (0.0 to 1.0) strictly INSIDE the object.\n"
            "   - 'facing_direction': The object's 3D orientation (e.g., 'facing camera', 'facing left', 'facing red car', 'N/A').\n"
            "   - 'reason': Why it's extracted, in at most 8 words.\n"
            "Respond ONLY with valid JSON.\n\n"
        )

        self.prompt_line2_orientation = (
            "Analyze the image and the question. Determine the required perspective. You MUST EXHAUSTIVELY extract ALL key objects involved in the question. Include every reference object, target object, and landmark mentioned.\n"
            "Use the exact wording of an answer option when it names the object. Always include every named reference object.\n"
            "OUTPUT FORMAT: Return a JSON object with strictly two keys:\n"
            "When the question compares named parts of one parent object (for example a red light and a green light), extract EACH named part as its own object rather than merging them into the parent.\n"
            "1. 'target_perspective': If asking from a specific object's viewpoint, output its EXACT 'name'. Otherwise, output 'camera'.\n"
            "2. 'objects': A JSON array containing EVERY involved object. Do not omit any. Each object MUST have:\n"
            "   - 'name': Object name.\n"
            "   - 'point_2d': [x, y] normalized coords (0.0 to 1.0). CRITICAL: This point MUST be located on a solid, unoccluded, and fully visible surface of the object. Do not place it on transparent parts, edges, or background.\n"
            "   - 'anchor_parts': What direction-defining parts are visible? (e.g., 'grille and left doors', 'only flat casing', 'both eyes').\n"
            "   - 'visible_profile': Determine its geometric pose. Strictly choose from: 'frontal', 'side_profile', 'back', 'diagonal_front', 'diagonal_back', or 'unknown'.\n"
            "   - '2d_pointing_direction': Where its front/opening points IN THE 2D IMAGE. Strictly choose from: 'left', 'right', 'camera', 'away', 'up', 'down', or 'unknown'.\n"
            "   - 'reason': Why it is extracted, in at most 8 words.\n"
            "Respond ONLY with valid JSON.\n\n"
        )

        self.prompt_line3_bbox = (
            "Analyze the image and the question. You MUST EXHAUSTIVELY identify and extract ALL objects required to answer the spatial or relative positioning question. Every reference point, subject, and target mentioned or logically implied MUST be included.\n"
            "Use the exact wording of an answer option when it names the object. Always include every named reference object. Keep names and reasons compact.\n"
            "OUTPUT FORMAT: Return a JSON object with strictly ONE key:\n"
            "1. 'objects': A JSON array containing EVERY relevant object. Do not omit any. Each object MUST have:\n"
            "   - 'name': Object name (with sequential number if multiple exist).\n"
            "   - 'bbox_2d': [x_min, y_min, x_max, y_max] normalized bounding box (0.0 to 1.0).\n"
            "   - 'point_2d': [x, y] normalized coords (0.0 to 1.0).\n"
            "   - 'reason': Why it is extracted, in at most 8 words.\n"
            "Respond ONLY with valid JSON.\n\n"
        )

    def _denormalize_point(self, point_norm, img_w, img_h):
        x = min(max(0, int(round(float(point_norm[0]) * img_w))), img_w - 1)
        y = min(max(0, int(round(float(point_norm[1]) * img_h))), img_h - 1)
        return [x, y]

    def _denormalize_bbox(self, bbox_norm, img_w, img_h):
        return [
            min(max(0, int(round(float(bbox_norm[0]) * img_w))), img_w - 1),
            min(max(0, int(round(float(bbox_norm[1]) * img_h))), img_h - 1),
            min(max(0, int(round(float(bbox_norm[2]) * img_w))), img_w - 1),
            min(max(0, int(round(float(bbox_norm[3]) * img_h))), img_h - 1)
        ]

    def _generate_response(self, messages, max_tokens, enable_thinking=False):
        """核心生成方法：通过开关灵活控制是否启用思维链"""
        
        # preprocess_inputs 里的 enable_thinking 决定了是否在 prompt 里强插 <think>
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        # 动态构建 generate 参数字典
        generate_kwargs = {"max_new_tokens": max_tokens}
        if enable_thinking:
            generate_kwargs["enable_thinking"] = True
            generate_kwargs["enable_thinking_budget"] = True
            generate_kwargs["thinking_budget"] = max(256, int(max_tokens * 0.75))

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids, 
                pixel_values=pixel_values, 
                grid_thws=grid_thws,
                **generate_kwargs
            )

        input_len = input_ids.shape[1]
        if len(outputs[0]) > input_len and torch.equal(outputs[0][:input_len], input_ids[0]):
            generated_tokens = outputs[0][input_len:]
        else:
            generated_tokens = outputs[0]
            
        output_text = self.model.text_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        del input_ids, pixel_values, grid_thws, outputs
        torch.cuda.empty_cache()
        return output_text

    def generate_text(self, messages, max_tokens, enable_thinking=False):
        """Public text-generation interface shared by the question-only router."""
        return self._generate_response(messages, max_tokens, enable_thinking)

    def extract_objects_3d(self, image: Image.Image, question_text: str, options_text: str, task_route: str = "line1_base"):
        """阶段 1：感知提取 (要求快且结构化，严格关闭 thinking)"""
        if task_route == "line2_orientation": system_prompt = self.prompt_line2_orientation
        elif task_route == "line3_bbox": system_prompt = self.prompt_line3_bbox
        else: system_prompt = self.prompt_line1_base

        user_prompt = f"Question:\n{question_text}\nOptions:\n{options_text}\n\nOutput JSON:"
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": system_prompt + "\n\n" + user_prompt}]}]

        # 🔴 关键修改：提取阶段关闭 thinking
        response_text = self._generate_response(messages, max_tokens=2048, enable_thinking=False)

        diagnostics = {
            "parse_status": "error",
            "parse_error": "",
            "raw_characters": len(response_text or ""),
            "trailing_text_present": False,
            "objects_returned": 0,
            "objects_accepted": 0,
            "generation_may_be_truncated": False,
        }
        try:
            clean_text = response_text.strip()
            # 兼容处理（以防万一模型自己吐出了 think 标签）
            if "</think>" in clean_text:
                clean_text = clean_text.split("</think>")[-1].strip()
            
            parsed_data, trailing_text = parse_first_json_object(clean_text)
            diagnostics["parse_status"] = "ok"
            diagnostics["trailing_text_present"] = bool(trailing_text)

            target_perspective = parsed_data.get("target_perspective", "camera") if task_route == "line2_orientation" else "camera"
            objects_list = parsed_data.get("objects", [])
            if not isinstance(objects_list, list):
                raise ValueError("'objects' must be a JSON array.")
            diagnostics["objects_returned"] = len(objects_list)

            cleaned_data = []
            for item in objects_list:
                if not isinstance(item, dict): continue
                point = item.get("point_2d") or item.get("point")
                
                if task_route == "line3_bbox":
                    bbox = item.get("bbox_2d")
                    if bbox and len(bbox) == 4 and point and len(point) == 2:
                        cleaned_data.append({
                            "name": item.get("name", "Unknown"),
                            "bbox_2d": self._denormalize_bbox(bbox, image.width, image.height),
                            "point_2d": self._denormalize_point(point, image.width, image.height),
                            "reason": item.get("reason", "")
                        })
                elif task_route == "line2_orientation":
                    if point and len(point) == 2:
                        cleaned_data.append({
                            "name": item.get("name", "Unknown"),
                            "point_2d": self._denormalize_point(point, image.width, image.height),
                            "anchor_parts": item.get("anchor_parts", ""),
                            "visible_profile": item.get("visible_profile", "unknown"),
                            "2d_pointing_direction": item.get("2d_pointing_direction", "unknown"),
                            "reason": item.get("reason", "")
                        })
                else: 
                    if point and len(point) == 2:
                        cleaned_data.append({
                            "name": item.get("name", "Unknown"),
                            "point_2d": self._denormalize_point(point, image.width, image.height),
                            "facing_direction": item.get("facing_direction", "Unknown"),
                            "reason": item.get("reason", "")
                        })
            diagnostics["objects_accepted"] = len(cleaned_data)
            return target_perspective, cleaned_data, response_text, diagnostics
        except Exception as e:
            diagnostics["parse_error"] = str(e)
            stripped = (response_text or "").rstrip()
            diagnostics["generation_may_be_truncated"] = bool(stripped) and not stripped.endswith("}")
            print(f"[ERROR] JSON 解析失败: {e}\nRaw Output: {response_text[:200]}")
            return "camera", [], response_text, diagnostics

    def perform_final_reasoning(self, image: Image.Image, question_text: str, options_text: str,
                                scene_graph_data: list, spatial_relations_text: str, task_route: str = "line1_base"):
        """阶段 4：纯净推理 (需要逻辑推导，开启 thinking)"""
        
        if task_route == "line1_base":
            context = "--- Auxiliary 3D Scene Graph ---\n"
            context += "* Camera is at (0, 0, 0). Coordinates are in meters.\n"
            context += "* +X is Right, +Y is Down, +Z is Forward (Depth).\n\n"
            for obj in scene_graph_data:
                pos = obj.get("global_3d_position", {})
                context += f"- Object: '{obj.get('name')}'\n"
                context += f"  Image point (pixels, origin top-left): {obj.get('point_2d')}\n"
                context += f"  Location: X={pos.get('X_meters')}, Y={pos.get('Y_meters')}, Z={pos.get('Z_meters')}\n"
                context += f"  Orientation: {obj.get('facing_direction', 'Unknown')}\n"
            
            # 🔴 关键修复：允许 Line 1 接收外部注入的先验知识
            if spatial_relations_text.strip():
                context += "\n" + spatial_relations_text.strip() + "\n"
                
            context += "--------------------------------\n"
        else:
            context = "--- Auxiliary Scene Graph ---\n"
            if task_route == "line3_bbox":
                for obj in scene_graph_data:
                    context += (
                        f"- Object: '{obj.get('name')}' | point={obj.get('point_2d')} | "
                        f"bbox={obj.get('bbox_2d')} | depth={obj.get('depth_meters')} | "
                        f"estimated_size=(w={obj.get('Physical_Width')}, h={obj.get('Physical_Height')})\n"
                    )
            elif task_route == "line2_orientation":
                for obj in scene_graph_data:
                    context += (
                        f"- Object: '{obj.get('name')}' | point={obj.get('point_2d')} | "
                        f"Profile: {obj.get('visible_profile')} | "
                        f"Points to: {obj.get('2d_pointing_direction')} | "
                        f"3D={obj.get('global_3d_position')}\n"
                    )
            
            # Line 2 和 3 会在这里同时接收“几何空间关系”和“外部先验知识”
            if spatial_relations_text.strip():
                context += "\n--- PRE-CALCULATED SPATIAL RELATIONSHIPS ---\n"
                context += spatial_relations_text.strip() + "\n"
            context += "--------------------------------\n"

        if task_route == "line1_base":
            reasoning_system_prompt = (
                "You are a spatial reasoning expert. Use the image and the provided 3D Scene Graph to answer the question.\n"
                "Pay close attention to object orientations and 3D coordinates (X,Y,Z) to resolve egocentric (camera-view) vs allocentric (object-view) relations.\n"
                "A larger Z-value means the object is farther away from the camera (more to the rear).\n"
                "For image-plane words such as left, right, above, below, top, bottom, or corner, use image points first: x grows rightward and y grows downward. Do not replace a 2D relation with camera depth.\n"
                "For distance to a named reference, compare positions relative to that reference. If it is absent, inspect the image instead of substituting distance from the camera.\n"
            )
        else:
            reasoning_system_prompt = (
                "You are an expert spatial reasoning assistant. Use the image, the visual features, and the PRE-CALCULATED SPATIAL RELATIONSHIPS (if provided) to answer the question.\n"
                "CRITICAL RULES & GUIDELINES:\n"
                "Treat pre-calculated geometry as noisy auxiliary evidence and verify it against the image and answer options.\n"
                "Objects sharing a frontal or side profile in the image are not necessarily parallel in 3D and need not share a compass direction. Use scene layout and visible structure before making that inference.\n"
                "For image-plane words such as left, right, above, below, top, bottom, or corner, use image points first: x grows rightward and y grows downward. Do not replace a 2D relation with camera depth.\n"
                "For distance to a named reference, compare positions relative to that reference. If it is absent, inspect the image instead of substituting distance from the camera.\n"
               
        )

        reasoning_system_prompt += (
            "Output a <think> block for step-by-step reasoning. Immediately after </think>, "
            "output only the exact choice number as <answer>N</answer>, where N is 1-4. "
            "Do not repeat the reasoning after </think>.\n"
        )

        user_prompt = f"{context}\nQuestion:\n{question_text}\nOptions:\n{options_text}\n\nFinal answer:"
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": reasoning_system_prompt + "\n\n" + user_prompt}]}
        ]

        return self._generate_response(messages, max_tokens=2048, enable_thinking=True)
