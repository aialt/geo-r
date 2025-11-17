from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
# from open_r1.vlm_modules.vlm_module import VLMBaseModule
from vlm_modules.vlm_module import VLMBaseModule
# from vlm_module import VLMBaseModule
from PIL import Image



############################################################################################
import json
import re
import ast

def extract_loc_from_string(text):
    try:
        data = ast.literal_eval(text)
        return data.get('LOC', None)
    except (SyntaxError, ValueError):
        return None

import re
import json
def extract_loc_from_list(s):
    # 示例函数，可以替换成你自己的解析逻辑
    match = re.search(r"\[([-\d.]+),\s*([-\d.]+)\]", s)
    return [float(match.group(1)), float(match.group(2))] if match else None


def extract_coordinates_from_json_block(text):
 
    # 清理 ```json 包裹
    cleaned = re.sub(r'```json\n|\n```|```', '', text.strip())

    # 1. 尝试解析标准 JSON
    try:
        data = json.loads(cleaned)
        for key in ["Coordinates", "LOC"]:
            if key in data and isinstance(data[key], list) and len(data[key]) == 2:
                return data[key]
    except:
        pass  # 非 JSON 格式，进入文本解析模式

    # 2. 匹配 (lat, lon)
    match_paren = re.search(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', cleaned)
    if match_paren:
        return [float(match_paren.group(1)), float(match_paren.group(2))]

    # 3. 匹配 [Lat. xx, Long. yy]
    match_labelled = re.search(
        r'\[\s*Lat\.?\s*[:=]?\s*([-\d.]+)\s*,\s*Long\.?\s*[:=]?\s*([-\d.]+)\s*\]',
        cleaned,
        re.IGNORECASE
    )
    if match_labelled:
        return [float(match_labelled.group(1)), float(match_labelled.group(2))]

    return None




from math import radians, cos, sin, asin, sqrt 

def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # 地球平均半径，单位：km
    return c * r

def distance_to_reward(coord1, coord2):
    """
    分段线性映射 reward:
    - 0km → 1.0
    - 750km → 0.5
    - 3000km → 0.2
    - 20000km → 0.0
    """
    distance = haversine_distance(coord1, coord2)
    
    if distance <= 750:
        # 映射 [0, 750] → [1.0, 0.5]
        return round(1.0 - (distance / 750) * 0.5, 4)
    elif distance <= 3000:
        # 映射 [750, 3000] → [0.5, 0.2]
        return round(0.5 - ((distance - 750) / (3000 - 750)) * 0.3, 4)
    elif distance <= 20000:
        # 映射 [3000, 20000] → [0.2, 0.0]
        return round(0.2 - ((distance - 3000) / (20000 - 3000)) * 0.2, 4)
    else:
        return 0.0
    
############################################################################################

# def distance_to_reward(coord1, coord2):
#     """
#     分段线性映射 reward:
#     - 0km    → 1.0
#     - 750km  → 0.5
#     - 3000km → 0.2
#     - 20000km → -1.0
#     其中：
#     - 0–750km   保持 [1.0 → 0.5]
#     - 750–3000km 保持 [0.5 → 0.2]
#     - 3000–20000km 线性映射 [0.0 → -1.0]
#     """
#     distance = haversine_distance(coord1, coord2)
    
#     if distance <= 750:
#         # 映射 [0, 750] → [1.0, 0.5]
#         return round(1.0 - (distance / 750) * 0.5, 4)
#     elif distance <= 3000:
#         # 映射 [750, 3000] → [0.5, 0.2]
#         return round(0.5 - ((distance - 750) / (3000 - 750)) * 0.3, 4)
#     elif distance <= 20000:
#         # 映射 [3000, 20000] → [0.0, -1.0]
#         ratio = (distance - 3000) / (20000 - 3000)
#         return round(0.0 - ratio * 1.0, 4)
#     else:
#         # 超出 20000km，固定最小值
#         return -1.0



class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                
                ############################################################################################
                # return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
                # ##############################################
                # return "{Question} You are given an image. Use visual clues to infer the most likely geographic coordinates (latitude and longitude) of the location shown in the image."
                # ##############################################
                return "{Question}"
                ############################################################################################
        
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards
    

    @staticmethod
    def loc_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json


        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'


        content_list = [] 
        for i, (content, sol) in enumerate(zip(contents, solution)):

            # image_grid_thw = kwargs.get("image_grid_thw")[i]
            # image_path = kwargs.get("image_path")[i][0]
            # image = Image.open(image_path)
            # image_width, image_height = image.size
            # input_height = int(image_grid_thw[1]*14)
            # input_width = int(image_grid_thw[2]*14)
            # sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            # sol = json.loads(sol.strip())
            # Try symbolic verification first
            # try:
            #     content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            #     if content_answer_match:
            #         content_answer = content_answer_match.group(1).strip()
            #         bbox_match = re.search(bbox_pattern, content_answer)
            #         if bbox_match:
            #             bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            #             bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
            #             # if iou(bbox, sol) > 0.5:
            #             #     reward = 1.0
            #             reward = iou(bbox, sol)
            # except Exception:
            #     pass  # Continue to next verification method if this fails

            reward = 0.0
            try:
                # print('++++++++++++++++++++++++++++++++++++++')
                # print('contents ',content)
                # print('--------------------------------------')
                # print('sol ',sol) 
                # print('++++++++++++++++++++++++++++++++++++++')

                # ############################
                # import pdb; pdb.set_trace()
                # ############################

                # answer = extract_loc_from_list(sol)
                answer = sol
                # pred = extract_loc_from_list(content)
                pred = extract_loc_from_string(content)
            
                
                print('++++++++++++++++++++++++++++++++++++++')
                print('GT-LOC', answer)
                print('PredLOC', pred)
                print('++++++++++++++++++++++++++++++++++++++')


                if pred == None:
                    content_list.append(content)
                    with open("saved_contents.txt", "w", encoding="utf-8") as f:
                        for item in content_list:
                            f.write(item + "\n\n")
                        

                reward = distance_to_reward(pred, answer)

            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 


        # print('problem: ', problem)
        print('##########################-----###########################')
        print('LOC Reward: ', rewards)
        print('##########################-----###########################')
        return rewards




    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
