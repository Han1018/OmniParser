import os
import re
import ast
import sys
import pdb
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch.distributed as dist
from accelerate.utils import gather_object
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from utils.utils import save_json
from prompt import CN_FIND_BBOX_FROM_LIST_PROMPT, GET_OUTPUT_PROMPT, CN_FIND_BBOX_FROM_LIST_IMAGE_PROMPT, \
    CN_FIND_BBOX_FROM_LIST_IMAGE_PROMPT_UPDATE, PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT
import ollama
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import matplotlib.pyplot as plt
import io
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

import logging
logging.basicConfig(level=logging.INFO)
# import pdb

# OmniParser
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
import importlib
import utils
importlib.reload(utils)

device = 'cuda'
model_path='weights/icon_detect/model.pt'

som_model = get_yolo_model(model_path)
som_model.to(device)
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)

def broadcast_value(value, src=0, local_rank=0):
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()

def get_bbox(bbox, img_size, xy_int):
    x1, y1, w, h = bbox
    weight, height = img_size

    # x1y1wh to x1y1x2y2
    bbox = [x1, y1, x1 + w, y1 + h]

    # normalisation
    bbox = [bbox[0] / weight, bbox[1] / height, 
            bbox[2] / weight, bbox[3] / height]
    if xy_int:
        bbox = [int(item * 1000) for item in bbox]
    return bbox

def pointinbbox(pred_point, gt_bbox):
    # pred_point: [x, y] in [0, 1]
    # gt_bbox: [x1, y1, x2, y2] in [0, 1]
    if (gt_bbox[0] <= pred_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= pred_point[1] <= gt_bbox[3]):
        return True
    else:
        return False

def draw_point_bbox(image_path, point=None, bbox=None, radius=5, line=3):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    if point is not None:
        x, y = point[0] * width, point[1] * height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue', outline='blue')
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line)

    image_draw = np.array(image)
    return image_draw

def calculate_screenspot_metrics(results):
    metrics = {}
    for type in results:
        num_step = 0
        num_success = 0

        for step in results[type]:
            num_step += 1
            num_success += step["acc"]

        metrics[f"{type} Success Rate"] = num_success / num_step

    for key, value in metrics.items():
        print(f"[{key}]: {value}")
    return metrics

def load_validation_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def filter_items(items, filter_type):
    """
    過濾輸入的 list of items，只取出 item's type 等於 filter_type 的 item.
    """
    return [item for item in items if item.get("type") == filter_type]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_find_bbox_input(prompt_origin, base64_image, dino_labled_img):
    messages = [
        {
            "role": "developer",
            "content": [
                # {"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}
                {"type": "text", "text": '''You are an expert at completing instructions on GUI screens. 
    You will be presented with two images. The first is the original screenshot. The second is the same screenshot with some numeric tags. You will also be provided with some descriptions of the bbox, and your task is to choose the numeric bbox idx you want to click in order to complete the user instruction.'''}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": prompt_origin

                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{dino_labled_img}",
                    }
                },
            ],
        }
    ]
    return messages

def get_extract_bbox_input(bbox_candidate : str):
    messages = [
        {'role': 'system', 'content': GET_OUTPUT_PROMPT},
        {'role': 'user', 'content': bbox_candidate}
    ]
    return messages
    
def extract_bbox(code_str):
    """
    將包含 markdown code block (```json ... ```) 的字串轉成 json object,
    如果有例外發生則回傳空 json (empty dict).
    """
    try:
        # 找出 code block 中的內容
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, code_str, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
    except Exception:
        print("字串不符合預期的 markdown code block 格式.")
        pass
    return {}

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox: [x1, y1, x2, y2]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [round(x,4) for x in point]
    return point

def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def reformat_messages(parsed_content_list):
    screen_info = ""
    for idx, element in enumerate(parsed_content_list):
        element['idx'] = idx
        if element['type'] == 'text':
            screen_info += f'''<p id={idx} class="text" alt="{element['content']}"> </p>\n'''
            # screen_info += f'ID: {idx}, Text: {element["content"]}\n'
        elif element['type'] == 'icon':
            screen_info += f'''<img id={idx} class="icon" alt="{element['content']}"> </img>\n'''
            # screen_info += f'ID: {idx}, Icon: {element["content"]}\n'
    return screen_info

@torch.no_grad()
def validate_screenspot(name, val_file):
    answers_unique = []
    generated_texts_unique = []
    outputs_unique = []

    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    val_data = load_validation_data(val_file)
    metric = 0
    
    # if use filter 0
    if name in ['filter_0']:
        val_data = val_data['desktop']['icon'] + val_data['mobile']['icon'] + val_data['web']['icon']
    
    for i, item in enumerate(tqdm(val_data)):
        torch.cuda.empty_cache()
        
        # get image
        if name in ['filter_0']:
            image_path = item['img_path']
        else:
            image_path = "/home/han/Documents/repos/ShowUI/datasets/ScreenSpot/images/" + item['img_url']
        # image_path = item['img_path'] # for filter 0
        image = Image.open(image_path).convert('RGB')
        base64_image = convert_pil_image_to_base64(image)
        
        if name in ['filter_0']:
            meta = item['meta']
        else:
            meta = item
        # meta = item
        # meta = item['meta'] # for filter 0  
        
            
        # config
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        BOX_TRESHOLD = 0.05
        try:
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.5}, use_paddleocr=True)
            text, ocr_bbox = ocr_bbox_rslt

            dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)
            
            screen_info = reformat_messages(parsed_content_list)
            prompt_origin = PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT.format(meta['task'], screen_info)
            
            # # filter parsed_content_list by icon/text
            # parsed_content_list = filter_items(parsed_content_list, item['data_type']) 
            # parsed_content_list_str = json.dumps(parsed_content_list, ensure_ascii=False, indent=4)
            
            # find similar bbox
            messages = get_find_bbox_input(prompt_origin, base64_image, dino_labled_img)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            response_text = completion.choices[0].message.content
            
                
            try:
                response_text = ast.literal_eval(response_text)
                
                icon_id = response_text['Click BBox ID']
                analysis = response_text['Analysis']
                bbox_caption = parsed_content_list[int(icon_id)].get('content', None)
                bbox = label_coordinates[str(icon_id)]
                click_point = str([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
            except:
                bbox_caption = 'error'
                analysis = "An error occurred"
                click_point = "[0, 0]"
                print('error parsing, use regex to parse!!!')

            # pdb.set_trace() 

        except Exception as e:
            analysis = "An error occurred"
            bbox_caption = 'error'
            click_point = "[0, 0]"
            print(f"An error occurred: {e}")
        
        # show
        print(f"--Instruction: {meta['task']}")
        print(f"--Answer: {bbox_caption}")
        # pdb.set_trace()

        outputs = {"split": meta['split'], 'data_type': meta['data_type'],
                    "anno_id": meta['id'], "img_path": image_path, "instruction": meta['task'], "sentence": click_point, "Analysis" : analysis, "bbox caption": bbox_caption,
                    "bbox": meta['bbox'], 
                    "meta": meta}
        generated_texts_unique.append(click_point)
        answers_unique.append(meta['bbox'])
        outputs_unique.append(outputs)
        
        # if i % 3 == 2:
        #     break

    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)
    outputs_unique = gather_object(outputs_unique)

    results = {}
    for pred_i, ans_i, output_i in tqdm(zip(generated_texts_unique, answers_unique, outputs_unique)):
        anno_id = output_i['anno_id']
        split_i = output_i['split']
        if split_i not in results:
            results[split_i] = {}

        type_i = output_i['data_type']
        if type_i not in results[split_i]:
            results[split_i][type_i] = []

        step_result = output_i.copy()

        img_size = output_i['meta']['img_size']
        gt_bbox = get_bbox(ans_i, img_size, False)
        step_result['gt_bbox'] = gt_bbox

        try:
            pred_point = ast.literal_eval(pred_i)
            step_result['pred_point'] = pred_point

            if pointinbbox(pred_point, gt_bbox):
                step_result["acc"] = 1
            else:
                step_result["acc"] = 0
                
        except Exception as e:
            print(e)
            print(f"format wrong with {anno_id}'s prediction: {pred_i}")
            step_result["acc"] = 0

        results[split_i][type_i].append(step_result)

    eval_dict = {}
    for split in results.keys():
        print("==="*10)
        print(f"{split}")
        print("==="*10)
        eval_dict[split] = calculate_screenspot_metrics(results[split])

    score_all = [value for split in eval_dict.values() for value in split.values()]
    metric = sum(score_all) / len(score_all)
    eval_dict['Avg Success Rate'] = metric

    save_json(results, os.path.join("./", f'{name}_screenspot_omniparser_tmp_dict.json'))
    save_json(eval_dict, os.path.join("./", f'{name}_screenspot_omniparser_res_dict.json'))

    # metric = broadcast_value(metric, src=0, local_rank=local_rank)
    return metric

if __name__ == '__main__':
    # validate_screenspot('/home/han/Documents/repos/ShowUI/datasets/ScreenSpot/metadata/hf_test_full.json')
    validate_screenspot(name = 'filter_0', val_file='./screenspot_omniparser_tmp_dict_filter_0_ori.json')
    validate_screenspot(name = 'full_screenspot', val_file='/home/han/Documents/repos/ShowUI/datasets/ScreenSpot/metadata/hf_test_full.json')