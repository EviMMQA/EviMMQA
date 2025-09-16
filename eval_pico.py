import argparse
import torch
import os
import json
import re
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IGNORE_INDEX
import transformers
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant."):
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_single_data(data, model, tokenizer, image_processor, args):
    idx = data["id"]
    image_files = data.get("image") or data.get("images")
    qs = data["conversations"][0]["value"]
    cur_prompt = args.extra_prompt + qs

    # 设置对话模式
    args.conv_mode = "qwen_1_5"
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    input_ids = preprocess_qwen([data["conversations"][0], {'from': 'gpt', 'value': None}], tokenizer, has_image=True).cuda()
    if image_files:
        if len(image_files) >0:
            images = []
            for image_file in image_files:
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                images.append(image_tensor.half().cuda())
        else:
            images = None
    else:
        images = None
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # import pdb; pdb.set_trace()
    model.to(device='cuda')

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=128,
            use_cache=True
        )

    # if args.return_gating_logit is not None:
    #     gating_logits = dict(gating_logit=[i.fea for i in fea_hooks],
    #                                     images=images,
    #                                     input_ids=input_ids,
    #                                     output_ids=output_ids)
    #     # gating_logits = dict(gating_logit=[i.fea for i in fea_hooks],
    #     #                                 images=images if images is None else images.detach().cpu(),
    #     #                                 input_ids=input_ids.detach().cpu(),
    #     #                                 output_ids=output_ids.detach().cpu())
    #     print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, torch.cat(images, dim=0).shape if images is not None else [])
    #     # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
    #     print('The number of hooks is:', len(fea_hooks))

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()

    # return outputs, data['conversations'][1]['value'], gating_logits
    return outputs, data['conversations'][1]['value']


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-qwen-7b-ft-lora-combine")
    # parser.add_argument("--model-path", type=str, default="checkpoints/llava-next-interleave-qwen-0.5b")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--model-base", type=str, default="checkpoints/llava-qwen-7b-ft-lora-combine")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument('--input_file', type=str, default='outputs_prepare_rag/test_rag_llava_6144.json')
    parser.add_argument("--image-folder", type=str, default="dataset/articles")
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--return_gating_logit', type=str, default='pico')
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.start == 0:
        output_file = os.path.join(args.output_folder, f"{model_name}.jsonl")
    else:
        output_file = os.path.join(args.output_folder, f"{model_name}_{str(args.start)}.jsonl")
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            already_processed = sum(1 for _ in f)  
    else:
        already_processed = 0
    
    data = read_json(args.input_file)

    new_data = data[already_processed + args.start:]


    batch_results = []
    for idx, item in enumerate(tqdm(new_data)):
        response, answer = eval_single_data(item, model, tokenizer, image_processor, args) 
        item.pop("conversations", None)  
        item.pop("images", None)  
        item.pop("image", None) 
        
        item['predict'] = response[0]
        item['answer'] = answer
        batch_results.append(item)

        if len(batch_results) == args.batch_size:
            with open(output_file, 'a') as f:
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            batch_results = []  

    if batch_results:
        with open(output_file, 'a') as f:
            for result in batch_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

