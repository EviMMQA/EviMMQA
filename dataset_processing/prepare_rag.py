import os
import re
import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import textwrap
def chunk_text_by_punctuation(text, max_length=128):
    pattern = r'(?<=[.!?])\s+|\n'

    sentences = re.split(pattern, text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        len_current_chunk = len(tokenizer(current_chunk).input_ids)
        len_sentence = len(tokenizer(sentence).input_ids)
        if len_current_chunk + len_sentence <= max_length:  # +1是考虑空格
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def clean_context(context):
    texts = context.split('<image>')
    if texts:
        texts_splitted = chunk_text_by_punctuation(texts[0])

    if len(texts) > 1:
        for t in texts[1:]:
            if t:
                texts_splitted.append(f'<image>{t}')
    return texts_splitted


def get_images_id(texts_splitted):
    images_id = {}
    id = 0
    for i,text in enumerate(texts_splitted):
        if '<image>' in text:
            images_id[i] = id
            id += 1
    return images_id


def get_text_features(model, tokenizer, question, texts_splitted):
    prefix = 'summarize:'
    question = prefix + question
    texts_splitted = [prefix + t for t in texts_splitted]
    input_ids = tokenizer(
        [question] + texts_splitted,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length',
    ).input_ids.cuda()

    text_features_list = []
    for i in input_ids:
        text_outputs = model.model.embed_tokens(i.unsqueeze(0))
        text_features = text_outputs.mean(dim=1)  
        text_features_list.append(text_features)
    text_features = torch.cat(text_features_list)
    return text_features


def count_lenth(tokenizer, text):
    text_length = len(tokenizer(text).input_ids)
    if '<image>' in text:
        text_length += 256
    return text_length

def get_idx_list(tokenizer, sorted_idx, texts_splitted, max_length):
    max_id = sorted_idx[0]
    context_length = count_lenth(tokenizer, texts_splitted[max_id])
    idx_list =[max_id]

    for idx in sorted_idx[1:]:
        new_context = texts_splitted[idx]
        new_context_length = count_lenth(tokenizer, new_context)
        if context_length + new_context_length <= max_length:
            context_length = context_length + new_context_length
            idx_list.append(idx)
        else:
            break
    idx_list_sorted = sorted(idx_list)
    return idx_list_sorted


def get_new_context(model, tokenizer, question, texts_splitted, images_list, max_length):
    text_features = get_text_features(model, tokenizer, question, texts_splitted)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    logits_per_text = text_features[:1] @ text_features[1:].t()
    text_features = text_features.to('cuda:0')

    texts_sim = logits_per_text[0]
    sorted_idx = torch.argsort(texts_sim, descending=True).tolist()
    idx_list = get_idx_list(tokenizer, sorted_idx, texts_splitted, max_length)
    
    new_image = []
    new_context = ''
    images_id = get_images_id(texts_splitted)
    for idx in idx_list:
        new_context += texts_splitted[idx] + ' '
        if '<image>' in new_context:
            new_image_id = images_id[idx]
            new_image.append(images_list[new_image_id])
    return new_context, new_image




if __name__ == "__main__":
    model_path = "./checkpoints/llava-v1.5-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )


    min_num_images = 0
    max_context_length = 64
        
    FILEPATH = 'test_gpt_format.json'
    with open(FILEPATH) as file:
        lines = json.load(file)

    batch_size = 100  
    batch_samples = []
    outputs = []
    max_length = 128

    for max_length in [5120, 6144]:
        ans_file_path = f'./outputs_prepare_rag/rag_llava_{max_length}.json'

        for idx, sample in enumerate(tqdm(lines)):
            import pdb; pdb.set_trace() 
            batch_samples.append(sample)
            
            if len(batch_samples) == batch_size or idx == len(lines) - 1:
                batch_contexts = [s['context'].replace('</s>', '') for s in batch_samples]
                batch_questions = [s['conversations'][0]['value'] for s in batch_samples]
                batch_images_lists = [s['images'].copy() for s in batch_samples]

                batch_texts_splitted = [clean_context(context) for context in batch_contexts]
                
                batch_new_context = []
                batch_new_images = []
                
                for i in range(len(batch_samples)):
                    question = batch_questions[i]
                    texts_splitted = batch_texts_splitted[i]
                    images_list = batch_images_lists[i]
                    new_context, new_image = get_new_context(model, tokenizer, question, texts_splitted, images_list, max_length)
                    batch_new_context.append(new_context)
                    batch_new_images.append(new_image)
                
                for i, sample in enumerate(batch_samples):
                    sample_with_rag = sample.copy()
                    sample_with_rag['conversations'][0]['value'] = batch_new_context[i]
                    sample_with_rag['images'] = batch_new_images[i]
                    outputs.append(json.dumps(sample_with_rag) + "\n")
                
                with open(ans_file_path, 'a') as file:
                    file.writelines(outputs)
                batch_samples = []
                outputs = []