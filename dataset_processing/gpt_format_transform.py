from template import template
import json
import glob
import os
import re
import csv
import tiktoken
import pandas as pd
import string


all_files_position = 'articles'

def load_txt(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return ""

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_text_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def concatenate_sections(csv_file, section):
    result = []
    patterns = {
        'method': re.compile(r'main text|method', re.IGNORECASE),
        'result': re.compile(r'result', re.IGNORECASE)
    }
    pattern = patterns.get(section, None)

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            if pattern:
                if pattern.search(row[0]):
                    text = row[3].replace('\n', ' ')
                    concatenated = f"{row[1]} {row[2]} \n{text} \n"
                    result.append(concatenated)
            else:
                text = row[3].replace('\n', ' ')
                concatenated = f"{row[0]} {row[1]} {row[2]} \n{text} \n"
                result.append(concatenated)
    
    return ''.join(result)

def get_context(context_folder):
    name = context_folder.split('_')[-1]
    context_file = f'{context_folder}/{name}.csv'
    context = concatenate_sections(context_file, 'all')
    return context


def process_csv_files(file_path):
    result_dict = {}
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        key = row["Id"]
        caption = row['Caption'] if pd.notna(row['Caption']) else ''
        supplementary = row['Supplementary'] if pd.notna(row['Supplementary']) else ''
        if caption and not caption.endswith(tuple(string.punctuation)):
            caption = f'{caption}.'
        if supplementary and not supplementary.endswith(tuple(string.punctuation)):
            supplementary = f'{supplementary}.'
        value = f"{caption} {supplementary}".strip()
        result_dict[key] = value
    return result_dict

def read_table_csv(file_path):
    df = pd.read_csv(file_path)
    csv_str = df.to_string(index=False)  
    return csv_str


def get_image_context(images_folder, images):
    jpg_files_dict = {}
    for id, image in images.items():
        image_path = f'{images_folder}/{id}.jpg'
        pseudo_image_path = f'{images_folder}/pseudo_{id}.jpg'
        content_csv_path = f'{images_folder}/{id}.csv'
        content_txt_path = f'{images_folder}/{id}.txt'

        if os.path.exists(pseudo_image_path):
            jpg_files_dict[id] = {"name": f'pseudo_{id}.jpg', "position": pseudo_image_path, "caption": image}
        else:
            jpg_files_dict[id] = {"name": f'{id}.jpg', "position": image_path, "caption": image}

        if os.path.exists(content_csv_path):
            jpg_files_dict[id]["content"] = read_table_csv(content_csv_path)
        elif os.path.exists(content_txt_path):
            jpg_files_dict[id]["content"] = load_txt(content_txt_path)
        else:
            jpg_files_dict[id]["content"] = ""

    return jpg_files_dict

def get_images(images_folder):
    csv_file = [file_name for file_name in os.listdir(images_folder) if file_name.endswith("_table.csv")][0]
    images = process_csv_files(f'{images_folder}/{csv_file}')
    import pdb; pdb.set_trace() 
    jpg_files_dict = get_image_context(images_folder, images)
    return jpg_files_dict

def gpt_format_transform(data):
    for item in data:
        context_folder = f"{all_files_position}/{item['context']}"
        context = get_context(context_folder)
        jpg_files_dict = get_images(context_folder)
        
        images = []
        table_all = ""
        chart_all = ""
        for id, id_content in jpg_files_dict.items():
            if 'table' in id:
                table_all += f"[{id_content['name']}] <image>\nCaption: {id_content['caption']}\nExtracted Content: {id_content['content']}\n"
            else:
                chart_all += f"[{id_content['name']}] <image>\nCaption: {id_content['caption']}\nExtracted Content: {id_content['content']}\n"
            images.append(id_content['position'])

        option_str = ""
        for option in item['options']:
            option_str += f"{option}\n"

        item['conversations'] = [
            {
                "from": "human",
                "value": template.format(context=context, tables=table_all, charts=chart_all, question=item['question'], options=option_str)
            },  
            {
                "from": "gpt",
                "value": item['answer']
            }
        ]
        item['images'] = images
    return data

if __name__ == "__main__":
    data = load_json("test.json")
    data = gpt_format_transform(data)
    save_json(data, "test_gpt_format.json")