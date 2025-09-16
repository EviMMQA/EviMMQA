import re
import json
import os
import pandas as pd
import argparse

def extract_answer(text):
    text = text.replace('*', '').replace('[', '').replace('"', '').replace('\n', ' ')
    if len(text) == 1:
        return text
    result = ''
    match1 = re.search(r"answer is: ([A-D])| answer: \s*([A-D]) |answer is:\s*([A-D])|Answer: ([A-D])|Answer:([A-D])|The answer is ([A-D])|Answers: ([A-D])", text, re.IGNORECASE)
    match2 = re.search(r'([A-D])\.| ([A-D]) \.|([A-D])\)|([A-D])\:|([A-D]) ', text)

    if match1:
        result = match1.group(1) or match1.group(2) or match1.group(3) or match1.group(4) or match1.group(5) or match1.group(6) or match1.group(7)
    elif match2:
        result = match2.group(1)
    return result


def read_jsonl(jsonl_file):
    df = pd.read_json(jsonl_file, lines=True)
    data_list = []
    
    for i, data in df.iterrows():
        sample_id = data["ID"]
        
        if "answer" in data and "output" in data:
            answer = data["answer"].strip()
            response = extract_answer(data["output"].strip())
        else:
            answer = data["answer"].strip()
            response = extract_answer(data["response"].strip())
        
        
        data_list.append((sample_id, answer, response))
    
    return data_list

def compute_accuracy(data_list, filtered_ids=None):
    data_dict = {id_: (answer, response) for id_, answer, response in data_list}
    
    if filtered_ids is None:
        correct = sum(1 for _, (answer, response) in data_dict.items() if answer == response)
        total = len(data_dict)
        return round(correct / total, 2) if total > 0 else 0

    filtered_data = {id_: (answer, response) for id_, (answer, response) in data_dict.items() if id_ in filtered_ids}
    correct_filtered = sum(1 for _, (answer, response) in filtered_data.items() if answer == response)
    total_filtered = len(filtered_data)
    
    return round(correct_filtered / total_filtered, 2) if total_filtered > 0 else 0

import pandas as pd

def get_file_acc(file):
    all_results = []    
    data_list = read_jsonl(file)

    test_count = [len(data_list)]
    for count in test_count:
        results = {"file": file}
        results["count"] = count
        if count > len(data_list):
            continue
        
        test_data_list = data_list[:count]
        results["total"] = compute_accuracy(test_data_list)
        
        for type, ids in pico_ids.items():
            results[f"{type}"] = compute_accuracy(test_data_list, filtered_ids=ids)
        results["filtered"] = compute_accuracy(test_data_list, filtered_ids=hallucination_ids)
        results["not_filtered"] = compute_accuracy(test_data_list, filtered_ids=not_hallucination_ids)

        all_results.append(results)
    return all_results

def process_path(path, output_csv="results.csv"):
    all_results = []

    if os.path.isdir(path): 
        jsonl_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl")]
        for file in jsonl_files:
            print(f"Processing: {file}")
            results = get_file_acc(file)
            all_results.extend(results)
            print(f"Processed: {file}")
    
    elif os.path.isfile(path) and path.endswith(".jsonl"):  # 处理单个文件
        results = get_file_acc(path)
        all_results.extend(results)
        print(f"Processed: {path}")

    else:
        return

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"all results saved to {output_csv}")

def load_ids_from_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input path from command line.")
    parser.add_argument(
        "--input_path",default="./results/test"
    )
    
    args = parser.parse_args()
    input_path = args.input_path
    pico_ids = load_ids_from_file("./dataset/test_categorized.json")
    process_path(input_path, output_csv=f"{input_path}/results_report.csv")
    print(f"Processed: {input_path}")
    