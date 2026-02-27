import json
import random
from collections import defaultdict

json_path = "train.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

context_to_items = defaultdict(list)

for item in data:
    context_to_items[item["context"]].append(item)

all_contexts = list(context_to_items.keys())
print(f"Total unique contexts: {len(all_contexts)}")

random.seed(42)
random.shuffle(all_contexts)

split_index = int(len(all_contexts) * 6 / 7)

train_contexts = set(all_contexts[:split_index])
val_contexts = set(all_contexts[split_index:])

train_data = []
val_data = []

for context, items in context_to_items.items():
    if context in train_contexts:
        train_data.extend(items)
    else:
        val_data.extend(items)

def get_stats(dataset):
    unique_contexts = set(item["context"] for item in dataset)
    unique_ids = set(item["ID"] for item in dataset)
    return len(unique_contexts), len(unique_ids)

train_context_count, train_id_count = get_stats(train_data)
val_context_count, val_id_count = get_stats(val_data)

with open("train_split.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("val_split.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print("Split completed and saved.")
