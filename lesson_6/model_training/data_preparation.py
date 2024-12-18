import json
from transformers import AutoTokenizer

json_file = '../output_data.json'
train_file_path = '../data/train.jsonl'
test_file_path = '../data/test.jsonl'
val_file_path = '../data/valid.jsonl'

model_id = "meta-llama/Llama-3.1-8B-Instruct"

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

# Инструкции и описание функции
hard_instructions = """You are a document analyzer. Classify the following chunk of text into the following classes with a rating for each class:
- specification: Contains technical specifications from the manufacturer.
- tires: Contains information about the tires on and in the car.
- condition: Describes the condition of the car, its current damages, repaired damages, or paint thickness measurements.

Provide a rating between 0 and 1 for each class, such that the sum of all ratings equals 1."""

tool_description = {
    "type": "function",
    "function": {
        "name": "classify_doc",
        "description": "Provide a rating between 0 and 1 for each class, such that the sum of all ratings equals 1.",
        "parameters": {
            "type": "object",
            "properties": {
                "ratings": {
                    "type": "dict",
                    "description": "Class of contents of the provided document",
                    "enum": [
                        "specification",
                        "tires",
                        "condition"
                    ]
                }
            },
            "required": ["ratings"]
        }
    }
}

# Делим данные на train и test
split_index = int(0.8 * len(data))  # 80% для train
split_index_2 = split_index + (len(data) - split_index) // 2
train_data = data[:split_index]
test_data = data[split_index:split_index_2]
val_data = data[split_index_2:]

def convert_to_format(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            chunk = item.get('chunk')
            ratings = item.get('validation')

            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{hard_instructions}\n\nChunk: {chunk}"
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_id",
                                "type": "function",
                                "function": {
                                    "name": "classify_doc",
                                    "arguments": json.dumps({"ratings": ratings})
                                }
                            }
                        ]
                    }
                ],
                "tools": [tool_description]
            }

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

convert_to_format(train_data, train_file_path)
convert_to_format(test_data, test_file_path)
convert_to_format(val_data, val_file_path)
