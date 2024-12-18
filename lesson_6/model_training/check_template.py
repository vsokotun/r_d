import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

json_file = '../output_data.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id,
                                            torch_dtype=torch.bfloat16,
                                            device_map="auto")

item = data[0]

# Извлекаем нужные поля
chunk = item.get("chunk")
ratings = item.get("validation")

tools = [{
            'type': 'function',
            'function': {
                'name': 'classify_doc',
                'description': 'Provide a rating between 0 and 1 for each class, such that the sum of all ratings equals 1.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'ratings': {
                            'type': 'dict',
                            'description': 'Class of contents of the provided document',
                            'enum': ['specification', 'tires', 'condition']
                        }
                    },
                    'required': ['ratings'],
                },
            },
        }]

prompt=[{
            "role": "user",
            "content": f"You are a document analyzer. Classify the following chunk of text into the following classes with a rating for each class:"
                       f"- specification: Contains technical specifications from the manufacturer."
                       f"- tires: Contains information about the tires on and in the car."
                       f"- condition: Describes the condition of the car, its current damages, repaired damages, or paint thickness measurements."
                       f"/nProvide a rating between 0 and 1 for each class, such that the sum of all ratings equals 1."
                       f"Chunk:"
                       f"{chunk}"
        }]


tokenized_sequence = tokenizer.apply_chat_template(
    prompt,
    tools=tools,
    return_tensors="pt",
    add_generation_prompt=True,
    tokenize=False)

tokenized_chat = tokenizer.apply_chat_template(
    prompt,
    tools=tools,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    tokenize=True)

tokenized_chat.to(model.device)

out = model.generate(**tokenized_chat, max_new_tokens=128)

# print(out)
generated_text = out[0, tokenized_chat['input_ids'].shape[0]:]

print(tokenizer.decode(generated_text))

"""Результат: 
<|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

{
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
            "required": [
                "ratings"
            ]
        }
    }
}

You are a document analyzer. Classify the following chunk of text into the following classes with a rating for each class:- specification: Contains technical specifications from the manufacturer.- tires: Contains information about the tires on and in the car.- condition: Describes the condition of the car, its current damages, repaired damages, or paint thickness measurements./nProvide a rating between 0 and 1 for each class, such that the sum of all ratings equals 1.Chunk:# VOLKSWAGEN PASSAT

### PASSAT VARIANT

FIN **WVWZZZ3CZKE115975**

Kilometerstand **157.847 km**

Leistung **-**

Erstzulassungsdatum **16.07.2019**<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|python_tag|>{"name": "classify_doc", "parameters": {"ratings": {"specification": 0.0, "tires": 0.0, "condition": 1.0}}}<|eom_id|>

"""