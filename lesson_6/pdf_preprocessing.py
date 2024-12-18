import os
import re
import json
import ollama

def pdf_to_markdown(pdf_path):
    """Приводимо текст з PDF до Markdown."""
    import pymupdf4llm
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text

def split_md_file(content: str) -> list[str]:
    """Розбиваємо на частини по заголоку другого рівня."""
    parts = re.split(r'(?<=\n)(?=## )', content.strip())
    return [p.strip() for p in parts if p.strip()]

def classify_chunk(chunk: str):
    """Використовуючи ollama формуємо запит до моделі з викликом функції"""
    response = ollama.chat(
        model='llama3.1:8b-instruct-q8_0',
        messages=[{
            "role": "user",
            "content": f"""
            You are a document analyzer. Classify the following chunk of text into the following classes with a rating for each class:
            - specification: Contains technical specifications from the manufacturer.
            - tires: Contains information about the tires on and in the car.
            - condition: Describes the condition of the car, its current damages, repaired damages, or paint thickness measurements.

            Provide a rating between 0 and 1 for each class, such that the sum of all ratings equals 1.

            Chunk:
            {chunk}

            Provide your answer as a JSON object in this format:
            {{
                "ratings": {{
                    "specification": 0.XX,
                    "tires": 0.XX,
                    "condition": 0.XX,
                }}
            }}
            """
        }],
        tools=[{
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
        }],
        stream=False,
        options={"temperature": 0.2}
    )
    return response

def classify_markdown_file(markdown_file_path: str, output_json: str):
    """Функція для виклику моделі одного Markdown-файлу і запису результату в JSON"""
    results = []

    with open(markdown_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        chunks = split_md_file(text)

        for index, chunk in enumerate(chunks):
            print(f"Processing chunk {index + 1} of {len(chunks)} from file: {markdown_file_path}")
            answer = classify_chunk(chunk)

            try:
                if 'tool_calls' in answer['message']:
                    ratings = answer['message']['tool_calls'][0]['function']['arguments']['ratings']

                    results.append({
                        "file": os.path.basename(markdown_file_path),
                        "chunk_index": index + 1,
                        "chunk": chunk,
                        "response": answer,
                        "ratings": ratings
                    })
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error extracting ratings: {e}")
                print(f"Response: {answer}")
                results.append({
                    "file": os.path.basename(markdown_file_path),
                    "chunk_index": index + 1,
                    "chunk": chunk,
                    "response": answer,
                    "error": str(e)
                })

    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as json_file:
            existing_results = json.load(json_file)
    else:
        existing_results = []

    existing_results.extend(results)

    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(existing_results, json_file, ensure_ascii=False, indent=4)

def get_processed_files(output_json):
    """Тут ми перевіряємо, чи не був документ ще опрацьований, щоб уникнути дублікатів."""
    if not os.path.exists(output_json):
        return set()
    with open(output_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {entry["file"] for entry in data}


if __name__ == "__main__":
    pdf_folder = "pdf"
    output_folder = "mds"
    output_json = "classification_results.json"

    os.makedirs(output_folder, exist_ok=True)

    processed_files = get_processed_files(output_json)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            markdown_filename = f"{os.path.splitext(pdf_file)[0]}.txt"
            if markdown_filename in processed_files:
                print(f"Skipping already processed file: {pdf_file}")
                continue

            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing PDF file: {pdf_file}")

            markdown_text = pdf_to_markdown(pdf_path)

            markdown_file_path = os.path.join(output_folder, markdown_filename)
            if not os.path.exists(markdown_file_path):
                with open(markdown_file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)

            classify_markdown_file(markdown_file_path, output_json)

    print(f"Classification completed for all PDFs. Results saved in {output_json}")

