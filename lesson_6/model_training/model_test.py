import json
import ollama


def classify_chunk(chunk: str):
    """Направляємо запити в модель для валідації."""
    response = ollama.chat(
        model="llama",
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
                    "condition": 0.XX
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
    try:
        tool_calls = response['message'].get('tool_calls', [])
        if tool_calls:
            arguments = tool_calls[0]['function']['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return arguments.get("ratings")
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error parsing response: {e}")
    return None


def extract_ground_truth(assistant_message):
    """Витягуємо еталонні рейтинги з assistant_message."""
    try:
        arguments = assistant_message["tool_calls"][0]["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return arguments["ratings"]
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error extracting ground truth: {e}")
        return None


def compare_ratings(predicted, ground_truth, threshold=0.4):
    """Порівнюємо вивід моделі з еталоном. Для того щоб оштрафувати нульові аутпути, сприймаємо 0 як -1."""
    correct = 0
    total = len(ground_truth)

    adjusted_predicted = {key: (-1 if value == 0 else value) for key, value in predicted.items()}
    adjusted_ground_truth = {key: (-1 if value == 0 else value) for key, value in ground_truth.items()}

    for key in adjusted_ground_truth:
        if abs(adjusted_predicted.get(key, -1) - adjusted_ground_truth[key]) <= threshold:
            correct += 1
    return correct, total


def process_test_data(test_jsonl_path):
    with open(test_jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    results = []
    total_correct = 0
    total_predictions = 0

    for i, entry in enumerate(data[:20]):
        messages = entry["messages"]
        user_message = next(msg for msg in messages if msg["role"] == "user")["content"]
        assistant_message = next(msg for msg in messages if msg["role"] == "assistant")

        ground_truth = extract_ground_truth(assistant_message)
        if not ground_truth:
            print(f"Skipping chunk {i + 1}: Unable to extract ground truth.")
            continue

        chunk = user_message.split("Chunk:")[-1].strip()

        print(f"Processing chunk {i + 1}...")
        predicted = classify_chunk(chunk)

        if predicted:
            correct, total = compare_ratings(predicted, ground_truth)
            total_correct += correct
            total_predictions += total

            differences = {key: abs(predicted.get(key, 0) - ground_truth[key]) for key in ground_truth}
            results.append({
                "chunk_index": i + 1,
                "chunk": chunk,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "differences": differences,
                "correct": correct,
                "total": total
            })
        else:
            results.append({
                "chunk_index": i + 1,
                "chunk": chunk,
                "predicted": None,
                "ground_truth": ground_truth,
                "error": "Model did not return valid predictions"
            })

    accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
    return results, accuracy


def print_results(results, accuracy):
    """Виведення чанків і оцінка точності"""
    for result in results:
        print(f"Chunk Index: {result['chunk_index']}")
        #print(f"Chunk: {result['chunk']}")
        print(f"Predicted Ratings: {result['predicted']}")
        print(f"Ground Truth Ratings: {result['ground_truth']}")
        if "differences" in result:
            print(f"Differences: {result['differences']}")
        if "error" in result:
            print(f"Error: {result['error']}")
        print("-" * 50)
    print(f"Overall Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test_jsonl_path = "../data/test.jsonl"
    results, accuracy = process_test_data(test_jsonl_path)
    print_results(results, accuracy)
