import json
import os
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

with open("output_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

class RatingsValidation(BaseModel):
    condition: float
    specification: float
    tires: float

def check_ratings(chunk, ratings, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": """You are an assistant that evaluates the accuracy of ratings based on provided text chunks.
                    The chunk contains a part of an auto-inspection document.
                    It was divided up and appraised. 
                    On a scale of 0 to 1, it has been given a score that indicates to which category this part of the document belongs.
                    The sum of the scores should be 1.
                    The categories are: 
                    **specification** - information about the basic parameters of the car from the manufacturer. For example, engine displacement, color, interior.
                    **condition** - information about the condition of the car, existing damage, repaired damage, signs of accidents, paint thickness measurements.
                    **tires** - information about wheels, disks, tires.
                    Chunk can refer to several parts of the document simultaneously. In this case it receives equal evaluation.
                    """
                },
                {
                    "role": "user",
                    "content": (
                        f"Text: {chunk}\n\n"
                        f"Current ratings:\n"
                        f"Condition: {ratings['condition']}\n"
                        f"Specification: {ratings['specification']}\n"
                        f"Tires: {ratings['tires']}\n\n"
                        "Evaluate the accuracy of these ratings. Provide a corrected version if necessary. "
                    )
                }
            ],
            response_format=RatingsValidation
        )
        print(response.choices[0].message.parsed)
        return response.choices[0].message.parsed.model_dump()
    except Exception as e:
        return {"error": str(e)}


for item in data:
    if not item.get("validated", False):
        chunk = item["chunk"]
        ratings = item["response"][0]["function"]["arguments"]["ratings"]
        validation_result = check_ratings(chunk, ratings)

        item["validation"] = validation_result
        item["validated"] = True

        result = {
            "chunk": chunk,
            "ratings": ratings,
            "validation": validation_result
        }

        # Обновление файла после обработки каждого элемента
        with open("validated_ratings.json", "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=4, ensure_ascii=False)

        # Обновление исходного файла
        with open("output_data.json", "w", encoding="utf-8") as input_file:
            json.dump(data, input_file, indent=4, ensure_ascii=False)