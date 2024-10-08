import time
import logging
from jinja2 import Template
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

inputs = [
    {"id": 1, "question": "What is the capital of France?", "synthetic_answer": "Lyon"},
    {"id": 2, "question": "Who wrote 'To Kill a Mockingbird'?", "synthetic_answer": "Harper Lee"},
    {"id": 3, "question": "What is the chemical symbol for gold?", "synthetic_answer": "Au"},
    {"id": 4, "question": "In what year did World War II end?", "synthetic_answer": "1939"},
    {"id": 5, "question": "What is the largest planet in our solar system?", "synthetic_answer": "Jupiter"},
    {"id": 6, "question": "What is the capital of Argentina?", "synthetic_answer": "Buenos Aires"}
] * 20

prompt_template = Template("""
<TASK_DESCRIPTION>
You are an AI assistant tasked with evaluating the quality of an answer to a given question.
Please assess whether the provided answer is likely to be correct based on your knowledge.
</TASK_DESCRIPTION>

<OUTPUT_FORMAT>
Respond with a JSON object containing the following fields:
   - is_likely_correct: An integer (0 or 1) indicating whether the given answer is likely to be correct (1) or incorrect (0)
   - explanation: A brief explanation of why you believe the answer is likely correct or incorrect
</OUTPUT_FORMAT>

<EXAMPLE_RESPONSE>
Example response:
{
  "is_likely_correct": 1,
  "explanation": "The answer 'Paris' is likely correct for the question 'What is the capital of France?'. Paris is widely known as the capital city of France."
}
</EXAMPLE_RESPONSE>
""")

client = OpenAI()

def inference_sync(input: str, model: str = "gpt-4o-mini") -> Dict:
    logging.info(f"Processing input: {input['id']}")
    messages = [
        {"role": "system", "content": prompt_template.render()},
        {
            "role": "user",
            "content": Template(
                "Given the question: {{ question }} and the answer: {{ answer }}. "
                "Please assess if the answer is likely correct or not."
            ).render(question=input["question"], answer=input["synthetic_answer"])
        },
    ]
    
    response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=350,
            response_format={ 
                "type": "json_object"
            },
            messages=messages,
        )

    return response.choices[0].message.content.strip()

def sync_main():
    logging.info("Starting synchronous processing")
    results = []
    for input in inputs:
        result = inference_sync(input)
        results.append(result)
        logging.info(f"Processed input {input['id']}")
    logging.info("Finished synchronous processing")
    return results

# 30 inputs
#2024-09-09 13:49:16,135 - INFO - Finished synchronous processing
#2024-09-09 13:49:16,135 - INFO - Sync execution time: 39.87 seconds

# 120 inputs
#2024-09-10 10:17:41,559 - INFO - Sync execution time: 151.47 seconds

if __name__ == "__main__":
    logging.info("Script started")

    # Run sync version
    start_time = time.time()
    results = sync_main()
    sync_time = time.time() - start_time
    
    logging.info(f"Sync execution time: {sync_time:.2f} seconds")
    # logging.info("Sync results:")
    # for i, result in enumerate(results, 1):
    #     logging.info(f"Result {i}: {result}")

    logging.info("Script finished")