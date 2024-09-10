import time
import logging
from jinja2 import Template
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import Dict
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Configure logging. To minimize overhead, set the logging level to WARNING.
# This will reduce the amount of logging and improve performance.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()

inputs = [
    {"id": 1, "question": "What is the capital of France?", "synthetic_answer": "Lyon"},
    {"id": 2, "question": "Who wrote 'To Kill a Mockingbird'?", "synthetic_answer": "Harper Lee"},
    {"id": 3, "question": "What is the chemical symbol for gold?", "synthetic_answer": "Au"},
    {"id": 4, "question": "In what year did World War II end?", "synthetic_answer": "1939"},
    {"id": 5, "question": "What is the largest planet in our solar system?", "synthetic_answer": "Jupiter"},
    {"id": 6, "question": "What is the capital of Argentina?", "synthetic_answer": "Buenos Aires"}
] * 5

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

# Initialize the AsyncOpenAI client
client = AsyncOpenAI()

# Tenacity (retries) helps with transient failures, ensuring the operation eventually succeeds.
# Semaphores control concurrency, limiting the number of simultaneous operations.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def inference_async(input: str, model: str = "gpt-4o-mini") -> Dict:
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
    
    response = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=350,
            response_format={ 
                "type": "json_object"
            },
            messages=messages,
        )

    return response.choices[0].message.content.strip()


"""
This approach runs multiple asynchronous functions sequentially.
Each function will complete before the next one starts.
"""
async def async_loop_main():
    logging.info("Starting sequential asynchronous processing")
    results = []
    for input in inputs:
        result = await inference_async(input)
        results.append(result)
        logging.info(f"Processed input {input['id']} asynchronously")
    logging.info("Finished sequential asynchronous processing")
    return results


"""
This method runs multiple asynchronous functions concurrently.
It waits for all the functions to complete before proceeding.
"""
async def async_main():
    results = await asyncio.gather(*[inference_async(input) for input in inputs])
    return results


"""
To rate limit our requests, we can use a semaphore to control
the number of concurrent requests being processed at any given time.
"""
async def async_main_with_semaphore(max_concurrent=10):
    
    # Create a semaphore to limit the number of concurrent requests
    sem = asyncio.Semaphore(max_concurrent)
    
    async def bounded_inference(input):
        async with sem:
            return await inference_async(input)
    
    results = await asyncio.gather(*[bounded_inference(input) for input in inputs])
    return results


# 30 inputs
#2024-09-09 13:49:16,135 - INFO - Sync execution time: 39.87 seconds
#2024-09-09 15:19:01,614 - INFO - Async loop execution time: 40.78 seconds
#2024-09-09 15:07:48,083 - INFO - Async execution time: 1.55 seconds
#2024-09-09 15:07:54,761 - INFO - Async with semaphore execution time: 6.68 seconds

# 120 inputs
#2024-09-10 10:17:41,559 - INFO - Sync execution time: 151.47 seconds
#2024-09-10 10:13:35,122 - INFO - Future/Thread execution time: 13.12 seconds
#2024-09-10 10:45:46,011 - INFO - Async with semaphore execution time: 3.29 seconds


if __name__ == "__main__":
    logging.info("Script started")

    # Run async loop version
    start_time = time.time()
    results = asyncio.run(async_loop_main())
    async_loop_time = time.time() - start_time
    
    logging.info(f"Async loop execution time: {async_loop_time:.2f} seconds")

    # Run async main version
    start_time = time.time()
    results = asyncio.run(async_main())
    async_time = time.time() - start_time
    
    logging.info(f"Async execution time: {async_time:.2f} seconds")

    # Run async main with semaphore version
    start_time = time.time()
    results = asyncio.run(async_main_with_semaphore(max_concurrent=100))
    async_sem_time = time.time() - start_time
    
    logging.info(f"Async with semaphore execution time: {async_sem_time:.2f} seconds")

    # logging.info("Results:")
    # for i, result in enumerate(results, 1):
    #     logging.info(f"Result {i}: {result}")

    logging.info("Script finished")