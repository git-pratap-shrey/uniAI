import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

completion = client.chat.completions.create(
    model="qwen/qwen3-coder:free",  # or any OpenRouter-supported model
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    temperature=0.6,
    max_tokens=4096,
    top_p=0.95,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")