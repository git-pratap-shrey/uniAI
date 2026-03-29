import os
from dotenv import load_dotenv
from google import genai

load_dotenv() # This reads your .env file

# Now it will find the key in your environment
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) 

response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview", # Using the 500 RPD one
    contents="Hi!"
)

print(response.text)