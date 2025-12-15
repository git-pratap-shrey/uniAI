import google.generativeai as genai

genai.configure(api_key="AIzaSyBbdu3ZKgWagy6R-2jWq_Ggym8REQyZeFY")

# List all models available to your API key
for m in genai.list_models():
    print(m.name, "supports:", m.supported_generation_methods)
