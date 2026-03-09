import google.generativeai as genai

genai.configure(api_key="")

# List all models available to your API key
for m in genai.list_models():
    print(m.name, "supports:", m.supported_generation_methods)
