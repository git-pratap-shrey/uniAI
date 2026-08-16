import glob
import os
from source_code import models
from source_code.config.main import CONFIG

# Step 1: Collect all markdown files
wiki_path = os.path.join("source_code", "data", "year_2", "CYBER_SECURITY", "wiki", "*.md")
print(f"--- Searching for wiki files in: {wiki_path} ---")
files = glob.glob(wiki_path)

if not files:
    print(f"⚠ No markdown files found in '{wiki_path}'.")
    exit(1)

# Step 2: Concatenate with separators
context = ""
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        context += f"\n---\n# {f}\n" + file.read()

# Step 3: Send to LLM using the project's abstraction
print(f"--- Sending context from {len(files)} files to Gemini ---")

response = models.chat(
    prompt=f"{context}\n\nQuestion: What is Encryption and Decryption?",
    provider="gemini",
    model="gemini-3.1-flash-lite-preview", # Specific model for Gemini
)

print("\n--- Response ---")
print(response)
