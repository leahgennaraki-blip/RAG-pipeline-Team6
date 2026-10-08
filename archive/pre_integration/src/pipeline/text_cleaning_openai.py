import os
from openai import OpenAI
from tqdm import tqdm

# --- Step 1: Initialize OpenAI client ---
client = OpenAI(api_key="YOUR_API_KEY_HERE")

# --- Step 2: Input/Output folders ---
input_folder = "/content/drive/MyDrive/txt files"
output_folder = "/content/drive/MyDrive/LLM_cleaned_output"
os.makedirs(output_folder, exist_ok=True)

# --- Step 3: GPT system prompt for cleaning ---
system_prompt = """You are a text cleaner for historical book data.
Remove page numbers, footnotes, headers, footers, copyright info, publisher info, and table of contents.
Keep ONLY: book title, author(s), chapter names, chapter text.
Keep the original order and structure. Clean up line breaks and spacing.
Return cleaned plain text.
"""

# --- Step 4: Functions ---
def chunk_text(text, chunk_size=12000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def clean_text_with_gpt(text_chunk):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_chunk}
        ],
        temperature=0.2,
        max_tokens=4096,
        timeout=60
    )
    return response.choices[0].message.content

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    chunks = chunk_text(raw_text)
    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        print(f" ⏳ Chunk {i+1}/{len(chunks)}...")
        try:
            cleaned = clean_text_with_gpt(chunk)
            cleaned_chunks.append(cleaned)
        except Exception as e:
            print(f" ❌ Error on chunk {i+1}: {e}")
    cleaned_text = "\n\n".join(cleaned_chunks)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f" ✅ Saved to {output_path}")

# --- Step 5: Process all .txt files in input folder ---
all_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
for filename in tqdm(all_files, desc="Cleaning books"):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    print(f"\n📘 Cleaning: {filename}")
    process_file(input_path, output_path)
