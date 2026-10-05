import os
import json
import faiss
import numpy as np
from openai import OpenAI

# --- Step 1: Configuration ---
INDEX_SAVE_PATH = "/content/drive/MyDrive/RAG_file/faiss_index"
METADATA_PATH = "/content/drive/MyDrive/RAG_file/chunks_with_metadata.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 40
MAX_TOKENS = 2048
OPENAI_API_KEY = "YOUR_API_KEY_HERE"

# --- Step 2: Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Step 3: Load FAISS Index and Metadata ---
print("🔍 Loading FAISS index and metadata...")
index = faiss.read_index(os.path.join(INDEX_SAVE_PATH, "index.faiss"))
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f]

# --- Step 4: Function definitions ---
def get_query_embedding(query):
    response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def search_index(query, top_k=TOP_K):
    query_embedding = get_query_embedding(query)
    distances, indices = index.search(query_embedding, top_k)
    results = [metadata[idx] for idx in indices[0] if idx != -1]
    return results

def format_contexts_and_sources(results):
    context_texts = []
    sources = set()
    for result in results:
        context_texts.append(result.get("chunk", ""))
        sources.add(result.get("source", "Unknown Source"))
    context_combined = "\n\n---\n\n".join(context_texts)
    return context_combined, sorted(sources)

def generate_response(query, context):
    messages = [
        {"role": "system", "content": "You are a historical assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nPlease provide a detailed answer based on the above context."}
    ]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=MAX_TOKENS)
    return response.choices[0].message.content

def answer_query(query):
    results = search_index(query)
    context, sources = format_contexts_and_sources(results)
    response = generate_response(query, context)
    return response, sources

# --- Step 5: Interactive Loop ---
print("Welcome! Ask questions about Suriname's history based on provided texts.")
print("Type 'quit' to exit.\n")

while True:
    user_query = input("Please enter your question: ")
    if user_query.lower() == 'quit':
        print("Goodbye!")
        break
    answer, sources = answer_query(user_query)
    print("\nAnswer:\n", answer)
    print("\nSources used:")
    for src in sources:
        print(f"- {src}")
    follow_up = input("\nDo you have another question? (yes to continue, 'quit' to exit) ").strip().lower()
    if follow_up == 'quit':
        print("Goodbye!")
        break
