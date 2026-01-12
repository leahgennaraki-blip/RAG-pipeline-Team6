# RAG-pipeline
This repository contains the code used in my master's thesis for processing historical book PDFs, cleaning the extracted text with AI, and implementing a Retrieval-Augmented Generation (RAG) pipeline for querying historical content. 

### Required Packages
```bash
pip install pymupdf tqdm openai faiss-cpu

Note: faiss-cpu is for FAISS vector search. Use faiss-gpu if you have GPU support.


Step 1: Convert PDF to Text (1_pdf_to_txt.py)

This script extracts text from PDFs and performs basic cleaning, such as:
Normalizing Unicode characters
Fixing unusual numeric characters
Removing hyphenation at line breaks
Removing extra whitespace and newlines

Usage
Upload your PDF file via Google Colab or adjust the script to use a local path.
Run the script: pdf_to_txt.py
The cleaned text will be saved as <original_filename>_cleaned.txt.

Step 2: Clean Text using OpenAI GPT (2_text_cleaning_openai.py)

This step uses GPT-4o-mini to remove irrelevant elements such as:
Page numbers
Footnotes
Headers and footers
Copyright and publisher info
Table of contents

It retains only:
Book title
Author(s)
Chapter names
Chapter text

Instructions-
Place your .txt files from Step 1 into input_folder.
Update your OpenAI API key in the script.
Run the script: text_cleaning_openai.py

Cleaned .txt files will be saved in the specified output_folder.

Step 3: RAG Pipeline

This step sets up a Retrieval-Augmented Generation (RAG) pipeline using:
FAISS for vector-based retrieval
OpenAI GPT-4o for generating detailed answers based on retrieved context

Features
Load precomputed FAISS index and metadata
Query the historical book texts interactively
Retrieve top-k relevant chunks and generate a detailed AI answer
Display sources used for transparency

Usage

Ensure you have:
A FAISS index (index.faiss) saved at INDEX_SAVE_PATH
Metadata JSONL (chunks_with_metadata.jsonl) containing chunks and sources
Update your OpenAI API key in the script.

Run the script: python 3_rag_pipeline.py
Type your question and receive a detailed AI-generated response.

Notes & Recommendations

Always keep your OpenAI API key secure. Consider using environment variables or a .env file.
The chunk size in Step 2 can be adjusted depending on token limits for GPT.
For large datasets, FAISS is recommended for efficient vector retrieval.

This pipeline is designed specifically for historical texts (e.g., Suriname history books) but can be adapted for other corpora.
