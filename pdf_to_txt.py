# --- Step 1: Install Dependencies ---
# pip install pymupdf

# --- Step 2: Imports ---
import fitz  # PyMuPDF
import unicodedata
import re
from pathlib import Path
from google.colab import files

# --- Step 3: Functions for Cleaning and Fixing Unicode ---
def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def fix_unicode_numbers(text):
    weird_digits = ""  # Add any special digits if needed
    normal_digits = "0123456789"
    return text.translate(str.maketrans(weird_digits, normal_digits))

def preprocess_text(text):
    text = normalize_unicode(text)
    text = fix_unicode_numbers(text)
    text = text.replace("-\n", "")  # undo hyphenation at line breaks
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# --- Step 4: PDF Upload ---
print("📂 Please upload your PDF file from Google Drive...")
uploaded = files.upload()
pdf_filename = next(iter(uploaded))
output_txt_path = pdf_filename.replace(".pdf", "_cleaned.txt")

# --- Step 5: Process the PDF ---
doc = fitz.open(pdf_filename)
cleaned_text = ""
for page in doc:
    raw_text = page.get_text("text")
    processed = preprocess_text(raw_text)
    cleaned_text += processed + "\n\n"

# --- Step 6: Save Output ---
Path(output_txt_path).write_text(cleaned_text, encoding="utf-8")
print(f"\n✅ Done! Cleaned text saved as: {output_txt_path}")
