import re
import unicodedata
import pandas as pd
from pathlib import Path

# Run this script from the overall project folder.
project_dir = Path.cwd()

input_file = project_dir / "assets" / "Team6_comparative_analysis.csv"
output_dir = project_dir / "docuscope_input"
output_dir.mkdir(parents=True, exist_ok=True)

# Read the exported Google Sheet.
df = pd.read_csv(input_file, dtype=str, keep_default_na=False)

# Row 0 contains the model labels, so remove it.
data = df.iloc[1:].copy().reset_index(drop=True)

# Keep only actual question rows.
data = data[data["Question"].str.strip().ne("")].copy()


def clean_text(value):
    """Lightly normalise text and remove common web/search-result debris."""
    text = str(value)

    # Normalise Unicode variants.
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible characters and normalise non-breaking spaces.
    text = (
        text.replace("\ufeff", "")  # Byte-order mark
        .replace("\u200b", "")      # Zero-width space
        .replace("\u200c", "")      # Zero-width non-joiner
        .replace("\u200d", "")      # Zero-width joiner
        .replace("\xa0", " ")       # Non-breaking space
    )

    # Standardise line breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep Markdown link labels, but remove their URLs.
    text = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", text)

    # Remove bare URLs.
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove standalone search-result counters such as "+1" or "+5".
    text = re.sub(r"(?m)^\s*\+\d+\s*$", "", text)

    # Remove source-name-plus-counter fragments, such as "Amazon +5".
    text = re.sub(r"(?m)^\s*[\w&.'’ -]+\s+\+\d+\s*$", "", text)

    # Remove lines consisting only of a web domain.
    text = re.sub(
        r"(?mi)^\s*(?:https?://)?(?:www\.)?"
        r"[a-z0-9][a-z0-9.-]*\.(?:com|org|net|edu|gov|nl|in|co\.uk)"
        r"(?:/[^\s]*)?\s*$",
        "",
        text,
    )

    # Normalise repeated spaces and tabs without processing text line by line.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces directly before or after line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Reduce excessive blank lines to one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def split_into_chunks(responses, max_tokens=200):
    """
    Combine complete responses into chunks of approximately max_tokens
    whitespace-separated words, without cutting sentences.

    A single sentence longer than max_tokens is kept intact in its own chunk.
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    for response in responses:
        # Split only after sentence-ending punctuation followed by whitespace.
        # The final text fragment is retained even if it has no final punctuation.
        sentences = re.split(r"(?<=[.!?])\s+", response.strip())

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # "Tokens" here means whitespace-separated words.
            sentence_token_count = len(re.findall(r"\S+", sentence))

            # Start a new chunk only when adding this complete sentence
            # would exceed the target and the current chunk has content.
            if (
                current_chunk
                and current_token_count + sentence_token_count > max_tokens
            ):
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_token_count = 0

            # Keep the complete sentence, even if it alone exceeds 200 words.
            current_chunk.append(sentence)
            current_token_count += sentence_token_count

    # Save the final incomplete chunk.
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

# Output filename prefix and corresponding source CSV column.
models = {
    "gpt_4": {
        "column": "A",
    },
    "gpt_5_2": {
        "column": "Unnamed: 3",
    },
    "rag": {
        "column": "B",
    },
}

for filename, model_info in models.items():
    source_column = model_info["column"]

    # Check that the expected column exists before proceeding.
    if source_column not in data.columns:
        raise KeyError(
            f"Column '{source_column}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    # Clean answers and discard missing or empty responses.
    # The original CSV/question order is retained.
    responses = data[source_column].map(clean_text)
    responses = responses[responses.ne("")].reset_index(drop=True)

    # Number of output fragments per model.
    # Create chunks of approximately 200 whitespace-separated words.
    # Chunks are created dynamically; there is no fixed number of files.
    chunks = split_into_chunks(responses, max_tokens=200)

    for part_number, chunk_text in enumerate(chunks, start=1):
        # Example: gpt_4_001.txt, gpt_4_002.txt, etc.
        output_path = output_dir / f"{filename}_{part_number:03d}.txt"
        output_path.write_text(chunk_text, encoding="utf-8")

        token_count = len(re.findall(r"\S+", chunk_text))

        print(
            f"Created: {output_path} "
            f"({token_count} whitespace-separated tokens)"
        )