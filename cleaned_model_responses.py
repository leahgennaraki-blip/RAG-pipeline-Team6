import re
import unicodedata
from pathlib import Path

import pandas as pd

# Run this script from the overall project folder.
PROJECT_DIR = Path.cwd()
INPUT_FILE = PROJECT_DIR / "assets" / "Team6_comparative_analysis.csv"
OUTPUT_DIR = PROJECT_DIR / "docuscope_input"

# Output filename -> source CSV column.
MODELS = {
    "gpt_4": "A",
    "gpt_5_2": "Unnamed: 3",
    "rag": "B",
}

# Invisible characters to strip, and non-breaking space to normalise.
INVISIBLE_CHARS = str.maketrans({
    "\ufeff": None,  # Byte-order mark
    "\u200b": None,  # Zero-width space
    "\u200c": None,  # Zero-width non-joiner
    "\u200d": None,  # Zero-width joiner
    "\xa0": " ",     # Non-breaking space
})

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
BARE_URL = re.compile(r"https?://\S+|www\.\S+")
COUNTER_LINE = re.compile(r"(?m)^\s*(?:[\w&.'’ -]+\s+)?\+\d+\s*$")  # "+5", "Amazon +5"
DOMAIN_LINE = re.compile(
    r"(?mi)^\s*(?:https?://)?(?:www\.)?"
    r"[a-z0-9][a-z0-9.-]*\.(?:com|org|net|edu|gov|nl|in|co\.uk)"
    r"(?:/\S*)?\s*$"
)


def clean_text(value):
    """Lightly normalise text and remove common web/search-result debris."""
    text = unicodedata.normalize("NFKC", str(value)).translate(INVISIBLE_CHARS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = MARKDOWN_LINK.sub(r"\1", text)  # Keep link labels, drop URLs
    text = BARE_URL.sub("", text)
    text = COUNTER_LINE.sub("", text)
    text = DOMAIN_LINE.sub("", text)

    text = re.sub(r"[ \t]+", " ", text)    # Collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)   # Trim spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text) # Max one blank line
    return text.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

    # Row 0 holds the model labels; keep only rows with a question.
    data = df.iloc[1:]
    data = data[data["Question"].str.strip().ne("")]

    missing = [col for col in MODELS.values() if col not in data.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {list(data.columns)}")

    for name, column in MODELS.items():
        responses = data[column].map(clean_text)
        responses = responses[responses.ne("")]

        output_path = OUTPUT_DIR / f"{name}.txt"
        output_path.write_text("\n\n".join(responses), encoding="utf-8")

        word_count = sum(len(r.split()) for r in responses)
        print(f"Created: {output_path} ({len(responses)} responses, {word_count} words)")


if __name__ == "__main__":
    main()