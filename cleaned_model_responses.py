import re
import unicodedata
import pandas as pd
from pathlib import Path

# Run this script from the overall project folder.
project_dir = Path.cwd()

input_file = project_dir / "assets" / "Team6_comparative_analysis.csv"
output_dir = project_dir / "docuscope_input"
output_dir.mkdir(parents=True, exist_ok=True)

# Read the exported Google Sheet
df = pd.read_csv(input_file, dtype=str, keep_default_na=False)

# Row 0 contains the model labels, so remove it.
data = df.iloc[1:].copy().reset_index(drop=True)

# Keep only actual question rows
data = data[data["Question"].str.strip().ne("")].copy()

def clean_text(value):
    """Lightly normalise text while retaining linguistic content."""
    text = str(value)

    # Normalise Unicode variants
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible characters and normalise non-breaking spaces
    text = (
        text.replace("\ufeff", "")  # Byte-order mark
            .replace("\u200b", "")  # Zero-width space
            .replace("\u200c", "")  # Zero-width non-joiner
            .replace("\u200d", "")  # Zero-width joiner
            .replace("\xa0", " ")   # Non-breaking space
    )

    # Standardise line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove repeated spaces/tabs within lines but retain paragraph structure
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Reduce excessive blank lines to one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# Output filename and corresponding source CSV column
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

    # Check that the expected column exists before proceeding
    if source_column not in data.columns:
        raise KeyError(
            f"Column '{source_column}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    # Clean answers and discard missing/empty responses
    responses = data[source_column].map(clean_text)
    responses = responses[responses.ne("")]

    # Combine every response from this model into one document.
    # A blank line separates original responses without adding labels or IDs.
    combined_text = "\n\n".join(responses)

    # Write one UTF-8 plain-text file per model
    output_path = output_dir / f"{filename}.txt"
    output_path.write_text(combined_text, encoding="utf-8")

    print(f"Created: {output_path} ({len(responses)} responses combined)")