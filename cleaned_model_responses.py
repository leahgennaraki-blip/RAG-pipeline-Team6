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

# Create question numbers: 1, 2, ..., 24
data["question_number"] = range(1, len(data) + 1)

def clean_text(value):
    """Remove invisible characters and surrounding whitespace."""
    return (
        str(value)
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\xa0", " ")
        .strip()
    )

# Output filename, doc_id prefix, and corresponding source column
models = {
    "gpt_4": {
        "column": "A",
        "doc_id_prefix": "gpt4",
    },
    "gpt_5_2": {
        "column": "Unnamed: 3",
        "doc_id_prefix": "gpt5.2",
    },
    "rag": {
        "column": "B",
        "doc_id_prefix": "rag",
    },
}

for filename, model_info in models.items():
    source_column = model_info["column"]
    prefix = model_info["doc_id_prefix"]

    output = pd.DataFrame({
        "doc_id": prefix + "_" + data["question_number"].astype(str),
        "text": data[source_column].map(clean_text),
    })

    # Omit missing or blank model responses
    output = output[output["text"].ne("")]

    output_path = output_dir / f"{filename}.csv"
    output.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Created: {output_path}")