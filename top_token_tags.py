"""Plot the 20 most frequent DocuScope token tags per model.

Reads the token_csv files from the latest DocuScope run in
docuscope_input_output/ and counts how many tokens received each tag.

Run this script from the overall project folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path.cwd()
DOCUSCOPE_OUTPUT = PROJECT_DIR / "docuscope_input_output"
OUTPUT_DIR = PROJECT_DIR / "top_token_tags_output"

TOP_N = 20

# DocuScope's catch-all "Other..." tags (e.g. OtherOrphanedOther) are not
# rhetorical categories; set to False to include them in the ranking.
EXCLUDE_OTHER_TAGS = True

# Token CSV filename prefix -> plot label.
MODELS = {"gpt_4": "GPT-4o", "gpt_5_2": "GPT 5.2", "rag": "RAG"}
MODEL_COLOURS = {"gpt_4": "#2a78d6", "gpt_5_2": "#eb6834", "rag": "#1baf7a"}

# Columns of a DocuScope token CSV (the files have no header row).
TOKEN_COLUMNS = ["token", "token_lower", "token_type", "tag", "position_in_pattern"]


def latest_run_folder():
    runs = sorted(DOCUSCOPE_OUTPUT.glob("docuscope_input-*"))
    if not runs:
        raise FileNotFoundError(f"No DocuScope runs found in {DOCUSCOPE_OUTPUT}")
    return runs[-1]  # Folder names end in a timestamp, so the last is newest


def top_tags(token_file):
    """Return the TOP_N tags by token count, plus the model's total token count."""
    # Not read with pd.read_csv: DocuScope lowercases curly quotes to a bare,
    # unescaped " in token_lower, which breaks CSV quoting. The last three
    # columns never contain commas, so split each line from the right instead.
    with open(token_file, encoding="utf-8") as f:
        rows = [line.rstrip("\n").rsplit(",", 3) for line in f if line.strip()]
    tokens = pd.DataFrame(rows, columns=["tokens", *TOKEN_COLUMNS[2:]])
    tags = tokens["tag"]
    tags = tags[tags.ne("")]  # Untagged tokens (mostly punctuation)
    if EXCLUDE_OTHER_TAGS:
        tags = tags[~tags.str.startswith("Other")]

    counts = tags.value_counts().head(TOP_N)
    return counts, len(tokens)


def plot(results):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 8))

    for ax, (model, (counts, total_tokens)) in zip(axes, results.items()):
        counts = counts.iloc[::-1]  # Highest count at the top
        bars = ax.barh(counts.index, counts.values, color=MODEL_COLOURS[model])

        labels = [f"{n:,} ({n / total_tokens:.1%})" for n in counts.values]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)

        ax.set_title(
            f"{MODELS[model]}  ({total_tokens:,} tokens)", loc="left", fontweight="bold"
        )
        ax.set_xlabel("Tagged tokens (share of all tokens)")
        ax.set_xlim(0, counts.max() * 1.3)  # Room for the bar labels
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Top {TOP_N} DocuScope token tags per model", fontsize=13)
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token_dir = latest_run_folder() / "token_csv"

    results = {
        model: top_tags(token_dir / f"{model}_tokens.csv") for model in MODELS
    }

    # Table with the same numbers as the plot.
    table = pd.concat(
        {
            MODELS[model]: pd.DataFrame({
                "tag": counts.index,
                "tokens": counts.values,
                "share": (counts.values / total_tokens).round(4),
            })
            for model, (counts, total_tokens) in results.items()
        },
        axis=1,
    )
    table.index = range(1, len(table) + 1)
    table.index.name = "rank"
    table.to_csv(OUTPUT_DIR / f"top_{TOP_N}_token_tags.csv")

    fig = plot(results)
    for extension in ("png", "pdf"):
        plot_path = OUTPUT_DIR / f"top_{TOP_N}_token_tags.{extension}"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
