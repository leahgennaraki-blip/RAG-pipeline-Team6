"""Plot the normalized DocuScope category values of each model's top 20 categories.

Reads N_docuscope_input.csv (share of all tokens per category) from the latest
DocuScope run in docuscope_input_output/. Every category in at least one
model's top 20 gets a row; a model's dot is only drawn where the category is
in that model's own top 20.

Run this script from the overall project folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path.cwd()
DOCUSCOPE_OUTPUT = PROJECT_DIR / "docuscope_input_output"
OUTPUT_DIR = PROJECT_DIR / "docuscope_categories_output"

TOP_N = 20

# DocuScope filename -> plot label.
MODELS = {"gpt_4.txt": "GPT-4o", "gpt_5_2.txt": "GPT 5.2", "rag.txt": "RAG"}
MODEL_COLOURS = {"gpt_4.txt": "#2a78d6", "gpt_5_2.txt": "#eb6834", "rag.txt": "#1baf7a"}
MODEL_MARKERS = {"gpt_4.txt": "o", "gpt_5_2.txt": "s", "rag.txt": "D"}

# Columns of the DocuScope CSV that are not categories.
NON_CATEGORY_COLUMNS = ["Tokens", "Group"]


def latest_run_folder():
    runs = sorted(DOCUSCOPE_OUTPUT.glob("docuscope_input-*"))
    if not runs:
        raise FileNotFoundError(f"No DocuScope runs found in {DOCUSCOPE_OUTPUT}")
    return runs[-1]  # Folder names end in a timestamp, so the last is newest


def load_top_categories():
    """Return a category x model table of normalized values, NaN outside each top 20."""
    values = pd.read_csv(latest_run_folder() / "csv" / "N_docuscope_input.csv")
    values = values.set_index("Filename").drop(columns=NON_CATEGORY_COLUMNS).T
    values = values[list(MODELS)]

    top = pd.DataFrame({
        model: values[model].nlargest(TOP_N) for model in MODELS
    })
    # Largest categories at the top of the plot.
    return top.loc[top.mean(axis=1).sort_values().index]


def plot(top):
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(top) + 1.8))
    rows = range(len(top))

    # Light connector showing the spread between models per category.
    ax.hlines(rows, top.min(axis=1), top.max(axis=1), color="#d4d3cf", lw=2, zorder=1)

    # Shift each model slightly within its row so near-equal values don't hide
    # each other.
    offsets = {model: (i - 1) * 0.2 for i, model in enumerate(MODELS)}
    for model, label in MODELS.items():
        ax.scatter(
            top[model], [r + offsets[model] for r in rows], label=label, s=55, zorder=2,
            color=MODEL_COLOURS[model], marker=MODEL_MARKERS[model],
            edgecolor="white", linewidth=1.2,
        )

    ax.set_yticks(rows, top.index)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Share of all tokens")
    ax.set_xlim(0, top.max().max() * 1.08)
    ax.grid(axis="x", color="#ecebe8", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.set_title(
        f"Top {TOP_N} DocuScope categories per model (normalized)",
        loc="left", fontweight="bold", fontsize=13, pad=28,
    )
    ax.text(
        0, 1.02, "Dots are only drawn for categories in that model's top "
        f"{TOP_N}.", transform=ax.transAxes, fontsize=9, color="#52514e",
    )
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top = load_top_categories()

    table = top.iloc[::-1].rename(columns=MODELS).round(4)
    table.index.name = "category"
    table.to_csv(OUTPUT_DIR / f"top_{TOP_N}_docuscope_categories.csv")

    fig = plot(top)
    for extension in ("png", "pdf"):
        plot_path = OUTPUT_DIR / f"top_{TOP_N}_docuscope_categories.{extension}"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
