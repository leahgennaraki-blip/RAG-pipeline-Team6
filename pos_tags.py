"""Plot the part-of-speech (POS) tag distribution per model.

Tags every model's text in docuscope_input/ with spaCy's universal POS tags
and counts how many tokens received each tag, leaving out punctuation. Each
value is the tag's share of the model's (non-punctuation) tokens.

A second plot shows the difference in each tag's share between every pair of
models. Its shares leave out SPACE (line breaks) too: GPT 5.2's text has far
more of them, which would otherwise shrink all its other shares.

Needs spaCy and its large English model:
    pip install spacy
    python -m spacy download en_core_web_lg
Run this script from the overall project folder.
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import spacy

PROJECT_DIR = Path.cwd()
TEXT_INPUT = PROJECT_DIR / "docuscope_input"
OUTPUT_DIR = PROJECT_DIR / "pos_tags_output"

SPACY_MODEL = "en_core_web_lg"

# Text filename prefix -> plot label.
MODELS = {"gpt_4": "GPT-4o", "gpt_5_2": "GPT 5.2", "rag": "RAG"}
MODEL_COLOURS = {"gpt_4": "#2a78d6", "gpt_5_2": "#eb6834", "rag": "#1baf7a"}

# Model pairs compared in the delta plot, as (first, second): first - second.
PAIRS = [("gpt_5_2", "gpt_4"), ("rag", "gpt_4"), ("rag", "gpt_5_2")]

# Tags that aren't word classes, left out of the delta plot and its shares.
NON_WORD_TAGS = ["SPACE", "PUNCT"]


def pos_counts(nlp, model):
    """Return the token count per POS tag, plus the model's total token count."""
    text = (TEXT_INPUT / f"{model}.txt").read_text(encoding="utf-8")
    counts = Counter(token.pos_ for token in nlp(text) if not token.is_punct)
    return pd.Series(counts), sum(counts.values())


def plot(shares):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Group the models' bars per tag, centred on the tag's tick.
    width = 0.8 / len(MODELS)
    for i, (model, label) in enumerate(MODELS.items()):
        offset = (i - (len(MODELS) - 1) / 2) * width
        ax.bar([x + offset for x in range(len(shares))], shares[model], width,
               label=label, color=MODEL_COLOURS[model])

    ax.set_xticks(range(len(shares)), shares.index)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("POS tag")
    ax.set_ylabel("Share of tokens (excluding punctuation)")
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax.set_title("POS tag distribution per model", loc="left", fontweight="bold",
                 fontsize=13)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def word_class_deltas(counts):
    """Return each tag's share difference per model pair, in percentage points.

    Shares are of the model's tokens without NON_WORD_TAGS.
    """
    words = counts.drop(NON_WORD_TAGS, errors="ignore")
    shares = words / words.sum()
    return pd.DataFrame({
        f"{MODELS[first]} - {MODELS[second]}": (shares[first] - shares[second]) * 100
        for first, second in PAIRS
    })


def plot_deltas(deltas):
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(12, 2.6 * len(PAIRS)),
                             sharex=True, sharey=True)

    for ax, (first, second), column in zip(axes, PAIRS, deltas):
        # Colour each bar by the model with the higher share.
        colours = [MODEL_COLOURS[first] if d > 0 else MODEL_COLOURS[second]
                   for d in deltas[column]]
        ax.bar(range(len(deltas)), deltas[column], 0.6, color=colours)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.tick_params(axis="x", length=0)

        ax.set_ylabel("Difference\n(percentage points)")
        ax.grid(axis="y", color="#ecebe8", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.set_title(f"{MODELS[first]} vs {MODELS[second]}  (above 0: "
                     f"{MODELS[first]} higher, below 0: {MODELS[second]} higher)",
                     loc="left", fontweight="bold", fontsize=10)

    axes[-1].set_xticks(range(len(deltas)), deltas.index)
    axes[-1].set_xlabel("POS tag")
    fig.suptitle(
        "POS tag share differences between models. Shares of tokens excluding "
        "punctuation and line breaks (SPACE).",
        x=0.01, ha="left", fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load(SPACY_MODEL)

    results = {model: pos_counts(nlp, model) for model in MODELS}
    counts = pd.DataFrame({model: c for model, (c, _) in results.items()}).fillna(0)
    shares = counts / pd.Series({model: total for model, (_, total) in results.items()})
    # Most frequent tags (averaged over the models) first.
    order = shares.mean(axis=1).sort_values(ascending=False).index
    counts, shares = counts.loc[order].astype(int), shares.loc[order]

    # Table with the same numbers as the plot.
    table = pd.concat(
        {
            MODELS[model]: pd.DataFrame({
                "tokens": counts[model],
                "share": shares[model].round(4),
            })
            for model in MODELS
        },
        axis=1,
    )
    table.insert(0, "description", [spacy.explain(tag) for tag in table.index])
    table.index.name = "tag"
    table.to_csv(OUTPUT_DIR / "pos_tag_distribution.csv")

    fig = plot(shares)
    for extension in ("png", "pdf"):
        plot_path = OUTPUT_DIR / f"pos_tag_distribution.{extension}"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {plot_path}")
    plt.close(fig)

    deltas = word_class_deltas(counts)
    deltas.round(2).rename_axis("tag").to_csv(OUTPUT_DIR / "pos_tag_deltas.csv")
    fig = plot_deltas(deltas)
    for extension in ("png", "pdf"):
        plot_path = OUTPUT_DIR / f"pos_tag_deltas.{extension}"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
