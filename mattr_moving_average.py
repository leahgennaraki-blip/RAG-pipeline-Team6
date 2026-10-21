"""Plot the moving-average type-token ratio (MATTR) of every model's text.

Each text in docuscope_input/ is lemmatized with spaCy, keeping only
alphabetic tokens (lowercased). A window of WINDOW lemmas then moves through
the text one lemma at a time. Per window, the type-token ratio is

    distinct lemmas in the window / WINDOW

and the model's MATTR is the mean of all its windows. A higher value means
more varied vocabulary. Plots the windows along the text, and the models'
MATTR scores side by side.

Needs spaCy and its large English model:
    pip install spacy
    python -m spacy download en_core_web_lg
Run this script from the overall project folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import spacy

PROJECT_DIR = Path.cwd()
TEXT_INPUT = PROJECT_DIR / "docuscope_input"
OUTPUT_DIR = PROJECT_DIR / "mattr_output"

SPACY_MODEL = "en_core_web_lg"

WINDOW = 500  # Lemmas per window

# Text filename prefix -> plot label.
MODELS = {"gpt_4": "GPT-4", "gpt_5_2": "GPT 5.2", "rag": "RAG"}
MODEL_COLOURS = {"gpt_4": "#2a78d6", "gpt_5_2": "#eb6834", "rag": "#1baf7a"}


def lemmas(nlp, model):
    text = (TEXT_INPUT / f"{model}.txt").read_text(encoding="utf-8")
    return [token.lemma_.lower() for token in nlp(text) if token.is_alpha]


def moving_ttr(tokens):
    """Return the type-token ratio of every WINDOW-lemma window, in text order.

    A text shorter than WINDOW gets a single value: its whole-text ratio.
    """
    if len(tokens) < WINDOW:
        return [len(set(tokens)) / len(tokens)] if tokens else [0.0]
    return [
        len(set(tokens[i:i + WINDOW])) / WINDOW
        for i in range(len(tokens) - WINDOW + 1)
    ]


def mattr(nlp, model):
    """Return one row per window (its position and TTR), plus the model's MATTR."""
    tokens = lemmas(nlp, model)
    ttr = moving_ttr(tokens)
    windows = pd.DataFrame({"first_lemma": range(len(ttr)), "ttr": ttr})
    # Position the window by its middle lemma, as a share of the text, so the
    # models line up in one plot despite their different lengths.
    middle = windows["first_lemma"] + min(WINDOW, len(tokens)) // 2
    windows.insert(1, "position", middle / max(len(tokens) - 1, 1))
    windows.insert(0, "model", MODELS[model])
    return windows, windows["ttr"].mean()


def plot(results):
    fig, ax = plt.subplots(figsize=(14, 4.5))

    for model, (windows, overall) in results.items():
        colour = MODEL_COLOURS[model]
        ax.plot(windows["position"], windows["ttr"], color=colour, lw=1.4, zorder=3,
                label=f"{MODELS[model]} (MATTR {overall:.4f})")
        ax.axhline(overall, color=colour, lw=1, ls=":", zorder=2)

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Position in text (window middle)")
    ax.set_ylabel("Type-token ratio in window")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.set_title(
        f"MATTR: moving window of {WINDOW} by model. Dotted: mean over all "
        "windows (the model's MATTR).",
        loc="left", fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_scores(results):
    """Bar chart of each model's MATTR, highest first."""
    scores = sorted(results.items(), key=lambda item: item[1][1], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        [MODELS[model] for model, _ in scores],
        [overall for _, (_, overall) in scores],
        color=[MODEL_COLOURS[model] for model, _ in scores],
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)

    ax.set_ylabel("MATTR")
    ax.set_ylim(0, max(overall for _, (_, overall) in scores) * 1.2)  # Room for the bar labels
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"MATTR per model (window = {WINDOW})", loc="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load(SPACY_MODEL)

    results = {model: mattr(nlp, model) for model in MODELS}

    table = pd.concat([windows for windows, _ in results.values()])
    table.to_csv(OUTPUT_DIR / f"mattr_moving_average_{WINDOW}.csv", index=False)

    scores = pd.DataFrame({
        "model": [MODELS[model] for model in results],
        "mattr": [round(overall, 4) for _, overall in results.values()],
    })
    scores.to_csv(OUTPUT_DIR / f"mattr_{WINDOW}.csv", index=False)
    for row in scores.itertuples():
        print(f"{row.model}: MATTR {row.mattr:.4f}")

    fig = plot(results)
    for extension in ("png", "pdf"):
        plot_path = OUTPUT_DIR / f"mattr_moving_average_{WINDOW}.{extension}"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {plot_path}")
    plt.close(fig)

    fig = plot_scores(results)
    for extension in ("png", "pdf"):
        plot_path = OUTPUT_DIR / f"mattr_{WINDOW}.{extension}"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
