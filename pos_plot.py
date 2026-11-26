#!/usr/bin/env python3
"""Plot the POS tag distribution per model from pos_analysis.py's output.

Reads pos_analysis_output/pos_pooled_values.csv (written by pos_analysis.py),
so the figure shows exactly the pooled percentages reported in Table 4.
Run pos_analysis.py first, then, from the repository root:
    python -m pip install pandas matplotlib
    python pos_plot.py

Writes pos_tag_distribution.png and .pdf next to the input CSV. SPACE tokens
are plotted too (they are part of the denominator, as in pos_analysis.py).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

# Same model labels and tag set as pos_analysis.py.
MODELS = ("GPT-4o", "GPT-5.2", "RAG")
TAGS = ("PROPN", "NUM", "NOUN", "VERB", "AUX", "ADJ", "ADV", "DET",
        "ADP", "PRON", "CCONJ", "SCONJ", "PART", "SYM", "X", "INTJ")
# SPACE is in pos_analysis.py's denominator but not in its TAGS, so add it here.
PLOT_TAGS = (*TAGS, "SPACE")
MODEL_COLOURS = dict(zip(MODELS, ("#2a78d6", "#eb6834", "#1baf7a")))


def load_shares(path):
    """Return a tag x model table of pooled shares, most frequent tags first."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run pos_analysis.py first.")
    pooled = pd.read_csv(path)
    missing = set(MODELS) - set(pooled["model"])
    if missing:
        raise ValueError(f"Models missing from {path.name}: {sorted(missing)}")
    shares = pooled.pivot(index="tag", columns="model", values="share")
    shares = shares.loc[list(PLOT_TAGS), list(MODELS)]
    # Most frequent tags (averaged over the models) first.
    return shares.loc[shares.mean(axis=1).sort_values(ascending=False).index]


def plot(shares):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Group the models' bars per tag, centred on the tag's tick.
    width = 0.8 / len(MODELS)
    for i, model in enumerate(MODELS):
        offset = (i - (len(MODELS) - 1) / 2) * width
        ax.bar([x + offset for x in range(len(shares))], shares[model], width,
               label=model, color=MODEL_COLOURS[model])

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path,
                        help="pos_analysis.py output folder. "
                             "Default: REPO_ROOT/pos_analysis_output")
    args = parser.parse_args()
    output = args.output_dir or args.repo_root.resolve() / "pos_analysis_output"

    fig = plot(load_shares(output / "pos_pooled_values.csv"))
    for extension in ("png", "pdf"):
        plot_path = output / f"pos_tag_distribution.{extension}"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()