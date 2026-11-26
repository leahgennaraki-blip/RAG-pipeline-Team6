"""Plot a 25-sentence moving average of seven DocuScope categories per model.

For each model, the text is split into sentences and every DocuScope token is
assigned to its sentence. A window of WINDOW whole sentences then moves
through the text one sentence at a time. Per window, each category's value is

    pattern matches starting in the window / tokens in the window

which is the same normalization as DocuScope's N_ csv, just per window.

Each window is also assigned a question, by two rules (QUESTION_RULES): the
question whose response most of its tokens come from (majority), and the
question its middle sentence comes from (middle). Every plot is made once
per rule; dashed lines mark where the window's question changes. The
responses are rebuilt from the source CSV with common/responses.py's
cleaning, since the .txt files don't mark where one response ends.

Needs NLTK for sentence splitting: pip install nltk
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import nltk
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.docuscope_io import align_tokens, read_tokens, select_run
from common.models import COLOURS as MODEL_COLOURS
from common.plotting import (
    majority_question_position, middle_question_position, question_axis,
    question_slots, slot_position,
)
from common.responses import numbered_responses

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "moving_average"

WINDOW = 25  # Sentences per window

# Token CSV / text filename prefix -> plot label.
MODELS = {"gpt_4": "GPT-4o", "gpt_5_2": "GPT 5.2", "rag": "RAG"}

# DocuScope category -> prefix of its tags in the token CSVs
# (e.g. NarrSceneShiftNeutral belongs to Narrative).
CATEGORIES = {
    "Narrative": "Narr",
    "Description": "Descript",
    "Negative": "Neg",
    "Exposition": "Expo",
    "Citation": "Citation",
    "Forceful": "Forceful",
    "Confidence": "Confidence",
}

# The overview grid is drawn once per group, so it doesn't get too wide.
# Group name -> filename suffix and its categories.
OVERVIEW_GROUPS = {
    "": ["Narrative", "Description", "Negative", "Exposition"],
    "_citation_forceful_confidence": ["Citation", "Forceful", "Confidence"],
}

# How a window is assigned to a question. Every plot is made once per rule.
QUESTION_RULES = {
    "majority": "the question most of its tokens come from",
    "middle": "the question its middle sentence comes from",
}
QUESTION_AXIS_LABELS = {
    "majority": "Question (windows spread evenly over the question most of their tokens "
                "come from, in text order)",
    "middle": "Question (window placed by where its middle sentence falls in that answer)",
}


def split_sentences(text):
    """Split text into sentences, treating line breaks as boundaries too.

    Line breaks are boundaries so headings like "Conclusion:" don't merge
    into the next sentence.
    """
    sentences = []
    for line in text.splitlines():
        if line.strip():
            sentences.extend(nltk.sent_tokenize(line))
    return sentences


def moving_average(model, token_dir):
    tokens = read_tokens(token_dir / f"{model}_tokens.csv")

    # Responses are separated by line breaks, so splitting each response
    # gives the same sentences as splitting the whole text.
    sentences, question_of_sentence = [], []
    for question, response in numbered_responses(model):
        response_sentences = split_sentences(response)
        sentences.extend(response_sentences)
        question_of_sentence.extend([question] * len(response_sentences))

    tokens["sentence"], unmatched = align_tokens(tokens["token"], sentences)
    if len(unmatched) > 0.01 * len(tokens):
        print(f"Warning: {model}: {len(unmatched)} of {len(tokens)} tokens did not "
              "line up with the text; sentence assignment may be off.")

    for category, prefix in CATEGORIES.items():
        tokens[category] = tokens["pattern_start"] & tokens["tag"].str.startswith(prefix)

    # Per-sentence counts, then sum WINDOW sentences and divide by the window's
    # tokens (not an average of per-sentence rates, which would weight a short
    # heading as much as a long sentence).
    per_sentence = tokens.groupby("sentence").agg(
        tokens=("token", "size"), **{c: (c, "sum") for c in CATEGORIES}
    )
    n_sentences = len(per_sentence)
    if n_sentences < WINDOW:
        print(f"Skipping {model}: only {n_sentences} sentences (window is {WINDOW}).")
        return None, None, None

    windows = per_sentence.rolling(WINDOW).sum().dropna()
    result = windows[list(CATEGORIES)].div(windows["tokens"], axis=0)
    result.insert(0, "window_tokens", windows["tokens"].astype(int))

    # Rolling windows are labelled by their last sentence; position the
    # window by its middle sentence, as a share of the text.
    middle = per_sentence.index.get_indexer(result.index) - WINDOW // 2
    middle_ids = per_sentence.index[middle]
    result.insert(0, "first_sentence", middle - WINDOW // 2)
    result.insert(1, "position", middle / (n_sentences - 1))

    # The window's question by each rule in QUESTION_RULES. Majority ties go
    # to the earlier question.
    question = pd.Series(question_of_sentence).loc[per_sentence.index]
    tokens_per_question = pd.get_dummies(question).mul(per_sentence["tokens"], axis=0)
    result.insert(2, "majority_question", tokens_per_question.rolling(WINDOW).sum().loc[
        result.index].idxmax(axis=1).astype(int))
    result.insert(3, "middle_question", pd.Series(question_of_sentence)[middle_ids].values)
    # Positions aligned to the questions, so the models line up in one plot.
    result.insert(4, "majority_question_position",
                  majority_question_position(result["majority_question"]))
    result.insert(5, "middle_question_position",
                  middle_question_position(question_of_sentence, middle_ids))
    result.insert(0, "model", MODELS[model])

    # Whole-text values for reference (same as DocuScope's N_ csv).
    overall = {c: tokens[c].sum() / len(tokens) for c in CATEGORIES}
    # Sentences per answer, for the plots whose question widths follow length.
    question_lengths = pd.Series(question_of_sentence).value_counts().sort_index()
    return result.reset_index(drop=True), overall, question_lengths


def question_segments(windows, rule):
    """Return (first position, last position, question) for each run of windows
    assigned to the same question by the rule."""
    column = f"{rule}_question"
    run = windows[column].ne(windows[column].shift()).cumsum()
    return [
        (group["position"].iloc[0], group["position"].iloc[-1], group[column].iloc[0])
        for _, group in windows.groupby(run)
    ]


def plot(results, rule, categories):
    fig, axes = plt.subplots(
        len(results), len(categories), figsize=(4.6 * len(categories), 3.3 * len(results)),
        sharex=True, sharey="col", squeeze=False,
    )

    for row, (model, (windows, overall)) in zip(axes, results.items()):
        colour = MODEL_COLOURS[model]
        segments = question_segments(windows, rule)

        for ax, category in zip(row, categories):
            # Dashed partitions where the window's question changes.
            for start, _end, _q in segments[1:]:
                ax.axvline(start, color="#b5b4af", lw=0.7, ls="--", zorder=1)

            ax.plot(windows["position"], windows[category], color=colour, lw=1.4, zorder=3)
            ax.axhline(overall[category], color=colour, lw=1, ls=":", zorder=2)

            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.grid(axis="y", color="#ecebe8", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        row[0].set_ylabel(f"{MODELS[model]}\nshare of tokens in window",
                          fontweight="bold")

    # Same y-range for all models per category, so the rows are comparable.
    for col, category in zip(axes.T, categories):
        top = max(windows[category].max() for windows, _ in results.values())
        col[0].set_ylim(0, top * 1.05)

    # Number every question above its segment, cycling through three
    # heights so the labels of short neighbouring segments don't overlap.
    for row, (windows, _) in zip(axes, results.values()):
        for ax in row:
            for i, (start, end, question) in enumerate(question_segments(windows, rule)):
                ax.annotate(
                    str(question), xy=((start + end) / 2, 1),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 1 + 7 * (i % 3)), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color="#52514e",
                )

    for ax, category in zip(axes[0], categories):
        ax.set_title(category, loc="left", fontweight="bold", pad=27)
    for ax in axes[-1]:
        ax.set_xlabel("Position in text (window middle)")

    fig.suptitle(
        f"DocuScope categories, {WINDOW}-sentence moving window. Dotted: whole-text "
        f"value.\nEach window is assigned {QUESTION_RULES[rule]} (numbered); dashed "
        "lines mark where that changes.",
        x=0.01, ha="left", fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_category(results, category, rule, question_lengths=None):
    """One plot for a category with a line per model, the x-axis aligned by question.

    With question_lengths, each question is as wide as its mean answer length.
    """
    fig, ax = plt.subplots(figsize=(14, 4.5))
    slots = question_slots(results, question_lengths)
    question_axis(ax, slots)

    for model, (windows, overall) in results.items():
        colour = MODEL_COLOURS[model]
        ax.plot(slot_position(windows[f"{rule}_question_position"], slots), windows[category],
                color=colour, lw=1.4, zorder=3, label=MODELS[model])
        ax.axhline(overall[category], color=colour, lw=1, ls=":", zorder=2)

    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(QUESTION_AXIS_LABELS[rule] + (
        "" if question_lengths is None
        else "\nQuestion width: its mean answer length over the models, in sentences"
    ))
    ax.set_ylabel("Share of tokens in window")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.set_title(
        f"{category}: DocuScope {WINDOW}-sentence moving window by model. "
        f"Dotted: whole-text value.\nEach window is assigned {QUESTION_RULES[rule]}.",
        loc="left", fontweight="bold",
    )
    fig.tight_layout()
    return fig


def main():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            raise SystemExit(
                "Could not download NLTK's punkt_tab sentence splitter. On macOS "
                "with python.org Python, an SSL error here usually means running "
                '"/Applications/Python 3.14/Install Certificates.command" once.'
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token_dir = select_run() / "token_csv"

    results, question_lengths = {}, {}
    for model in MODELS:
        windows, overall, lengths = moving_average(model, token_dir)
        if windows is not None:
            results[model] = (windows, overall)
            question_lengths[model] = lengths

    table = pd.concat([windows for windows, _ in results.values()])
    table.to_csv(OUTPUT_DIR / f"docuscope_moving_average_{WINDOW}.csv", index=False)

    for rule in QUESTION_RULES:
        for group, categories in OVERVIEW_GROUPS.items():
            fig = plot(results, rule, categories)
            for extension in ("png", "pdf"):
                plot_path = OUTPUT_DIR / f"docuscope_moving_average_{WINDOW}_{rule}{group}.{extension}"
                fig.savefig(plot_path, dpi=200, bbox_inches="tight")
                print(f"Saved: {plot_path}")
            plt.close(fig)

        # Equal question widths, and widths following the answers' length.
        for lengths, suffix in ((None, ""), (question_lengths, "_by_length")):
            for category in CATEGORIES:
                fig = plot_category(results, category, rule, lengths)
                for extension in ("png", "pdf"):
                    plot_path = OUTPUT_DIR / (
                        f"docuscope_moving_average_{WINDOW}_models_compared_"
                        f"{category.lower()}_{rule}{suffix}.{extension}"
                    )
                    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
                    print(f"Saved: {plot_path}")
                plt.close(fig)


if __name__ == "__main__":
    main()

