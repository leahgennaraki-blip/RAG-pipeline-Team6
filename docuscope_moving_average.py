"""Plot a 25-sentence moving average of four DocuScope categories per model.

For each model, the text is split into sentences and every DocuScope token is
assigned to its sentence. A window of WINDOW whole sentences then moves
through the text one sentence at a time. Per window, each category's value is

    pattern matches starting in the window / tokens in the window

which is the same normalization as DocuScope's N_ csv, just per window.

Each window is also assigned a question, by two rules (QUESTION_RULES): the
question whose response most of its tokens come from (majority), and the
question its middle sentence comes from (middle). Every plot is made once
per rule; dashed lines mark where the window's question changes. The
responses are rebuilt from the source CSV with cleaned_model_responses.py's
cleaning, since the .txt files don't mark where one response ends.

Needs NLTK for sentence splitting: pip install nltk
Run this script from the overall project folder.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import nltk
import numpy as np
import pandas as pd

from cleaned_model_responses import INPUT_FILE, MODELS as RESPONSE_COLUMNS, clean_text

PROJECT_DIR = Path.cwd()
DOCUSCOPE_INPUT = PROJECT_DIR / "docuscope_input"
DOCUSCOPE_OUTPUT = PROJECT_DIR / "docuscope_input_output"
OUTPUT_DIR = PROJECT_DIR / "docuscope_moving_average_output"

WINDOW = 25  # Sentences per window

# Token CSV / text filename prefix -> plot label.
MODELS = {"gpt_4": "GPT-4", "gpt_5_2": "GPT 5.2", "rag": "RAG"}
MODEL_COLOURS = {"gpt_4": "#2a78d6", "gpt_5_2": "#eb6834", "rag": "#1baf7a"}

# DocuScope category -> prefix of its tags in the token CSVs
# (e.g. NarrSceneShiftNeutral belongs to Narrative).
CATEGORIES = {
    "Narrative": "Narr",
    "Description": "Descript",
    "Negative": "Neg",
    "Exposition": "Expo",
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

# How many characters ahead to look for a token that doesn't line up with the
# text right away (e.g. a character DocuScope dropped or changed).
ALIGN_LOOKAHEAD = 50


def latest_run_folder():
    runs = sorted(DOCUSCOPE_OUTPUT.glob("docuscope_input-*"))
    if not runs:
        raise FileNotFoundError(f"No DocuScope runs found in {DOCUSCOPE_OUTPUT}")
    return runs[-1]  # Folder names end in a timestamp, so the last is newest


def load_responses(model):
    """Return the model's cleaned responses as (question number, text) pairs.

    Rebuilt the same way as cleaned_model_responses.py, and checked against
    the .txt DocuScope read, so the two can't silently drift apart.
    """
    df = pd.read_csv(INPUT_FILE, sep=";", dtype=str, keep_default_na=False)
    data = df.iloc[1:]  # Row 0 holds the model labels
    data = data[data["Question"].str.strip().ne("")]

    responses = data[RESPONSE_COLUMNS[model]].map(clean_text).tolist()
    pairs = [(q, text) for q, text in enumerate(responses, start=1) if text]

    text = (DOCUSCOPE_INPUT / f"{model}.txt").read_text(encoding="utf-8")
    if "\n\n".join(r for _, r in pairs) != text:
        raise ValueError(
            f"{model}.txt doesn't match the responses in {INPUT_FILE.name}; "
            "re-run cleaned_model_responses.py (and DocuScope) first."
        )
    return pairs


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


def first_field(head):
    """Return the token (first column) from the "token,token_lower" part of a row."""
    # Tokens containing a comma are quoted, e.g. ",",","
    quoted = re.match(r'"((?:[^"]|"")*)",', head)
    if quoted:
        return quoted.group(1).replace('""', '"')
    return head.split(",", 1)[0]


def read_tokens(token_file):
    """Return one row per token: token text and tag, plus whether it starts a pattern.

    Not read with pd.read_csv: DocuScope lowercases curly quotes to a bare,
    unescaped " in token_lower, which breaks CSV quoting. The last three
    columns never contain commas, so split each line from the right instead.
    """
    rows = []
    with open(token_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            head, _token_type, tag, position = line.rstrip("\n").rsplit(",", 3)
            rows.append({
                "token": first_field(head),
                "tag": tag,
                "pattern_start": tag != "" and position == "0",
            })
    return pd.DataFrame(rows)


def assign_sentences(tokens, sentences):
    """Return the sentence index of every token.

    Matches tokens to the sentence text character by character, ignoring
    whitespace, since DocuScope's tokens don't keep it.
    """
    chars, sentence_of_char = [], []
    for i, sentence in enumerate(sentences):
        compact = re.sub(r"\s+", "", sentence)
        chars.append(compact)
        sentence_of_char.extend([i] * len(compact))
    text = "".join(chars)

    sentence_ids, pos, unmatched = [], 0, 0
    for token in tokens:
        token = re.sub(r"\s+", "", token)
        found = text.find(token, pos, pos + ALIGN_LOOKAHEAD + len(token)) if token else -1
        if found == -1:
            # Keep it in the current sentence. Only count tokens with letters
            # or digits: DocuScope changes some punctuation (straight quotes
            # become curly, "###" becomes six "#" tokens), which is harmless.
            if re.search(r"\w", token):
                unmatched += 1
        else:
            pos = found + len(token)
        sentence_ids.append(sentence_of_char[min(max(pos - 1, 0), len(text) - 1)])

    return sentence_ids, unmatched


def middle_question_position(question_of_unit, middle_ids):
    """Return each window's position on an x-axis aligned to the questions,
    placed by its middle unit (sentence or lemma).

    Question q spans q - 0.5 to q + 0.5, and the window sits inside it by
    where its middle unit falls within that question's answer.
    question_of_unit: the question of every unit, in text order.
    """
    question = pd.Series(question_of_unit)
    index_in_question = question.groupby(question).cumcount()
    question_length = question.map(question.value_counts())
    return (
        question[middle_ids].values - 0.5
        + (index_in_question[middle_ids].values + 0.5) / question_length[middle_ids].values
    )


def majority_question_position(window_questions):
    """Return each window's position on an x-axis aligned to the questions,
    placed by its majority question.

    Question q spans q - 0.5 to q + 0.5, and the windows assigned to it are
    spread evenly over that span in text order. A question that no window
    mostly consists of gets no windows.
    """
    question = pd.Series(window_questions).reset_index(drop=True)
    rank = question.groupby(question).cumcount()
    count = question.map(question.value_counts())
    return (question - 0.5 + (rank + 0.5) / count).values


def moving_average(model, token_dir):
    tokens = read_tokens(token_dir / f"{model}_tokens.csv")

    # Responses are separated by line breaks, so splitting each response
    # gives the same sentences as splitting the whole text.
    sentences, question_of_sentence = [], []
    for question, response in load_responses(model):
        response_sentences = split_sentences(response)
        sentences.extend(response_sentences)
        question_of_sentence.extend([question] * len(response_sentences))

    tokens["sentence"], unmatched = assign_sentences(tokens["token"], sentences)
    if unmatched > 0.01 * len(tokens):
        print(f"Warning: {model}: {unmatched} of {len(tokens)} tokens did not "
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


def question_slots(results, question_lengths=None):
    """Return each question's span on the x-axis, as (start, width) Series
    indexed by question.

    Without question_lengths, question q spans q - 0.5 to q + 0.5. With them
    ({model: answer length per question}), a question is as wide as its mean
    answer length over the models. Uses the questions found by either rule,
    so both rules' plots get the same axis.
    """
    questions = pd.Index(sorted(set().union(*(
        windows[column] for windows, _ in results.values()
        for column in ("majority_question", "middle_question")
    ))))
    if question_lengths is None:
        return pd.Series(questions - 0.5, index=questions), pd.Series(1.0, index=questions)
    widths = pd.DataFrame(question_lengths).mean(axis=1).loc[questions]
    return widths.cumsum() - widths, widths


def slot_position(question_position, slots):
    """Move positions from the one-wide question spans (q - 0.5 to q + 0.5)
    into the given spans, keeping where they sit within their question."""
    starts, widths = slots
    question_position = np.asarray(question_position)
    question = np.floor(question_position + 0.5).astype(int)
    within = question_position - (question - 0.5)
    return starts.loc[question].values + within * widths.loc[question].values


def question_axis(ax, slots):
    """Lay out the x-axis by question, with every other question shaded and
    dashed lines between them."""
    starts, widths = slots
    for i, q in enumerate(starts.index):
        if q % 2 == 0:
            ax.axvspan(starts[q], starts[q] + widths[q], color="#f4f3f0", lw=0, zorder=0)
        if i > 0:
            ax.axvline(starts[q], color="#b5b4af", lw=0.7, ls="--", zorder=1)
    ax.set_xlim(starts.iloc[0], starts.iloc[-1] + widths.iloc[-1])
    ax.set_xticks(starts + widths / 2, starts.index)
    ax.tick_params(axis="x", length=0)


def plot(results, rule):
    fig, axes = plt.subplots(
        len(results), len(CATEGORIES), figsize=(4.6 * len(CATEGORIES), 3.3 * len(results)),
        sharex=True, sharey="col", squeeze=False,
    )

    for row, (model, (windows, overall)) in zip(axes, results.items()):
        colour = MODEL_COLOURS[model]
        segments = question_segments(windows, rule)

        for ax, category in zip(row, CATEGORIES):
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
    for col, category in zip(axes.T, CATEGORIES):
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

    for ax, category in zip(axes[0], CATEGORIES):
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
    token_dir = latest_run_folder() / "token_csv"

    results, question_lengths = {}, {}
    for model in MODELS:
        windows, overall, lengths = moving_average(model, token_dir)
        if windows is not None:
            results[model] = (windows, overall)
            question_lengths[model] = lengths

    table = pd.concat([windows for windows, _ in results.values()])
    table.to_csv(OUTPUT_DIR / f"docuscope_moving_average_{WINDOW}.csv", index=False)

    for rule in QUESTION_RULES:
        fig = plot(results, rule)
        for extension in ("png", "pdf"):
            plot_path = OUTPUT_DIR / f"docuscope_moving_average_{WINDOW}_{rule}.{extension}"
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
