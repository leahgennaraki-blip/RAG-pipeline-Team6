"""Question-aligned x-axes for the moving-window plots (DocuScope and MATTR).

A window is placed on an axis where question q spans q - 0.5 to q + 0.5,
either by its middle unit (sentence or lemma) or spread evenly over the
question most of it comes from.
"""
import numpy as np
import pandas as pd


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
