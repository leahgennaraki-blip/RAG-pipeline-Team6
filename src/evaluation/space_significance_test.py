"""Test whether GPT-4o, GPT 5.2 and RAG differ significantly in their share of
SPACE tokens (spaCy's tag for whitespace, mostly line breaks).

significance_tests.py leaves SPACE out of the POS tag tests, since it marks
layout (paragraphs, list items, headings) rather than a word class. This
script tests it on its own, with the same per-response shares and the same
tests as the POS tags there (Friedman, pairwise Wilcoxon with Holm), by
reusing significance_tests.py's functions. As the only feature in its
family, its Benjamini-Hochberg q-value equals its Friedman p-value.

Writes significance_tests_output/space_tests.csv, with the same columns as
pos_tags_tests.csv, so the two tables can be combined later (e.g. to plot
SPACE alongside the POS tags). The per-response SPACE shares the test uses
are already in pos_tags_per_response_values.csv.

Needs spaCy (with en_core_web_lg), NLTK and SciPy:
    pip install spacy nltk scipy
    python -m spacy download en_core_web_lg
Run this script from the overall project folder.
"""
import spacy

# Allow direct execution from the reorganized evaluation directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "text_analysis"))

from docuscope_moving_average import load_responses
from significance_tests import (
    MODELS, OUTPUT_DIR, SPACY_MODEL, save, spacy_features, summarize, test_features,
)

FEATURES = ["SPACE"]


def space_tests(pos):
    """Return the test results for SPACE, in pos_tags_tests.csv's format.

    pos: the per-response POS shares from significance_tests.spacy_features.
    """
    table = test_features(pos, FEATURES)
    table.insert(1, "description", table["feature"].map(spacy.explain))
    # pos_tags_tests.csv has a note column for tags too rare to test; keep it
    # (empty) so the two tables have the same columns.
    if "note" not in table:
        table.insert(table.columns.get_loc("significant pairs"), "note", None)
    return table


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load(SPACY_MODEL)
    responses = {model: load_responses(model) for model in MODELS}

    print("Tagging the responses with spaCy (about a minute)...", flush=True)
    pos, _ = spacy_features(nlp, responses)

    table = space_tests(pos)
    save(table, "space_tests")
    summarize("space", table)


if __name__ == "__main__":
    main()
