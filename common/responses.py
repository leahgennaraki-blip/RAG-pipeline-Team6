"""Read and clean the model responses from data/Team6_comparative_analysis.csv."""
import re
import unicodedata

import pandas as pd

from common.models import LABELS, MODEL_KEYS, SOURCE_COLUMNS
from common.paths import DOCUSCOPE_INPUT, RESPONSES_CSV

EXPECTED_QUESTIONS = 24

# Invisible characters to strip, and non-breaking space to normalise.
INVISIBLE_CHARS = str.maketrans({
    "﻿": None,  # Byte-order mark
    "​": None,  # Zero-width space
    "‌": None,  # Zero-width non-joiner
    "‍": None,  # Zero-width joiner
    "\xa0": " ",     # Non-breaking space
})

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
BARE_URL = re.compile(r"https?://\S+|www\.\S+")
COUNTER_LINE = re.compile(r"(?m)^\s*(?:[\w&.'’ -]+\s+)?\+\d+\s*$")  # "+5", "Amazon +5"
DOMAIN_LINE = re.compile(
    r"(?mi)^\s*(?:https?://)?(?:www\.)?"
    r"[a-z0-9][a-z0-9.-]*\.(?:com|org|net|edu|gov|nl|in|co\.uk)"
    r"(?:/\S*)?\s*$"
)


def clean_text(value):
    """Lightly normalise text and remove common web/search-result debris."""
    text = unicodedata.normalize("NFKC", str(value)).translate(INVISIBLE_CHARS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = MARKDOWN_LINK.sub(r"\1", text)  # Keep link labels, drop URLs
    text = BARE_URL.sub("", text)
    text = COUNTER_LINE.sub("", text)
    text = DOMAIN_LINE.sub("", text)

    text = re.sub(r"[ \t]+", " ", text)    # Collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)   # Trim spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text) # Max one blank line
    return text.strip()


def source_rows():
    """The question rows of the source CSV (row 0 holds model labels)."""
    df = pd.read_csv(RESPONSES_CSV, sep=";", dtype=str, keep_default_na=False)
    data = df.iloc[1:]
    data = data[data["Question"].str.strip().ne("")]
    missing = [col for col in SOURCE_COLUMNS.values() if col not in data.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {list(data.columns)}")
    return data


def cleaned_responses(key):
    """The model's cleaned responses in question order (empty strings kept)."""
    return source_rows()[SOURCE_COLUMNS[key]].map(clean_text).tolist()


def check_docuscope_input(key, texts):
    """Fail if docuscope_input/<key>.txt no longer matches the responses."""
    text = (DOCUSCOPE_INPUT / f"{key}.txt").read_text(encoding="utf-8")
    if "\n\n".join(texts) != text:
        raise ValueError(
            f"{key}.txt doesn't match the responses in {RESPONSES_CSV.name}; re-run "
            "0_preprocessing/prepare_docuscope_input.py (and DocuScope) first."
        )


def numbered_responses(key):
    """The model's non-empty responses as (question number, text) pairs,
    checked against the .txt file DocuScope read."""
    pairs = [(q, text) for q, text in enumerate(cleaned_responses(key), start=1) if text]
    check_docuscope_input(key, [text for _, text in pairs])
    return pairs


def matched_responses(check_input=False):
    """{label: 24 cleaned responses} for the question-matched tests.

    Every model must answer every question. With check_input, also require
    the DocuScope .txt files to match.
    """
    data = source_rows()
    if len(data) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected {EXPECTED_QUESTIONS} questions; found {len(data)}.")
    result = {}
    for key in MODEL_KEYS:
        texts = data[SOURCE_COLUMNS[key]].map(clean_text).tolist()
        if any(not text for text in texts):
            raise ValueError(f"{LABELS[key]}: an answer is empty; cannot form matched data.")
        if check_input:
            check_docuscope_input(key, texts)
        result[LABELS[key]] = texts
    return result
