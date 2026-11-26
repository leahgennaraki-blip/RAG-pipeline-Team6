"""Read DocuScope exports and line their tokens up with the source text."""
import re
from pathlib import Path

import pandas as pd

from common.paths import DOCUSCOPE_EXPORT

# How many characters ahead to look for a token that doesn't line up with the
# text right away (e.g. a character DocuScope dropped or changed).
ALIGN_LOOKAHEAD = 50


def select_run(requested=None):
    """The requested DocuScope export run, or the newest one."""
    if requested is not None:
        run = Path(requested).resolve()
    else:
        runs = sorted(DOCUSCOPE_EXPORT.glob("docuscope_input-*"))
        if not runs:
            raise FileNotFoundError(f"No DocuScope runs found in {DOCUSCOPE_EXPORT}")
        run = runs[-1]  # Folder names end in a timestamp, so the last is newest
    if not (run / "token_csv").is_dir():
        raise FileNotFoundError(f"Missing token_csv directory in {run}")
    return run


def first_field(head):
    """Return the token (first column) from the "token,token_lower" part of a row."""
    # Tokens containing a comma are quoted, e.g. ",",","
    quoted = re.match(r'"((?:[^"]|"")*)",', head)
    return quoted.group(1).replace('""', '"') if quoted else head.split(",", 1)[0]


def read_tokens(path):
    """Return one row per token: token text and tag, plus whether it starts a pattern.

    Not read with pd.read_csv: DocuScope lowercases curly quotes to a bare,
    unescaped " in token_lower, which breaks CSV quoting. The last three
    columns never contain commas, so split each line from the right instead.
    """
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                head, _token_type, tag, position = line.rstrip("\n").rsplit(",", 3)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: malformed token row.") from exc
            rows.append({
                "token": first_field(head), "tag": tag,
                "pattern_start": bool(tag) and position == "0",
            })
    if not rows:
        raise ValueError(f"Empty token export: {path}")
    return pd.DataFrame(rows)


def align_tokens(tokens, units):
    """Return the unit index (0-based) of every token, and the indices of
    lexical tokens that didn't line up with the text.

    units are consecutive pieces of the text (responses or sentences).
    Matches tokens to the text character by character, ignoring whitespace,
    since DocuScope's tokens don't keep it. A token that doesn't line up stays
    in the current unit. Only tokens with letters or digits count as
    unmatched: DocuScope changes some punctuation (straight quotes become
    curly, "###" becomes six "#" tokens), which is harmless.
    """
    chars, unit_of_char = [], []
    for i, unit in enumerate(units):
        compact = re.sub(r"\s+", "", unit)
        chars.append(compact)
        unit_of_char.extend([i] * len(compact))
    text = "".join(chars)

    unit_ids, unmatched, pos = [], [], 0
    for index, token in enumerate(tokens):
        compact = re.sub(r"\s+", "", token)
        found = text.find(compact, pos, pos + ALIGN_LOOKAHEAD + len(compact)) if compact else -1
        if found >= 0:
            pos = found + len(compact)
        elif re.search(r"\w", compact):
            unmatched.append(index)
        unit_ids.append(unit_of_char[min(max(pos - 1, 0), len(text) - 1)])
    return unit_ids, unmatched
