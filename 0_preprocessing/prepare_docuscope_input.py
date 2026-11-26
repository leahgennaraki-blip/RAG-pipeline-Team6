"""Write each model's cleaned responses to data/docuscope_input/<key>.txt.

These .txt files are what DocuScope tags (a manual step), and what the
MATTR and DocuScope analyses check their own cleaned responses against.
Responses are joined by one blank line; empty responses are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import MODEL_KEYS
from common.paths import DOCUSCOPE_INPUT
from common.responses import cleaned_responses


def main():
    DOCUSCOPE_INPUT.mkdir(parents=True, exist_ok=True)

    for key in MODEL_KEYS:
        responses = [text for text in cleaned_responses(key) if text]

        output_path = DOCUSCOPE_INPUT / f"{key}.txt"
        output_path.write_text("\n\n".join(responses), encoding="utf-8")

        word_count = sum(len(r.split()) for r in responses)
        print(f"Created: {output_path} ({len(responses)} responses, {word_count} words)")


if __name__ == "__main__":
    main()
