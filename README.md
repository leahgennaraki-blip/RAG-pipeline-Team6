# RAG pipeline: Team 6

Code and outputs for the paper's comparison of GPT-4o, GPT-5.2 and a RAG system on 24 questions.
Each analysis lives in its own folder and writes to its own `output/` folder.

| Folder | What it produces | Paper |
|---|---|---|
| `0_preprocessing/` | PDF → text → RAG answers (Colab), and the DocuScope input files | — |
| `1_pos/` | POS tag comparison (`output/table4.tex`) | Table 4 |
| `2_mattr/` | Lexical diversity (`output/analysis/mattr_table.tex`) and moving-window plots | MATTR table |
| `3_docuscope/` | DocuScope categories (`output/analysis/table6.tex`, `appendix_vi.tex`) and plots | Table 6, Appendix VI |
| `4_grading/` | Seeded randomiser used to assign answers to reviewers; the reviewers' gradings and their summaries (`qualitative_starter.py`) | Grading |
| `common/` | Code shared by the analyses (paths, model labels, cleaning, stats, plot axes) | — |
| `data/` | Inputs: the model responses CSV, DocuScope input and export | — |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Reproduce the results

Scripts can be run from any folder.

```bash
# 1. Write the cleaned responses that DocuScope tags
python 0_preprocessing/prepare_docuscope_input.py

# 2. Manual: run DocuScope on data/docuscope_input/ and put its export in
#    data/docuscope_export/ (the committed export used dictionary ds_dict 2024.01.11)

# 3. Analyses (independent of each other)
python 1_pos/pos_analysis.py && python 1_pos/pos_plot.py
python 2_mattr/mattr_analysis.py && python 2_mattr/mattr_moving_average.py
python 3_docuscope/docuscope_analysis.py
python 3_docuscope/docuscope_moving_average.py
python 3_docuscope/top_token_tags.py
```

`--from-saved` on the three `*_analysis.py` scripts rebuilds the tables from an earlier run's saved output, without re-tagging the text.

## Notes

- The source responses were collected by hand into `data/Team6_comparative_analysis.csv`. `_debris.csv` is the uncleaned version and isn't used by any script.
- `0_preprocessing/pdf_to_txt.py`, `text_cleaning_openai.py` and `rag_pipeline.py` are Colab scripts. They need Google Drive paths and an OpenAI API key, and the FAISS index they read is not in this repository.
- The DocuScope scripts use the newest run in `data/docuscope_export/`. `docuscope_analysis.py --run-dir` selects a specific one.
