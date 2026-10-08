# Team 6 — AI, historical texts and collective memory

`restructured` consolidates the group's research and is the intended future main branch. Inclusion of a result here does not mean it has been selected for the paper.

## Directory guide

| Location | Contents |
| --- | --- |
| src/pipeline/ | Original PDF extraction, LLM cleaning and RAG scripts |
| src/text_analysis/ | Response cleaning, DocuScope, MATTR, POS and rating extrema |
| src/evaluation/ | Rating aggregation, significance tests, SPACE test, randomization |
| data/responses/ | Current semicolon-delimited response table and source with debris |
| data/qualitative/ | Raw ratings, cleaned ratings, combined rating tables |
| data/processed/docuscope_input/ | Three cleaned model-response texts |
| data/processed/docuscope_runs/ | Imported DocuScope run and token-level exports |
| outputs/docuscope/ | Categories, token tags and moving-average results |
| outputs/mattr/, outputs/pos/ | MATTR and POS result tables and figures |
| outputs/significance_tests/ | Statistical tables and per-response values |
| outputs/human_evaluation/summary_stats/ | Individual-rater summaries |
| outputs/randomised_forms/ | Existing HTML selection tool and optional generated forms |
| archive/ | Previous variants and scripts before path adaptation |
| assets/ | Original response table retained; current analysis uses data/responses/ |
| docs/ | Integration decisions, source provenance, validation and figure inventory |

Matching CSV/PDF/PNG results stay together by analysis and filename. Numerical rating summaries are distinct from close reading, despite the inherited qualitative directory names.

## Running analyses

Install `requirements.txt` in your Python environment. NLP scripts also require the spaCy `en_core_web_lg` model and NLTK sentence-tokenization resources. DocuScope is an external processing step; an existing export is included.

From the project root:

```bash
python src/text_analysis/cleaned_model_responses.py
# If input texts change, rerun DocuScope externally before continuing.
python src/text_analysis/docuscope_categories_plot.py
python src/text_analysis/top_token_tags.py
python src/text_analysis/docuscope_moving_average.py
python src/text_analysis/mattr_moving_average.py
python src/text_analysis/pos_tags.py
python src/evaluation/significance_tests.py
python src/evaluation/space_significance_test.py
```

For human-rating summaries:

```bash
python src/evaluation/qualitative_starter.py
python src/text_analysis/low_high_per_model.py
```

Commands regenerate their output files. Save a result snapshot first when retaining every version matters. Significance tests include 10,000 permutations and 5,000 bootstrap samples.

The optional `src/evaluation/generate_randomised_grading_forms.py` writes five ordering lists into `outputs/randomised_forms/generated/`. It displays model names, omits answer text and has no fixed seed; it is not a validated blinded-evaluation protocol.

## RAG reproduction boundary

The original pipeline requires Colab/Google Drive configuration, API credentials and external FAISS index/metadata. The corpus and index-construction workflow are not included. Its original experimental parameters have been retained.

## Adding work

Copy new work into the corresponding folder on this branch and retain its source branch. Record the source commit/path, preserve differing older versions in `archive/`, and keep scripts, inputs and results consistent. Mark selected manuscript figures in [figure inventory](docs/figure_inventory.csv).

See [integration notes](docs/integration_notes.md), [exact provenance](docs/file_provenance.json) and [validation](docs/validation_report.json). No source branch or research file has been deleted by this integration.
