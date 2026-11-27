# Team 6 — AI, historical texts and collective memory

`restructured` consolidates the group's research and is the intended future main branch. Inclusion of a result here does not mean it has been selected for the paper.

## 2026-11-27 — Drive update

On 27 November we compared 123 Drive files with the 29 October import on restructured. We added 16 files (nine criterion-heatmap files, four RAG rating plots and three legends) and updated 92 (two R scripts, six Typst layouts and 84 PDFs). The six rating datasets matched the October originals. Nine files in plots/old matched figures already archived in Git. The [shared appendices](https://docs.google.com/document/d/1SkKVLeOai-l9Fh4gg6tW7rtvPGqMSx309lvrHVGFeKU/edit) are linked for reference. The team still needs to confirm manuscript figure selection. [Bilingual file-by-file report](docs/drive_sync_2026-11-27.md).

## 2026-10-29 — Google Drive research materials imported

Copied the `analysis/final/` materials from the group Drive into this branch: Pearson correlation and paired-samples t-test R scripts, six numbered rating datasets (`r1.csv`–`r6.csv`), correlation plots and their Typst layout sources, and overall/per-criterion rating-distribution plots. Earlier `processing.Rmd`, processing reports and quantitative write-up files were preserved separately under `archive/drive_import/analysis/`.

These uploads fill the previously identified code and summary-plot gaps. They are imported for team review; figure accuracy, statistical validity, anonymisation and the final manuscript version remain unconfirmed. Separate exported correlation/t-test result tables and a dedicated t-test effect/interval plot have not been identified in the imported set. Rating-distribution plots should not be labelled as t-test result plots.

| Imported material | Repository location |
| --- | --- |
| Correlation and paired t-test code | `src/evaluation/r/` |
| Numbered individual ratings | `data/qualitative/anonymised_ratings/` |
| Original correlation and summary PDFs | `outputs/human_evaluation/drive_import/plots/` |
| Typst layouts for combined correlation figures | `outputs/human_evaluation/drive_import/` |
| Original R scripts before path adaptation | `archive/drive_import/original_scripts/` |
| Earlier analysis source and reports | `archive/drive_import/analysis/` |

The R scripts now resolve repository paths and write regenerated figures to `outputs/human_evaluation/reruns/`, preserving the imported figures. Statistical calculations were not changed or rerun. The source IDs, paths and SHA-256 hashes are recorded in [the import manifest](docs/drive_import_2026-10-29.json).

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

## Research deliverables status

This checklist reflects the group's meeting requirements. “Present” means that the relevant material is currently identifiable in this branch; it does not mean that the method or result has been selected for the paper.

| Requirement | Current status | Where to look / remaining issue |
| --- | --- | --- |
| MATTR code and visualisations | **Present** | `src/text_analysis/mattr_moving_average.py`, `outputs/mattr/`; confirm which `majority`, `middle` or `by_length` figure is final. |
| POS-tagging code and visualisations | **Present** | `src/text_analysis/pos_tags.py`, `outputs/pos/`; `SPACE` is handled separately by `src/evaluation/space_significance_test.py`. |
| DocuScope code and visualisations | **Present, with an external dependency** | `src/text_analysis/`, `data/processed/docuscope_runs/`, `outputs/docuscope/`; the DocuScope application/version and complete rerun procedure are not stored here. |
| Summary statistics code and plots | **Code and plots present; review pending** | Existing Python summaries are supplemented by `src/evaluation/r/paired_t.R` and `outputs/human_evaluation/drive_import/plots/`: `rating_dist.pdf`, `criteria_ratings_4o.pdf`, `criteria_ratings_5.2.pdf`. |
| Correlation statistics code and plots | **Code and plots imported; review pending** | `src/evaluation/r/correlation.R` computes inter-rater Pearson correlation and RMSE at question-by-criterion, criterion and question levels. Six plot groups are under `outputs/human_evaluation/drive_import/plots/`. A separate exported results table is not included. |
| Paired-samples t-test code and plots | **Code imported; result exports/plots still to confirm** | `src/evaluation/r/paired_t.R` includes paired tests, Bonferroni correction and effect sizes. Imported plots show rating distributions; a standalone t-test result table and dedicated effect/interval figure have not been identified. |
| Text cleaning code | **Present** | `src/pipeline/pdf_to_txt.py`, `src/pipeline/text_cleaning_openai.py`, `src/text_analysis/cleaned_model_responses.py`; confirm which response dataset is the canonical input. |
| RAG pipeline code | **Present but not fully reproducible** | `src/pipeline/rag_pipeline.py`; FAISS index, metadata, source corpus and index-building workflow remain external. The manuscript says top-k = 20 while the code uses `TOP_K = 40`; this needs confirmation. |
| Comparative dataset / model responses | **Present, with versions to reconcile** | `data/responses/Team6_comparative_analysis.csv` is the current analysis input; `assets/` and `source_with_debris.csv` are retained legacy/source versions. Confirm the canonical file and model labels (`GPT-4`/`GPT-4o`). |
| Anonymised individual ratings | **Numbered files imported; confirmation pending** | `data/qualitative/anonymised_ratings/r1.csv`–`r6.csv` are the inputs for the R scripts. They are separate from earlier raw/cleaned ratings. Confirm numbering and pairing with the team; `r6` has no GPT5 rows. No identity mapping has been added. |

### Remaining review and export tasks

1. Confirm whether the imported Drive `final/` set is the manuscript version; verify figures with their authors.
2. Confirm the numbered ratings, missing-value handling and the paired sample. The R script pairs by rater, question and criterion; not every rater has both models.
3. Confirm whether standalone correlation and t-test result tables should be exported. The imported R scripts calculate/print these values but do not save separate CSV result tables.
4. Confirm whether a dedicated t-test effect/interval plot is required in addition to the available rating-distribution plots.
5. Record the canonical response dataset, model naming convention and actual RAG top-k setting.

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

### Imported R analyses

Install R and the `tidyverse`, `ggpubr` and `ggtext` packages, then run from the repository root:

```bash
Rscript src/evaluation/r/correlation.R
Rscript src/evaluation/r/paired_t.R
```

The scripts read `data/qualitative/anonymised_ratings/` and write to `outputs/human_evaluation/reruns/`. Runtime package versions are not locked, and these analyses have not been executed as part of this import. The correlation script covers GPT4/GPT5; the paired model comparison uses GPT-4o/GPT-5.2, not RAG.

The six Typst layouts retain their relative `plots/` references. Compile them from `outputs/human_evaluation/drive_import/` if needed, choosing a new output filename to preserve the imported PDF. The archived R Markdown source is a historical version, not an additional step in the current pipeline.

## RAG reproduction boundary

The original pipeline requires Colab/Google Drive configuration, API credentials and external FAISS index/metadata. The corpus and index-construction workflow are not included. Its original experimental parameters have been retained.

## Adding work

Copy new work into the corresponding folder on this branch and retain its source branch. Record the source commit/path, preserve differing older versions in `archive/`, and keep scripts, inputs and results consistent. Mark selected manuscript figures in [figure inventory](docs/figure_inventory.csv).

See [integration notes](docs/integration_notes.md), [exact provenance](docs/file_provenance.json) and [validation](docs/validation_report.json). No source branch or research file has been deleted by this integration.
