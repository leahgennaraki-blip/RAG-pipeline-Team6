# Integration notes

## Source snapshots

| Branch | Commit | Role |
| --- | --- | --- |
| restructured | 52a1fd1 | Target baseline, 35 files |
| question-dividers | 9563769 | Latest analysis and figure variants |
| qualitative_starter | 9fecf66 | Older analysis and outputs |
| alt_randomization | 91d55d4 | Optional ordering utility |
| main | b46c519 | Current main baseline |

`file_provenance.json` records full commits, source paths, exact preserved copies and SHA-256 hashes. `active_destination` identifies adapted scripts; `destination` identifies the exact original. Source branches are not rewritten or deleted.

## Decisions

- Current response data goes to `data/responses/`. The comma-delimited legacy `assets/` file stays intact. The semicolon-delimited current table is needed for question-boundary reconstruction and must match the cleaned texts.
- Import missing cleaning code, DocuScope texts/token exports, all result families, rater summaries and the SPACE test.
- Adopt the newer question-boundary implementations and GPT-4o plot labels, including Citation/Forceful/Confidence and POS delta plots. Preserve older differing files in `archive/qualitative_starter/`.
- Adapt paths and local imports; preserve original source scripts under `archive/before_path_adaptation/` and replaced target versions under `archive/pre_integration/`.
- Keep scientific calculations and RAG parameters unchanged. Complete the package list without claiming locked dependency versions.
- Keep existing tracked OS files to respect the no-deletion request. Ignore future OS/cache files; do not import source `.pyc` or `.DS_Store` files.
- Keep tables alongside figures so matching CSV/PDF/PNG outputs remain easy to locate.

## Manuscript decisions pending

Available local first drafts describe the memory study, Suriname case study, RAG and human assessment, with a general reference to NLP metrics. They do not establish the final selection of individual figures. Candidate figures are therefore marked pending in `figure_inventory.csv`, rather than being classified as brainstorming.

- Methodology draft says top-k = 20; code says TOP_K = 40. Confirm the actual experiment setting. Neither is silently changed.
- New plots say GPT-4o; older source labels and identifiers still say GPT-4 / gpt_4. Preserve data identifiers and confirm model provenance.
- Numerical human-rating summaries are separate from the close-reading analysis.
- The ordering utility prints model names and has no stored random seed. Its inclusion does not establish blind evaluation.
- The rating script excludes RAG from some column summaries; this inherited methodological choice is preserved.
- Source corpus, FAISS index/metadata and index-building code remain external.

## Figure variants

`majority` assigns a window to the question contributing most tokens/lemmas. `middle` uses the middle sentence/lemma. `by_length` is an alternative question-axis layout from the source implementation. These are alternative views, not independent experiments. Current candidates are in `outputs/`; older versions are in `archive/`.

## Validation boundary

`validation_report.json` records preservation, syntax, imports, response-text consistency and isolated execution checks. Saved NLP results are imported without rerunning expensive statistical calculations. RAG API calls are not executed. Path adaptation is not an independent audit of the scientific methodology.
