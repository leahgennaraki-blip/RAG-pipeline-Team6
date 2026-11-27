# Imported human-rating figures

Copied from the group Drive `analysis/final/` on 2026-10-29 and updated on 2026-11-27. Figure contents and statistical results await team review.

| Files/folders | Purpose |
| --- | --- |
| plots/0a_qxc_gpt4, plots/0b_qxc_gpt5 | Inter-rater correlation at the question-by-criterion level |
| plots/1a_c_gpt4, plots/1b_c_gpt5 | Inter-rater correlation of criterion means |
| plots/2a_q_gpt4, plots/2b_q_gpt5 | Inter-rater correlation of question means |
| 0a.typ through 2b.typ | Layout sources combining the matching pairwise PDFs |
| plots/0a.pdf through plots/2b.pdf | Combined correlation figures supplied in Drive |
| plots/rating_dist.pdf | Overall model-rating distributions |
| plots/criteria_ratings_4o.pdf, plots/criteria_ratings_5.2.pdf | Per-criterion rating distributions |

Correlation figures correspond to correlation.R. Rating-distribution figures correspond to paired_t.R but do not directly visualize t-test statistics. Original source files and download hashes are listed in the import manifest. Rerunning the adapted R scripts writes to a separate reruns directory.

The November update adds criterion heatmaps, RAG rating distributions and plot legends. See the [file-by-file update report](../../../docs/drive_sync_2026-11-27.md).
