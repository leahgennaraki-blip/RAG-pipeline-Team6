"""Correlate the qualitative grading criteria with selected DocuScope categories,
separately for each model, with Pearson, Spearman and Kendall.

Reads the mean grade per model x question x criterion
(global_model_question_criterion_means.csv) and every DocuScope category's
share of a response's tokens (docuscope_categories_per_response_values.csv,
written by significance_tests.py), joins them per response and computes the
correlation between every criterion and category within each model (24
questions each), so differences between the models don't drive the result.

Correlation methods
    - Pearson's r: strength of a straight-line relationship between the raw
      values. Its p-value assumes roughly normal data, which the grades
      (ordinal, means of 1-5 ratings, RAG near the ceiling) and the category
      shares (right-skewed) are not, and with 24 points a single outlier can
      move r a lot.
    - Spearman's rho: Pearson on the ranks. Measures any monotonic
      relationship (one goes up as the other does, not necessarily in a
      straight line), needs no normality and is robust to skew and outliers.
      The usual choice for ordinal ratings.
    - Kendall's tau-b: share of concordant minus discordant pairs of
      responses, with a correction for ties. Smaller in size than rho for
      the same data (|tau| is roughly 2/3 of |rho|), but handles the many tied
      grades better and has more reliable p-values for small samples.

Stars mark correlations whose p-value, Benjamini-Hochberg corrected over all
model x criterion x category pairs of a method, is below ALPHA. Where a
criterion got the same grade for every question of a model, the correlation is
undefined and the cell is marked n/a.

Writes to docuscope_criterion_correlation_output/: per method, the table of
correlations, p- and q-values and the heatmaps.

Run this script from the overall project folder, after significance_tests.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path.cwd()
GRADINGS_CSV = (PROJECT_DIR / "qualitative_starter_final_gradings" / "global_gradings"
                / "global_model_question_criterion_means.csv")
CATEGORIES_CSV = (PROJECT_DIR / "significance_tests_output"
                  / "docuscope_categories_per_response_values.csv")
OUTPUT_DIR = PROJECT_DIR / "docuscope_criterion_correlation_output"

ALPHA = 0.05

# Model name in the gradings -> model name in the DocuScope values (plot label).
MODELS = {"GPT4": "GPT-4o", "GPT5": "GPT 5.2", "RAG": "RAG"}

CATEGORIES = ["Description", "Forceful", "Narrative", "Negative"]

# Method -> (label, test returning the correlation and its p-value).
# scipy's kendalltau computes tau-b.
METHODS = {
    "pearson": ("Pearson's r", stats.pearsonr),
    "spearman": ("Spearman's ρ", stats.spearmanr),
    "kendall": ("Kendall's τ-b", stats.kendalltau),
}


def load_responses():
    """Return one row per model x question, a column per criterion and category."""
    grades = pd.read_csv(GRADINGS_CSV)
    grades["model"] = grades["Model"].map(MODELS)
    grades["question"] = grades["Question"].str.removeprefix("Q").astype(int)
    grades = grades.pivot_table(
        index=["model", "question"], columns="criterion", values="mean_value",
    )

    categories = pd.read_csv(CATEGORIES_CSV).set_index(["model", "question"])
    responses = grades.join(categories[CATEGORIES], how="inner")
    if len(responses) != len(grades):
        raise ValueError(f"Only {len(responses)} of {len(grades)} graded responses "
                         "have DocuScope values")
    return responses, list(grades.columns)


def benjamini_hochberg(p_values):
    """BH q-values; NaN p-values are skipped and stay NaN."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    order = np.argsort(p[valid])
    ranked = p[valid][order] * valid.sum() / np.arange(1, valid.sum() + 1)
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1)
    q[valid] = adjusted
    return q


def correlate(responses, criteria, method="pearson"):
    """Return a long table of the correlation, p and BH-corrected q per model x pair."""
    test = METHODS[method][1]
    rows = []
    for model in MODELS.values():
        group = responses.loc[model]
        for criterion in criteria:
            for category in CATEGORIES:
                if group[criterion].nunique() < 2:
                    r, p = np.nan, np.nan  # Constant grades: r is undefined
                else:
                    r, p = test(group[criterion], group[category])
                rows.append({"model": model, "criterion": criterion,
                             "category": category, "r": r, "p": p})
    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg(table["p"])
    return table


def plot(table, criteria, method="pearson"):
    method_label = METHODS[method][0]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 0.55 * len(criteria) + 2.2),
                             sharey=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#ecebe8")

    for ax, model in zip(axes, MODELS.values()):
        group = table[table["model"].eq(model)]
        r = group.pivot(index="criterion", columns="category", values="r")
        q = group.pivot(index="criterion", columns="category", values="q")
        r, q = r.loc[criteria, CATEGORIES], q.loc[criteria, CATEGORIES]
        image = ax.imshow(np.ma.masked_invalid(r.to_numpy()), cmap=cmap,
                          vmin=-1, vmax=1, aspect="auto")

        for i, criterion in enumerate(criteria):
            for j, category in enumerate(CATEGORIES):
                value = r.loc[criterion, category]
                if np.isnan(value):
                    label, colour = "n/a", "#8a8984"
                else:
                    star = "*" if q.loc[criterion, category] < ALPHA else ""
                    label = f"{value:.2f}{star}".replace("0.", ".")
                    colour = "white" if abs(value) > 0.6 else "#1f1e1c"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=colour)

        ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=40, ha="right")
        ax.set_yticks(range(len(criteria)), criteria)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.set_title(model, fontsize=11)

    colourbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colourbar.set_label(method_label)
    colourbar.outline.set_visible(False)

    fig.suptitle(
        f"Grading criteria vs. DocuScope categories per model ({method_label})",
        x=0.06, ha="left", fontweight="bold", fontsize=13, y=1.06,
    )
    fig.text(
        0.06, 0.985, "Per response, n = 24 questions per model. "
        f"* = q < {ALPHA} (Benjamini-Hochberg over all pairs). "
        "n/a = same grade for every question.",
        fontsize=9, color="#52514e",
    )
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    responses, criteria = load_responses()

    for method, (label, _) in METHODS.items():
        table = correlate(responses, criteria, method)
        table.to_csv(OUTPUT_DIR / f"criterion_category_{method}_per_model.csv",
                     index=False, float_format="%.6g")

        fig = plot(table, criteria, method)
        for extension in ("png", "pdf"):
            plot_path = (OUTPUT_DIR
                         / f"criterion_category_{method}_per_model.{extension}")
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {plot_path}")
        plt.close(fig)

        significant = table[table["q"] < ALPHA].sort_values("r", key=abs,
                                                            ascending=False)
        print(f"\n{label}: significant correlations (q < {ALPHA}): "
              f"{len(significant)}")
        if len(significant):
            print(significant.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
