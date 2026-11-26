#!/usr/bin/env python3
"""Reproduce Table 4: pooled POS percentages and question-matched tests.

    python -m spacy download en_core_web_lg
    python 1_pos/pos_analysis.py

To reuse the observations an earlier full run saved, without spaCy:
    python 1_pos/pos_analysis.py --from-saved

Each response is tagged separately with spaCy en_core_web_lg. Descriptives
pool its integer POS counts and retained-token denominator across all 24
responses per model; tests use the individual question-matched proportions.
Keep every token for which token.is_punct is False, including
SPACE in the denominator; do not lowercase, lemmatize or remove stopwords.
No artificial blank-line tokens are added between answers, and descriptives
are NOT the unweighted average of answer percentages.
SPACE and PUNCT are excluded from the displayed/tested features.
Features present in fewer than 10 of the 72 responses are descriptive only.
Bonferroni correction is applied separately across ALL eligible Friedman
tests and ALL three pairwise Wilcoxon tests per eligible feature. For the
study data, these family sizes are 15 and 45. There is no BH/Holm correction.
Pairwise significance stars are shown only when the corrected overall test
is significant, matching Table 4's post-hoc presentation (including SCONJ).

The script does not invoke or alter MATTR or DocuScope analyses. Test validity
still depends on independent question blocks; shared conversational context
can introduce dependence. Wilcoxon location inference also assumes
symmetric paired differences. Failure to reject does not establish equality.
"""

import argparse
import sys
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import MODELS
from common.paths import load_saved
from common.responses import EXPECTED_QUESTIONS, matched_responses
from common.stats import latex_effect

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

TAGS = ("PROPN", "NUM", "NOUN", "VERB", "AUX", "ADJ", "ADV", "DET",
        "ADP", "PRON", "CCONJ", "SCONJ", "PART", "SYM", "X", "INTJ")
MIN_RESPONSES = 10
SPACY_MODEL = "en_core_web_lg"
PAIRS = tuple(combinations(MODELS, 2))


def pos_counts(doc):
    """Match pos_tags.py's is_punct filter, retaining SPACE tokens."""
    return Counter(token.pos_ for token in doc if not token.is_punct)


def recompute():
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("Install spaCy and en_core_web_lg, or use --from-saved.") from exc
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError("Run: python -m spacy download en_core_web_lg") from exc
    texts_by_model = matched_responses()
    response_rows = []
    for model, texts in texts_by_model.items():
        print(f"Tagging {model}...", flush=True)
        nlp.max_length = max(nlp.max_length, max(map(len, texts)) + 1)
        for question, doc in enumerate(nlp.pipe(texts), start=1):
            counts = pos_counts(doc)
            total = sum(counts.values())
            if not total:
                raise ValueError(f"{model} Q{question}: no retained response tokens.")
            response_rows.append({
                "model": model, "question": question, "tokens": total,
                **{tag: counts[tag] / total for tag in TAGS},
                "SPACE": counts["SPACE"] / total,
                "PUNCT": counts["PUNCT"] / total,
                **{f"{tag}_count": counts[tag] for tag in (*TAGS, "SPACE", "PUNCT")},
            })
    metadata = {"source": "source responses", "spacy_version": spacy.__version__,
                "spacy_model": SPACY_MODEL, "model_version": nlp.meta.get("version")}
    responses = pd.DataFrame(response_rows)
    return pool_responses(responses), responses, metadata


def pool_responses(responses):
    """Pooled count / pooled denominator, reproducing Table 4 exactly."""
    rows = []
    for model in MODELS:
        group = responses.loc[responses["model"].eq(model)]
        denominators = pd.to_numeric(group["tokens"], errors="raise").to_numpy()
        if not np.isfinite(denominators).all() or not (denominators > 0).all():
            raise ValueError(f"{model}: invalid response denominator.")
        total = int(denominators.sum())
        reconstructed_total = np.zeros(len(group), dtype=np.int64)
        for tag in (*TAGS, "SPACE", "PUNCT"):
            shares = pd.to_numeric(group[tag], errors="raise").to_numpy()
            if f"{tag}_count" in group:
                counts = pd.to_numeric(group[f"{tag}_count"], errors="raise").to_numpy()
                if not np.isfinite(counts).all() or not np.equal(counts, np.rint(counts)).all():
                    raise ValueError(f"{model}/{tag}: non-integer POS count.")
                counts = counts.astype(np.int64)
            else:
                # Original saved shares have six significant digits. Recover
                # their integer numerators, then check the rounding tolerance.
                counts = np.rint(shares * denominators).astype(np.int64)
            if (counts < 0).any() or not np.allclose(
                counts / denominators, shares, rtol=0, atol=5.1e-7
            ):
                raise ValueError(f"{model}/{tag}: shares do not match integer counts.")
            reconstructed_total += counts
            rows.append({"model": model, "tag": tag, "count": int(counts.sum()),
                         "tokens": total, "share": float(counts.sum() / total)})
        if not np.array_equal(reconstructed_total, denominators):
            raise ValueError(f"{model}: POS counts do not sum to each response denominator.")
    return pd.DataFrame(rows)


def read_saved(output):
    """Reuse the per-response observations of a full run."""
    responses = load_saved(output, "pos_per_response_values.csv")
    return pool_responses(responses), responses, {"source": "saved observations"}


def validate(pooled, responses):
    if set(pooled["model"]) != set(MODELS) or set(responses["model"]) != set(MODELS):
        raise ValueError("Expected exactly GPT-4o, GPT-5.2 and RAG.")
    if pooled.duplicated(["model", "tag"]).any():
        raise ValueError("Duplicate pooled model/tag entry.")
    if responses.duplicated(["model", "question"]).any():
        raise ValueError("Duplicate model/question observation.")
    expected = set(range(1, EXPECTED_QUESTIONS + 1))
    for model in MODELS:
        group = responses.loc[responses["model"].eq(model)]
        if set(group["question"]) != expected:
            raise ValueError(f"{model}: expected questions 1 through 24, once each.")
        if not set(TAGS).issubset(pooled.loc[pooled["model"].eq(model), "tag"]):
            raise ValueError(f"{model}: missing pooled POS categories.")
    for tag in TAGS:
        if tag not in responses:
            responses[tag] = 0.0
        responses[tag] = pd.to_numeric(responses[tag], errors="raise")
    values = responses[list(TAGS)].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Response shares must be finite and in [0, 1].")
    shares = pd.to_numeric(pooled["share"], errors="raise").to_numpy()
    if not np.isfinite(shares).all() or not ((shares >= 0) & (shares <= 1)).all():
        raise ValueError("Pooled shares must be finite and in [0, 1].")
    totals = pd.to_numeric(responses["tokens"], errors="raise").to_numpy()
    if not np.isfinite(totals).all() or not (totals > 0).all():
        raise ValueError("Response denominators must be positive.")
    return {model: responses.loc[responses["model"].eq(model)]
            .set_index("question").sort_index() for model in MODELS}


def rank_biserial(differences):
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def analyse(pooled, responses):
    wide = validate(pooled, responses)
    corpus = pooled.set_index(["model", "tag"])["share"]
    occurrences = {tag: sum((wide[m][tag] > 0).sum() for m in MODELS) for tag in TAGS}
    eligible = [tag for tag in TAGS if occurrences[tag] >= MIN_RESPONSES]
    n_omnibus, n_pairwise = len(eligible), len(eligible) * len(PAIRS)
    rows = []
    for tag in TAGS:
        row = {"tag": tag, "n_questions": EXPECTED_QUESTIONS,
               "responses_with_feature": int(occurrences[tag]),
               "tested": tag in eligible,
               **{f"{m} pooled percent": float(corpus.loc[(m, tag)] * 100)
                  for m in MODELS}}
        if tag not in eligible:
            row["note"] = f"Not tested: fewer than {MIN_RESPONSES} responses"
            rows.append(row)
            continue
        values = [wide[m][tag].to_numpy() for m in MODELS]
        matrix = np.column_stack(values)
        if np.all(matrix == matrix[:, :1]):
            q, p = 0.0, 1.0
        else:
            q, p = stats.friedmanchisquare(*values)
        row.update({"friedman_chi2": float(q), "friedman_p": float(p),
                    "friedman_p_bonferroni": min(1.0, float(p) * n_omnibus),
                    "kendalls_w": float(q / (EXPECTED_QUESTIONS * (len(MODELS) - 1)))})
        row["overall_significant"] = row["friedman_p_bonferroni"] < .05
        for first, second in PAIRS:
            name = f"{first} vs {second}"
            d = wide[first][tag].to_numpy() - wide[second][tag].to_numpy()
            # Same default zero handling and p-value selection as the original
            # analysis. Keep full-precision values when recomputing from text.
            if np.all(d == 0):
                statistic, p_pair = 0.0, 1.0
            else:
                result = stats.wilcoxon(d, zero_method="wilcox",
                                        alternative="two-sided", method="auto")
                statistic, p_pair = float(result.statistic), float(result.pvalue)
            row[f"{name}: rank-biserial r"] = rank_biserial(d)
            row[f"{name}: wilcoxon_statistic"] = statistic
            row[f"{name}: p"] = p_pair
            row[f"{name}: p (Bonferroni)"] = min(1.0, p_pair * n_pairwise)
            row[f"{name}: significant"] = (
                row["overall_significant"] and row[f"{name}: p (Bonferroni)"] < .05
            )
        rows.append(row)
    return pd.DataFrame(rows), eligible


def caption(eligible):
    return (
        r"POS composition and matched comparisons across 24 questions "
        r"(72 responses). Pooled shares are percentages of each system's "
        r"non-punctuation tokens, including SPACE in the denominator. "
        r"Tests use the 24 question-matched response proportions. "
        r"Kendall's $W\in[0,1]$ measures the consistency of system rankings "
        r"across questions, with 1 indicating perfect consistency. Stars on "
        r"$W$ denote Friedman $p$-values after Bonferroni correction across "
        + str(len(eligible)) + r" eligible tags. Pairwise $r\in[-1,1]$ is the "
        r"matched-pairs rank-biserial effect size: positive values indicate "
        r"higher proportions for the first-named system and negative values "
        r"for the second; $r=\pm1$ means all nonzero paired differences have "
        r"the same direction. Stars on $r$ denote two-sided Wilcoxon $p$-values "
        r"after Bonferroni correction across all "
        + str(len(eligible) * len(PAIRS)) + r" pairwise comparisons. "
        r"Pairwise stars are shown only when the corrected overall test is "
        r"also significant. "
        r"$^{*}p_{\mathrm{adj}}<.05$, $^{**}p_{\mathrm{adj}}<.01$, "
        r"$^{***}p_{\mathrm{adj}}<.001$. Tags occurring in fewer than 10 "
        r"responses are not tested. SPACE and PUNCT are excluded from this "
        r"family of tests."
    )


def make_latex(table, eligible):
    lines = [
        r"% Requires \usepackage{booktabs}. Use table* for a two-column span.",
        r"\begin{table}[tbp]", r"\centering", r"\caption{" + caption(eligible) + "}",
        r"\label{tab:pos}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{Pooled share (\%)} & Overall"
        r" & \multicolumn{3}{c}{Pairwise effect ($r$)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-8}",
        r"Tag & GPT-4o & GPT-5.2 & RAG & $W$ & 4o--5.2 & 4o--RAG & 5.2--RAG \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [row["tag"]] + [f"{row[f'{m} pooled percent']:.2f}" for m in MODELS]
        if row["tested"]:
            cells.append(latex_effect(row["kendalls_w"], row["friedman_p_bonferroni"]))
            for first, second in PAIRS:
                name = f"{first} vs {second}"
                cells.append(latex_effect(row[f"{name}: rank-biserial r"],
                                          row[f"{name}: p (Bonferroni)"]
                                          if row["overall_significant"] else 1.0))
        else:
            cells.append(r"\multicolumn{4}{l}{\emph{Not tested: fewer than 10 responses}}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="Default: 1_pos/output")
    parser.add_argument("--from-saved", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or OUTPUT_DIR
    pooled, responses, metadata = read_saved(output) if args.from_saved else recompute()
    results, eligible = analyse(pooled, responses)
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / "pos_pooled_values.csv", index=False, float_format="%.15g")
    responses.sort_values(["model", "question"]).to_csv(
        output / "pos_per_response_values.csv", index=False, float_format="%.15g"
    )
    results.to_csv(output / "pos_tests.csv", index=False, float_format="%.15g")
    (output / "table4.tex").write_text(make_latex(results, eligible), encoding="utf-8")
    metadata.update({
        "scipy_version": scipy.__version__, "n_questions": EXPECTED_QUESTIONS,
        "minimum_responses_with_feature": MIN_RESPONSES, "eligible_tags": eligible,
        "friedman_correction": "Bonferroni", "friedman_family_size": len(eligible),
        "wilcoxon_correction": "Bonferroni",
        "wilcoxon_family_size": len(eligible) * len(PAIRS),
        "pairwise_significance_gate": "Bonferroni-corrected Friedman p < .05",
        "denominator": "tokens satisfying not token.is_punct; SPACE retained",
        "pooled_descriptives": "sum of response POS counts / sum of response denominators",
        "wilcoxon_zero_method": "wilcox", "wilcoxon_method": "auto",
    })
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nEligible POS tags: {len(eligible)}; pairwise family: {3 * len(eligible)}.")
    if len(eligible) != 15:
        print("NOTE: this input differs from the paper's 15 eligible tags; "
              "the generated caption uses the actual family size.")
    print(results.to_string(index=False))
    print(f"\nSaved Table 4 LaTeX and CSV results to {output.resolve()}")


if __name__ == "__main__":
    main()
