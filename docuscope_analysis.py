#!/usr/bin/env python3
"""Build Table 6 and Appendix VI using the matched-analysis layout of Tables 4/5.

Upload this script beside cleaned_model_responses.py in the repository root.
    python -m pip install numpy pandas scipy
    python docuscope_analysis.py

DocuScope itself must already have produced docuscope_input/ and a run under
docuscope_input_output/. This script analyzes those exports; it does not run
DocuScope or change its dictionary. --run-dir can select a particular run.

Descriptives: 100 * category pattern starts / all DocuScope corpus tokens.
Inference: 24 question-matched response proportions for each category.
Exclude categories present in fewer than 10 of the 72 responses.
Friedman tests + Kendall's W; two-sided paired Wilcoxon signed-rank tests
+ matched-pairs rank-biserial r. Bonferroni corrections use all eligible
categories separately for omnibus tests and all three pairs per category.
For the current data, the correction multipliers are 23 and 69.
Pairwise significance stars require both the category's corrected overall
test and the pair's corrected test to pass .05; numeric p-values are saved
for every comparison, including categories failing the overall test.

The main LaTeX table retains the original Table 6 display rule: W >= .30
and a significant corrected omnibus test. Corrections occur BEFORE this
display filter. --all-categories includes all categories in the main table.
A full-category LaTeX table and complete result CSV are always saved.
Appendix VI uses LAT token occupancy (every token carrying a LAT / all
DocuScope tokens), not category pattern starts. Its testing family is
separate: all eligible LATs and their three pairs, currently 148 and 444.
It includes tags passing the corrected omnibus test and at least one
corrected pairwise test, without a W threshold. Longtable LaTeX, all-tag
results, and response counts/proportions are written alongside Table 6.

--from-saved reconstructs integer counts from the original repository's
rounded response proportions and checks them against the DocuScope exports.
Default mode realigns the token exports to the cleaned source responses,
using the original character-matching rule and auditing unmatched tokens.

The Table 6 category values are pattern frequencies, not percentages of
tokens carrying any category tag. They need not sum to 100%. Appendix VI
instead reports LAT token occupancy. Matched inference assumes
independent question blocks; shared conversation context can violate this.
"""

import argparse
import json
import re
from importlib import util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODELS = ("GPT-4o", "GPT-5.2", "RAG")
MODEL_KEYS = ("gpt_4", "gpt_5_2", "rag")
ALIASES = {"GPT-4": "GPT-4o", "GPT 5.2": "GPT-5.2"}
PAIRS = tuple(combinations(MODELS, 2))
EXPECTED_QUESTIONS = 24
MIN_RESPONSES = 10
ALPHA = .05
MIN_DISPLAY_W = .30
ALIGN_LOOKAHEAD = 50
MAX_UNMATCHED_FRACTION = .01
CATEGORY_PREFIXES = {
    "Character": "Char", "Citation": "Citation", "Confidence": "Confidence",
    "Contingent": "Contingent", "Description": "Descript", "Exposition": "Expo",
    "First Person": "FP", "Facilitate": "Facilitate", "Forceful": "Forceful",
    "Future": "Future", "Inquiry": "Inquiry", "Interactive": "Inter",
    "Metadiscourse": "Metadiscourse", "Narrative": "Narr", "Negative": "Neg",
    "Positive": "Pos", "Public": "Pub", "Reasoning": "Reasoning",
    "Responsibility": "Responsibility", "Special Topics": "SpecialTopics",
    "Strategic": "Strat", "Uncertainty": "Uncertainty", "Updates": "Updates",
}
CATEGORIES = tuple(CATEGORY_PREFIXES)


def select_run(root, requested):
    if requested is not None:
        run = requested.resolve()
    else:
        runs = sorted((root / "docuscope_input_output").glob("docuscope_input-*"))
        if not runs:
            raise FileNotFoundError("No DocuScope export run found.")
        run = runs[-1]
    if not (run / "token_csv").is_dir():
        raise FileNotFoundError(f"Missing token_csv directory in {run}")
    return run


def export_table(run, prefix):
    paths = sorted((run / "csv").glob(f"{prefix}_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {prefix}_ export in {run / 'csv'}, got {paths}.")
    table = pd.read_csv(paths[0]).set_index("Filename")
    filenames = [f"{key}.txt" for key in MODEL_KEYS]
    if table.index.duplicated().any() or not set(filenames).issubset(table.index):
        raise ValueError(f"{paths[0]}: missing or duplicate model filenames.")
    return table.loc[filenames, ["Tokens", *CATEGORIES]]


def load_responses(root):
    helper = root / "cleaned_model_responses.py"
    spec = util.spec_from_file_location("docuscope_repository_cleaning", helper)
    cleaning = util.module_from_spec(spec)
    spec.loader.exec_module(cleaning)
    frame = pd.read_csv(root / "assets" / "Team6_comparative_analysis.csv",
                        sep=";", dtype=str, keep_default_na=False)
    frame = frame.iloc[1:]  # First data row contains the model labels.
    frame = frame.loc[frame["Question"].str.strip().ne("")]
    if len(frame) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected 24 questions; found {len(frame)}.")
    result = {}
    for key, label in zip(MODEL_KEYS, MODELS):
        texts = frame[cleaning.MODELS[key]].map(cleaning.clean_text).tolist()
        if any(not text for text in texts):
            raise ValueError(f"{label}: an answer is empty.")
        tagged_text = (root / "docuscope_input" / f"{key}.txt").read_text("utf-8")
        if "\n\n".join(texts) != tagged_text:
            raise ValueError(
                f"{key}.txt does not match the cleaned source responses. "
                "Regenerate the text inputs and rerun DocuScope first."
            )
        result[label] = texts
    return result


def first_field(head):
    quoted = re.match(r'"((?:[^"]|"")*)",', head)
    return quoted.group(1).replace('""', '"') if quoted else head.split(",", 1)[0]


def read_tokens(path):
    """Read the headerless exports despite malformed quotes in token_lower."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                head, token_type, tag, position = line.rstrip("\n").rsplit(",", 3)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: malformed token row.") from exc
            rows.append({
                "token": first_field(head), "tag": tag,
                "pattern_start": bool(tag) and position == "0",
            })
    if not rows:
        raise ValueError(f"Empty token export: {path}")
    return pd.DataFrame(rows)


def category_of(tag):
    for category, prefix in sorted(CATEGORY_PREFIXES.items(),
                                   key=lambda item: -len(item[1])):
        if tag.startswith(prefix):
            return category
    if tag and not tag.startswith("Other"):
        raise ValueError(f"Unrecognised DocuScope tag: {tag!r}")
    return None


def assign_responses(tokens, responses):
    """Original matching rule, with an explicit audit of lexical mismatches."""
    parts, question_of_char = [], []
    for question, response in enumerate(responses, 1):
        compact = re.sub(r"\s+", "", response)
        parts.append(compact)
        question_of_char.extend([question] * len(compact))
    text = "".join(parts)
    assignments, audit, position = [], [], 0
    for index, token in enumerate(tokens):
        compact = re.sub(r"\s+", "", token)
        found = (text.find(compact, position, position + ALIGN_LOOKAHEAD + len(compact))
                 if compact else -1)
        if found >= 0:
            position = found + len(compact)
        question = question_of_char[min(max(position - 1, 0), len(text) - 1)]
        assignments.append(question)
        # Preserve original treatment: unmatched punctuation is not counted
        # as a lexical alignment failure. It stays at the current position.
        if found < 0 and re.search(r"\w", compact):
            audit.append({"token_index": index, "token": token, "question": question})
    return assignments, audit


def verify_exports(counts, count_export, norm_export):
    """Require exact pooled counts and the same DocuScope denominators."""
    pooled_rows = []
    for key, model in zip(MODEL_KEYS, MODELS):
        group = counts.loc[counts["model"].eq(model)]
        totals = group[["tokens", *CATEGORIES]].sum()
        filename = f"{key}.txt"
        expected = count_export.loc[filename]
        if int(totals["tokens"]) != int(expected["Tokens"]):
            raise ValueError(f"{model}: token total does not match C_ export.")
        np.testing.assert_array_equal(
            totals[list(CATEGORIES)].to_numpy(dtype=int),
            expected[list(CATEGORIES)].to_numpy(dtype=int),
            err_msg=f"{model}: category pattern counts differ from C_ export",
        )
        shares = totals[list(CATEGORIES)].astype(float) / totals["tokens"]
        if int(norm_export.loc[filename, "Tokens"]) != int(totals["tokens"]):
            raise ValueError(f"{model}: token total does not match N_ export.")
        np.testing.assert_allclose(
            shares.to_numpy(), norm_export.loc[filename, list(CATEGORIES)].to_numpy(float),
            rtol=0, atol=1e-14, err_msg=f"{model}: shares differ from N_ export",
        )
        pooled_rows.append({"model": model, "tokens": int(totals["tokens"]), **shares})
    return pd.DataFrame(pooled_rows)


def recompute(root, run):
    responses = load_responses(root)
    rows, lat_rows, alignment_rows, unmatched_rows = [], [], [], []
    for key, model in zip(MODEL_KEYS, MODELS):
        tokens = read_tokens(run / "token_csv" / f"{key}_tokens.csv")
        tokens["question"], unmatched = assign_responses(tokens["token"], responses[model])
        fraction = len(unmatched) / len(tokens)
        alignment_rows.append({
            "model": model, "tokens": len(tokens), "unmatched_lexical_tokens": len(unmatched),
            "unmatched_fraction": fraction, "allowed_fraction": MAX_UNMATCHED_FRACTION,
        })
        if fraction > MAX_UNMATCHED_FRACTION:
            raise ValueError(
                f"{model}: {len(unmatched)}/{len(tokens)} lexical tokens failed alignment."
            )
        unmatched_rows.extend({"model": model, **item} for item in unmatched)
        tokens["category"] = tokens["tag"].map(category_of)
        for question, group in tokens.groupby("question"):
            found = group.loc[group["pattern_start"], "category"].dropna().value_counts()
            tags = group.loc[group["tag"].ne("") & ~group["tag"].str.startswith("Other"), "tag"]
            rows.append({
                "model": model, "question": int(question), "tokens": len(group),
                **{category: int(found.get(category, 0)) for category in CATEGORIES},
            })
            lat_rows.append({"model": model, "question": int(question),
                             "tokens": len(group), **tags.value_counts().to_dict()})
    lat_counts = pd.DataFrame(lat_rows).fillna(0)
    lat_counts[lat_counts.columns[3:]] = lat_counts.iloc[:, 3:].astype(int)
    return pd.DataFrame(rows), lat_counts, pd.DataFrame(alignment_rows), pd.DataFrame(
        unmatched_rows, columns=["model", "token_index", "token", "question"]
    )


def from_saved(root, family="categories"):
    path = root / "significance_tests_output" / f"docuscope_{family}_per_response_values.csv"
    saved = pd.read_csv(path)
    saved["model"] = saved["model"].replace(ALIASES)
    counts = saved[["model", "question", "tokens"]].copy()
    feature_counts = {}
    features = CATEGORIES if family == "categories" else tuple(saved.columns[3:])
    for category in features:
        products = saved[category].to_numpy() * saved["tokens"].to_numpy()
        rounded = np.rint(products)
        if not np.isfinite(products).all() or np.max(np.abs(products - rounded)) > .01:
            raise ValueError(f"{category}: cannot reconstruct integer counts from saved shares.")
        feature_counts[category] = rounded.astype(int)
    return pd.concat([counts, pd.DataFrame(feature_counts, index=counts.index)], axis=1)


def validate_counts(counts, features=CATEGORIES):
    if set(counts["model"]) != set(MODELS) or counts.duplicated(["model", "question"]).any():
        raise ValueError("Missing model or duplicate model/question observations.")
    expected = set(range(1, EXPECTED_QUESTIONS + 1))
    for model in MODELS:
        if set(counts.loc[counts["model"].eq(model), "question"]) != expected:
            raise ValueError(f"{model}: expected questions 1 through 24 exactly once.")
    numeric = counts[["tokens", *features]].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (numeric < 0).any()
            or not np.equal(numeric, np.floor(numeric)).all()
            or (counts["tokens"] <= 0).any()):
        raise ValueError("Counts must be finite nonnegative integers and token totals positive.")
    if counts[list(features)].sum(axis=1).gt(counts["tokens"]).any():
        raise ValueError("Feature counts cannot exceed the response's token total.")
    shares = counts.copy()
    shares[list(features)] = counts[list(features)].div(counts["tokens"], axis=0)
    return shares


def lat_pooled_shares(counts, run):
    """Check token-occupancy counts directly against the token exports."""
    features = tuple(counts.columns[3:])
    rows = []
    for key, model in zip(MODEL_KEYS, MODELS):
        tokens = read_tokens(run / "token_csv" / f"{key}_tokens.csv")
        tag_counts = tokens.loc[
            tokens["tag"].ne("") & ~tokens["tag"].str.startswith("Other"), "tag"
        ].value_counts()
        for tag in tag_counts.index:
            category_of(tag)  # Fail if an export uses an unrecognised category.
        if not set(tag_counts.index).issubset(features):
            raise ValueError(f"{model}: LAT response data omit an exported tag.")
        totals = counts.loc[counts["model"].eq(model), ["tokens", *features]].sum()
        if int(totals["tokens"]) != len(tokens):
            raise ValueError(f"{model}: LAT token denominator does not match export.")
        expected = tag_counts.reindex(features, fill_value=0).to_numpy(dtype=int)
        np.testing.assert_array_equal(totals[list(features)].to_numpy(dtype=int), expected,
                                      err_msg=f"{model}: pooled LAT counts do not match export")
        rows.append({"model": model, "tokens": len(tokens),
                     **(totals[list(features)].astype(float) / len(tokens))})
    return pd.DataFrame(rows)


def signed_rank(first, second):
    differences = np.round(np.asarray(first) - np.asarray(second), 12)
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0., 1., 0., "all differences zero"
    ranks = stats.rankdata(np.abs(nonzero))
    effect = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    exact = (len(nonzero) == len(differences) and len(nonzero) <= 50
             and len(np.unique(np.abs(nonzero))) == len(nonzero))
    method = "exact" if exact else "auto"
    result = stats.wilcoxon(differences, alternative="two-sided",
                            zero_method="wilcox", method=method)
    return float(result.statistic), float(result.pvalue), float(effect), method


def compare(shares, pooled, features=CATEGORIES, label="category"):
    wide = {model: shares.loc[shares["model"].eq(model)].set_index("question").sort_index()
            for model in MODELS}
    descriptive = pooled.set_index("model")
    rows = []
    for category in features:
        arrays = [wide[model][category].to_numpy() for model in MODELS]
        present = sum(int((values > 0).sum()) for values in arrays)
        row = {label: category, "n_questions": EXPECTED_QUESTIONS,
               "responses_with_feature": present, "tested": present >= MIN_RESPONSES}
        for model in MODELS:
            row[f"{model} pooled_pct"] = 100 * descriptive.loc[model, category]
            row[f"{model} median"] = float(np.median(wide[model][category]))
        if row["tested"]:
            values = np.column_stack(arrays)
            if np.all(values == values[:, :1]):
                chi2, p = 0., 1.
            else:
                chi2, p = stats.friedmanchisquare(*arrays)
            row.update(friedman_chi2=float(chi2), friedman_p=float(p),
                       kendalls_w=float(chi2 / (EXPECTED_QUESTIONS * (len(MODELS) - 1))))
            for first, second in PAIRS:
                name = f"{first} vs {second}"
                stat, p_pair, effect, method = signed_rank(wide[first][category], wide[second][category])
                row.update({f"{name} statistic": stat, f"{name} p": p_pair,
                            f"{name} r": effect, f"{name} method": method})
        rows.append(row)
    result = pd.DataFrame(rows)
    eligible = int(result["tested"].sum())
    if eligible == 0:
        raise ValueError("No features meet the minimum response frequency for testing.")
    result["friedman_p_bonferroni"] = np.minimum(result["friedman_p"] * eligible, 1.)
    for first, second in PAIRS:
        name = f"{first} vs {second}"
        result[f"{name} p_bonferroni"] = np.minimum(result[f"{name} p"] * (3 * eligible), 1.)
        result[f"{name} significant"] = (
            (result["friedman_p_bonferroni"] < ALPHA)
            & (result[f"{name} p_bonferroni"] < ALPHA)
        )
    result["omnibus_family_size"] = eligible
    result["pairwise_family_size"] = 3 * eligible
    return result.sort_values(["kendalls_w", label], ascending=[False, True],
                              na_position="last").reset_index(drop=True)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def marked(value, p):
    suffix = stars(p)
    return "$" + f"{value:.2f}" + ("^{" + suffix + "}" if suffix else "") + "$"


def make_latex(results, all_categories=False):
    eligible = int(results["tested"].sum())
    overall_count = int((results["tested"] & (results["friedman_p_bonferroni"] < ALPHA)).sum())
    selected = results if all_categories else results.loc[
        results["tested"] & (results["kendalls_w"] >= MIN_DISPLAY_W)
        & (results["friedman_p_bonferroni"] < ALPHA)
    ]
    selection = ("All categories are shown." if all_categories else
                 f"Of the {overall_count} categories passing the corrected overall test, "
                 f"the {len(selected)} with " + r"$W\geq .30$ are displayed; "
                 "corrections use the full eligible category family before this display filter.")
    caption = (
        "DocuScope category composition and matched comparisons across 24 questions "
        "(72 responses). Pooled shares are 100 times the number of category pattern "
        "starts divided by all tokens counted by DocuScope in each system's corpus; "
        "they are not percentages of tokens carrying a category tag. Tests use "
        "the 24 question-matched per-response proportions. Categories present in "
        "fewer than 10 responses are excluded from testing. "
        r"$W$ is Kendall's concordance effect size; its stars denote Friedman "
        f"$p$-values after Bonferroni correction across {eligible} eligible categories. "
        r"Pairwise $r$ is the matched-pairs rank-biserial effect size, with positive "
        "values indicating higher proportions for the first-named system. Its stars "
        "denote two-sided paired Wilcoxon $p$-values after Bonferroni correction "
        f"across all {3 * eligible} pairwise comparisons. "
        "Pairwise significance stars additionally require a significant corrected "
        "overall test for that category. "
        r"$^{*}p_{\mathrm{adj}}<.05$, $^{**}p_{\mathrm{adj}}<.01$, and "
        r"$^{***}p_{\mathrm{adj}}<.001$. " + selection
    )
    lines = [
        r"% Requires \usepackage{booktabs}. For two columns, use table* if needed.",
        r"\begin{table}[tbp]", r"\centering", r"\caption{" + caption + "}",
        r"\label{tab:docuscope}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{Pooled share (\%)} & \multicolumn{1}{c}{Overall}"
        r" & \multicolumn{3}{c}{Pairwise effect ($r$)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-8}",
        r"Category & GPT-4o & GPT-5.2 & RAG & $W$ & 4o--5.2 & 4o--RAG & 5.2--RAG \\",
        r"\midrule",
    ]
    for _, row in selected.iterrows():
        cells = [row["category"], *[f"{row[f'{model} pooled_pct']:.2f}" for model in MODELS]]
        if row["tested"]:
            cells.append(marked(row["kendalls_w"], row["friedman_p_bonferroni"]))
            cells.extend(marked(row[f"{a} vs {b} r"],
                                row[f"{a} vs {b} p_bonferroni"]
                                if row["friedman_p_bonferroni"] < ALPHA else 1.)
                         for a, b in PAIRS)
        else:
            cells.append(r"\multicolumn{4}{c}{Not tested: fewer than 10 responses}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def make_appendix_latex(results):
    """Longtable showing significant LATs, grouped by DocuScope category."""
    significant_columns = [f"{a} vs {b} significant" for a, b in PAIRS]
    selected = results.loc[
        (results["friedman_p_bonferroni"] < ALPHA)
        & results[significant_columns].any(axis=1)
    ].sort_values(["category", "tag"])
    eligible = int(results["tested"].sum())
    overall_count = int((results["tested"] & (results["friedman_p_bonferroni"] < ALPHA)).sum())
    caption = (
        f"DocuScope LAT composition and matched comparisons: {len(selected)} of "
        f"{eligible} eligible tags, grouped by category. Of {overall_count} tags "
        f"passing the corrected overall test, {len(selected)} also pass at least "
        "one corrected pairwise test and are displayed. Pooled shares are "
        "percentages of all DocuScope tokens carrying the specified LAT. "
        "Tests use the 24 question-matched per-response token proportions. "
        "Tags occurring in fewer than 10 of the 72 responses are excluded from testing. "
        r"$W$ is Kendall's concordance effect size; its stars denote Friedman "
        f"$p$-values after Bonferroni correction across all {eligible} eligible tags. "
        r"Pairwise $r$ is the matched-pairs rank-biserial effect size; positive "
        "values indicate higher proportions for the first-named system. "
        "Pairwise stars denote two-sided Wilcoxon $p$-values after Bonferroni "
        f"correction across all {3 * eligible} pairwise comparisons and additionally "
        "require a significant corrected omnibus test. A tag is listed only "
        "when the corrected omnibus test and at least one corrected pairwise "
        "test are significant; no minimum $W$ is imposed. "
        r"$^{*}p_{\mathrm{adj}}<.05$, $^{**}p_{\mathrm{adj}}<.01$, and "
        r"$^{***}p_{\mathrm{adj}}<.001$."
    )
    if selected["rare"].any():
        caption += (
            r" $^{\dagger}$ identifies a tag present in fewer than 20 responses, "
            "or with a zero per-response median in every system; interpret these "
            "sparse results cautiously."
        )
    header = [
        r"\toprule",
        r" & \multicolumn{3}{c}{Pooled share (\%)} & \multicolumn{1}{c}{Overall}"
        r" & \multicolumn{3}{c}{Pairwise effect ($r$)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-8}",
        r"Category / LAT & GPT-4o & GPT-5.2 & RAG & $W$ & 4o--5.2 & 4o--RAG & 5.2--RAG \\",
        r"\midrule",
    ]
    lines = [
        r"% Requires \usepackage{booktabs,longtable}. Use in one-column mode.",
        r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\begin{longtable}{@{}p{0.34\textwidth}rrrrrrr@{}}",
        r"\caption{" + caption + r"}\label{tab:docuscope-lats}\\",
        *header, r"\endfirsthead",
        r"\caption[]{DocuScope LAT composition and matched comparisons (continued).}\\",
        *header, r"\endhead",
        r"\midrule", r"\multicolumn{8}{r}{Continued on next page}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    for category, group in selected.groupby("category", sort=True):
        lines.extend([r"\addlinespace[3pt]",
                      r"\multicolumn{8}{@{}l}{\textbf{" + category + r"}}\\*"])
        for _, row in group.iterrows():
            # Allow wrapping at CamelCase boundaries while preserving the
            # dictionary tag name verbatim when the typeset text is read.
            tag = re.sub(r"(?<=[a-z])(?=[A-Z])", r"\\allowbreak{}", row["tag"])
            if row["rare"]:
                tag += r"$^{\dagger}$"
            cells = [r"\hspace*{.5em}" + tag,
                     *[f"{row[f'{model} pooled_pct']:.2f}" for model in MODELS],
                     marked(row["kendalls_w"], row["friedman_p_bonferroni"])]
            cells.extend(marked(row[f"{a} vs {b} r"], row[f"{a} vs {b} p_bonferroni"])
                         for a, b in PAIRS)
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-dir", type=Path, help="Specific DocuScope export run.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from-saved", action="store_true")
    parser.add_argument("--all-categories", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    run = select_run(root, args.run_dir)
    output = args.output_dir or root / "docuscope_analysis_output"
    if args.from_saved:
        counts = from_saved(root)
        lat_counts = from_saved(root, "tags")
        alignment = unmatched = None
    else:
        counts, lat_counts, alignment, unmatched = recompute(root, run)
    shares = validate_counts(counts)
    pooled = verify_exports(counts, export_table(run, "C"), export_table(run, "N"))
    results = compare(shares, pooled)
    lat_features = tuple(lat_counts.columns[3:])
    lat_shares = validate_counts(lat_counts, lat_features)
    lat_pooled = lat_pooled_shares(lat_counts, run)
    lat_results = compare(lat_shares, lat_pooled, lat_features, "tag")
    lat_results.insert(1, "category", lat_results["tag"].map(category_of))
    lat_results["rare"] = (
        (lat_results["responses_with_feature"] < 20)
        | lat_results[[f"{model} median" for model in MODELS]].eq(0).all(axis=1)
    )
    output.mkdir(parents=True, exist_ok=True)
    tables = {"docuscope_response_counts.csv": counts,
              "docuscope_per_response_values.csv": shares,
              "docuscope_pooled_shares.csv": pooled,
              "docuscope_tests.csv": results,
              "docuscope_lat_response_counts.csv": lat_counts,
              "docuscope_lat_per_response_values.csv": lat_shares,
              "docuscope_lat_pooled_shares.csv": lat_pooled,
              "docuscope_lat_tests.csv": lat_results}
    if alignment is not None:
        tables.update({"alignment_summary.csv": alignment,
                       "unmatched_lexical_tokens.csv": unmatched})
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.15g")
    (output / "table6.tex").write_text(make_latex(results, args.all_categories), encoding="utf-8")
    (output / "table6_all_categories.tex").write_text(make_latex(results, True), encoding="utf-8")
    (output / "appendix_vi.tex").write_text(make_appendix_latex(lat_results), encoding="utf-8")
    metadata = {
        "source_mode": "saved proportions reconstructed to counts" if args.from_saved else "token exports",
        "docuscope_run": str(run), "questions": EXPECTED_QUESTIONS,
        "minimum_responses": MIN_RESPONSES, "bonferroni_omnibus_family": int(results.tested.sum()),
        "bonferroni_pairwise_family": 3 * int(results.tested.sum()),
        "categories_passing_omnibus": int((results.friedman_p_bonferroni < ALPHA).sum()),
        "pairwise_significance_requires_omnibus": True,
        "lat_bonferroni_omnibus_family": int(lat_results.tested.sum()),
        "lat_bonferroni_pairwise_family": 3 * int(lat_results.tested.sum()),
        "lat_display_rows": int(lat_results[
            [f"{a} vs {b} significant" for a, b in PAIRS]
        ].any(axis=1).sum()),
        "python_packages": {"numpy": np.__version__, "pandas": pd.__version__,
                            "scipy": __import__("scipy").__version__},
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"DocuScope run: {run}")
    print(f"Bonferroni families: {int(results.tested.sum())} omnibus; "
          f"{3 * int(results.tested.sum())} pairwise.")
    if alignment is not None:
        print(alignment.to_string(index=False))
    print(results[["category", "kendalls_w", "friedman_p_bonferroni"]].to_string(index=False))
    print(f"LAT Bonferroni families: {int(lat_results.tested.sum())} omnibus; "
          f"{3 * int(lat_results.tested.sum())} pairwise.")
    print(f"Appendix VI: {metadata['lat_display_rows']} significant tags.")
    print(f"Saved results, Table 6 and Appendix VI LaTeX to {output.resolve()}")


if __name__ == "__main__":
    main()
