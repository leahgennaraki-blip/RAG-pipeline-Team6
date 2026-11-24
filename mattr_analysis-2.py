#!/usr/bin/env python3
"""Generate the revised MATTR table and question-reset MATTR-100 plot.

Place this file in the repository root beside cleaned_model_responses.py.

Default: recompute from assets/Team6_comparative_analysis.csv.
    python -m pip install numpy pandas scipy spacy matplotlib
    python -m spacy download en_core_web_lg
    python mattr_analysis.py

Rebuild from the repository's existing saved scores, without spaCy:
    python mattr_analysis.py --from-saved
After a full run, --from-saved also reuses the saved window-level curves.
Original repository summary scores alone cannot reconstruct those curves.
Outputs include mattr_100_by_question.pdf, a PNG preview, and window CSV.

Descriptives: mean MATTR-500 over each model's whole concatenated corpus.
Inference: ONLY per-response MATTR-100, paired across the same 24 questions.
One Friedman test; three two-sided Wilcoxon signed-rank tests; Bonferroni
correction across these three pairs. MATTR is a separate testing family.
No MATTR-500 significance tests or bootstrap confidence intervals are run.

The corpus and individual answers are lemmatized separately, deliberately
matching mattr_moving_average.py and significance_tests.py respectively.
Alphabetic tokens only; lowercase lemmas; stopwords retained; stride = 1.
Tests assume independent question blocks. Wilcoxon location comparisons
also assume symmetric paired differences. Shared conversation context may
violate independence; these tests do not remove that limitation.
"""

import argparse
from collections import Counter
from importlib import util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODELS = ("GPT-4o", "GPT-5.2", "RAG")
MODEL_KEYS = ("gpt_4", "gpt_5_2", "rag")
ALIASES = {"GPT-4": "GPT-4o", "GPT 5.2": "GPT-5.2"}
EXPECTED_QUESTIONS = 24
CORPUS_WINDOW = 500
RESPONSE_WINDOW = 100
SPACY_MODEL = "en_core_web_lg"
COLOURS = {"GPT-4o": "#2a78d6", "GPT-5.2": "#eb6834", "RAG": "#1baf7a"}


def window_ttr(tokens, window):
    """TTR in every overlapping full window, with a stride of one.

    Refuse short texts rather than silently substitute a different window.
    """
    if window < 1 or len(tokens) < window:
        raise ValueError(
            f"Need at least {window} tokens for this MATTR; got {len(tokens)}."
        )
    counts = Counter(tokens[:window])
    distinct = [len(counts)]
    for start in range(1, len(tokens) - window + 1):
        outgoing = tokens[start - 1]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[start + window - 1]] += 1
        distinct.append(len(counts))
    return np.asarray(distinct, dtype=float) / window


def mattr(tokens, window):
    """The mean of the window TTR values."""
    return float(window_ttr(tokens, window).mean())


def load_responses(repo_root):
    """Use the repository's actual cleaning function and column mapping."""
    cleaning_path = repo_root / "cleaned_model_responses.py"
    if not cleaning_path.is_file():
        raise FileNotFoundError(f"Missing repository helper: {cleaning_path}")
    spec = util.spec_from_file_location("mattr_repository_cleaning", cleaning_path)
    cleaning = util.module_from_spec(spec)
    spec.loader.exec_module(cleaning)
    source = repo_root / "assets" / "Team6_comparative_analysis.csv"
    data = pd.read_csv(source, sep=";", dtype=str, keep_default_na=False)
    # The source's first data row contains model labels, not a question.
    data = data.iloc[1:]
    data = data.loc[data["Question"].str.strip().ne("")]
    if len(data) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} questions; found {len(data)} in {source}."
        )
    responses = {}
    for key, model in zip(MODEL_KEYS, MODELS):
        texts = data[cleaning.MODELS[key]].map(cleaning.clean_text).tolist()
        if any(not text for text in texts):
            raise ValueError(f"{model}: an answer is empty; cannot form matched data.")
        responses[model] = texts
    return responses


def recompute(repo_root):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "Install spaCy and en_core_web_lg (see this script's opening "
            "instructions), or use --from-saved."
        ) from exc
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"Missing spaCy model: run python -m spacy download {SPACY_MODEL}"
        ) from exc
    responses = load_responses(repo_root)
    corpus_rows, response_rows, window_rows = [], [], []
    for model in MODELS:
        texts = responses[model]
        corpus_text = "\n\n".join(texts)
        nlp.max_length = max(nlp.max_length, len(corpus_text) + 1)
        print(f"Processing {model}...", flush=True)
        # Process the joined text in one pass, exactly as the original plot
        # script does. Do NOT concatenate individually lemmatized answers:
        # spaCy's contextual analyses can differ at response boundaries.
        corpus_lemmas = [
            token.lemma_.lower() for token in nlp(corpus_text) if token.is_alpha
        ]
        corpus_rows.append({
            "model": model,
            "mattr_500": mattr(corpus_lemmas, CORPUS_WINDOW),
            "tokens": len(corpus_lemmas),
            "windows": len(corpus_lemmas) - CORPUS_WINDOW + 1,
        })
        for question, doc in enumerate(nlp.pipe(texts), start=1):
            lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
            if len(lemmas) < RESPONSE_WINDOW:
                raise ValueError(
                    f"{model}, question {question}: fewer than 100 retained tokens."
                )
            ratios = window_ttr(lemmas, RESPONSE_WINDOW)
            response_rows.append({
                "model": model, "question": question, "tokens": len(lemmas),
                "mattr_100": float(ratios.mean()),
            })
            # Each answer starts at lemma zero: no window can cross answers.
            window_rows.extend({
                "model": model, "question": question,
                "first_lemma": start, "ttr_100": float(value),
            } for start, value in enumerate(ratios))
    windows = pd.DataFrame(window_rows)
    windows.attrs["spacy_version"] = spacy.__version__
    windows.attrs["model_version"] = nlp.meta.get("version", "unknown")
    return pd.DataFrame(corpus_rows), pd.DataFrame(response_rows), windows


def read_saved(repo_root, output=None):
    """Use the ORIGINAL whole-corpus scores, not the permutation output."""
    # Prefer a complete, coherent cache from this revised script.
    if output is not None and all((output / name).is_file() for name in (
        "mattr_500_corpus.csv", "mattr_100_per_response_values.csv",
        "mattr_100_windows.csv",
    )):
        return (
            pd.read_csv(output / "mattr_500_corpus.csv"),
            pd.read_csv(output / "mattr_100_per_response_values.csv"),
            pd.read_csv(output / "mattr_100_windows.csv"),
        )
    corpus = pd.read_csv(repo_root / "mattr_output" / "mattr_500.csv")
    corpus = corpus.rename(columns={"mattr": "mattr_500"})
    responses = pd.read_csv(
        repo_root / "significance_tests_output" / "mattr_100_per_response_values.csv"
    )
    for table in (corpus, responses):
        table["model"] = table["model"].replace(ALIASES)
    return corpus, responses, None


def validate_windows(windows, responses):
    """Ensure every plotted segment matches one complete response."""
    keys = ["model", "question"]
    expected = responses.set_index(keys)
    groups = windows.groupby(keys, sort=False)
    if set(groups.groups) != set(expected.index):
        raise ValueError("Window data and response data have different question sets.")
    if windows.duplicated(keys + ["first_lemma"]).any():
        raise ValueError("Duplicate window position within a response.")
    values = windows["ttr_100"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError("Window ratios must be finite and between 0 and 1.")
    for key, group in groups:
        row = expected.loc[key]
        starts = group["first_lemma"].sort_values().to_numpy()
        required = np.arange(int(row["tokens"]) - RESPONSE_WINDOW + 1)
        if not np.array_equal(starts, required):
            raise ValueError(f"{key}: missing, invalid or cross-boundary window.")
        if not np.isclose(group["ttr_100"].mean(), row["mattr_100"],
                          rtol=0, atol=5e-7):
            raise ValueError(f"{key}: curve mean does not match its MATTR-100 score.")


def plot_mattr_100(windows, responses, output):
    """Separate window trajectories for all 72 responses; gaps at boundaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    validate_windows(windows, responses)
    questions = sorted(responses["question"].unique())
    widths = responses.groupby("question")["tokens"].mean().loc[questions].to_numpy()
    edges = np.r_[0.0, np.cumsum(widths)]
    centres = (edges[:-1] + edges[1:]) / 2
    grouped = windows.groupby(["model", "question"], sort=False)
    means = responses.set_index(["model", "question"])["mattr_100"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(17, 6))
    fig.subplots_adjust(left=.055, right=.985, bottom=.23, top=.78)
    for i, q in enumerate(questions):
        left, right = edges[i:i + 2]
        if i % 2:
            ax.axvspan(left, right, color="#f4f5f7", lw=0, zorder=0)
        if i:
            ax.axvline(left, color="#b5b8be", lw=.65, ls="--", zorder=1)
        gap = .055 * (right - left)
        for model in MODELS:
            part = grouped.get_group((model, q)).sort_values("first_lemma")
            # Spread windows in text order within the shared question band.
            # Make a separate plotting call for every answer: never join them.
            x = (np.linspace(left + gap, right - gap, len(part))
                 if len(part) > 1 else np.array([centres[i]]))
            ax.plot(x, part["ttr_100"], color=COLOURS[model], lw=.85,
                    alpha=.92, zorder=3)
            ax.scatter([centres[i]], [means.loc[(model, q)]], s=26,
                       color=COLOURS[model], edgecolors="white", linewidths=.55,
                       zorder=5)
    low = max(0, np.floor(windows["ttr_100"].min() * 20) / 20 - .025)
    high = min(1, np.ceil(windows["ttr_100"].max() * 20) / 20 + .025)
    ax.set_ylim(low, high)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks(centres, [str(q) for q in questions])
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlabel("Question", labelpad=9)
    ax.set_ylabel("Type-token ratio in 100-lemma window", labelpad=10)
    ax.grid(axis="y", color="#e2e5e9", lw=.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#9ca3af")
    fig.text(.055, .94, "MATTR-100 within individual responses",
             fontsize=17, fontweight="bold", color="#17212e")
    fig.text(.055, .89,
             "100-lemma windows restart at every question; no window crosses an answer boundary.",
             fontsize=11, color="#485260")
    handles = [Line2D([0], [0], color=COLOURS[m], lw=2, label=m) for m in MODELS]
    handles.append(Line2D([0], [0], color="#555b64", marker="o", ls="",
                          markersize=5, label="Dots: per-response MATTR-100"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .87),
               ncol=4, frameon=False, columnspacing=2.2)
    fig.text(.055, .105,
             "Question width: mean response length across systems. Within each question, windows follow relative text order.",
             fontsize=9.5, color="#485260")
    fig.text(.055, .065,
             "Each dot is the mean of that answer's window ratios, and the observation used in the paired statistical tests.",
             fontsize=9.5, color="#485260")
    fig.text(.055, .025,
             "Separate window calculations do not establish statistical independence between answers.",
             fontsize=9, color="#66707d")
    for ext in ("pdf", "png"):
        fig.savefig(output / f"mattr_100_by_question.{ext}", dpi=220,
                    facecolor="white")
    plt.close(fig)


def validate(corpus, responses):
    for table, score in ((corpus, "mattr_500"), (responses, "mattr_100")):
        if set(table["model"]) != set(MODELS):
            raise ValueError("Expected exactly GPT-4o, GPT-5.2 and RAG.")
        values = pd.to_numeric(table[score], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
            raise ValueError(f"{score}: scores must be finite and between 0 and 1.")
        table[score] = values
    if len(corpus) != len(MODELS) or corpus["model"].duplicated().any():
        raise ValueError("Expected exactly one corpus score per system.")
    if responses.duplicated(["model", "question"]).any():
        raise ValueError("Duplicate model/question response.")
    expected = set(range(1, EXPECTED_QUESTIONS + 1))
    for model in MODELS:
        rows = responses.loc[responses["model"].eq(model)]
        if set(rows["question"]) != expected:
            raise ValueError(f"{model}: expected questions 1 through 24, once each.")
    lengths = pd.to_numeric(responses["tokens"], errors="raise").to_numpy()
    if not np.isfinite(lengths).all() or (lengths < RESPONSE_WINDOW).any():
        raise ValueError("Every response must have at least 100 retained tokens.")
    return responses.pivot(
        index="question", columns="model", values="mattr_100"
    ).sort_index().loc[:, list(MODELS)]


def infer(wide):
    """Friedman and paired Wilcoxon use MATTR-100 only."""
    values = wide.to_numpy()
    n, k = values.shape
    if np.all(values == values[:, :1]):
        chi2, p = 0.0, 1.0
    else:
        chi2, p = stats.friedmanchisquare(*values.T)
    overall = {
        "n_questions": n, "friedman_chi2": float(chi2), "df": k - 1,
        "friedman_p": float(p), "kendalls_w": float(chi2 / (n * (k - 1))),
    }
    pairs = list(combinations(MODELS, 2))
    rows = []
    for first, second in pairs:
        # Round only subtraction noise, not the input scores. This avoids
        # spuriously separating mathematically tied absolute differences.
        differences = np.round(wide[first].to_numpy() - wide[second].to_numpy(), 12)
        nonzero = differences[differences != 0]
        if len(nonzero) == 0:
            statistic, p_raw, effect, method = 0.0, 1.0, 0.0, "all differences zero"
        else:
            ranks = stats.rankdata(np.abs(nonzero), method="average")
            effect = (
                ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()
            ) / ranks.sum()
            # The current 24-question data have no zeros or tied absolute
            # differences, so their two-sided p-values are exact.
            exact = (
                len(nonzero) == len(differences)
                and len(np.unique(np.abs(nonzero))) == len(nonzero)
                and len(nonzero) <= 50
            )
            method = "exact" if exact else "auto"
            result = stats.wilcoxon(
                differences, zero_method="wilcox", alternative="two-sided",
                method=method,
            )
            statistic, p_raw = float(result.statistic), float(result.pvalue)
        rows.append({
            "first": first, "second": second, "n_questions": n,
            "nonzero_pairs": len(nonzero), "wilcoxon_statistic": statistic,
            "rank_biserial_r": float(effect), "p_raw": p_raw,
            "p_bonferroni": min(1.0, len(pairs) * p_raw),
            "wilcoxon_method": method,
        })
    return overall, pd.DataFrame(rows)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def latex_effect(value, p):
    suffix = stars(p)
    return "$" + f"{value:.2f}" + ("^{" + suffix + "}" if suffix else "") + "$"


def make_latex(corpus, overall, pairwise):
    scores = corpus.set_index("model")["mattr_500"]
    cells = [f"{scores[model]:.4f}" for model in MODELS]
    cells.append(latex_effect(overall["kendalls_w"], overall["friedman_p"]))
    cells.extend(
        latex_effect(row.rank_biserial_r, row.p_bonferroni)
        for row in pairwise.itertuples(index=False)
    )
    row = "MATTR & " + " & ".join(cells) + r" \\"
    return r"""% Requires \usepackage{booktabs} in the preamble.
% For a two-column manuscript, change table to table* if necessary.
\begin{table}[tbp]
\centering
\caption{Corpus-level lexical diversity and matched comparisons across
24 questions (72 responses). The three system columns report mean MATTR
over all overlapping 500-lemma windows in each system's concatenated
responses, in question order. All inferential results use only the
24 question-matched, per-response MATTR-100 scores. $W$ is Kendall's
concordance effect size for the Friedman test. Pairwise $r$ is the
matched-pairs rank-biserial effect size; positive values indicate higher
scores for the first-named system and negative values for the second.
Stars on $W$ denote the overall Friedman $p$-value (one test, unadjusted);
stars on $r$ denote two-sided paired Wilcoxon $p$-values after Bonferroni
correction across the three MATTR comparisons.
$^{*}p<.05$, $^{**}p<.01$, and $^{***}p<.001$, using adjusted $p$-values
for pairwise comparisons.}
\label{tab:mattr}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{3}{c}{Corpus MATTR-500}
 & \multicolumn{1}{c}{Overall}
 & \multicolumn{3}{c}{Pairwise effect ($r$)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-8}
Measure & GPT-4o & GPT-5.2 & RAG & $W$
 & 4o--5.2 & 4o--RAG & 5.2--RAG \\
\midrule
""" + row + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: REPO_ROOT/mattr_analysis_output")
    parser.add_argument("--from-saved", action="store_true",
                        help="Use existing corpus-500 and per-response-100 CSVs.")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output_dir or repo_root / "mattr_analysis_output"
    corpus, responses, windows = (
        read_saved(repo_root, output) if args.from_saved else recompute(repo_root)
    )
    wide = validate(corpus, responses)
    overall, pairwise = infer(wide)
    scores = corpus.set_index("model")["mattr_500"]
    summary = {"measure": "MATTR",
               **{f"{model} corpus MATTR-500": scores[model] for model in MODELS},
               **overall}
    for row in pairwise.itertuples(index=False):
        name = f"{row.first} vs {row.second}"
        summary[f"{name}: rank-biserial r"] = row.rank_biserial_r
        summary[f"{name}: p (Bonferroni)"] = row.p_bonferroni
    output.mkdir(parents=True, exist_ok=True)
    if windows is not None:
        validate_windows(windows, responses)
    tables = {
        "mattr_500_corpus.csv": corpus,
        "mattr_100_per_response_values.csv": responses.sort_values(["model", "question"]),
        "mattr_100_overall.csv": pd.DataFrame([overall]),
        "mattr_100_pairwise.csv": pairwise,
        "mattr_table.csv": pd.DataFrame([summary]),
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False, float_format="%.15g")
    (output / "mattr_table.tex").write_text(
        make_latex(corpus, overall, pairwise), encoding="utf-8"
    )
    if windows is not None:
        windows.to_csv(output / "mattr_100_windows.csv", index=False,
                       float_format="%.15g")
        plot_mattr_100(windows, responses, output)
        if windows.attrs:
            import json
            (output / "nlp_versions.json").write_text(
                json.dumps({"model": SPACY_MODEL, **windows.attrs}, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"\nSaved MATTR-100 PDF/PNG plot and window-level CSV to {output.resolve()}")
    else:
        print("\nThe original summary CSVs contain no window-level curves. "
              "Run without --from-saved once to generate the plot.")
    print("\nCorpus MATTR-500 (descriptive only):")
    print(scores.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nMATTR-100 Friedman: chi2({overall['df']})="
          f"{overall['friedman_chi2']:.6g}, W={overall['kendalls_w']:.6g}, "
          f"p={overall['friedman_p']:.6g}")
    print(pairwise.to_string(index=False))
    if args.from_saved:
        print("\nUsed previously saved scores; did not rerun text preprocessing.")
    print(f"\nSaved CSV results and mattr_table.tex to {output.resolve()}")


if __name__ == "__main__":
    main()
