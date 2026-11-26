"""Test whether GPT-4o, GPT 5.2 and RAG differ significantly in MATTR, POS tags,
DocuScope token tags and DocuScope categories.

All three models answered the same 24 questions, so the unit of analysis is
the response and every test is paired by question. Treating the ~20,000
tokens per model as independent observations (chi-square / log-likelihood on
the pooled counts) would make almost every difference "significant", since
words within one answer are not independent; testing per response avoids that.

POS tags, DocuScope tags and DocuScope categories
    Per response, every feature's share of the response's tokens (computed the
    same way as in pos_tags.py, top_token_tags.py and DocuScope's N_ csv).
    - Friedman test (rank-based repeated-measures test): do the three models
      differ at all? Effect size: Kendall's W (0 = no agreement, 1 = the
      models are ranked the same way in every question).
    - Wilcoxon signed-rank test per pair of models, as the follow-up. Effect
      size: matched-pairs rank-biserial correlation r (-1 to 1; positive
      means the first model of the pair is higher).
    - Correction for multiple testing: the Friedman p-values are corrected
      with Benjamini-Hochberg over all features of a family (q-value), the
      pairwise p-values with Holm over the three pairs of a feature. A pair
      counts as significant only if both its feature's q and its own Holm p
      are below ALPHA.

MATTR
    MATTR is defined on a 500-lemma window over the whole text, and many
    responses are shorter than that, so it can't be computed per response.
    - Permutation test: within every question, the model labels of the
      responses are shuffled, each model's text is rebuilt in question order,
      and its MATTR recomputed. The p-value is the share of shuffles giving
      a difference at least as large as the observed one. Omnibus statistic:
      variance of the three MATTRs; pairwise: absolute difference (shuffling
      only between the two models), Holm-corrected over the three pairs.
    - Bootstrap: questions are resampled with replacement (the same questions
      for every model) for 95% confidence intervals of each MATTR and of
      every pairwise difference.
    - Check: MATTR per response with a WINDOW_PER_RESPONSE-lemma window,
      tested with Friedman and Wilcoxon like the other features.

Writes to significance_tests_output/: one results table per feature family
(sorted by Friedman p) plus the per-response values the tests used.

Needs spaCy (with en_core_web_lg), NLTK and SciPy:
    pip install spacy nltk scipy
    python -m spacy download en_core_web_lg
Run this script from the overall project folder, after DocuScope.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy import stats

from docuscope_moving_average import (
    assign_sentences, latest_run_folder, load_responses, read_tokens,
)

PROJECT_DIR = Path.cwd()
OUTPUT_DIR = PROJECT_DIR / "significance_tests_output"

SPACY_MODEL = "en_core_web_lg"

ALPHA = 0.05
WINDOW = 500  # Same as mattr_moving_average.py
WINDOW_PER_RESPONSE = 100  # Every response has at least this many lemmas
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 5_000
SEED = 6

# Features found in fewer responses (out of all models' responses together)
# are too rare to test; mostly zeros carry no information about ranks.
MIN_RESPONSES = 10

# POS tags that aren't word classes: SPACE is a line break, and PUNCT is
# already left out by pos_tags.py. They still count in the share's total, so
# the shares match pos_tag_distribution.csv.
UNTESTED_POS_TAGS = {"SPACE", "PUNCT"}

# Text filename prefix -> label, as in the other scripts.
MODELS = {"gpt_4": "GPT-4o", "gpt_5_2": "GPT 5.2", "rag": "RAG"}
PAIRS = list(combinations(MODELS, 2))

# DocuScope category -> prefix of its tags. Summing the token CSVs this way
# reproduces DocuScope's C_ csv exactly. Longest prefix wins (e.g. "Pos" is
# not a prefix of any other category's tags, but checking long first is safer).
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


# ---------------------------------------------------------------------------
# Per-response features
# ---------------------------------------------------------------------------

def spacy_features(nlp, responses):
    """Return per-response POS shares, lemma lists and token totals.

    responses: {model: [(question, text), ...]}
    """
    pos_rows, lemmas = [], {}
    for model, pairs in responses.items():
        lemmas[model] = {}
        for question, doc in zip((q for q, _ in pairs), nlp.pipe(t for _, t in pairs)):
            tokens = [t for t in doc if not t.is_punct]  # As in pos_tags.py
            counts = pd.Series([t.pos_ for t in tokens]).value_counts()
            pos_rows.append({"model": model, "question": question,
                             "tokens": len(tokens), **(counts / len(tokens))})
            # As in mattr_moving_average.py
            lemmas[model][question] = [t.lemma_.lower() for t in doc if t.is_alpha]
    pos = pd.DataFrame(pos_rows).fillna(0.0)
    return pos, lemmas


def category_of(tag):
    for category, prefix in sorted(CATEGORY_PREFIXES.items(), key=lambda c: -len(c[1])):
        if tag.startswith(prefix):
            return category
    return None  # DocuScope's "Other..." tags


def docuscope_features(responses):
    """Return per-response DocuScope tag shares and category shares.

    Tags: tokens carrying the tag / tokens, as in top_token_tags.py ("Other"
    tags left out). Categories: pattern matches / tokens, as in DocuScope's
    N_ csv.
    """
    token_dir = latest_run_folder() / "token_csv"
    tag_rows, category_rows = [], []
    for model, pairs in responses.items():
        tokens = read_tokens(token_dir / f"{model}_tokens.csv")
        # assign_sentences matches tokens to any list of text pieces; here the
        # pieces are whole responses.
        response_ids, unmatched = assign_sentences(tokens["token"], [t for _, t in pairs])
        if unmatched > 0.01 * len(tokens):
            raise ValueError(f"{model}: {unmatched} tokens did not line up with the text.")
        tokens["question"] = [pairs[i][0] for i in response_ids]
        tokens["category"] = tokens["tag"].map(category_of)

        for question, group in tokens.groupby("question"):
            n = len(group)
            tags = group.loc[group["tag"].ne("") & ~group["tag"].str.startswith("Other"), "tag"]
            categories = group.loc[group["pattern_start"], "category"].dropna()
            base = {"model": model, "question": question, "tokens": n}
            tag_rows.append({**base, **(tags.value_counts() / n)})
            category_rows.append({**base, **(categories.value_counts() / n)})

    tags = pd.DataFrame(tag_rows).fillna(0.0)
    categories = pd.DataFrame(category_rows).fillna(0.0)
    for category in CATEGORY_PREFIXES:  # Keep categories no model used
        if category not in categories:
            categories[category] = 0.0
    return tags, categories


# ---------------------------------------------------------------------------
# Friedman + Wilcoxon for per-response shares
# ---------------------------------------------------------------------------

def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation of x - y (zero differences dropped)."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def holm(p_values):
    """Holm-adjusted p-values, in the original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg q-values, in the original order (NaN stays NaN)."""
    p = pd.Series(p_values, dtype=float)
    valid = p.dropna()
    return pd.Series(stats.false_discovery_control(valid), index=valid.index).reindex(p.index)


def test_features(per_response, features, pooled_weights="tokens"):
    """Return one row per feature with Friedman and pairwise Wilcoxon results.

    per_response: one row per model x question, a column per feature (shares).
    """
    wide = {m: per_response[per_response["model"].eq(m)].set_index("question")
            for m in MODELS}
    questions = sorted(set.intersection(*(set(w.index) for w in wide.values())))
    wide = {m: w.loc[questions] for m, w in wide.items()}

    rows = []
    for feature in features:
        values = {m: wide[m][feature].to_numpy() for m in MODELS}
        row = {"feature": feature, "n_questions": len(questions),
               "responses_with_feature": int(sum((v > 0).sum() for v in values.values()))}
        # Share over all of the model's tokens (what the existing plots show),
        # and the median share per response (what the tests compare).
        for m, label in MODELS.items():
            weights = wide[m][pooled_weights].to_numpy()
            row[f"{label} pooled share"] = (values[m] * weights).sum() / weights.sum()
        for m, label in MODELS.items():
            row[f"{label} median share"] = np.median(values[m])

        if row["responses_with_feature"] < MIN_RESPONSES:
            row["note"] = f"not tested: in fewer than {MIN_RESPONSES} responses"
            rows.append(row)
            continue

        chi2, p = stats.friedmanchisquare(*values.values())
        row.update({"friedman_chi2": chi2, "friedman_p": p,
                    "kendalls_w": chi2 / (len(questions) * (len(MODELS) - 1))})

        pair_p = []
        for a, b in PAIRS:
            name = f"{MODELS[a]} vs {MODELS[b]}"
            if np.all(values[a] == values[b]):
                p_pair = 1.0
            else:
                p_pair = stats.wilcoxon(values[a], values[b]).pvalue
            pair_p.append(p_pair)
            row[f"{name}: median diff"] = np.median(values[a] - values[b])
            row[f"{name}: rank-biserial r"] = rank_biserial(values[a], values[b])
            row[f"{name}: p"] = p_pair
        for (a, b), p_adj in zip(PAIRS, holm(pair_p)):
            row[f"{MODELS[a]} vs {MODELS[b]}: p (Holm)"] = p_adj
        rows.append(row)

    table = pd.DataFrame(rows)
    table.insert(table.columns.get_loc("kendalls_w"), "friedman_q (BH)",
                 benjamini_hochberg(table["friedman_p"]))
    table["significant pairs"] = table.apply(significant_pairs, axis=1)
    return table.sort_values("friedman_p", na_position="last").reset_index(drop=True)


def significant_pairs(row):
    """Describe the significant pairs, e.g. "GPT-4o > RAG; GPT 5.2 > RAG"."""
    if not row.get("friedman_q (BH)", 1) < ALPHA:
        return ""
    found = []
    for a, b in PAIRS:
        name = f"{MODELS[a]} vs {MODELS[b]}"
        if row[f"{name}: p (Holm)"] < ALPHA:
            higher, lower = (a, b) if row[f"{name}: rank-biserial r"] > 0 else (b, a)
            found.append(f"{MODELS[higher]} > {MODELS[lower]}")
    return "; ".join(found)


# ---------------------------------------------------------------------------
# MATTR
# ---------------------------------------------------------------------------

def mattr(ids, window=WINDOW):
    """MATTR of a sequence of integer lemma ids, as in mattr_moving_average.py.

    Counts the distinct lemmas of every window incrementally: sliding one
    step drops the window's first lemma (one type less if it doesn't occur
    again inside the window) and adds the next one (one type more if it
    didn't occur in the window yet).
    """
    ids = np.asarray(ids)
    n = len(ids)
    if n < window:
        return len(set(ids.tolist())) / n
    positions = np.arange(n)
    order = np.lexsort((positions, ids))  # Group by lemma, in text order
    same = ids[order][1:] == ids[order][:-1]
    previous = np.full(n, -1)
    following = np.full(n, n)
    previous[order[1:][same]] = order[:-1][same]
    following[order[:-1][same]] = order[1:][same]

    starts = np.arange(n - window)
    leaves = following[starts] >= starts + window
    enters = previous[starts + window] <= starts
    distinct = len(set(ids[:window].tolist())) + np.concatenate(
        ([0], np.cumsum(enters.astype(int) - leaves.astype(int))))
    return distinct.mean() / window


def mattr_tests(lemmas, rng):
    """Permutation tests and bootstrap intervals for the whole-text MATTR."""
    vocabulary = {}
    ids = {m: {q: np.array([vocabulary.setdefault(l, len(vocabulary)) for l in ls])
               for q, ls in per_q.items()}
           for m, per_q in lemmas.items()}
    questions = sorted(set.intersection(*(set(d) for d in ids.values())))
    models = list(MODELS)
    # responses[q][k]: lemma ids of model k's answer to question q
    responses = [[ids[m][q] for m in models] for q in questions]

    def scores(assignment, question_order=None):
        """MATTR per model; assignment[q][k] = which response model k gets."""
        question_order = range(len(questions)) if question_order is None else question_order
        return np.array([
            mattr(np.concatenate([responses[q][assignment[q][k]] for q in question_order]))
            for k in range(len(models))
        ])

    identity = np.tile(np.arange(len(models)), (len(questions), 1))
    observed = scores(identity)

    # Omnibus: shuffle all three labels within every question.
    observed_var = observed.var()
    exceed = sum(
        scores(rng.permuted(identity, axis=1)).var() >= observed_var - 1e-12
        for _ in range(N_PERMUTATIONS)
    )
    omnibus_p = (exceed + 1) / (N_PERMUTATIONS + 1)

    # Pairwise: swap the two models' responses in a random half of the questions.
    pair_rows = []
    for a, b in PAIRS:
        i, j = models.index(a), models.index(b)
        observed_diff = observed[i] - observed[j]
        exceed = 0
        for _ in range(N_PERMUTATIONS):
            assignment = identity.copy()
            swap = rng.random(len(questions)) < 0.5
            assignment[swap, i], assignment[swap, j] = j, i
            s = scores(assignment)
            exceed += abs(s[i] - s[j]) >= abs(observed_diff) - 1e-12
        pair_rows.append({"pair": f"{MODELS[a]} vs {MODELS[b]}",
                          "difference": observed_diff,
                          "p": (exceed + 1) / (N_PERMUTATIONS + 1)})

    # Bootstrap: resample questions, the same ones for every model.
    boot = np.array([
        scores(identity, rng.integers(0, len(questions), len(questions)))
        for _ in range(N_BOOTSTRAP)
    ])
    ci = np.percentile(boot, [2.5, 97.5], axis=0)

    per_model = pd.DataFrame({
        "model": [MODELS[m] for m in models],
        "mattr": observed,
        "ci_low": ci[0], "ci_high": ci[1],
    })
    per_model["omnibus permutation p"] = omnibus_p

    pairs = pd.DataFrame(pair_rows)
    pairs["p (Holm)"] = holm(pairs["p"])
    for (a, b), idx in zip(PAIRS, pairs.index):
        diff = boot[:, models.index(a)] - boot[:, models.index(b)]
        pairs.loc[idx, ["ci_low", "ci_high"]] = np.percentile(diff, [2.5, 97.5])
    pairs["significant"] = (pairs["p (Holm)"] < ALPHA) & (omnibus_p < ALPHA)
    return per_model, pairs


def mattr_per_response(lemmas):
    """MATTR of every response on its own, with a shorter window."""
    rows = []
    for model, per_q in lemmas.items():
        for question, ls in per_q.items():
            if len(ls) < WINDOW_PER_RESPONSE:
                raise ValueError(f"{model} Q{question} has only {len(ls)} lemmas.")
            ids = pd.factorize(pd.Series(ls))[0]
            rows.append({"model": model, "question": question, "tokens": len(ls),
                         f"mattr_{WINDOW_PER_RESPONSE}": mattr(ids, WINDOW_PER_RESPONSE)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def save(table, name, float_format="%.6g"):
    path = OUTPUT_DIR / f"{name}.csv"
    table.to_csv(path, index=False, float_format=float_format)
    print(f"Saved: {path}")


def summarize(title, table):
    tested = table["friedman_p"].notna().sum()
    significant = table[table["significant pairs"].ne("")]
    print(f"\n{title}: {tested} tested, {len(significant)} with significant differences")
    for _, row in significant.iterrows():
        print(f"  {row['feature']:<40} q={row['friedman_q (BH)']:.4f}  {row['significant pairs']}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    nlp = spacy.load(SPACY_MODEL)
    responses = {model: load_responses(model) for model in MODELS}

    print("Tagging the responses with spaCy (about a minute)...", flush=True)
    pos, lemmas = spacy_features(nlp, responses)
    tags, categories = docuscope_features(responses)

    # MATTR
    print(f"MATTR permutation tests and bootstrap ({N_PERMUTATIONS:,} shuffles; "
          "a few minutes)...", flush=True)
    per_model, pairs = mattr_tests(lemmas, rng)
    save(per_model, f"mattr_{WINDOW}_permutation_per_model")
    save(pairs, f"mattr_{WINDOW}_permutation_pairwise")
    print(per_model.to_string(index=False))
    print(pairs.to_string(index=False))

    per_response_mattr = mattr_per_response(lemmas)
    families = {
        f"mattr_{WINDOW_PER_RESPONSE}": (
            per_response_mattr, [f"mattr_{WINDOW_PER_RESPONSE}"]),
        "pos_tags": (pos, [c for c in pos.columns[3:] if c not in UNTESTED_POS_TAGS]),
        "docuscope_tags": (tags, list(tags.columns[3:])),
        "docuscope_categories": (categories, list(CATEGORY_PREFIXES)),
    }
    for name, (per_response, features) in families.items():
        save(per_response.assign(model=per_response["model"].map(MODELS))
             .sort_values(["model", "question"]), f"{name}_per_response_values")
        table = test_features(per_response, features)
        if name == "pos_tags":
            table.insert(1, "description", table["feature"].map(spacy.explain))
        if name == "docuscope_tags":
            table.insert(1, "category", table["feature"].map(category_of))
        save(table, f"{name}_tests")
        summarize(name, table)


if __name__ == "__main__":
    main()
