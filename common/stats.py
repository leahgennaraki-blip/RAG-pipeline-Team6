"""Paired tests and significance formatting shared by the analyses."""
import numpy as np
from scipy import stats


def signed_rank(first, second):
    """Two-sided paired Wilcoxon signed-rank test with matched-pairs rank-biserial r.

    Returns (statistic, p, r, method). Subtraction noise is rounded away so
    mathematically tied differences stay tied. The p-value is exact when
    there are no zeros or ties and at most 50 pairs.
    """
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


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def latex_effect(value, p):
    """A LaTeX cell like $0.42^{**}$."""
    suffix = stars(p)
    return "$" + f"{value:.2f}" + ("^{" + suffix + "}" if suffix else "") + "$"
