"""The statistics of a validation study: medians, a Wilcoxon test and the A12 effect size.

The input is the table of every independent run of the study, one row per run of a contender on a
problem, with the columns `contender`, `problem`, `Run`, `Seed`, `TimeMs` and one per quality
indicator. Lower values are better unless the indicator is maximized (`catalogue.is_maximized`):
"better" below means lower or higher accordingly.

One contender, the pivot (the tuned configuration), is compared with each of the others, problem by
problem: a Wilcoxon rank-sum test (the runs are independent) says whether the difference is
significant, and Vargha and Delaney's A12 how large it is, the same as Evolver's
`scripts/effect_size_tables.py`.

Over all the problems at once, the algorithms are ranked on each problem by their median, and the
average ranks are compared as Demšar proposes (J. Mach. Learn. Res. 7, 2006): Friedman's test says
whether they differ at all, Nemenyi's critical difference which pairs differ (the critical
difference plot), and Holm's procedure which algorithms differ from the pivot.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from evolver_studio.catalogue import is_maximized

ALPHA = 0.05
# The thresholds of |A12 - 0.5| for a small, a medium and a large effect.
MAGNITUDE_THRESHOLDS = ((0.21, "large"), (0.14, "medium"), (0.06, "small"))
PIVOT_BETTER = "pivot better"
PIVOT_WORSE = "pivot worse"
NO_DIFFERENCE = "no significant difference"
NON_INDICATOR_COLUMNS = ("contender", "problem", "Run", "Seed", "TimeMs")
NO_CORRECTION = "none"
HOLM = "holm"
BENJAMINI_HOCHBERG = "benjamini-hochberg"


def indicator_names(runs: pd.DataFrame) -> list[str]:
    """The quality indicators of a table of runs.

    Args:
        runs: The runs table.

    Returns:
        Its columns that are not the contender, the problem, the run, the seed or the time.
    """
    return [column for column in runs.columns if column not in NON_INDICATOR_COLUMNS]


def medians(runs: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """The median of an indicator over the runs of each contender on each problem.

    Args:
        runs: The runs table.
        indicator: The indicator.

    Returns:
        A table with a row per problem and a column per contender, in order of appearance.
    """
    return _grouped(runs, indicator, lambda values: values.median())


def interquartile_ranges(runs: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """The interquartile range of an indicator, laid out as `medians`.

    Args:
        runs: The runs table.
        indicator: The indicator.

    Returns:
        A table with a row per problem and a column per contender.
    """
    return _grouped(runs, indicator, lambda values: values.quantile(0.75) - values.quantile(0.25))


def best_contenders(median_table: pd.DataFrame, maximize: bool = False) -> pd.Series:
    """The contender with the best median on each problem.

    Args:
        median_table: The table `medians` returns.
        maximize: Whether higher values of the indicator are better.

    Returns:
        The name of the best contender, by problem; the first one on a tie.
    """
    return median_table.idxmax(axis=1) if maximize else median_table.idxmin(axis=1)


def a12(pivot: np.ndarray, other: np.ndarray, maximize: bool = False) -> float:
    """The probability that a run of the pivot is better than one of the other.

    Ties count one half, so 0.5 is no difference.

    Args:
        pivot: The pivot's values of an indicator.
        other: The other contender's.
        maximize: Whether higher values are better (else lower ones are).

    Returns:
        The A12 statistic.
    """
    pivot_values = np.asarray(pivot, dtype=float)[:, None]
    other_values = np.asarray(other, dtype=float)[None, :]
    better = (pivot_values > other_values if maximize else pivot_values < other_values).sum()
    ties = (pivot_values == other_values).sum()
    return float((better + 0.5 * ties) / (pivot_values.size * other_values.size))


def magnitude(value: float) -> str:
    """The magnitude of an A12 value: negligible, small, medium or large.

    Args:
        value: An A12 value.

    Returns:
        The label.
    """
    # Rounded, so that 0.71 is as far from 0.5 as the threshold 0.21 despite floating point.
    distance = round(abs(value - 0.5), 10)
    for threshold, label in MAGNITUDE_THRESHOLDS:
        if distance >= threshold:
            return label
    return "negligible"


def compare_with_pivot(
    runs: pd.DataFrame,
    indicator: str,
    pivot: str,
    alpha: float = ALPHA,
    maximize: bool | None = None,
    correction: str = NO_CORRECTION,
) -> pd.DataFrame:
    """Compare the pivot with each other contender on each problem.

    Args:
        runs: The runs table.
        indicator: The indicator to compare on.
        pivot: The pivot contender.
        alpha: The significance level of the Wilcoxon test.
        maximize: Whether higher values are better; None looks it up in the catalogue.
        correction: How the p-values of all the comparisons are adjusted for being many
            (`NO_CORRECTION`, `HOLM` or `BENJAMINI_HOCHBERG`, see `adjust_p_values`); the verdicts
            follow the adjusted ones.

    Returns:
        One row per problem and other contender, with the columns `problem`, `contender`,
        `pivot_median`, `median`, `p_value`, `adjusted_p_value`, `a12`, `magnitude` and
        `verdict`: the pivot is better or worse when the test is significant and its A12 is above
        or below 0.5, and there is no significant difference otherwise.
    """
    if maximize is None:
        maximize = is_maximized(indicator)
    rows = []
    for problem, problem_runs in runs.groupby("problem", sort=False):
        pivot_values = _values(problem_runs, pivot, indicator)
        for contender in dict.fromkeys(problem_runs["contender"]):
            if contender == pivot:
                continue
            other_values = _values(problem_runs, contender, indicator)
            if pivot_values.size == 0 or other_values.size == 0:
                continue
            effect = a12(pivot_values, other_values, maximize)
            rows.append(
                {
                    "problem": problem,
                    "contender": contender,
                    "pivot_median": float(np.median(pivot_values)),
                    "median": float(np.median(other_values)),
                    "p_value": _rank_sum_p_value(pivot_values, other_values),
                    "a12": effect,
                    "magnitude": magnitude(effect),
                }
            )
    columns = [
        "problem",
        "contender",
        "pivot_median",
        "median",
        "p_value",
        "adjusted_p_value",
        "a12",
        "magnitude",
        "verdict",
    ]
    table = pd.DataFrame(
        rows, columns=[c for c in columns if c not in ("adjusted_p_value", "verdict")]
    )
    table.insert(5, "adjusted_p_value", adjust_p_values(table["p_value"].to_numpy(), correction))
    table["verdict"] = [
        _verdict(p, effect, alpha) for p, effect in zip(table["adjusted_p_value"], table["a12"])
    ]
    return table[columns]


def adjust_p_values(p_values: np.ndarray, correction: str = HOLM) -> np.ndarray:
    """Adjust p-values for being many comparisons at once.

    Args:
        p_values: The p-values.
        correction: `NO_CORRECTION`; `HOLM`, Holm's step-down procedure, which keeps the
            probability of any false difference (the family-wise error) below the significance
            level; or `BENJAMINI_HOCHBERG`, which keeps the expected share of false differences
            among those found (the false discovery rate) below it, and so finds more.

    Returns:
        The adjusted p-values, in the same order.
    """
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)
    if correction == NO_CORRECTION or count == 0:
        return p_values.copy()
    if correction == HOLM:
        order = np.argsort(p_values, kind="stable")
        adjusted = np.empty(count)
        running = 0.0
        for step, index in enumerate(order):
            running = max(running, min(1.0, (count - step) * p_values[index]))
            adjusted[index] = running
        return adjusted
    if correction == BENJAMINI_HOCHBERG:
        order = np.argsort(p_values, kind="stable")[::-1]
        adjusted = np.empty(count)
        running = 1.0
        for position, index in enumerate(order):
            rank = count - position
            running = min(running, p_values[index] * count / rank)
            adjusted[index] = min(1.0, running)
        return adjusted
    raise ValueError(f"Unknown correction: {correction}")


def verdict_counts(comparison: pd.DataFrame) -> pd.DataFrame:
    """Count on how many problems the pivot is better, worse or not different.

    Args:
        comparison: The table `compare_with_pivot` returns.

    Returns:
        A row per contender, with the number of problems where the pivot is better, equivalent and
        worse (the columns `better`, `equal` and `worse`).
    """
    counts = (
        comparison.groupby(["contender", "verdict"], sort=False).size().unstack(fill_value=0)
    ).reindex(columns=[PIVOT_BETTER, NO_DIFFERENCE, PIVOT_WORSE], fill_value=0)
    counts.columns = ["better", "equal", "worse"]
    return counts.reindex(list(dict.fromkeys(comparison["contender"])))


def average_ranks(median_table: pd.DataFrame, maximize: bool = False) -> pd.Series:
    """The average rank of each contender over the problems, ranking them by their median.

    Args:
        median_table: The table `medians` returns; problems where a contender has no value are
            left out.
        maximize: Whether higher values are better.

    Returns:
        The average rank (1 is the best) of each contender, best first; ties share the average of
        their ranks.
    """
    complete = median_table.dropna()
    ranks = complete.rank(axis=1, ascending=not maximize, method="average")
    return ranks.mean().sort_values(kind="stable")


def friedman_test(median_table: pd.DataFrame) -> tuple[float, float] | None:
    """Friedman's test of whether the contenders differ over the problems.

    Args:
        median_table: The table `medians` returns; problems where a contender has no value are
            left out.

    Returns:
        The statistic and its p-value; None with fewer than three contenders or two problems,
        when the test is not defined.
    """
    complete = median_table.dropna()
    if complete.shape[1] < 3 or complete.shape[0] < 2:
        return None
    if (complete.nunique(axis=1) == 1).all():
        return 0.0, 1.0
    result = stats.friedmanchisquare(*(complete[column] for column in complete.columns))
    return float(result.statistic), float(result.pvalue)


def critical_difference(contenders: int, problems: int, alpha: float = ALPHA) -> float:
    """Nemenyi's critical difference of average ranks, as in Demšar's critical difference plot.

    Two contenders whose average ranks differ by more than it are significantly different.

    Args:
        contenders: How many contenders are ranked.
        problems: On how many problems.
        alpha: The significance level.

    Returns:
        The critical difference.
    """
    q_alpha = stats.studentized_range.ppf(1 - alpha, contenders, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(contenders * (contenders + 1) / (6.0 * problems)))


def nonsignificant_groups(ranks: pd.Series, difference: float) -> list[list[str]]:
    """The groups of contenders whose average ranks differ by less than the critical difference.

    These are the bars of a critical difference plot: the contenders joined by one are not
    significantly different. Only the largest groups are kept (none is inside another), and a
    contender different from all the others is in none.

    Args:
        ranks: The average ranks `average_ranks` returns, best first.
        difference: The critical difference.

    Returns:
        The groups, each in rank order, from the best ranked.
    """
    ordered = ranks.sort_values(kind="stable")
    names = list(ordered.index)
    values = ordered.to_numpy()
    groups: list[list[str]] = []
    last_end = -1
    for start in range(len(values)):
        end = start
        while end + 1 < len(values) and values[end + 1] - values[start] < difference:
            end += 1
        if end > start and end > last_end:
            groups.append(names[start : end + 1])
            last_end = end
    return groups


def holm_against_pivot(
    ranks: pd.Series, pivot: str, problems: int, alpha: float = ALPHA
) -> pd.DataFrame:
    """Holm's procedure: which contenders' average ranks differ from the pivot's.

    Each contender is compared with the pivot by z = (R_i - R_pivot) / sqrt(k(k + 1) / 6N), and
    the p-values are adjusted for the number of comparisons with Holm's step-down procedure.

    Args:
        ranks: The average ranks `average_ranks` returns.
        pivot: The pivot.
        problems: How many problems the ranks are over.
        alpha: The significance level.

    Returns:
        One row per other contender, in the order of the ranks: `contender`, `rank`, `z`,
        `p_value`, `adjusted_p_value` and `significant`; empty when the pivot is not ranked.
    """
    columns = ["contender", "rank", "z", "p_value", "adjusted_p_value", "significant"]
    if pivot not in ranks.index or len(ranks) < 2 or problems < 1:
        return pd.DataFrame(columns=columns)
    contenders = len(ranks)
    error = np.sqrt(contenders * (contenders + 1) / (6.0 * problems))
    others = ranks.drop(pivot)
    z = (others - ranks[pivot]) / error
    p_values = 2 * stats.norm.sf(np.abs(z))
    adjusted = adjust_p_values(p_values, HOLM)
    return pd.DataFrame(
        {
            "contender": others.index,
            "rank": others.to_numpy(),
            "z": z.to_numpy(),
            "p_value": p_values,
            "adjusted_p_value": adjusted,
            "significant": adjusted < alpha,
        },
        columns=columns,
    )


def signed_rank_against_pivot(
    median_table: pd.DataFrame,
    pivot: str,
    maximize: bool = False,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Wilcoxon's signed-rank test of the pivot against each contender, over the problems.

    Each problem is a pair (the two medians); Demšar recommends this test to compare two
    algorithms over several problems. The p-values are adjusted with Holm's procedure, for the
    comparisons with each contender.

    Args:
        median_table: The table `medians` returns; problems where either has no value are left
            out of that comparison.
        pivot: The pivot.
        maximize: Whether higher values are better.
        alpha: The significance level.

    Returns:
        One row per other contender: `contender`, `problems`, `pivot_better` and `pivot_worse`
        (on how many problems its median is better or worse), `p_value`, `adjusted_p_value` and
        `verdict` (as in `compare_with_pivot`); empty when the pivot is not in the table.
    """
    columns = [
        "contender",
        "problems",
        "pivot_better",
        "pivot_worse",
        "p_value",
        "adjusted_p_value",
        "verdict",
    ]
    if pivot not in median_table.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for contender in median_table.columns:
        if contender == pivot:
            continue
        pair = median_table[[pivot, contender]].dropna()
        # Positive: the pivot is better.
        gain = (pair[pivot] - pair[contender]) * (1 if maximize else -1)
        nonzero = gain[gain != 0]
        if nonzero.empty:
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(nonzero, zero_method="wilcox").pvalue)
        rows.append(
            {
                "contender": contender,
                "problems": len(pair),
                "pivot_better": int((gain > 0).sum()),
                "pivot_worse": int((gain < 0).sum()),
                "p_value": p_value,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=columns)
    table["adjusted_p_value"] = adjust_p_values(table["p_value"].to_numpy(), HOLM)
    table["verdict"] = [
        NO_DIFFERENCE
        if p >= alpha or better == worse
        else (PIVOT_BETTER if better > worse else PIVOT_WORSE)
        for p, better, worse in zip(
            table["adjusted_p_value"], table["pivot_better"], table["pivot_worse"]
        )
    ]
    return table[columns]


def aligned_friedman_test(median_table: pd.DataFrame) -> tuple[float, float] | None:
    """The Friedman aligned-ranks test (Hodges and Lehmann), as in García et al. (2010).

    Each value is aligned by subtracting its problem's mean, and all the aligned values are
    ranked together, so that problems are compared with each other too: more power than
    Friedman's test with few problems. Its statistic follows a chi-squared distribution with
    k - 1 degrees of freedom.

    Args:
        median_table: The table `medians` returns; problems where a contender has no value are
            left out.

    Returns:
        The statistic and its p-value; None with fewer than three contenders or two problems.
    """
    complete = median_table.dropna()
    problems, contenders = complete.shape
    if contenders < 3 or problems < 2:
        return None
    values = complete.to_numpy(dtype=float)
    aligned = values - values.mean(axis=1, keepdims=True)
    ranks = stats.rankdata(aligned.ravel()).reshape(aligned.shape)
    total = problems * contenders
    by_contender = (ranks.sum(axis=0) ** 2).sum()
    by_problem = (ranks.sum(axis=1) ** 2).sum()
    numerator = (contenders - 1) * (
        by_contender - (contenders * problems**2 / 4.0) * (total + 1) ** 2
    )
    denominator = total * (total + 1) * (2 * total + 1) / 6.0 - by_problem / contenders
    if denominator <= 0:
        return 0.0, 1.0
    statistic = float(numerator / denominator)
    return statistic, float(stats.chi2.sf(statistic, contenders - 1))


def quade_test(median_table: pd.DataFrame) -> tuple[float, float] | None:
    """Quade's test: Friedman's, with each problem weighted by how much the contenders differ.

    The problems are ranked by the range of their values, and a contender's ranks within a
    problem count more on problems where the contenders differ more. Its statistic follows an F
    distribution with k - 1 and (n - 1)(k - 1) degrees of freedom (García et al., 2010).

    Args:
        median_table: The table `medians` returns; problems where a contender has no value are
            left out.

    Returns:
        The statistic and its p-value; None with fewer than three contenders or two problems.
    """
    complete = median_table.dropna()
    problems, contenders = complete.shape
    if contenders < 3 or problems < 2:
        return None
    values = complete.to_numpy(dtype=float)
    within = np.apply_along_axis(stats.rankdata, 1, values)
    weights = stats.rankdata(values.max(axis=1) - values.min(axis=1))
    scores = weights[:, None] * (within - (contenders + 1) / 2.0)
    total = (scores**2).sum()
    between = (scores.sum(axis=0) ** 2).sum() / problems
    if total - between <= 0:
        return (0.0, 1.0) if between == 0 else (float("inf"), 0.0)
    statistic = float((problems - 1) * between / (total - between))
    return statistic, float(
        stats.f.sf(statistic, contenders - 1, (problems - 1) * (contenders - 1))
    )


@dataclass(frozen=True)
class BayesianComparison:
    """The posterior of the Bayesian signed-rank test of the pivot against a contender.

    Attributes:
        contender: The contender.
        pivot_better: The probability that the pivot is better (by more than the ROPE).
        equivalent: The probability that they are practically equivalent (within the ROPE).
        pivot_worse: The probability that the pivot is worse.
        samples: Samples of the posterior, one row per sample: the probabilities of the pivot
            being worse, equivalent and better (for the simplex plot).
    """

    contender: str
    pivot_better: float
    equivalent: float
    pivot_worse: float
    samples: np.ndarray


def bayesian_signed_rank(
    median_table: pd.DataFrame,
    pivot: str,
    contender: str,
    rope: float,
    maximize: bool = False,
    samples: int = 20000,
    prior: float = 0.5,
    seed: int = 1,
    relative: bool = True,
) -> BayesianComparison | None:
    """Benavoli et al.'s Bayesian signed-rank test of the pivot against a contender (JMLR 2017).

    Over the problems, as Wilcoxon's signed-rank test, but it answers with probabilities: that
    the pivot is better, practically equivalent (their difference within the region of practical
    equivalence, the ROPE) or worse. The posterior is a Dirichlet process with a pseudo-
    observation at zero difference (weight `prior`), sampled `samples` times.

    Args:
        median_table: The table `medians` returns.
        pivot: The pivot.
        contender: The contender to compare it with.
        rope: Half the width of the region of practical equivalence: a share of the medians when
            `relative`, else in the indicator's units.
        maximize: Whether higher values are better.
        samples: How many samples of the posterior.
        prior: The weight of the pseudo-observation.
        seed: The seed of the sampling, so that the answer is the same every time.
        relative: Whether each problem's difference is divided by the mean of the two medians,
            so that problems whose indicator values differ in scale weigh alike, and the ROPE is
            a share (0.01 is 1%).

    Returns:
        The comparison; None when they share no problem.
    """
    pair = median_table[[pivot, contender]].dropna()
    if pair.empty:
        return None
    # Positive: the pivot is better.
    gain = ((pair[pivot] - pair[contender]) * (1 if maximize else -1)).to_numpy(dtype=float)
    if relative:
        scale = (pair[pivot].abs() + pair[contender].abs()).to_numpy(dtype=float) / 2
        gain = np.divide(gain, scale, out=np.zeros_like(gain), where=scale > 0)
    points = np.concatenate([[0.0], gain])
    # Walsh averages (z_i + z_j) / 2 against the ROPE; one exactly on its edge counts one half.
    sums = points[:, None] + points[None, :]
    better = (sums > 2 * rope) + 0.5 * (sums == 2 * rope)
    worse = (sums < -2 * rope) + 0.5 * (sums == -2 * rope)
    within = 1.0 - better - worse
    weights = np.random.default_rng(seed).dirichlet(
        np.concatenate([[prior], np.ones(len(gain))]), samples
    )
    posterior = np.column_stack(
        [np.einsum("si,ij,sj->s", weights, region, weights) for region in (worse, within, better)]
    )
    winners = np.bincount(posterior.argmax(axis=1), minlength=3) / samples
    return BayesianComparison(
        contender=contender,
        pivot_worse=float(winners[0]),
        equivalent=float(winners[1]),
        pivot_better=float(winners[2]),
        samples=posterior,
    )


def indicator_correlations(runs: pd.DataFrame) -> pd.DataFrame:
    """How the indicators agree: Spearman's correlation between each pair, within each problem.

    Each indicator is first turned so that higher is better, so a positive correlation means
    the two agree on which runs are good. The correlation is computed over the runs of each
    problem and averaged over the problems: pooling the problems would mix their scales (two
    indicators that agree on every problem, as the hypervolume and the normalized one, would not
    seem to).

    Args:
        runs: The runs table.

    Returns:
        A square table, one row and column per indicator.
    """
    names = indicator_names(runs)
    oriented = runs[["problem"]].assign(
        **{name: runs[name] * (1 if is_maximized(name) else -1) for name in names}
    )
    per_problem = [
        group[names].corr(method="spearman")
        for _, group in oriented.groupby("problem", sort=False)
        if len(group) > 2
    ]
    if not per_problem:
        return pd.DataFrame(np.nan, index=names, columns=names)
    return (
        pd.concat(per_problem)
        .groupby(level=0, sort=False)
        .mean()
        .reindex(index=names, columns=names)
    )


MEDIAN_RUN = "median"
BEST_RUN = "best"
WORST_RUN = "worst"


def chosen_run(
    runs: pd.DataFrame,
    indicator: str,
    problem: str,
    contender: str,
    which: str = MEDIAN_RUN,
    maximize: bool | None = None,
) -> int | None:
    """The run of a contender on a problem that shows it: its median, best or worst by an indicator.

    The median run is an actual run: with an even number of runs, the better of the two middle
    ones.

    Args:
        runs: The runs table.
        indicator: The indicator that orders the runs.
        problem: The problem.
        contender: The contender.
        which: `MEDIAN_RUN`, `BEST_RUN` or `WORST_RUN`.
        maximize: Whether higher values are better; None looks it up in the catalogue.

    Returns:
        The run's number, or None when the contender has no run with a value on the problem.
    """
    if maximize is None:
        maximize = is_maximized(indicator)
    mask = (runs["problem"] == problem) & (runs["contender"] == contender)
    values = runs.loc[mask, ["Run", indicator]].dropna()
    if values.empty:
        return None
    # Best first; a stable sort keeps the order of the runs among equal values.
    ordered = values.sort_values(indicator, ascending=not maximize, kind="stable")
    position = {BEST_RUN: 0, WORST_RUN: len(ordered) - 1}.get(which, (len(ordered) - 1) // 2)
    return int(ordered["Run"].iloc[position])


def _grouped(runs: pd.DataFrame, indicator: str, statistic) -> pd.DataFrame:
    problems = list(dict.fromkeys(runs["problem"]))
    contenders = list(dict.fromkeys(runs["contender"]))
    table = pd.DataFrame(index=problems, columns=contenders, dtype=float)
    for (problem, contender), values in runs.groupby(["problem", "contender"], sort=False):
        table.loc[problem, contender] = statistic(values[indicator])
    return table


def _values(problem_runs: pd.DataFrame, contender: str, indicator: str) -> np.ndarray:
    values = problem_runs.loc[problem_runs["contender"] == contender, indicator]
    return values.dropna().to_numpy(dtype=float)


def _rank_sum_p_value(pivot_values: np.ndarray, other_values: np.ndarray) -> float:
    """The two-sided p-value of a Wilcoxon rank-sum test, 1 when all the values are equal."""
    combined = np.concatenate([pivot_values, other_values])
    if np.all(combined == combined[0]):
        return 1.0
    return float(stats.mannwhitneyu(pivot_values, other_values, alternative="two-sided").pvalue)


def _verdict(p_value: float, effect: float, alpha: float) -> str:
    if p_value >= alpha or effect == 0.5:
        return NO_DIFFERENCE
    return PIVOT_BETTER if effect > 0.5 else PIVOT_WORSE
