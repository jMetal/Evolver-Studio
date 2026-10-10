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
) -> pd.DataFrame:
    """Compare the pivot with each other contender on each problem.

    Args:
        runs: The runs table.
        indicator: The indicator to compare on.
        pivot: The pivot contender.
        alpha: The significance level of the Wilcoxon test.
        maximize: Whether higher values are better; None looks it up in the catalogue.

    Returns:
        One row per problem and other contender, with the columns `problem`, `contender`,
        `pivot_median`, `median`, `p_value`, `a12`, `magnitude` and `verdict`: the pivot is better
        or worse when the test is significant and its A12 is above or below 0.5, and there is no
        significant difference otherwise.
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
            p_value = _rank_sum_p_value(pivot_values, other_values)
            rows.append(
                {
                    "problem": problem,
                    "contender": contender,
                    "pivot_median": float(np.median(pivot_values)),
                    "median": float(np.median(other_values)),
                    "p_value": p_value,
                    "a12": effect,
                    "magnitude": magnitude(effect),
                    "verdict": _verdict(p_value, effect, alpha),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "problem",
            "contender",
            "pivot_median",
            "median",
            "p_value",
            "a12",
            "magnitude",
            "verdict",
        ],
    )


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
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values))
    running = 0.0
    for step, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - step) * p_values[index]))
        adjusted[index] = running
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
