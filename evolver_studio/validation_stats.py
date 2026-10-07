"""The statistics of a validation study: medians, a Wilcoxon test and the A12 effect size.

The input is the table of every independent run of the study, one row per run of a contender on a
problem, with the columns `contender`, `problem`, `Run`, `Seed`, `TimeMs` and one per quality
indicator. The indicators are all minimized.

One contender, the pivot (the tuned configuration), is compared with each of the others, problem by
problem: a Wilcoxon rank-sum test (the runs are independent) says whether the difference is
significant, and Vargha and Delaney's A12 how large it is, the same as Evolver's
`scripts/effect_size_tables.py`.
"""

import numpy as np
import pandas as pd
from scipy import stats

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


def best_contenders(median_table: pd.DataFrame) -> pd.Series:
    """The contender with the lowest median on each problem.

    Args:
        median_table: The table `medians` returns.

    Returns:
        The name of the best contender, by problem; the first one on a tie.
    """
    return median_table.idxmin(axis=1)


def a12(pivot: np.ndarray, other: np.ndarray) -> float:
    """The probability that a run of the pivot is better (lower) than one of the other.

    Ties count one half, so 0.5 is no difference.

    Args:
        pivot: The pivot's values of an indicator.
        other: The other contender's.

    Returns:
        The A12 statistic.
    """
    pivot_values = np.asarray(pivot, dtype=float)[:, None]
    other_values = np.asarray(other, dtype=float)[None, :]
    better = (pivot_values < other_values).sum()
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
    runs: pd.DataFrame, indicator: str, pivot: str, alpha: float = ALPHA
) -> pd.DataFrame:
    """Compare the pivot with each other contender on each problem.

    Args:
        runs: The runs table.
        indicator: The indicator to compare on.
        pivot: The pivot contender.
        alpha: The significance level of the Wilcoxon test.

    Returns:
        One row per problem and other contender, with the columns `problem`, `contender`,
        `pivot_median`, `median`, `p_value`, `a12`, `magnitude` and `verdict`: the pivot is better
        or worse when the test is significant and its A12 is above or below 0.5, and there is no
        significant difference otherwise.
    """
    rows = []
    for problem, problem_runs in runs.groupby("problem", sort=False):
        pivot_values = _values(problem_runs, pivot, indicator)
        for contender in dict.fromkeys(problem_runs["contender"]):
            if contender == pivot:
                continue
            other_values = _values(problem_runs, contender, indicator)
            if pivot_values.size == 0 or other_values.size == 0:
                continue
            effect = a12(pivot_values, other_values)
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
