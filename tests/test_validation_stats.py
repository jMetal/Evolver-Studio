"""Tests for the statistics of a validation study."""

import numpy as np
import pandas as pd
import pytest

from evolver_studio.validation_stats import (
    NO_DIFFERENCE,
    PIVOT_BETTER,
    PIVOT_WORSE,
    a12,
    best_contenders,
    compare_with_pivot,
    indicator_names,
    interquartile_ranges,
    magnitude,
    medians,
    verdict_counts,
)


def _runs(values: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
    """A runs table from {(problem, contender): the EP value of each run}."""
    rows = [
        {"contender": contender, "problem": problem, "Run": run, "Seed": run, "TimeMs": 1, "EP": v}
        for (problem, contender), runs in values.items()
        for run, v in enumerate(runs, start=1)
    ]
    return pd.DataFrame(rows)


LOW = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
HIGH = [value + 1 for value in LOW]


class TestA12:
    def test_should_be_one_when_the_pivot_is_always_lower(self):
        # Act / Assert
        assert a12(np.array(LOW), np.array(HIGH)) == 1.0

    def test_should_be_zero_when_the_pivot_is_always_higher(self):
        # Act / Assert
        assert a12(np.array(HIGH), np.array(LOW)) == 0.0

    def test_should_count_higher_values_as_better_for_a_maximized_indicator(self):
        # Act / Assert
        assert a12(np.array(HIGH), np.array(LOW), maximize=True) == 1.0

    def test_should_count_ties_as_one_half(self):
        # Act / Assert
        assert a12(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5

    @pytest.mark.parametrize(
        ("value", "label"),
        [
            (0.5, "negligible"),
            (0.55, "negligible"),
            (0.56, "small"),
            (0.64, "medium"),
            (0.71, "large"),
            (0.29, "large"),
            (0.44, "small"),
            (1.0, "large"),
        ],
    )
    def test_should_label_the_magnitude_by_its_distance_from_one_half(self, value, label):
        # Act / Assert
        assert magnitude(value) == label


class TestSummaryTables:
    @pytest.fixture
    def runs(self) -> pd.DataFrame:
        return _runs(
            {
                ("P1", "tuned"): LOW,
                ("P1", "NSGA-II"): HIGH,
                ("P2", "tuned"): HIGH,
                ("P2", "NSGA-II"): LOW,
            }
        )

    def test_should_give_the_median_per_problem_and_contender(self, runs: pd.DataFrame):
        # Act
        table = medians(runs, "EP")

        # Assert
        assert list(table.index) == ["P1", "P2"]
        assert list(table.columns) == ["tuned", "NSGA-II"]
        assert table.loc["P1", "tuned"] == pytest.approx(0.145)
        assert table.loc["P1", "NSGA-II"] == pytest.approx(1.145)

    def test_should_give_the_interquartile_range(self, runs: pd.DataFrame):
        # Act
        table = interquartile_ranges(runs, "EP")

        # Assert
        # the quartiles of 0.10 ... 0.19 are 0.1325 and 0.1775; shifting all by 1 keeps the range
        assert table.loc["P1", "tuned"] == pytest.approx(0.045)
        assert table.loc["P1", "NSGA-II"] == pytest.approx(0.045)

    def test_should_pick_the_contender_with_the_lowest_median_per_problem(self, runs):
        # Act
        best = best_contenders(medians(runs, "EP"))

        # Assert
        assert best.to_dict() == {"P1": "tuned", "P2": "NSGA-II"}

    def test_should_pick_the_highest_median_for_a_maximized_indicator(self, runs):
        # Act
        best = best_contenders(medians(runs, "EP"), maximize=True)

        # Assert
        assert best.to_dict() == {"P1": "NSGA-II", "P2": "tuned"}

    def test_should_list_the_indicators_of_the_table(self, runs: pd.DataFrame):
        # Act / Assert
        assert indicator_names(runs) == ["EP"]


class TestCompareWithPivot:
    @pytest.fixture
    def comparison(self) -> pd.DataFrame:
        runs = _runs(
            {
                ("P1", "tuned"): LOW,
                ("P1", "NSGA-II"): HIGH,
                ("P2", "tuned"): HIGH,
                ("P2", "NSGA-II"): LOW,
                ("P3", "tuned"): LOW,
                ("P3", "NSGA-II"): LOW[::-1],
            }
        )
        return compare_with_pivot(runs, "EP", "tuned")

    def test_should_compare_the_pivot_with_each_other_contender_on_each_problem(self, comparison):
        # Assert
        assert list(comparison["problem"]) == ["P1", "P2", "P3"]
        assert set(comparison["contender"]) == {"NSGA-II"}

    def test_should_say_the_pivot_is_better_when_significantly_lower(self, comparison):
        # Act
        row = comparison[comparison["problem"] == "P1"].iloc[0]

        # Assert
        assert row["verdict"] == PIVOT_BETTER
        assert row["a12"] == 1.0
        assert row["magnitude"] == "large"
        assert row["p_value"] < 0.05

    def test_should_say_the_pivot_is_worse_when_significantly_higher(self, comparison):
        # Act
        row = comparison[comparison["problem"] == "P2"].iloc[0]

        # Assert
        assert row["verdict"] == PIVOT_WORSE
        assert row["a12"] == 0.0

    def test_should_say_the_pivot_is_better_when_significantly_higher_on_a_maximized_indicator(
        self,
    ):
        # Arrange
        runs = _runs({("P1", "tuned"): HIGH, ("P1", "NSGA-II"): LOW})

        # Act
        row = compare_with_pivot(runs, "EP", "tuned", maximize=True).iloc[0]

        # Assert
        assert row["verdict"] == PIVOT_BETTER
        assert row["a12"] == 1.0

    def test_should_find_no_difference_between_equal_samples(self, comparison):
        # Act
        row = comparison[comparison["problem"] == "P3"].iloc[0]

        # Assert
        assert row["verdict"] == NO_DIFFERENCE
        assert row["a12"] == 0.5

    def test_should_not_fail_when_every_value_is_equal(self):
        # Arrange
        runs = _runs({("P1", "tuned"): [1.0] * 5, ("P1", "NSGA-II"): [1.0] * 5})

        # Act
        comparison = compare_with_pivot(runs, "EP", "tuned")

        # Assert
        assert comparison.iloc[0]["p_value"] == 1.0
        assert comparison.iloc[0]["verdict"] == NO_DIFFERENCE

    def test_should_skip_a_problem_where_a_contender_has_no_runs(self):
        # Arrange: NSGA-II could not run on P2
        runs = _runs({("P1", "tuned"): LOW, ("P1", "NSGA-II"): HIGH, ("P2", "tuned"): LOW})

        # Act
        comparison = compare_with_pivot(runs, "EP", "tuned")

        # Assert
        assert list(comparison["problem"]) == ["P1"]

    def test_should_count_the_verdicts_per_contender(self, comparison):
        # Act
        counts = verdict_counts(comparison)

        # Assert
        assert counts.loc["NSGA-II"].to_dict() == {"better": 1, "equal": 1, "worse": 1}
