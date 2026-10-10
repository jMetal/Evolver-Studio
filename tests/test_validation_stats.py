"""Tests for the statistics of a validation study."""

import numpy as np
import pandas as pd
import pytest

from evolver_studio.validation_stats import (
    BEST_RUN,
    MEDIAN_RUN,
    NO_DIFFERENCE,
    PIVOT_BETTER,
    PIVOT_WORSE,
    WORST_RUN,
    a12,
    average_ranks,
    best_contenders,
    chosen_run,
    compare_with_pivot,
    critical_difference,
    friedman_test,
    holm_against_pivot,
    indicator_names,
    interquartile_ranges,
    magnitude,
    medians,
    nonsignificant_groups,
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


def _median_table() -> pd.DataFrame:
    """Five problems where A is always best, B second and C last; D ties B on two of them."""
    return pd.DataFrame(
        {
            "A": [0.1, 0.1, 0.1, 0.1, 0.1],
            "B": [0.2, 0.2, 0.2, 0.2, 0.2],
            "C": [0.4, 0.4, 0.4, 0.4, 0.4],
            "D": [0.2, 0.2, 0.3, 0.3, 0.3],
        },
        index=[f"P{i}" for i in range(1, 6)],
    )


class TestRanking:
    def test_should_average_the_ranks_of_each_problem_best_first(self):
        # Act
        ranks = average_ranks(_median_table())

        # Assert: ties share the average of their ranks (B and D are 2.5 on P1 and P2)
        assert list(ranks.index) == ["A", "B", "D", "C"]
        assert ranks["A"] == 1.0
        assert ranks["B"] == pytest.approx((2.5 + 2.5 + 2 + 2 + 2) / 5)
        assert ranks["D"] == pytest.approx((2.5 + 2.5 + 3 + 3 + 3) / 5)
        assert ranks["C"] == 4.0

    def test_should_rank_the_highest_first_for_a_maximized_indicator(self):
        # Act
        ranks = average_ranks(_median_table(), maximize=True)

        # Assert
        assert list(ranks.index)[0] == "C"

    def test_should_find_a_difference_with_friedman(self):
        # Act
        result = friedman_test(_median_table())

        # Assert
        assert result is not None
        assert result[1] < 0.05

    def test_should_not_apply_friedman_to_two_algorithms(self):
        # Act / Assert
        assert friedman_test(_median_table()[["A", "B"]]) is None

    def test_should_compute_the_critical_difference_of_demsar(self):
        # Act: Demšar's q(0.05) is 2.569 for four algorithms
        difference = critical_difference(4, 5)

        # Assert
        assert difference == pytest.approx(2.569 * np.sqrt(4 * 5 / (6 * 5)), abs=1e-3)

    def test_should_join_the_algorithms_closer_than_the_critical_difference(self):
        # Arrange
        ranks = pd.Series({"A": 1.0, "B": 1.5, "C": 3.0, "D": 3.4})

        # Act
        groups = nonsignificant_groups(ranks, 1.0)

        # Assert: only the largest groups
        assert groups == [["A", "B"], ["C", "D"]]

    def test_should_join_no_algorithm_when_all_differ(self):
        # Act / Assert
        assert nonsignificant_groups(pd.Series({"A": 1.0, "B": 3.0}), 1.0) == []

    def test_should_adjust_the_comparisons_with_the_pivot_with_holm(self):
        # Arrange: C is far from the pivot A, B near it
        ranks = pd.Series({"A": 1.0, "B": 1.4, "C": 3.6})

        # Act
        holm = holm_against_pivot(ranks, "A", problems=20)

        # Assert: the smallest p-value is multiplied by 2, the next by 1 (and never decreases)
        by_name = holm.set_index("contender")
        assert set(by_name.index) == {"B", "C"}
        assert by_name.loc["C", "adjusted_p_value"] == pytest.approx(
            min(1.0, 2 * by_name.loc["C", "p_value"])
        )
        assert by_name.loc["B", "adjusted_p_value"] >= by_name.loc["C", "adjusted_p_value"]
        assert bool(by_name.loc["C", "significant"]) and not bool(by_name.loc["B", "significant"])


class TestChosenRun:
    @pytest.fixture
    def runs(self) -> pd.DataFrame:
        return _runs({("P1", "tuned"): [0.3, 0.1, 0.4, 0.2]})

    def test_should_choose_the_better_of_the_two_middle_runs_as_the_median(self, runs):
        # Act / Assert: ordered 0.1 (run 2), 0.2 (run 4), 0.3 (run 1), 0.4 (run 3)
        assert chosen_run(runs, "EP", "P1", "tuned", MEDIAN_RUN, maximize=False) == 4

    def test_should_choose_the_best_and_the_worst_run(self, runs):
        # Act / Assert
        assert chosen_run(runs, "EP", "P1", "tuned", BEST_RUN, maximize=False) == 2
        assert chosen_run(runs, "EP", "P1", "tuned", WORST_RUN, maximize=False) == 3

    def test_should_choose_the_highest_as_best_for_a_maximized_indicator(self, runs):
        # Act / Assert
        assert chosen_run(runs, "EP", "P1", "tuned", BEST_RUN, maximize=True) == 3

    def test_should_choose_none_for_a_contender_without_runs(self, runs):
        # Act / Assert
        assert chosen_run(runs, "EP", "P1", "nobody") is None
