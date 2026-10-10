"""Tests for the statistics of a validation study."""

import numpy as np
import pandas as pd
import pytest

from evolver_studio.validation_stats import (
    BENJAMINI_HOCHBERG,
    BEST_RUN,
    HOLM,
    MEDIAN_RUN,
    NO_CORRECTION,
    NO_DIFFERENCE,
    PIVOT_BETTER,
    PIVOT_WORSE,
    WORST_RUN,
    a12,
    adjust_p_values,
    aligned_friedman_test,
    average_ranks,
    bayesian_signed_rank,
    best_contenders,
    chosen_run,
    compare_with_pivot,
    critical_difference,
    friedman_test,
    holm_against_pivot,
    indicator_correlations,
    indicator_names,
    interquartile_ranges,
    magnitude,
    medians,
    nonsignificant_groups,
    quade_test,
    signed_rank_against_pivot,
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


class TestMultipleComparisons:
    P_VALUES = np.array([0.01, 0.04, 0.03, 0.20])

    def test_should_leave_the_p_values_as_they_are_without_correction(self):
        # Act / Assert
        assert list(adjust_p_values(self.P_VALUES, NO_CORRECTION)) == list(self.P_VALUES)

    def test_should_adjust_with_holm(self):
        # Act: sorted 0.01, 0.03, 0.04, 0.20 times 4, 3, 2, 1, never decreasing
        adjusted = adjust_p_values(self.P_VALUES, HOLM)

        # Assert
        assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])

    def test_should_adjust_with_benjamini_hochberg(self):
        # Act: p * 4 / rank, never increasing from the largest
        adjusted = adjust_p_values(self.P_VALUES, BENJAMINI_HOCHBERG)

        # Assert
        assert adjusted == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20])

    def test_should_decide_the_verdicts_on_the_adjusted_p_values(self):
        # Arrange: a difference on one problem, significant alone, not among many comparisons
        values = {("P1", "tuned"): LOW, ("P1", "A"): [v + 0.05 for v in LOW]}
        for index in range(2, 21):
            values[(f"P{index}", "tuned")] = LOW
            values[(f"P{index}", "A")] = LOW
        runs = _runs(values)

        # Act
        alone = compare_with_pivot(runs, "EP", "tuned", maximize=False)
        corrected = compare_with_pivot(runs, "EP", "tuned", maximize=False, correction=HOLM)

        # Assert
        assert alone.iloc[0]["verdict"] == PIVOT_BETTER
        assert corrected.iloc[0]["verdict"] == NO_DIFFERENCE
        assert corrected.iloc[0]["adjusted_p_value"] > corrected.iloc[0]["p_value"]


def _random_table() -> pd.DataFrame:
    """The table SAES's tests were checked against."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        rng.random((9, 5)) + np.array([0, 0.1, 0.2, 0.3, 0.05]), columns=list("ABCDE")
    )


class TestOtherRankingTests:
    def test_should_compute_the_aligned_friedman_test_as_saes(self):
        # Act
        statistic, p_value = aligned_friedman_test(_random_table())

        # Assert: SAES 1.5.0's friedman_aligned_rank on the same table
        assert statistic == pytest.approx(12.54233689)
        assert p_value == pytest.approx(0.01374265, abs=1e-8)

    def test_should_compute_quades_test_as_saes(self):
        # Act
        statistic, p_value = quade_test(_random_table())

        # Assert: SAES 1.5.0's quade on the same table
        assert statistic == pytest.approx(2.16042781)
        assert p_value == pytest.approx(0.09605242, abs=1e-8)

    def test_should_not_apply_them_to_two_algorithms(self):
        # Act / Assert
        assert aligned_friedman_test(_random_table()[["A", "B"]]) is None
        assert quade_test(_random_table()[["A", "B"]]) is None

    def test_should_compare_the_pivot_over_the_problems_with_the_signed_rank_test(self):
        # Arrange: the pivot better than B on all ten problems, and as good as C
        table = pd.DataFrame(
            {"pivot": np.arange(10) / 10, "B": np.arange(10) / 10 + 0.5, "C": np.arange(10) / 10}
        )

        # Act
        result = signed_rank_against_pivot(table, "pivot").set_index("contender")

        # Assert
        assert result.loc["B", "pivot_better"] == 10
        assert result.loc["B", "verdict"] == PIVOT_BETTER
        assert result.loc["C", "verdict"] == NO_DIFFERENCE


class TestBayesianSignedRank:
    def test_should_give_probabilities_that_add_up_to_one(self):
        # Act
        result = bayesian_signed_rank(_random_table(), "A", "B", rope=0.05, maximize=True)

        # Assert
        assert result.pivot_better + result.equivalent + result.pivot_worse == pytest.approx(1)
        assert result.samples.shape[1] == 3

    def test_should_agree_with_baycomp(self):
        # Act: baycomp 1.0.3's SignedRankTest.probs on the same medians gives 0.106, 0.056, 0.838
        rng = np.random.default_rng(5)
        table = pd.DataFrame(rng.random((9, 2)) + np.array([0, 0.1]), columns=["A", "B"])
        result = bayesian_signed_rank(
            table, "A", "B", rope=0.1, maximize=True, samples=50000, relative=False
        )

        # Assert: within the error of sampling
        assert result.pivot_better == pytest.approx(0.106, abs=0.01)
        assert result.equivalent == pytest.approx(0.056, abs=0.01)
        assert result.pivot_worse == pytest.approx(0.838, abs=0.01)

    def test_should_be_sure_when_the_pivot_is_always_much_better(self):
        # Arrange
        table = pd.DataFrame({"pivot": [0.1] * 10, "B": [0.9] * 10})

        # Act
        result = bayesian_signed_rank(table, "pivot", "B", rope=0.01)

        # Assert
        assert result.pivot_better > 0.95


class TestIndicatorCorrelations:
    def test_should_turn_every_indicator_so_that_higher_is_better(self):
        # Arrange: EP (minimized) and HV (maximized) that agree on which runs are good
        runs = pd.DataFrame(
            {
                "contender": ["A"] * 4,
                "problem": ["P"] * 4,
                "Run": [1, 2, 3, 4],
                "Seed": [1, 2, 3, 4],
                "TimeMs": [1] * 4,
                "EP": [0.1, 0.2, 0.3, 0.4],
                "HV": [0.9, 0.8, 0.7, 0.6],
            }
        )

        # Act
        correlations = indicator_correlations(runs)

        # Assert
        assert correlations.loc["EP", "HV"] == pytest.approx(1.0)


class TestRelativeBayesianSignedRank:
    def test_should_weigh_problems_of_different_scales_alike_with_relative_differences(self):
        # Arrange: the pivot half the other's (lower is better) on problems whose values are tiny
        table = pd.DataFrame({"pivot": [0.004] * 10, "B": [0.008] * 10})

        # Act
        absolute = bayesian_signed_rank(table, "pivot", "B", rope=0.01, relative=False)
        relative = bayesian_signed_rank(table, "pivot", "B", rope=0.01)

        # Assert: 0.004 is within an absolute ROPE of 0.01, but 67% of the medians' mean
        assert absolute.equivalent > 0.95
        assert relative.pivot_better > 0.95

    def test_should_not_mix_the_scales_of_the_problems(self):
        # Arrange: on each problem HV is a decreasing function of NHV, at different scales
        rows = []
        for problem, scale in (("P1", 1.0), ("P2", 10.0)):
            for run, nhv in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
                rows.append(
                    {
                        "contender": "A",
                        "problem": problem,
                        "Run": run,
                        "Seed": run,
                        "TimeMs": 1,
                        "NHV": nhv,
                        "HV": scale * (1 - nhv),
                    }
                )

        # Act
        correlations = indicator_correlations(pd.DataFrame(rows))

        # Assert
        assert correlations.loc["NHV", "HV"] == pytest.approx(1.0)
