"""Tests for what the configurations of a training's front have in common."""

import pytest

from evolver_studio.configuration import parse_configuration
from evolver_studio.front_analysis import (
    COLUMNS,
    DIFFERENT,
    SAME,
    UNKNOWN,
    numeric_values,
    summaries_table,
    summarize_front,
    value_counts,
)
from evolver_studio.parameter_space import parse_parameter_space

SPACE = parse_parameter_space(
    """
crossover:
  type: categorical
  globalSubParameters:
    crossoverProbability:
      type: double
      range: [0.0, 1.0]
  values:
    SBX:
      conditionalParameters:
        sbxDistributionIndex:
          type: double
          range: [5.0, 400.0]
    blxAlpha:
      conditionalParameters:
        blxAlphaCrossoverAlpha:
          type: double
          range: [0.0, 1.0]
selection:
  type: categorical
  values:
    tournament:
      conditionalParameters:
        selectionTournamentSize:
          type: integer
          range: [2, 10]
"""
)


def _front(*texts: str) -> list[dict[str, str]]:
    return [parse_configuration(text) for text in texts]


FRONT = _front(
    "--crossover SBX --crossoverProbability 0.90 --sbxDistributionIndex 20 --selection tournament "
    "--selectionTournamentSize 2",
    "--crossover SBX --crossoverProbability 0.92 --sbxDistributionIndex 40 --selection tournament "
    "--selectionTournamentSize 4",
    "--crossover blxAlpha --crossoverProbability 0.94 --blxAlphaCrossoverAlpha 0.5 "
    "--selection tournament --selectionTournamentSize 2",
)
DEFAULTS = {
    "crossover": "SBX",
    "crossoverProbability": "0.9",
    "sbxDistributionIndex": "20.0",
    "selection": "tournament",
    "selectionTournamentSize": "2",
}


def _by_name(summaries):
    return {summary.name: summary for summary in summaries}


class TestSummarizeFront:
    def test_should_find_the_value_most_configurations_choose_for_a_categorical_parameter(self):
        # Act
        crossover = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))["crossover"]

        # Assert
        assert crossover.kind == "categorical"
        assert crossover.typical == "SBX"
        assert crossover.agreement == pytest.approx(2 / 3)
        assert crossover.values == "SBX ×2, blxAlpha ×1"
        assert (crossover.active, crossover.total) == (3, 3)

    def test_should_say_how_many_configurations_the_parameter_is_active_in(self):
        """A conditional parameter only exists in the configurations whose parent allows it."""
        # Act
        summaries = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))

        # Assert
        assert (
            summaries["sbxDistributionIndex"].active,
            summaries["sbxDistributionIndex"].total,
        ) == (2, 3)
        assert summaries["blxAlphaCrossoverAlpha"].active == 1

    def test_should_summarize_a_numeric_parameter_by_its_median_and_range(self):
        # Act
        index = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))["sbxDistributionIndex"]

        # Assert
        assert index.kind == "numeric"
        assert index.typical == "30"
        assert index.values == "20 – 40"

    def test_should_give_full_agreement_when_every_configuration_chooses_the_same(self):
        # Act
        selection = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))["selection"]

        # Assert
        assert selection.agreement == 1.0

    def test_should_measure_a_numeric_agreement_against_the_range_of_the_space(self):
        # Arrange
        summaries = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))

        # Assert: the quartile range of 0.90, 0.92, 0.94 is 0.02 of a range of 1, and that of
        # 20 and 40 is 10 of a range of 395
        assert summaries["crossoverProbability"].agreement == pytest.approx(0.98)
        assert summaries["sbxDistributionIndex"].agreement == pytest.approx(1 - 10 / 395)

    def test_should_show_an_integer_without_decimals(self):
        # Act
        size = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))["selectionTournamentSize"]

        # Assert
        assert size.typical == "2" and size.values == "2 – 4"

    def test_should_order_the_parameters_by_agreement(self):
        # Act
        summaries = summarize_front(FRONT, SPACE, DEFAULTS)

        # Assert
        agreements = [s.agreement for s in summaries]
        assert agreements == sorted(agreements, reverse=True)
        assert summaries[0].agreement == 1.0


class TestVersusTheDefault:
    def test_should_say_a_categorical_parameter_is_the_default_only_if_every_configuration_is(self):
        # Act
        summaries = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))

        # Assert: one configuration uses blxAlpha, the default is SBX
        assert summaries["crossover"].versus_default == DIFFERENT
        assert summaries["selection"].versus_default == SAME

    def test_should_call_a_numeric_value_the_default_when_it_is_close_in_the_range(self):
        # Act
        summaries = _by_name(summarize_front(FRONT, SPACE, DEFAULTS))

        # Assert: 0.92 against 0.9 in a range of 1; 30 against 20 in a range of 395
        assert summaries["crossoverProbability"].versus_default == SAME
        assert summaries["sbxDistributionIndex"].versus_default == SAME

    def test_should_say_a_numeric_value_differs_when_it_is_far(self):
        # Arrange
        front = _front("--crossover SBX --crossoverProbability 0.2 --sbxDistributionIndex 300")

        # Act
        summaries = _by_name(summarize_front(front, SPACE, DEFAULTS))

        # Assert
        assert summaries["crossoverProbability"].versus_default == DIFFERENT
        assert summaries["sbxDistributionIndex"].versus_default == DIFFERENT

    def test_should_not_judge_without_a_default(self):
        # Act
        summaries = summarize_front(FRONT, SPACE, None)

        # Assert
        assert {s.versus_default for s in summaries} == {UNKNOWN}
        assert {s.default for s in summaries} == {None}


class TestParametersTheSpaceDoesNotKnow:
    def test_should_read_one_that_is_all_numbers_as_numeric_without_an_agreement(self):
        # Arrange: the training searched a space the page does not have
        front = _front("--mystery 3.5", "--mystery 4.5")

        # Act
        (mystery,) = summarize_front(front, SPACE, None)

        # Assert
        assert mystery.kind == "numeric" and mystery.typical == "4" and mystery.agreement is None

    def test_should_read_one_that_is_not_as_categorical(self):
        # Act
        (mystery,) = summarize_front(_front("--mystery a", "--mystery a"), SPACE, None)

        # Assert
        assert mystery.kind == "categorical" and mystery.agreement == 1.0


class TestTable:
    def test_should_lay_the_summaries_out_with_the_default_when_known(self):
        # Act
        table = summaries_table(summarize_front(FRONT, SPACE, DEFAULTS))

        # Assert
        assert list(table.columns) == list(COLUMNS)
        row = table.set_index("Parameter").loc["crossover"]
        assert row["Active in"] == "3 of 3" and row["Default"] == "SBX"

    def test_should_mark_an_unknown_default(self):
        # Act
        table = summaries_table(summarize_front(FRONT, SPACE, {}))

        # Assert
        assert set(table["Default"]) == {UNKNOWN}


class TestValues:
    def test_should_count_the_values_of_a_parameter_most_common_first(self):
        # Act / Assert
        assert value_counts(FRONT, "crossover") == [("SBX", 2), ("blxAlpha", 1)]
        assert value_counts(FRONT, "missing") == []

    def test_should_list_the_numbers_of_a_numeric_parameter(self):
        # Act / Assert
        assert numeric_values(FRONT, "sbxDistributionIndex") == [20.0, 40.0]
        assert numeric_values(FRONT, "crossover") == []
