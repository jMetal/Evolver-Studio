"""Tests for reading, completing and checking base-level configurations."""

import pytest

from evolver_studio.configuration import (
    complete_values,
    configuration_string,
    modified_values,
    parse_configuration,
    starting_values,
    values_outside_the_space,
)
from evolver_studio.evolver_client import jar_path
from evolver_studio.parameter_space import (
    CategoricalChoice,
    CategoricalParameter,
    RangeParameter,
    parse_parameter_space,
)
from evolver_studio.resource_files import default_configuration_text, parameter_space_text

# Evolver's default configurations, with the parameter space each is written against.
REAL_CONFIGURATIONS = [
    ("NSGAIIDouble.yaml", "NSGAIIDoubleDefault.txt"),
    ("MOEADDouble.yaml", "MOEADDoubleDefault.txt"),
    ("RVEADouble.yaml", "RVEADoubleDefault.txt"),
    ("RVEADouble.yaml", "RVEAStarDoubleDefault.txt"),
    ("RVEADouble.yaml", "IRVEADoubleDefault.txt"),
]


def _crossover_space() -> list:
    """A crossover with a global probability and an SBX-only index, then an integer range."""
    crossover = CategoricalParameter(
        "crossover",
        (
            CategoricalChoice("SBX", (RangeParameter("sbxDistributionIndex", "double", 5, 400),)),
            CategoricalChoice(
                "blxAlpha", (RangeParameter("blxAlphaCrossoverAlpha", "double", 0, 1),)
            ),
        ),
        global_sub_parameters=(RangeParameter("crossoverProbability", "double", 0, 1),),
    )
    return [crossover, RangeParameter("populationSize", "integer", 10, 100)]


class TestParseConfiguration:
    def test_should_read_name_value_pairs_in_order(self):
        # Act
        values = parse_configuration("--crossover SBX --crossoverProbability 0.9")

        # Assert
        assert values == {"crossover": "SBX", "crossoverProbability": "0.9"}
        assert list(values) == ["crossover", "crossoverProbability"]

    def test_should_read_only_the_first_line(self):
        # Act
        values = parse_configuration("\n--crossover SBX\n--crossover blxAlpha\n")

        # Assert
        assert values == {"crossover": "SBX"}

    def test_should_read_an_empty_text_as_no_values(self):
        # Act / Assert
        assert parse_configuration("  \n") == {}

    @pytest.mark.parametrize("text", ["--crossover", "crossover SBX"])
    def test_should_reject_text_that_is_not_name_value_pairs(self, text: str):
        # Act / Assert
        with pytest.raises(ValueError):
            parse_configuration(text)


class TestStartingValues:
    def test_should_start_from_the_first_choice_and_the_middle_of_each_range(self):
        # Act
        values = starting_values(_crossover_space())

        # Assert
        assert values == {
            "crossover": "SBX",
            "sbxDistributionIndex": "202.5",
            "blxAlphaCrossoverAlpha": "0.5",
            "crossoverProbability": "0.5",
            "populationSize": "55",
        }


class TestConfigurationString:
    def test_should_write_only_the_active_parameters_in_the_spaces_order(self):
        # Arrange
        values = complete_values(_crossover_space(), {"crossover": "blxAlpha"})

        # Act
        text = configuration_string(_crossover_space(), values)

        # Assert
        assert text == (
            "--crossover blxAlpha --crossoverProbability 0.5 "
            "--blxAlphaCrossoverAlpha 0.5 --populationSize 55"
        )

    def test_should_keep_the_values_the_configuration_gives(self):
        # Arrange
        given = parse_configuration("--crossover SBX --sbxDistributionIndex 20.0")

        # Act
        values = complete_values(_crossover_space(), given)

        # Assert
        assert values["sbxDistributionIndex"] == "20.0"
        assert values["populationSize"] == "55"


class TestValuesOutsideTheSpace:
    def test_should_accept_values_inside_the_space(self):
        # Arrange
        values = complete_values(_crossover_space(), {"crossoverProbability": "0.9"})

        # Act / Assert
        assert values_outside_the_space(_crossover_space(), values) == []

    def test_should_flag_an_unknown_choice_and_an_out_of_range_number(self):
        # Arrange
        values = complete_values(
            _crossover_space(),
            {"crossover": "SBX", "populationSize": "500", "sbxDistributionIndex": "x"},
        )

        # Act
        outside = values_outside_the_space(_crossover_space(), values)

        # Assert
        assert outside == ["sbxDistributionIndex", "populationSize"]

    def test_should_ignore_the_inactive_parameters(self):
        # Arrange: blxAlphaCrossoverAlpha is out of range, but only SBX's parameters are active
        values = complete_values(_crossover_space(), {"blxAlphaCrossoverAlpha": "7"})

        # Act / Assert
        assert values_outside_the_space(_crossover_space(), values) == []


class TestModifiedValues:
    def test_should_list_the_active_parameters_that_differ_from_the_reference(self):
        # Arrange
        reference = {
            "crossover": "SBX",
            "sbxDistributionIndex": "20.0",
            "crossoverProbability": "0.9",
        }
        values = complete_values(
            _crossover_space(), {**reference, "crossoverProbability": "0.8", "populationSize": "55"}
        )

        # Act
        modified = modified_values(_crossover_space(), values, reference)

        # Assert: populationSize is not in the reference, so it counts as modified too
        assert modified == ["crossoverProbability", "populationSize"]

    def test_should_compare_numbers_as_numbers(self):
        # Arrange
        reference = {"crossover": "SBX", "crossoverProbability": "0.9", "populationSize": "55"}
        values = complete_values(
            _crossover_space(),
            {**reference, "crossoverProbability": "0.90", "sbxDistributionIndex": "202.5"},
        )

        # Act
        modified = modified_values(_crossover_space(), values, reference)

        # Assert: only sbxDistributionIndex, missing from the reference
        assert modified == ["sbxDistributionIndex"]


class TestRealDefaultConfigurations:
    @pytest.mark.parametrize(("space_file", "configuration_file"), REAL_CONFIGURATIONS)
    def test_should_be_inside_their_parameter_space_and_round_trip(
        self, space_file: str, configuration_file: str
    ):
        """A default that lay outside its space would make the guided editor warn from the start."""
        jar = jar_path()
        if not jar.is_file():
            pytest.skip(f"Evolver jar not found at {jar}")

        # Arrange
        parameters = parse_parameter_space(parameter_space_text(jar, space_file))
        given = parse_configuration(default_configuration_text(jar, configuration_file))

        # Act
        values = complete_values(parameters, given)
        written = parse_configuration(configuration_string(parameters, values))

        # Assert
        assert values_outside_the_space(parameters, values) == []
        assert written == given
