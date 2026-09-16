"""Tests for parsing/serializing Evolver's YAML parameter space files."""

from pathlib import Path

import pytest
import yaml

from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.parameter_space import (
    CategoricalParameter,
    RangeParameter,
    parse_parameter_space,
    serialize_parameter_space,
    with_range,
    with_selected_choices,
)

PARAMETER_SPACES_DIR = Path("/Users/ajnebro/Softw/Evolver/src/main/resources/parameterSpaces")
REAL_PARAMETER_SPACE_FILES = [
    PARAMETER_SPACES_DIR / "NSGAIIDoubleReduced.yaml",
    *(
        PARAMETER_SPACES_DIR / filename
        for algorithm in BASE_ALGORITHMS
        for filename in algorithm.encodings.values()
    ),
]


class TestParseParameterSpace:
    def test_should_parse_a_range_parameter(self):
        """A double/integer parameter must carry its range."""
        # Arrange
        text = "populationSize:\n  type: integer\n  range: [10, 200]\n"

        # Act
        parameters = parse_parameter_space(text)

        # Assert
        assert parameters == [RangeParameter("populationSize", "integer", 10, 200)]

    def test_should_parse_a_scalar_list_categorical_as_numeric(self):
        """A list of numbers must be recognized as a numeric categorical."""
        # Arrange
        text = "offspringPopulationSize:\n  type: categorical\n  values: [1, 5, 10]\n"

        # Act
        (parameter,) = parse_parameter_space(text)

        # Assert
        assert isinstance(parameter, CategoricalParameter)
        assert parameter.uses_list_form is True
        assert parameter.values_are_numeric is True
        assert [c.value for c in parameter.choices] == ["1", "5", "10"]

    def test_should_parse_a_mapping_categorical_with_conditional_parameters(self):
        """A mapping-form categorical must expose each choice's own sub-parameters."""
        # Arrange
        text = (
            "selection:\n"
            "  type: categorical\n"
            "  values:\n"
            "    tournament:\n"
            "      conditionalParameters:\n"
            "        selectionTournamentSize:\n"
            "          type: integer\n"
            "          range: [2, 10]\n"
            "    random: {}\n"
        )

        # Act
        (parameter,) = parse_parameter_space(text)

        # Assert
        assert parameter.uses_list_form is False
        tournament = next(c for c in parameter.choices if c.value == "tournament")
        random_choice = next(c for c in parameter.choices if c.value == "random")
        assert tournament.conditional_parameters == (
            RangeParameter("selectionTournamentSize", "integer", 2, 10),
        )
        assert random_choice.conditional_parameters == ()

    def test_should_parse_global_sub_parameters(self):
        """globalSubParameters must be attached to the parent, active for every choice."""
        # Arrange
        text = (
            "crossover:\n"
            "  type: categorical\n"
            "  globalSubParameters:\n"
            "    crossoverProbability:\n"
            "      type: double\n"
            "      range: [0.0, 1.0]\n"
            "  values:\n"
            "    SBX: {}\n"
        )

        # Act
        (parameter,) = parse_parameter_space(text)

        # Assert
        assert parameter.global_sub_parameters == (
            RangeParameter("crossoverProbability", "double", 0.0, 1.0),
        )

    @pytest.mark.parametrize(
        ("yaml_type", "expected_kind"),
        [("integer", "integer"), ("int", "integer"), ("double", "double"), ("real", "double")],
    )
    def test_should_normalize_type_synonyms(self, yaml_type: str, expected_kind: str):
        """Evolver accepts int/integer and double/real interchangeably (confirmed in MOPSO.yaml)."""
        # Arrange
        text = f"x:\n  type: {yaml_type}\n  range: [0, 1]\n"

        # Act
        (parameter,) = parse_parameter_space(text)

        # Assert
        assert parameter.kind == expected_kind

    def test_should_raise_on_unknown_type(self):
        """An unrecognized `type` must fail loudly rather than being silently ignored."""
        # Arrange
        text = "weird:\n  type: mystery\n"

        # Act / Assert
        with pytest.raises(ValueError, match="mystery"):
            parse_parameter_space(text)


class TestSerializeParameterSpace:
    def test_should_round_trip_a_range_parameter(self):
        """A parsed-then-serialized range parameter must parse back identically."""
        # Arrange
        parameters = [RangeParameter("crossoverProbability", "double", 0.0, 1.0)]

        # Act
        reparsed = parse_parameter_space(serialize_parameter_space(parameters))

        # Assert
        assert reparsed == parameters

    def test_should_preserve_numeric_list_form(self):
        """A numeric scalar-list categorical must stay a numeric list, not become a mapping."""
        # Arrange
        text = "offspringPopulationSize:\n  type: categorical\n  values: [1, 5, 10]\n"
        parameters = parse_parameter_space(text)

        # Act
        yaml_text = serialize_parameter_space(parameters)
        raw = yaml.safe_load(yaml_text)

        # Assert
        assert raw["offspringPopulationSize"]["values"] == [1, 5, 10]

    def test_should_round_trip_conditional_parameters(self):
        """Nested conditional parameters must survive a serialize/parse round trip."""
        # Arrange
        text = (
            "selection:\n"
            "  type: categorical\n"
            "  values:\n"
            "    tournament:\n"
            "      conditionalParameters:\n"
            "        selectionTournamentSize:\n"
            "          type: integer\n"
            "          range: [2, 10]\n"
            "    random: {}\n"
        )
        parameters = parse_parameter_space(text)

        # Act
        reparsed = parse_parameter_space(serialize_parameter_space(parameters))

        # Assert
        assert reparsed == parameters


class TestWithSelectedChoices:
    def test_should_keep_only_the_selected_choices(self):
        """Narrowing a categorical must drop unselected choices, in original order."""
        # Arrange
        text = "selection:\n  type: categorical\n  values:\n    tournament: {}\n    random: {}\n"
        (parameter,) = parse_parameter_space(text)

        # Act
        narrowed = with_selected_choices(parameter, {"random"})

        # Assert
        assert [c.value for c in narrowed.choices] == ["random"]


class TestWithRange:
    def test_should_replace_the_bounds(self):
        """Narrowing a range must update both bounds."""
        # Arrange
        parameter = RangeParameter("crossoverProbability", "double", 0.0, 1.0)

        # Act
        narrowed = with_range(parameter, 0.6, 0.9)

        # Assert
        assert narrowed.lower_bound == 0.6
        assert narrowed.upper_bound == 0.9


class TestRealParameterSpaceFiles:
    @pytest.mark.parametrize("path", REAL_PARAMETER_SPACE_FILES, ids=lambda p: p.name)
    def test_should_parse_and_round_trip_evolvers_own_files(self, path: Path):
        """The parser/serializer must handle Evolver's real parameter space files."""
        if not path.exists():
            pytest.skip(f"Evolver checkout not found at {path}")

        # Arrange
        original_text = path.read_text()

        # Act
        parameters = parse_parameter_space(original_text)
        reparsed = parse_parameter_space(serialize_parameter_space(parameters))

        # Assert
        assert reparsed == parameters
