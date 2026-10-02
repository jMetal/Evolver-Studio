"""Tests for parsing/serializing Evolver's YAML parameter space files."""

import pytest
import yaml

from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.evolver_client import jar_path
from evolver_studio.parameter_space import (
    CategoricalChoice,
    CategoricalParameter,
    ParameterRow,
    ParameterSpaceSummary,
    RangeParameter,
    active_parameter_names,
    active_sub_parameters,
    count_parameters,
    filter_rows,
    parameter_rows,
    parse_parameter_space,
    serialize_parameter_space,
    summarize_parameter_space,
    with_range,
    with_selected_choices,
)
from evolver_studio.resource_files import parameter_space_text

# Evolver's real parameter space files, read from its jar.
REAL_PARAMETER_SPACE_FILES = [
    "NSGAIIDoubleReduced.yaml",
    *(filename for algorithm in BASE_ALGORITHMS for filename in algorithm.encodings.values()),
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
    @pytest.mark.parametrize("filename", REAL_PARAMETER_SPACE_FILES)
    def test_should_parse_and_round_trip_evolvers_own_files(self, filename: str):
        """The parser/serializer must handle Evolver's real parameter space files."""
        jar = jar_path()
        if not jar.is_file():
            pytest.skip(f"Evolver jar not found at {jar}")

        # Arrange
        original_text = parameter_space_text(jar, filename)

        # Act
        parameters = parse_parameter_space(original_text)
        reparsed = parse_parameter_space(serialize_parameter_space(parameters))

        # Assert
        assert reparsed == parameters


def _crossover_space() -> list:
    """A small space: a crossover with a global probability and an SBX-only index."""
    crossover = CategoricalParameter(
        "crossover",
        (
            CategoricalChoice("SBX", (RangeParameter("sbxDistributionIndex", "double", 5, 400),)),
            CategoricalChoice("blxAlpha"),
        ),
        global_sub_parameters=(RangeParameter("crossoverProbability", "double", 0, 1),),
    )
    return [crossover, RangeParameter("populationSize", "integer", 10, 100)]


class TestCountParameters:
    def test_should_count_sub_parameters_of_every_value(self):
        # Act
        total = count_parameters(_crossover_space())

        # Assert
        assert total == 4

    def test_should_match_evolvers_count_for_nsgaii_double(self):
        """Evolver's ParameterManagement.parameterFlattening gives 34 for NSGAIIDouble.yaml."""
        jar = jar_path()
        if not jar.is_file():
            pytest.skip(f"Evolver jar not found at {jar}")

        # Act
        total = count_parameters(
            parse_parameter_space(parameter_space_text(jar, "NSGAIIDouble.yaml"))
        )

        # Assert
        assert total == 34


class TestActiveParameters:
    def test_should_activate_the_conditional_parameters_of_the_chosen_value(self):
        # Act
        names = active_parameter_names(_crossover_space(), {"crossover": "SBX"})

        # Assert
        assert names == [
            "crossover",
            "crossoverProbability",
            "sbxDistributionIndex",
            "populationSize",
        ]

    def test_should_keep_only_global_sub_parameters_for_a_value_without_conditionals(self):
        # Act
        names = active_parameter_names(_crossover_space(), {"crossover": "blxAlpha"})

        # Assert
        assert names == ["crossover", "crossoverProbability", "populationSize"]

    def test_should_take_the_first_value_when_no_choice_is_given(self):
        # Act
        active = active_sub_parameters(_crossover_space()[0], {})

        # Assert
        assert [p.name for p in active] == ["crossoverProbability", "sbxDistributionIndex"]

    def test_should_have_no_sub_parameters_for_a_range_parameter(self):
        # Act
        active = active_sub_parameters(_crossover_space()[1], {})

        # Assert
        assert active == []


class TestParameterRows:
    def test_should_list_each_parameter_before_its_sub_parameters_conditional_first(self):
        """Conditional parameters of each value come first, then the global ones."""
        # Act
        rows = parameter_rows(_crossover_space())

        # Assert
        assert rows == [
            ParameterRow(0, "crossover", "categorical", "SBX, blxAlpha", "", None),
            ParameterRow(1, "sbxDistributionIndex", "double", "[5, 400]", "crossover = SBX", 0),
            ParameterRow(1, "crossoverProbability", "double", "[0, 1]", "crossover (any)", 0),
            ParameterRow(0, "populationSize", "integer", "[10, 100]", "", None),
        ]

    def test_should_have_one_row_per_parameter_of_a_real_file(self):
        """The rows must cover every parameter, as count_parameters counts them."""
        jar = jar_path()
        if not jar.is_file():
            pytest.skip(f"Evolver jar not found at {jar}")

        # Arrange
        parameters = parse_parameter_space(parameter_space_text(jar, "NSGAIIDouble.yaml"))

        # Act
        rows = parameter_rows(parameters)

        # Assert
        assert len(rows) == count_parameters(parameters)


class TestFilterRows:
    def test_should_keep_every_row_for_a_blank_filter(self):
        # Arrange
        rows = parameter_rows(_crossover_space())

        # Act
        kept = filter_rows(rows, "  ")

        # Assert
        assert kept == rows

    def test_should_keep_the_ancestors_of_a_matching_row(self):
        """A match must keep the parameter it hangs from, which explains when it applies."""
        # Arrange
        rows = parameter_rows(_crossover_space())

        # Act
        kept = filter_rows(rows, "DISTRIBUTION")

        # Assert
        assert [row.name for row in kept] == ["crossover", "sbxDistributionIndex"]

    def test_should_match_the_domain_and_the_condition(self):
        # Arrange
        rows = parameter_rows(_crossover_space())

        # Act
        kept = filter_rows(rows, "sbx")

        # Assert
        assert [row.name for row in kept] == ["crossover", "sbxDistributionIndex"]

    def test_should_keep_nothing_when_nothing_matches(self):
        # Act
        kept = filter_rows(parameter_rows(_crossover_space()), "archive")

        # Assert
        assert kept == []


class TestSummarizeParameterSpace:
    def test_should_count_parameters_by_kind_and_depth(self):
        # Act
        summary = summarize_parameter_space(_crossover_space())

        # Assert
        assert summary == ParameterSpaceSummary(
            total=4, top_level=2, categorical=1, numeric=3, max_depth=1
        )

    def test_should_summarize_an_empty_space(self):
        # Act
        summary = summarize_parameter_space([])

        # Assert
        assert summary == ParameterSpaceSummary(0, 0, 0, 0, 0)
