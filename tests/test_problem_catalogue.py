"""Tests for reading DescribeMain's problem catalogue and building a request's problem."""

import pytest

from evolver_studio.problem_catalogue import (
    Problem,
    ProblemArgument,
    format_arguments,
    format_problem_spec,
    number_of_objectives,
    parse_arguments_text,
    parse_problem_catalogue,
    problem_spec,
    problem_spec_arguments,
    problem_spec_name,
    problems_with_encoding,
)

MANIFEST = {
    "problemCatalogue": [
        {
            "name": "DTLZ2",
            "family": "DTLZ",
            "encoding": "Double",
            "numberOfObjectives": 3,
            "numberOfVariables": 12,
            "arguments": [
                {"name": "numberOfVariables", "type": "integer", "default": 12},
                {"name": "numberOfObjectives", "type": "integer", "default": 3},
            ],
        },
        {
            "name": "ZDT5",
            "family": "ZDT",
            "encoding": "Binary",
            "numberOfObjectives": 2,
            "numberOfVariables": 11,
            "arguments": [{"name": "numberOfVariables", "type": "integer", "default": 11}],
        },
        {
            "name": "KroAB100TSP",
            "family": "TSP",
            "encoding": "Permutation",
            "numberOfObjectives": 2,
            "numberOfVariables": 100,
            "arguments": None,
        },
        {
            "name": "ZDT1",
            "family": "ZDT",
            "encoding": "Double",
            "numberOfObjectives": 2,
            "numberOfVariables": 30,
            "arguments": [],
        },
    ]
}

ZCAT1 = Problem(
    name="ZCAT1",
    family="ZCAT",
    encoding="Double",
    number_of_objectives=2,
    number_of_variables=30,
    arguments=(
        ProblemArgument("numberOfObjectives", "integer", 2),
        ProblemArgument("complicatedParetoSet", "boolean", False),
        ProblemArgument("scale", "number", None),
    ),
)


class TestParseProblemCatalogue:
    def test_should_read_every_problem_with_its_arguments(self):
        # Act
        problems = parse_problem_catalogue(MANIFEST)

        # Assert
        assert problems is not None
        assert set(problems) == {"DTLZ2", "ZDT5", "KroAB100TSP", "ZDT1"}
        assert problems["DTLZ2"].arguments == (
            ProblemArgument("numberOfVariables", "integer", 12),
            ProblemArgument("numberOfObjectives", "integer", 3),
        )
        assert problems["DTLZ2"].number_of_objectives == 3

    def test_should_read_a_problem_without_arguments_as_an_empty_tuple(self):
        # Act
        problems = parse_problem_catalogue(MANIFEST)

        # Assert
        assert problems is not None
        assert problems["KroAB100TSP"].arguments == ()
        assert problems["ZDT1"].arguments == ()

    def test_should_return_none_for_a_manifest_without_catalogue(self):
        """Evolver 2.3's manifest lists the problems by name only."""
        # Act / Assert
        assert parse_problem_catalogue({"problems": ["ZDT1"]}) is None


class TestProblemsWithEncoding:
    def test_should_keep_only_the_problems_of_the_encoding_sorted(self):
        # Arrange
        problems = parse_problem_catalogue(MANIFEST)
        assert problems is not None

        # Act / Assert
        assert problems_with_encoding(problems, "Double") == ["DTLZ2", "ZDT1"]
        assert problems_with_encoding(problems, "Binary") == ["ZDT5"]
        assert problems_with_encoding(problems, "Permutation") == ["KroAB100TSP"]


class TestProblemSpec:
    def test_should_be_the_name_alone_without_arguments(self):
        # Act / Assert
        assert problem_spec("ZDT1", None) == "ZDT1"
        assert problem_spec("ZDT1", []) == "ZDT1"

    def test_should_be_a_class_and_args_map_with_arguments(self):
        # Act / Assert
        assert problem_spec("DTLZ2", [12, 5]) == {"class": "DTLZ2", "args": [12, 5]}

    @pytest.mark.parametrize(
        ("spec", "name", "arguments", "text"),
        [
            ("ZDT1", "ZDT1", [], "ZDT1"),
            ({"class": "DTLZ2", "args": [12, 5]}, "DTLZ2", [12, 5], "DTLZ2(12, 5)"),
            ({"class": "ZCAT1", "args": [2, True]}, "ZCAT1", [2, True], "ZCAT1(2, true)"),
            ({"class": "ZDT1"}, "ZDT1", [], "ZDT1"),
        ],
    )
    def test_should_read_back_the_name_and_arguments_of_a_spec(self, spec, name, arguments, text):
        # Act / Assert
        assert problem_spec_name(spec) == name
        assert problem_spec_arguments(spec) == arguments
        assert format_problem_spec(spec) == text

    def test_should_have_no_name_for_a_value_that_is_not_a_spec(self):
        # Act / Assert
        assert problem_spec_name(42) is None
        assert problem_spec_name({"args": [1]}) is None


class TestParseArgumentsText:
    def test_should_read_a_blank_text_as_no_arguments(self):
        # Act / Assert
        assert parse_arguments_text(ZCAT1, "  ") == []

    def test_should_convert_each_value_to_its_arguments_type(self):
        # Act
        values = parse_arguments_text(ZCAT1, "3, True, 0.5")

        # Assert
        assert values == [3, True, 0.5]
        assert isinstance(values[0], int)

    def test_should_reject_a_wrong_number_of_values(self):
        """ProblemSpec takes all the arguments or none."""
        # Act
        message = parse_arguments_text(ZCAT1, "3")

        # Assert
        assert isinstance(message, str)
        assert "takes 3 arguments" in message

    @pytest.mark.parametrize("text", ["three, true, 0.5", "3, yes, 0.5", "3, true, half"])
    def test_should_reject_a_value_of_the_wrong_type(self, text: str):
        # Act / Assert
        assert isinstance(parse_arguments_text(ZCAT1, text), str)

    def test_should_reject_arguments_for_a_problem_that_takes_none(self):
        # Arrange
        problem = Problem("ZDT1", "ZDT", "Double", 2, 30, ())

        # Act / Assert
        assert parse_arguments_text(problem, "30") == "ZDT1 takes no arguments."


class TestFormatArguments:
    def test_should_name_each_argument_with_its_default_when_known(self):
        # Act / Assert
        assert format_arguments(ZCAT1) == "numberOfObjectives=2, complicatedParetoSet=false, scale"


class TestNumberOfObjectives:
    def test_should_take_the_argument_when_the_problem_is_built_with_arguments(self):
        # Arrange
        problems = parse_problem_catalogue(MANIFEST)
        assert problems is not None

        # Act / Assert
        assert number_of_objectives(problems["DTLZ2"], (12, 5)) == 5

    def test_should_take_the_default_without_arguments(self):
        # Arrange
        problems = parse_problem_catalogue(MANIFEST)
        assert problems is not None

        # Act / Assert
        assert number_of_objectives(problems["DTLZ2"], ()) == 3
        assert number_of_objectives(problems["ZDT5"], (11,)) == 2
