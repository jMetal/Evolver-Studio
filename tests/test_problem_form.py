"""Tests for the pure parts of the form of a problem's arguments and reference front."""

from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.problem_catalogue import Problem, ProblemArgument
from evolver_studio.problem_form import (
    argument_values_text,
    defaults_help,
    initial_argument_values,
    initial_front,
    objectives_of,
    rows_digest,
)

WFG1 = Problem(
    name="WFG1",
    family="WFG",
    encoding="Double",
    number_of_objectives=2,
    number_of_variables=6,
    arguments=(
        ProblemArgument("numberOfPositionParameters", "integer", 2),
        ProblemArgument("numberOfDistanceParameters", "integer", 4),
        ProblemArgument("numberOfObjectives", "integer", 2),
    ),
)
ZCAT1 = Problem(
    name="ZCAT1",
    family="ZCAT",
    encoding="Double",
    number_of_objectives=2,
    number_of_variables=30,
    arguments=(
        ProblemArgument("numberOfObjectives", "integer", 2),
        ProblemArgument("complicatedParetoSet", "boolean", False),
    ),
)


@pytest.fixture
def fronts(tmp_path: Path) -> list[str]:
    """Two fronts of a problem, with 2 and 3 objectives, relative to `tmp_path`."""
    directory = tmp_path / "resources" / "referenceFronts"
    directory.mkdir(parents=True)
    (directory / "P.2D.csv").write_text("0.0,1.0\n1.0,0.0\n")
    (directory / "P.3D.csv").write_text("0.0,0.0,1.0\n1.0,0.0,0.0\n")
    return ["resources/referenceFronts/P.2D.csv", "resources/referenceFronts/P.3D.csv"]


class TestInitialArgumentValues:
    def test_should_start_from_the_values_of_a_valid_text(self):
        """Given the arguments of a row, the inputs start from them."""
        # Act
        values = initial_argument_values(WFG1, "4, 20, 3")

        # Assert
        assert values == [4, 20, 3]

    @pytest.mark.parametrize("text", ["", "4, 20", "4, x, 3"])
    def test_should_start_from_the_defaults_without_a_valid_text(self, text: str):
        """Given no arguments, or invalid ones, the inputs start from the defaults."""
        # Act
        values = initial_argument_values(WFG1, text)

        # Assert
        assert values == [2, 4, 2]


class TestArgumentValuesText:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [([4, 20, 3], "4, 20, 3"), ([4, None, 3], "4, , 3"), ([3, True], "3, true")],
    )
    def test_should_write_the_values_as_a_row_keeps_them(self, values: list, expected: str):
        """The values of the inputs become the comma-separated text of the row."""
        # Act & Assert
        assert argument_values_text(values) == expected


class TestObjectivesOf:
    @pytest.mark.parametrize(("problem", "values"), [(WFG1, [4, 20, 3]), (ZCAT1, [3, True])])
    def test_should_read_the_number_of_objectives_of_the_arguments(
        self, problem: Problem, values: list
    ):
        """Given all the arguments, the number of objectives is the one among them."""
        # Act & Assert
        assert objectives_of(problem, values) == 3

    def test_should_keep_the_default_while_a_value_is_missing(self):
        """Given an empty input, the number of objectives is the default one."""
        # Act & Assert
        assert objectives_of(WFG1, [4, 20, None]) == 2


class TestInitialFront:
    def test_should_keep_the_current_front_when_it_has_the_objectives(
        self, fronts: list[str], tmp_path: Path
    ):
        """Given the row's front with the problem's objectives, the selector starts from it."""
        # Act & Assert
        assert initial_front(fronts, fronts[1], 3, tmp_path) == fronts[1]

    def test_should_follow_the_number_of_objectives(self, fronts: list[str], tmp_path: Path):
        """Given a front of 2 objectives and a problem of 3, the selector offers the 3D front."""
        # Act & Assert
        assert initial_front(fronts, fronts[0], 3, tmp_path) == fronts[1]

    @pytest.mark.parametrize("objectives", [None, 5])
    def test_should_choose_none_when_no_front_fits(
        self, fronts: list[str], tmp_path: Path, objectives: int | None
    ):
        """Given several fronts and an unknown or unmatched number of objectives, none is chosen."""
        # Act & Assert
        assert initial_front(fronts, "", objectives, tmp_path) is None


class TestDefaultsHelp:
    def test_should_say_what_the_problem_is_built_with(self):
        """The checkbox's help gives the objectives and variables of the defaults."""
        # Act & Assert
        assert defaults_help(WFG1) == (
            "Without them, WFG1 is built with its defaults (2 objectives, 6 variables)."
        )


class TestRowsDigest:
    def test_should_change_when_a_row_changes(self):
        """Editing a row changes the digest, so that the form tells its rows from others."""
        # Arrange
        rows = pd.DataFrame([{"problem": "WFG1", "arguments": "", "reference_front": "a"}])
        edited = rows.assign(arguments="4, 20, 3")

        # Act & Assert
        assert rows_digest(rows) == rows_digest(rows.copy())
        assert rows_digest(rows) != rows_digest(edited)
