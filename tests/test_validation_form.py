"""Tests for the tables and checks of the Validation page."""

from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.problem_catalogue import Problem, ProblemArgument
from evolver_studio.validation_form import (
    PROBLEM_COLUMNS,
    configuration_errors,
    parse_problem_table,
    problem_table,
)

CATALOGUE = {
    "ZDT1": Problem("ZDT1", "ZDT", "Double", 2, 30, ()),
    "DTLZ2": Problem(
        "DTLZ2",
        "DTLZ",
        "Double",
        3,
        12,
        (
            ProblemArgument("numberOfVariables", "integer", 12),
            ProblemArgument("numberOfObjectives", "integer", 3),
        ),
    ),
    "ZDT5": Problem("ZDT5", "ZDT", "Binary", 2, 11, ()),
}


@pytest.fixture
def working_directory(tmp_path: Path) -> Path:
    fronts = tmp_path / "resources" / "referenceFronts"
    fronts.mkdir(parents=True)
    (fronts / "ZDT1.csv").write_text("0,1\n")
    (fronts / "DTLZ2.2D.csv").write_text("0,1\n")
    (fronts / "DTLZ2.3D.csv").write_text("0,0,1\n")
    return tmp_path


def _table(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(PROBLEM_COLUMNS))


class TestProblemTable:
    def test_should_give_a_row_per_problem_with_its_reference_front(self, working_directory: Path):
        # Act
        table = problem_table(["ZDT1", "DTLZ2"], CATALOGUE, working_directory)

        # Assert: DTLZ2 has fronts for several objectives; the catalogue says 3
        assert list(table["problem"]) == ["ZDT1", "DTLZ2"]
        assert list(table["reference_front"]) == [
            "resources/referenceFronts/ZDT1.csv",
            "resources/referenceFronts/DTLZ2.3D.csv",
        ]
        assert list(table["arguments"]) == ["", ""]

    def test_should_keep_what_the_user_edited_of_a_problem_that_stays(
        self, working_directory: Path
    ):
        # Arrange
        edited = _table(("DTLZ2", "12, 2", "resources/referenceFronts/DTLZ2.2D.csv"))

        # Act
        table = problem_table(["ZDT1", "DTLZ2"], CATALOGUE, working_directory, edited)

        # Assert
        row = table.set_index("problem").loc["DTLZ2"]
        assert (row["arguments"], row["reference_front"]) == (
            "12, 2",
            "resources/referenceFronts/DTLZ2.2D.csv",
        )

    def test_should_drop_the_rows_of_the_problems_no_longer_chosen(self, working_directory: Path):
        # Arrange
        edited = _table(("ZDT1", "", "x"), ("DTLZ2", "", "y"))

        # Act
        table = problem_table(["ZDT1"], CATALOGUE, working_directory, edited)

        # Assert
        assert list(table["problem"]) == ["ZDT1"]


class TestParseProblemTable:
    def test_should_read_the_problems_with_their_arguments(self, working_directory: Path):
        # Arrange
        table = _table(
            ("ZDT1", "", "resources/referenceFronts/ZDT1.csv"),
            ("DTLZ2", "12, 2", "resources/referenceFronts/DTLZ2.2D.csv"),
        )

        # Act
        problems, errors = parse_problem_table(table, CATALOGUE, "Double", working_directory)

        # Assert
        assert errors == []
        assert [(p.name, p.arguments) for p in problems] == [("ZDT1", ()), ("DTLZ2", (12, 2))]
        assert problems[1].label == "DTLZ2(12, 2)"

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            (("Nothing", "", "x"), "Nothing is not a problem Evolver knows."),
            (("ZDT5", "", "x"), "ZDT5 is a Binary problem, not a Double one."),
            (("DTLZ2", "12", "x"), "takes 2 arguments"),
            (("ZDT1", "", ""), "ZDT1 needs a reference front"),
            (("ZDT1", "", "resources/nowhere.csv"), "the reference front resources/nowhere.csv"),
            (
                ("ZDT1", "", "resources/referenceFronts/DTLZ2.3D.csv"),
                "the reference front has 3 objectives, the problem 2",
            ),
            (
                ("DTLZ2", "12, 2", "resources/referenceFronts/DTLZ2.3D.csv"),
                "the reference front has 3 objectives, the problem 2",
            ),
        ],
    )
    def test_should_say_what_is_wrong_with_a_row(
        self, working_directory: Path, row: tuple, message: str
    ):
        # Act
        problems, errors = parse_problem_table(_table(row), CATALOGUE, "Double", working_directory)

        # Assert
        assert problems == ()
        assert len(errors) == 1 and message in errors[0]


class TestConfigurationErrors:
    SPACE = """
algorithmResult:
  type: categorical
  values:
    population:
    externalArchive:
"""

    def test_should_accept_a_configuration_the_space_allows(self):
        # Act / Assert
        assert configuration_errors("--algorithmResult population", self.SPACE, "tuned") == []

    def test_should_reject_a_value_the_space_does_not_allow(self):
        # Act
        errors = configuration_errors("--algorithmResult nonsense", self.SPACE, "tuned")

        # Assert
        assert errors == ["tuned: the space does not allow the value of algorithmResult."]

    def test_should_reject_a_malformed_or_empty_configuration(self):
        # Act / Assert
        assert "pairs" in configuration_errors("--algorithmResult", self.SPACE, "tuned")[0]
        assert configuration_errors("  ", self.SPACE, "tuned") == [
            "tuned: the configuration is empty."
        ]
