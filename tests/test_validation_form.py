"""Tests for the tables and checks of the Validation page."""

from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.problem_catalogue import Problem, ProblemArgument
from evolver_studio.validation import Contender, StudyProblem, ValidationStudy
from evolver_studio.validation_form import (
    ENCODING_KEY,
    PASTED_ALGORITHM_KEY,
    PASTED_CONFIGURATION_KEY,
    PROBLEM_COLUMNS,
    PROBLEM_ROWS_KEY,
    TUNED_PASTED,
    TUNED_SOURCE_KEY,
    configuration_errors,
    defaults_key,
    form_state_from_study,
    parse_problem_table,
    problem_table,
    problems_key,
    tuned_name_key,
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


class TestFormStateFromStudy:
    STUDY = ValidationStudy(
        encoding="Double",
        problems=(
            StudyProblem("WFG2", (), "resources/referenceFronts/WFG2.2D.csv"),
            StudyProblem("DTLZ2", (12, 2), "resources/referenceFronts/DTLZ2.2D.csv"),
        ),
        contenders=(
            Contender("NSGA-II (tuned)", "NSGA-II", "--crossover blxAlpha"),
            Contender("NSGA-II", "NSGA-II", "--crossover SBX"),
            Contender("MOEA/D", "MOEAD", "--x 1"),
        ),
        pivot="NSGA-II (tuned)",
        population_size=100,
        max_evaluations=25000,
        runs=15,
        seed=1,
    )

    def test_should_set_the_problems_with_their_arguments_and_fronts(self):
        # Act
        state = form_state_from_study(self.STUDY)

        # Assert
        assert state[ENCODING_KEY] == "Double"
        assert state[problems_key("Double")] == ["WFG2", "DTLZ2"]
        rows = state[PROBLEM_ROWS_KEY]
        assert list(rows["problem"]) == ["WFG2", "DTLZ2"]
        assert list(rows["arguments"]) == ["", "12, 2"]
        assert rows["reference_front"].iloc[1] == "resources/referenceFronts/DTLZ2.2D.csv"

    def test_should_restore_the_pivot_as_a_pasted_configuration_and_the_others_as_defaults(self):
        # Act
        state = form_state_from_study(self.STUDY)

        # Assert
        assert state[TUNED_SOURCE_KEY] == TUNED_PASTED
        assert state[PASTED_ALGORITHM_KEY] == "NSGA-II"
        assert state[PASTED_CONFIGURATION_KEY] == "--crossover blxAlpha"
        assert state[tuned_name_key("NSGA-II")] == "NSGA-II (tuned)"
        assert state[defaults_key("Double")] == ["NSGA-II", "MOEA/D"]

    def test_should_not_restore_a_study_whose_pivot_is_missing(self):
        # Arrange
        study = ValidationStudy("Double", (), (), pivot="nobody")

        # Act / Assert
        with pytest.raises(ValueError):
            form_state_from_study(study)
