"""Tests for taking a configuration a training found to Validation and to Run algorithm."""

from pathlib import Path

import pytest

from evolver_studio.problem_catalogue import Problem, ProblemArgument
from evolver_studio.solve_form import (
    ALGORITHM_KEY,
    EVALUATIONS_KEY,
    LOADED_CONFIGURATION_KEY,
    POPULATION_KEY,
    PROBLEM_KEY,
    WEIGHT_VECTORS_KEY,
    reference_front_key,
)
from evolver_studio.training_handoff import solve_form_state, validation_form_state
from evolver_studio.training_results import TrainedConfiguration
from evolver_studio.training_runs import FinishedTraining
from evolver_studio.validation_form import (
    ENCODING_KEY,
    INDICATORS_KEY,
    PASTED_ALGORITHM_KEY,
    PASTED_CONFIGURATION_KEY,
    PROBLEM_ROWS_KEY,
    TUNED_PASTED,
    TUNED_SOURCE_KEY,
    defaults_key,
    problems_key,
    tuned_name_key,
)

CONFIGURATION = TrainedConfiguration({"EP": 0.1, "NHV": 0.2}, "--crossover blxAlpha")
CATALOGUE = {
    "ZDT4": Problem("ZDT4", "ZDT", "Double", 2, 10, ()),
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


def _training(**changes) -> FinishedTraining:
    training = FinishedTraining(
        run_id="20261007-120000",
        algorithm="NSGA-II",
        encoding="Double",
        problems=("ZDT4", "DTLZ2"),
        configurations=(CONFIGURATION,),
        run_dir=Path("cli-runner-runs/20261007-120000"),
        population_size=100,
        problem_specs=("ZDT4", {"class": "DTLZ2", "args": [12, 2]}),
        reference_fronts=(
            "resources/referenceFronts/ZDT4.csv",
            "resources/referenceFronts/DTLZ2.2D.csv",
        ),
        evaluations=(12000, 8000),
        indicators=("Epsilon", "NormalizedHypervolume"),
    )
    from dataclasses import replace

    return replace(training, **changes)


class TestValidationFormState:
    def test_should_put_the_configuration_in_as_the_tuned_one(self):
        # Act
        state = validation_form_state(_training(), CONFIGURATION)

        # Assert
        assert state is not None
        assert state[ENCODING_KEY] == "Double"
        assert state[TUNED_SOURCE_KEY] == TUNED_PASTED
        assert state[PASTED_ALGORITHM_KEY] == "NSGA-II"
        assert state[PASTED_CONFIGURATION_KEY] == "--crossover blxAlpha"
        assert state[tuned_name_key("NSGA-II")] == "NSGA-II (tuned)"

    def test_should_compare_it_with_the_default_of_the_same_algorithm(self):
        # Act
        state = validation_form_state(_training(), CONFIGURATION)

        # Assert
        assert state is not None and state[defaults_key("Double")] == ["NSGA-II"]

    def test_should_not_choose_among_the_variants_of_an_algorithm_with_several_defaults(self):
        # Act: RVEA has three default configurations
        state = validation_form_state(_training(algorithm="RVEA"), CONFIGURATION)

        # Assert
        assert state is not None and defaults_key("Double") not in state

    def test_should_leave_the_problems_for_the_user_to_choose(self):
        # Act
        state = validation_form_state(_training(), CONFIGURATION)

        # Assert
        assert state is not None
        assert state[problems_key("Double")] == []
        assert state[PROBLEM_ROWS_KEY].empty

    def test_should_take_the_population_and_the_indicators_of_the_training(self):
        # Act
        state = validation_form_state(_training(population_size=150), CONFIGURATION)

        # Assert
        assert state is not None
        assert state["validation_population"] == 150
        assert state[INDICATORS_KEY] == ["Epsilon", "NormalizedHypervolume"]

    def test_should_not_fill_what_it_does_not_know(self):
        # Act
        state = validation_form_state(_training(algorithm="Nope"), CONFIGURATION)

        # Assert
        assert state is None


class TestSolveFormState:
    NAMES = ["ZDT4", "DTLZ2", "ZDT1"]

    def test_should_fill_the_run_algorithm_form_with_the_configuration_on_a_problem(self):
        # Act
        state = solve_form_state(_training(), CONFIGURATION, 0, self.NAMES, 0, CATALOGUE)

        # Assert
        assert state is not None
        assert state[PROBLEM_KEY] == "ZDT4"
        assert state[ALGORITHM_KEY] == "NSGA-II"
        assert state[POPULATION_KEY] == 100
        assert state[EVALUATIONS_KEY] == 12000
        assert state[reference_front_key("ZDT4")] == "resources/referenceFronts/ZDT4.csv"
        assert state[LOADED_CONFIGURATION_KEY]["values"] == {"crossover": "blxAlpha"}

    def test_should_take_the_budget_and_the_front_of_the_problem_chosen(self):
        # Act
        state = solve_form_state(_training(), CONFIGURATION, 1, self.NAMES, 0, CATALOGUE)

        # Assert: DTLZ2 with its arguments
        assert state is not None
        assert state[PROBLEM_KEY] == "DTLZ2"
        assert state[EVALUATIONS_KEY] == 8000
        assert state["solve_problem_arguments_DTLZ2"] is True
        assert state["solve_problem_argument_DTLZ2_numberOfObjectives"] == 2

    def test_should_carry_the_weight_vectors_of_a_decomposition_algorithm(self):
        # Arrange
        training = _training(
            algorithm="MOEAD",
            extra_config={"weightVectorFilesDirectory": "resources/weightVectors"},
        )

        # Act
        state = solve_form_state(training, CONFIGURATION, 0, self.NAMES, 0, CATALOGUE)

        # Assert
        assert state is not None and state[WEIGHT_VECTORS_KEY] == "resources/weightVectors"

    @pytest.mark.parametrize("index", [-1, 2])
    def test_should_not_take_a_problem_the_training_did_not_have(self, index: int):
        # Act / Assert
        assert solve_form_state(_training(), CONFIGURATION, index, self.NAMES, 0, CATALOGUE) is None

    def test_should_not_take_a_problem_the_page_does_not_offer(self):
        # Act
        state = solve_form_state(_training(), CONFIGURATION, 0, ["ZDT1"], 0, CATALOGUE)

        # Assert
        assert state is None
