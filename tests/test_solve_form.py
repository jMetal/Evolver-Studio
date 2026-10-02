"""Tests for restoring the Run algorithm form from a past run's request."""

import pytest

from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.solve_form import (
    ALGORITHM_KEY,
    CONFIGURATION_VERSION_KEY,
    EVALUATIONS_KEY,
    FIX_SEED_KEY,
    LOADED_CONFIGURATION_KEY,
    NO_REFERENCE_FRONT,
    POPULATION_CHOICE_KEY,
    POPULATION_KEY,
    PROBLEM_KEY,
    RUNS_KEY,
    SEED_KEY,
    WEIGHT_VECTORS_KEY,
    encoding_key,
    form_state_from_request,
    indicators_key,
    reference_front_key,
)

PROBLEMS = ["ZDT1", "DTLZ2"]


def _request(**changes) -> dict:
    request = {
        "algorithmName": "NSGA-II",
        "encoding": "Double",
        "populationSize": 100,
        "yamlParameterSpaceFile": "NSGAIIDouble.yaml",
        "extraConfig": {},
        "configuration": "--crossover blxAlpha --crossoverProbability 0.8",
        "problem": "ZDT1",
        "referenceFrontFileName": "resources/referenceFronts/ZDT1.csv",
        "maxEvaluations": 5000,
        "numberOfIndependentRuns": 3,
        "seed": 7,
        "indicatorNames": ["Epsilon"],
        "outputDirectory": "solve-runs/x/output",
    }
    return {**request, **changes}


class TestFormStateFromRequest:
    def test_should_set_the_widgets_of_the_budget_the_problem_and_the_algorithm(self):
        # Act
        state = form_state_from_request(_request(), BASE_ALGORITHMS, PROBLEMS, 0)

        # Assert
        assert state[PROBLEM_KEY] == "ZDT1"
        assert state[ALGORITHM_KEY] == "NSGA-II"
        assert state[reference_front_key("ZDT1")] == "resources/referenceFronts/ZDT1.csv"
        assert state[POPULATION_KEY] == state[POPULATION_CHOICE_KEY] == 100
        assert state[EVALUATIONS_KEY] == 5000
        assert state[RUNS_KEY] == 3
        assert state[indicators_key(True)] == ["Epsilon"]

    def test_should_translate_the_registry_name_to_the_algorithms_display_name(self):
        # Act
        state = form_state_from_request(
            _request(algorithmName="MOEAD", extraConfig={"weightVectorFilesDirectory": "w"}),
            BASE_ALGORITHMS,
            PROBLEMS,
            0,
        )

        # Assert
        assert state[ALGORITHM_KEY] == "MOEA/D"
        assert state[WEIGHT_VECTORS_KEY] == "w"

    def test_should_set_the_encoding_only_for_an_algorithm_with_several(self):
        # Act
        nsgaii = form_state_from_request(
            _request(encoding="Permutation"), BASE_ALGORITHMS, PROBLEMS, 0
        )
        rvea = form_state_from_request(
            _request(algorithmName="RVEA", extraConfig={"weightVectorFilesDirectory": "w"}),
            BASE_ALGORITHMS,
            PROBLEMS,
            0,
        )

        # Assert
        assert nsgaii[encoding_key("NSGA-II")] == "Permutation"
        assert encoding_key("RVEA") not in rvea

    def test_should_fix_the_seed_only_when_the_run_had_one(self):
        # Act
        fixed = form_state_from_request(_request(), BASE_ALGORITHMS, PROBLEMS, 0)
        random = form_state_from_request(_request(seed=None), BASE_ALGORITHMS, PROBLEMS, 0)

        # Assert
        assert fixed[FIX_SEED_KEY] is True
        assert fixed[SEED_KEY] == 7
        assert random[FIX_SEED_KEY] is False
        assert SEED_KEY not in random

    def test_should_choose_no_reference_front_when_the_run_had_none(self):
        # Act
        state = form_state_from_request(
            _request(referenceFrontFileName=None, indicatorNames=[]),
            BASE_ALGORITHMS,
            PROBLEMS,
            0,
        )

        # Assert
        assert state[reference_front_key("ZDT1")] == NO_REFERENCE_FRONT
        assert state[indicators_key(False)] == []

    def test_should_carry_the_configuration_and_renew_the_configuration_widgets(self):
        # Act
        state = form_state_from_request(_request(), BASE_ALGORITHMS, PROBLEMS, 4)

        # Assert
        assert state[LOADED_CONFIGURATION_KEY] == {
            "algorithm": "NSGA-II",
            "encoding": "Double",
            "values": {"crossover": "blxAlpha", "crossoverProbability": "0.8"},
        }
        assert state[CONFIGURATION_VERSION_KEY] == 5

    @pytest.mark.parametrize(
        "changes",
        [
            {"algorithmName": "NoSuchAlgorithm"},
            {"problem": "NoSuchProblem"},
            {"configuration": "--odd"},
            {"populationSize": None},
        ],
    )
    def test_should_not_restore_a_request_it_cannot_make_sense_of(self, changes: dict):
        # Act
        state = form_state_from_request(
            {**_request(), **changes}
            if changes != {"populationSize": None}
            else {k: v for k, v in _request().items() if k != "populationSize"},
            BASE_ALGORITHMS,
            PROBLEMS,
            0,
        )

        # Assert
        assert state is None
