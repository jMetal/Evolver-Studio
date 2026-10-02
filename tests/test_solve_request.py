"""Tests for the solve request and its YAML."""

import dataclasses

import yaml

from evolver_studio.solve_request import SolveRequest, solve_request_to_yaml


def _request(**changes) -> SolveRequest:
    request = SolveRequest(
        algorithm_name="NSGA-II",
        encoding="Double",
        population_size=100,
        yaml_parameter_space_file="NSGAIIDouble.yaml",
        extra_config=None,
        configuration="--algorithmResult population --crossover SBX",
        problem="ZDT1",
        reference_front_file_name="resources/referenceFronts/ZDT1.csv",
        max_evaluations=25000,
        number_of_independent_runs=3,
        seed=7,
        indicator_names=["Epsilon", "NormalizedHypervolume"],
        status_frequency=500,
        output_directory="solve-runs/20260101-120000/output",
    )
    return dataclasses.replace(request, **changes)


class TestSolveRequestToYaml:
    def test_should_write_every_field_of_evolvers_record(self):
        # Act
        data = yaml.safe_load(solve_request_to_yaml(_request()))

        # Assert
        assert data == {
            "algorithmName": "NSGA-II",
            "encoding": "Double",
            "populationSize": 100,
            "yamlParameterSpaceFile": "NSGAIIDouble.yaml",
            "extraConfig": {},
            "configuration": "--algorithmResult population --crossover SBX",
            "problem": "ZDT1",
            "referenceFrontFileName": "resources/referenceFronts/ZDT1.csv",
            "maxEvaluations": 25000,
            "numberOfIndependentRuns": 3,
            "seed": 7,
            "indicatorNames": ["Epsilon", "NormalizedHypervolume"],
            "statusFrequency": 500,
            "outputDirectory": "solve-runs/20260101-120000/output",
        }

    def test_should_leave_out_what_is_unset(self):
        # Act
        data = yaml.safe_load(
            solve_request_to_yaml(
                _request(
                    seed=None,
                    reference_front_file_name=None,
                    indicator_names=[],
                    status_frequency=None,
                )
            )
        )

        # Assert
        assert "seed" not in data
        assert "referenceFrontFileName" not in data
        assert "statusFrequency" not in data
        assert data["indicatorNames"] == []

    def test_should_write_the_extra_configuration(self):
        # Act
        data = yaml.safe_load(
            solve_request_to_yaml(
                _request(extra_config={"weightVectorFilesDirectory": "resources/weightVectors"})
            )
        )

        # Assert
        assert data["extraConfig"] == {"weightVectorFilesDirectory": "resources/weightVectors"}


class TestValidationErrors:
    def test_should_accept_a_complete_request(self):
        # Act / Assert
        assert _request().validation_errors() == []

    def test_should_accept_no_indicators_without_a_reference_front(self):
        # Act / Assert
        assert (
            _request(reference_front_file_name=None, indicator_names=[]).validation_errors() == []
        )

    def test_should_ask_for_a_reference_front_when_there_are_indicators(self):
        # Act
        errors = _request(reference_front_file_name=None).validation_errors()

        # Assert
        assert len(errors) == 1
        assert "reference front" in errors[0]

    def test_should_reject_a_status_frequency_below_one(self):
        # Act
        errors = _request(status_frequency=0).validation_errors()

        # Assert
        assert len(errors) == 1
        assert "evaluation" in errors[0]

    def test_should_report_every_problem_found(self):
        # Act
        errors = _request(
            problem="",
            population_size=0,
            max_evaluations=0,
            number_of_independent_runs=0,
            configuration=" ",
            status_frequency=-1,
        ).validation_errors()

        # Assert
        assert len(errors) == 6
