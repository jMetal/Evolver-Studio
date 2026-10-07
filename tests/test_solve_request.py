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
        front_frequency=None,
        write_population=False,
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
                    front_frequency=None,
                )
            )
        )

        # Assert
        assert "seed" not in data
        assert "referenceFrontFileName" not in data
        assert "statusFrequency" not in data
        assert "frontFrequency" not in data
        assert "writePopulation" not in data
        assert data["indicatorNames"] == []

    def test_should_write_the_front_frequency_and_whether_to_write_the_population(self):
        # Act
        data = yaml.safe_load(
            solve_request_to_yaml(_request(front_frequency=1000, write_population=True))
        )

        # Assert
        assert data["frontFrequency"] == 1000
        assert data["writePopulation"] is True

    def test_should_not_write_writepopulation_when_it_is_off(self):
        # Act
        data = yaml.safe_load(solve_request_to_yaml(_request(front_frequency=1000)))

        # Assert
        assert data["frontFrequency"] == 1000
        assert "writePopulation" not in data

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

    def test_should_reject_a_front_frequency_below_one_and_the_population_without_a_front(self):
        # Act
        below_one = _request(front_frequency=0).validation_errors()
        without_front = _request(write_population=True).validation_errors()

        # Assert
        assert len(below_one) == 1
        assert len(without_front) == 1
        assert "live front" in without_front[0]

    def test_should_report_every_problem_found(self):
        # Act
        errors = _request(
            problem="",
            population_size=0,
            max_evaluations=0,
            number_of_independent_runs=0,
            configuration=" ",
            status_frequency=-1,
            front_frequency=-1,
        ).validation_errors()

        # Assert
        assert len(errors) == 7


class TestProblemArguments:
    def test_should_name_the_problem_with_its_arguments_when_it_has_some(self):
        # Act
        data = yaml.safe_load(
            solve_request_to_yaml(_request(problem="DTLZ2", problem_arguments=(12, 2)))
        )

        # Assert
        assert data["problem"] == {"class": "DTLZ2", "args": [12, 2]}

    def test_should_name_the_problem_alone_without_arguments(self):
        # Act
        data = yaml.safe_load(solve_request_to_yaml(_request(problem_arguments=())))

        # Assert
        assert data["problem"] == "ZDT1"


class TestFrontDelay:
    def test_should_write_the_pause_after_each_front_when_set(self):
        # Act
        data = yaml.safe_load(
            solve_request_to_yaml(_request(front_frequency=1000, front_delay_millis=300))
        )

        # Assert
        assert data["frontDelayMillis"] == 300

    def test_should_leave_out_the_pause_when_unset(self):
        # Act
        data = yaml.safe_load(solve_request_to_yaml(_request(front_frequency=1000)))

        # Assert
        assert "frontDelayMillis" not in data

    def test_should_need_the_live_front_to_pause_after_it(self):
        # Act
        errors = _request(front_frequency=None, front_delay_millis=300).validation_errors()

        # Assert
        assert errors == ["Pausing after each front needs the live front."]
