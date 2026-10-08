"""Tests for listing the finished training runs."""

from pathlib import Path

import yaml

from evolver_studio.training_runs import list_finished_trainings, read_outcome

VAR_CONF = "# Evaluation: 100\nEP=1 NHV=2 | --crossover SBX\nEP=3 NHV=4 | --crossover PCX\n"


def _training(
    working_directory: Path, run_id: str, status: str = "FINISHED", var_conf: str | None = VAR_CONF
) -> Path:
    run_dir = working_directory / "cli-runner-runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.yaml").write_text(
        yaml.safe_dump(
            {"status": status, "evaluationsDone": 1, "maxEvaluations": 1, "updatedAt": "x"}
        )
    )
    output = f"results/{run_id}"
    (run_dir / "request.yaml").write_text(yaml.safe_dump({"outputDirectory": output}))
    (run_dir / "base_level.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithmName": "NSGA-II",
                "encoding": "Double",
                "trainingProblemNames": ["ZDT4", {"class": "DTLZ2", "args": [12, 2]}],
            }
        )
    )
    if var_conf is not None:
        (working_directory / output).mkdir(parents=True)
        (working_directory / output / "VAR_CONF.txt").write_text(var_conf)
    return run_dir


class TestListFinishedTrainings:
    def test_should_list_a_finished_run_with_its_configurations(self, tmp_path: Path):
        # Arrange
        _training(tmp_path, "20261007-120000")

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert (training.algorithm, training.encoding) == ("NSGA-II", "Double")
        assert training.problems == ("ZDT4", "DTLZ2")
        assert [c.configuration for c in training.configurations] == [
            "--crossover SBX",
            "--crossover PCX",
        ]
        assert training.label.startswith("20261007-120000 · NSGA-II (Double) · ZDT4, DTLZ2")

    def test_should_list_the_most_recent_first(self, tmp_path: Path):
        # Arrange
        _training(tmp_path, "20261007-110000")
        _training(tmp_path, "20261007-120000")

        # Act
        runs = list_finished_trainings(tmp_path)

        # Assert
        assert [run.run_id for run in runs] == ["20261007-120000", "20261007-110000"]

    def test_should_leave_out_what_has_not_finished_or_has_no_configurations(self, tmp_path: Path):
        # Arrange
        _training(tmp_path, "a-running", status="RUNNING")
        _training(tmp_path, "b-failed", status="FAILED")
        _training(tmp_path, "c-no-file", var_conf=None)
        _training(tmp_path, "d-empty", var_conf="# nothing\n")
        cancelled = _training(tmp_path, "e-cancelled")
        (cancelled / "CANCELLED").write_text("")

        # Act / Assert
        assert list_finished_trainings(tmp_path) == []

    def test_should_list_nothing_without_a_runs_directory(self, tmp_path: Path):
        # Act / Assert
        assert list_finished_trainings(tmp_path) == []


class TestSeveralCheckpoints:
    def test_should_offer_only_the_final_front_of_a_training_with_several_checkpoints(
        self, tmp_path: Path
    ):
        # Arrange: Evolver appends a block per checkpoint to VAR_CONF.txt
        text = (
            "# Evaluation: 100\n# Time (min): 0.5\nEP=9 NHV=1 | --crossover SBX\n\n"
            "# Evaluation: 200\n# Time (min): 1.0\nEP=5 NHV=1 | --crossover PCX\n"
            "EP=4 NHV=2 | --crossover blxAlpha\n\n"
        )
        _training(tmp_path, "20261007-120000", var_conf=text)

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert [c.configuration for c in training.configurations] == [
            "--crossover PCX",
            "--crossover blxAlpha",
        ]


METADATA = """=== Meta-Optimization Experiment ===

--- Execution ---
Wall-clock time: 1h 2m 3s (3723456 ms)
Meta-evaluations performed: 2000
"""


class TestWhatATrainingRecords:
    def _complete(self, working_directory: Path, population: bool = False) -> None:
        run_dir = _training(working_directory, "20261007-120000")
        base_level = yaml.safe_load((run_dir / "base_level.yaml").read_text())
        base_level.update(
            {
                "populationSize": 100,
                "trainingReferenceFrontFileNames": [
                    "resources/referenceFronts/ZDT4.csv",
                    "resources/referenceFronts/DTLZ2.2D.csv",
                ],
                "trainingEvaluations": [12000, 8000],
                "indicatorNames": ["Epsilon", "NormalizedHypervolume"],
                "extraConfig": {"weightVectorFilesDirectory": "resources/weightVectors"},
                "yamlParameterSpaceFile": "/runs/base_parameter_space.yaml",
            }
        )
        (run_dir / "base_level.yaml").write_text(yaml.safe_dump(base_level))
        (run_dir / "meta_search.yaml").write_text(
            yaml.safe_dump(
                {
                    "algorithm": "AGE-MOEA",
                    "encoding": "tree",
                    "metaMaxComputingTimeMinutes": 90.0,
                    "metaPopulationSize": 80,
                }
            )
        )
        output = working_directory / "results" / "20261007-120000"
        output.mkdir(parents=True, exist_ok=True)
        (output / "METADATA.txt").write_text(METADATA)
        if population:
            (output / "POPULATION_INDICATORS.csv").write_text("Evaluation,SolutionId,EP\n")

    def test_should_say_what_was_tuned_and_with_what_budget(self, tmp_path: Path):
        # Arrange
        self._complete(tmp_path)

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert training.population_size == 100
        assert training.reference_fronts[1] == "resources/referenceFronts/DTLZ2.2D.csv"
        assert training.evaluations == (12000, 8000)
        assert training.indicators == ("Epsilon", "NormalizedHypervolume")
        assert training.extra_config == {"weightVectorFilesDirectory": "resources/weightVectors"}
        assert training.problem_specs[1] == {"class": "DTLZ2", "args": [12, 2]}
        assert training.parameter_space_file == Path("/runs/base_parameter_space.yaml")

    def test_should_say_how_the_meta_optimizer_searched(self, tmp_path: Path):
        # Arrange
        self._complete(tmp_path)

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert (training.meta_algorithm, training.meta_encoding) == ("AGE-MOEA", "tree")
        assert training.meta_population_size == 80
        assert training.meta_limit == "90 min"

    def test_should_say_the_limit_in_evaluations_otherwise(self, tmp_path: Path):
        # Arrange
        self._complete(tmp_path)
        run_dir = tmp_path / "cli-runner-runs" / "20261007-120000"
        (run_dir / "meta_search.yaml").write_text("algorithm: NSGA-II\nmetaMaxEvaluations: 2000\n")

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert training.meta_limit == "2,000 evaluations"
        assert training.meta_population_size is None

    def test_should_read_how_long_it_took_and_whether_it_wrote_its_population(self, tmp_path: Path):
        # Arrange
        self._complete(tmp_path, population=True)

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert training.outcome is not None
        assert training.outcome.wall_clock_seconds == 3723
        assert training.outcome.meta_evaluations == 2000
        assert training.has_population

    def test_should_still_list_a_training_whose_extra_files_are_missing(self, tmp_path: Path):
        # Arrange: only what the earlier versions of the app kept
        _training(tmp_path, "20261007-120000")

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert training.meta_algorithm is None and training.outcome is None
        assert not training.has_population and training.reference_fronts == ()


class TestReadOutcome:
    def test_should_read_it_from_metadata(self, tmp_path: Path):
        # Arrange
        (tmp_path / "METADATA.txt").write_text(METADATA)

        # Act / Assert
        outcome = read_outcome(tmp_path / "METADATA.txt")
        assert outcome is not None and outcome.wall_clock_seconds == 3723

    def test_should_know_nothing_without_the_file_or_the_execution_section(self, tmp_path: Path):
        # Arrange
        (tmp_path / "METADATA.txt").write_text("=== Meta-Optimization Experiment ===\n")

        # Act / Assert
        assert read_outcome(tmp_path / "METADATA.txt") is None
        assert read_outcome(tmp_path / "missing.txt") is None
