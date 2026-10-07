"""Tests for planning, writing and reading a validation study."""

from pathlib import Path

import pytest
import yaml

from evolver_studio.validation import (
    Contender,
    StudyProblem,
    ValidationStudy,
    collect_runs,
    default_contenders,
    list_studies,
    plan_jobs,
    read_manifest,
    study_is_written,
    write_study,
)

NSGAII_DEFAULT = "--algorithmResult population --crossover SBX"


@pytest.fixture
def working_directory(tmp_path: Path) -> Path:
    """A working directory with a two-objective front and weight vectors for 100 and 300."""
    fronts = tmp_path / "resources" / "referenceFronts"
    fronts.mkdir(parents=True)
    (fronts / "ZDT1.csv").write_text("0,1\n1,0\n")
    (fronts / "DTLZ2.3D.csv").write_text("0,0,1\n")
    weights = tmp_path / "resources" / "weightVectors"
    weights.mkdir()
    for name in ("W2D_100.dat", "W2D_300.dat", "W3D_91.dat"):
        (weights / name).write_text("")
    return tmp_path


def _study(**changes) -> ValidationStudy:
    study = ValidationStudy(
        encoding="Double",
        problems=(StudyProblem("ZDT1", (), "resources/referenceFronts/ZDT1.csv"),),
        contenders=(
            Contender("NSGA-II (tuned)", "NSGA-II", NSGAII_DEFAULT + " --tuned 1"),
            Contender("NSGA-II", "NSGA-II", NSGAII_DEFAULT),
            Contender("MOEA/D", "MOEAD", "--algorithmResult population"),
        ),
        pivot="NSGA-II (tuned)",
        runs=5,
        max_evaluations=1000,
    )
    from dataclasses import replace

    return replace(study, **changes)


class TestPlanJobs:
    def test_should_plan_a_job_per_contender_and_problem(
        self, working_directory: Path, tmp_path: Path
    ):
        # Act
        jobs, skipped = plan_jobs(_study(), tmp_path / "study", working_directory)

        # Assert
        assert [job.contender.name for job in jobs] == ["NSGA-II (tuned)", "NSGA-II", "MOEA/D"]
        assert skipped == []
        assert [job.number for job in jobs] == [1, 2, 3]

    def test_should_give_every_job_the_same_budget_and_seeds(
        self, working_directory: Path, tmp_path: Path
    ):
        # Act
        jobs, _ = plan_jobs(_study(population_size=100), tmp_path / "s", working_directory)

        # Assert
        assert {(j.request.population_size, j.request.seed) for j in jobs} == {(100, 1)}
        assert {j.request.number_of_independent_runs for j in jobs} == {5}
        assert {j.request.max_evaluations for j in jobs} == {1000}

    def test_should_ask_for_the_weight_vectors_only_of_the_algorithms_that_read_them(
        self, working_directory: Path, tmp_path: Path
    ):
        # Act
        jobs, _ = plan_jobs(_study(), tmp_path / "s", working_directory)
        extra = {job.contender.name: job.request.extra_config for job in jobs}

        # Assert
        assert extra["NSGA-II"] is None
        assert extra["MOEA/D"] == {"weightVectorFilesDirectory": "resources/weightVectors"}

    def test_should_use_the_parameter_space_of_the_study_encoding(
        self, working_directory: Path, tmp_path: Path
    ):
        # Act
        jobs, _ = plan_jobs(_study(), tmp_path / "s", working_directory)

        # Assert
        assert jobs[0].request.yaml_parameter_space_file == "NSGAIIDouble.yaml"
        assert jobs[0].request.encoding == "Double"

    def test_should_leave_out_an_algorithm_with_no_weight_vector_file_for_the_problem(
        self, working_directory: Path, tmp_path: Path
    ):
        # Arrange: a population of 100 has no file for three objectives
        problems = (StudyProblem("DTLZ2", (), "resources/referenceFronts/DTLZ2.3D.csv"),)

        # Act
        jobs, skipped = plan_jobs(
            _study(problems=problems, population_size=100), tmp_path / "s", working_directory
        )

        # Assert
        assert [job.contender.name for job in jobs] == ["NSGA-II (tuned)", "NSGA-II"]
        assert len(skipped) == 1
        assert skipped[0].contender == "MOEA/D"
        assert "no weight vector file for population size 100 and 3 objectives" in skipped[0].reason

    def test_should_pass_the_arguments_of_a_problem_on(
        self, working_directory: Path, tmp_path: Path
    ):
        # Arrange
        problems = (StudyProblem("DTLZ2", (12, 3), "resources/referenceFronts/DTLZ2.3D.csv"),)

        # Act
        jobs, _ = plan_jobs(_study(problems=problems), tmp_path / "s", working_directory)

        # Assert
        assert jobs[0].request.problem_arguments == (12, 3)


class TestWriteAndCollect:
    def test_should_write_a_request_per_job_and_a_manifest(self, working_directory: Path):
        # Arrange
        directory = working_directory / "validation-runs" / "20260101-000000"

        # Act
        jobs = write_study(_study(), directory, working_directory)

        # Assert
        request = yaml.safe_load((directory / "jobs" / "001" / "request.yaml").read_text())
        assert request["configuration"] == NSGAII_DEFAULT + " --tuned 1"
        assert request["outputDirectory"] == "validation-runs/20260101-000000/jobs/001/output"
        manifest = read_manifest(directory)
        assert manifest is not None
        assert manifest["pivot"] == "NSGA-II (tuned)"
        assert [job["contender"] for job in manifest["jobs"]] == [j.contender.name for j in jobs]

    def test_should_gather_the_runs_of_the_jobs_that_have_results(self, working_directory: Path):
        # Arrange: only the first job has finished
        directory = working_directory / "validation-runs" / "x"
        write_study(_study(), directory, working_directory)
        output = directory / "jobs" / "001" / "output"
        output.mkdir()
        (output / "INDICATORS.csv").write_text("Run,Seed,TimeMs,EP\n1,1,10,0.5\n2,2,11,0.6\n")

        # Act
        runs = collect_runs(directory, working_directory)

        # Assert
        assert list(runs.columns) == ["contender", "problem", "Run", "Seed", "TimeMs", "EP"]
        assert set(runs["contender"]) == {"NSGA-II (tuned)"}
        assert set(runs["problem"]) == {"ZDT1"}
        assert len(runs) == 2

    def test_should_gather_nothing_before_any_job_has_finished(self, working_directory: Path):
        # Arrange
        directory = working_directory / "validation-runs" / "x"
        write_study(_study(), directory, working_directory)

        # Act / Assert
        assert collect_runs(directory, working_directory).empty

    def test_should_not_read_a_directory_without_manifest(self, tmp_path: Path):
        # Act / Assert
        assert read_manifest(tmp_path) is None
        assert collect_runs(tmp_path, tmp_path).empty


class TestValidationErrors:
    def test_should_accept_a_complete_study(self):
        # Act / Assert
        assert _study().validation_errors() == []

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"problems": ()}, "Choose at least one problem."),
            ({"pivot": "nobody"}, "The pivot must be one of the contenders."),
            ({"indicators": ()}, "Choose at least one quality indicator."),
            ({"runs": 1}, "At least two runs are needed to compare contenders."),
            ({"population_size": 1}, "The population size must be at least 2."),
        ],
    )
    def test_should_say_what_is_missing(self, changes: dict, message: str):
        # Act / Assert
        assert message in _study(**changes).validation_errors()

    def test_should_need_two_contenders_with_distinct_names(self):
        # Arrange
        same = Contender("A", "NSGA-II", "--x 1")

        # Act
        errors = _study(contenders=(same, same), pivot="A").validation_errors()

        # Assert
        assert errors == ["Every contender needs its own name."]
        assert (
            "A study compares at least two contenders."
            in _study(contenders=(same,), pivot="A").validation_errors()
        )

    def test_should_need_a_reference_front_for_every_problem(self):
        # Act
        errors = _study(problems=(StudyProblem("ZDT1", (), ""),)).validation_errors()

        # Assert
        assert errors == ["Every problem needs a reference front: the indicators use it."]


class TestDefaultContenders:
    @staticmethod
    def _read(file_name: str) -> str:
        return f"--from {file_name}\nsecond line"

    def test_should_offer_every_algorithm_with_a_default_for_the_encoding(self):
        # Act
        names = [c.name for c in default_contenders("Double", self._read)]

        # Assert: RDE-MOEA has no default configuration, and RVEA has three variants
        assert names[:2] == ["NSGA-II", "MOEA/D"]
        assert {"RVEA", "RVEA*", "iRVEA", "PAES", "NSGA-III", "SSMOEA"} <= set(names)
        assert "RDE-MOEA" not in names

    def test_should_offer_only_nsgaii_and_paes_for_binary_and_permutation(self):
        # Act / Assert
        for encoding in ("Binary", "Permutation"):
            assert [c.name for c in default_contenders(encoding, self._read)] == [
                "NSGA-II",
                "PAES",
            ]

    def test_should_take_the_first_line_of_the_configuration_file(self):
        # Act
        nsgaii = default_contenders("Double", self._read)[0]

        # Assert
        assert nsgaii.configuration == "--from NSGAIIDoubleDefault.txt"
        assert nsgaii.algorithm == "NSGA-II"


class TestListStudies:
    def test_should_list_the_studies_with_a_manifest_most_recent_first(
        self, working_directory: Path
    ):
        # Arrange
        runs = working_directory / "validation-runs"
        write_study(_study(), runs / "20260101-000000", working_directory)
        write_study(_study(), runs / "20260102-000000", working_directory)
        (runs / "not-a-study").mkdir()

        # Act
        studies = list_studies(runs)

        # Assert
        assert [s.study_id for s in studies] == ["20260102-000000", "20260101-000000"]
        assert studies[0].label == "20260102-000000 · NSGA-II (tuned) · 1 problems"

    def test_should_list_nothing_without_a_directory(self, tmp_path: Path):
        # Act / Assert
        assert list_studies(tmp_path / "nothing") == []


class TestStudyIsWritten:
    def test_should_recognize_the_study_that_was_written_there(self, working_directory: Path):
        # Arrange
        directory = working_directory / "validation-runs" / "x"
        write_study(_study(), directory, working_directory)

        # Act / Assert
        assert study_is_written(_study(), directory, working_directory)

    @pytest.mark.parametrize(
        "changes", [{"runs": 6}, {"max_evaluations": 2000}, {"seed": 2}, {"population_size": 91}]
    )
    def test_should_not_recognize_a_study_planned_differently(
        self, working_directory: Path, changes: dict
    ):
        # Arrange
        directory = working_directory / "validation-runs" / "x"
        write_study(_study(), directory, working_directory)

        # Act / Assert
        assert not study_is_written(_study(**changes), directory, working_directory)

    def test_should_not_recognize_a_directory_without_a_study(self, working_directory: Path):
        # Act / Assert
        assert not study_is_written(_study(), working_directory / "nothing", working_directory)
