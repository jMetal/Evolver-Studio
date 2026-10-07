"""Tests for running a validation study end to end, with the real Evolver jar."""

import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

from evolver_studio.evolver_client import WORKING_DIRECTORY, jar_path, read_status
from evolver_studio.resource_files import default_configuration_text
from evolver_studio.runs import RunPhase, run_phase
from evolver_studio.validation import (
    StudyProblem,
    ValidationStudy,
    collect_runs,
    default_contenders,
    write_study,
)
from evolver_studio.validation_runner import cancel_study, start_study
from evolver_studio.validation_stats import compare_with_pivot

pytestmark = pytest.mark.skipif(not jar_path().is_file(), reason="Evolver jar not found")


def _study() -> ValidationStudy:
    contenders = tuple(
        c
        for c in default_contenders("Double", lambda f: default_configuration_text(jar_path(), f))
        if c.name in ("NSGA-II", "MOEA/D", "RVEA")
    )
    return ValidationStudy(
        encoding="Double",
        problems=(StudyProblem("ZDT1", (), "resources/referenceFronts/ZDT1.csv"),),
        contenders=contenders,
        pivot="NSGA-II",
        population_size=100,
        max_evaluations=2000,
        runs=4,
    )


def _wait(study_directory: Path, seconds: float = 120) -> None:
    deadline = time.monotonic() + seconds
    while run_phase(study_directory) in (RunPhase.STARTING, RunPhase.RUNNING):
        assert time.monotonic() < deadline, "the study did not finish"
        time.sleep(0.3)


@pytest.fixture
def study_directory() -> Path:
    directory = WORKING_DIRECTORY / "validation-runs" / "test-study"
    yield directory
    shutil.rmtree(directory, ignore_errors=True)
    try:
        directory.parent.rmdir()  # the directory of the studies, when this was the only one
    except OSError:
        pass


class TestRunAStudy:
    def test_should_run_every_job_and_compare_the_contenders(self, study_directory: Path):
        # Arrange
        study = _study()
        jobs = write_study(study, study_directory, WORKING_DIRECTORY)

        # Act
        start_study(study_directory, jar_path(), 3, WORKING_DIRECTORY)
        _wait(study_directory)

        # Assert
        assert run_phase(study_directory) is RunPhase.FINISHED
        status = read_status(study_directory / "status.yaml")
        assert (status.evaluations_done, status.max_evaluations) == (len(jobs), len(jobs))
        runs = collect_runs(study_directory, WORKING_DIRECTORY)
        assert len(runs) == len(jobs) * study.runs
        assert sorted(set(runs["contender"])) == ["MOEA/D", "NSGA-II", "RVEA"]
        # the same seeds for every contender, so that the comparison is paired
        assert len({tuple(seeds) for _, seeds in runs.groupby("contender")["Seed"]}) == 1
        comparison = compare_with_pivot(runs, "EP", "NSGA-II")
        assert set(comparison["contender"]) == {"MOEA/D", "RVEA"}
        assert comparison["p_value"].between(0, 1).all()

    def test_should_stop_the_study_and_its_jvms_when_cancelled(self, study_directory: Path):
        # Arrange: a study long enough to be cancelled
        study = replace(_study(), max_evaluations=50_000_000, runs=2)
        write_study(study, study_directory, WORKING_DIRECTORY)
        process = start_study(study_directory, jar_path(), 2, WORKING_DIRECTORY)
        time.sleep(3)
        assert process.poll() is None

        # Act
        cancel_study(process.pid)
        process.wait(timeout=30)

        # Assert
        assert process.returncode != 0
