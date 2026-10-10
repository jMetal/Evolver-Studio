"""Tests for tutorial S5, "Validating a configuration", with the real study it runs."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client, tutorial_validation
from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path
from evolver_studio.problem_catalogue import parse_problem_catalogue
from evolver_studio.runs import RunPhase, run_phase
from evolver_studio.tutorial_validation import (
    COMPARED_WITH,
    PIVOT_NAME,
    RUNS,
    STUDY_ID,
    tutorial_study,
)
from evolver_studio.validation import plan_jobs
from evolver_studio.validation_form import PROBLEM_ROWS_KEY, form_state_from_study

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"
RUN_TIMEOUT_SECONDS = 180
OPEN_KEY = "tutorial_open_validating_a_configuration"


@pytest.fixture
def jar() -> Path:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    return jar_path()


def _describes_problems() -> bool:
    result = describe(WORKING_DIRECTORY, jar_path())
    return parse_problem_catalogue(getattr(result, "value", {})) is not None


@pytest.fixture
def newer_jar(jar: Path) -> Path:
    if not _describes_problems():
        pytest.skip("The Evolver jar in use does not describe the problems")
    return jar


class TestTutorialStudy:
    def test_should_compare_the_tuned_configuration_with_three_default_algorithms(
        self, newer_jar: Path
    ):
        # Act
        study = tutorial_study(newer_jar)

        # Assert
        assert [c.name for c in study.contenders] == [PIVOT_NAME, *COMPARED_WITH]
        assert study.pivot == PIVOT_NAME
        assert study.validation_errors() == []
        assert "--algorithmResult externalArchive" in study.contenders[0].configuration

    def test_should_plan_a_job_for_every_algorithm_and_problem(self, newer_jar: Path, tmp_path):
        # Arrange
        (tmp_path / "resources").symlink_to(RESOURCES)

        # Act
        jobs, skipped = plan_jobs(tutorial_study(newer_jar), tmp_path / "s", tmp_path)

        # Assert: the population of 100 has a weight vector file for two objectives
        assert len(jobs) == 16
        assert skipped == []
        assert {job.request.number_of_independent_runs for job in jobs} == {RUNS}

    def test_should_give_dtlz2_its_two_objectives(self, newer_jar: Path):
        # Act
        problems = {p.name: p for p in tutorial_study(newer_jar).problems}

        # Assert
        assert problems["DTLZ2"].arguments == (12, 2)
        assert problems["DTLZ2"].reference_front.endswith("DTLZ2.2D.csv")


class TestOpenInValidation:
    def test_should_fill_the_validation_form_with_the_study(
        self, newer_jar: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The button 'Open in Validation' relies on it."""
        # Arrange
        (tmp_path / "resources").symlink_to(RESOURCES)
        monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=60)
        for key, value in form_state_from_study(tutorial_study(newer_jar)).items():
            app.session_state[key] = value

        # Act: through the app, so that the page's links to other pages resolve
        app.switch_page("pages/validation.py").run()

        # Assert
        assert not app.exception
        assert list(app.multiselect(key="validation_problems_Double").value) == [
            "WFG2",
            "WFG4",
            "ZDT1",
            "DTLZ2",
        ]
        assert list(app.session_state[PROBLEM_ROWS_KEY]["arguments"]) == ["", "", "", "12, 2"]
        assert app.multiselect(key="validation_defaults_Double").value == list(COMPARED_WITH)
        assert any("16 jobs of 15 runs each" in c.value for c in app.caption)
        assert not next(b for b in app.button if b.label == "Run the study").disabled


class TestWalkThrough:
    """Runs the real app through S5, with its study in a temporary working directory."""

    @pytest.fixture
    def app(self, newer_jar: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
        (tmp_path / "resources").symlink_to(RESOURCES)
        monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
        app.switch_page("pages/tutorials.py").run()
        app.button(key=OPEN_KEY).click().run()
        return app

    @staticmethod
    def _go_to_step(app: AppTest, title: str) -> None:
        while app.subheader[0].value != title:
            next(button for button in app.button if button.label == "Next →").click().run()

    @staticmethod
    def _run_study(app: AppTest, tmp_path: Path) -> None:
        app.button(key="tutorial_validation_run").click().run()
        directory = tmp_path / "validation-runs" / STUDY_ID
        deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
        while run_phase(directory) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the study did not finish"
            time.sleep(0.3)
        assert run_phase(directory) is RunPhase.FINISHED
        app.run()

    def test_should_ask_to_run_the_study_in_every_step_that_reads_its_results(self, app: AppTest):
        # Act
        asked = []
        for step in tutorial_validation.STEPS[2:6]:
            self._go_to_step(app, step.title)
            asked.append(any("has not been run yet" in info.value for info in app.info))

        # Assert
        assert asked == [True, True, True, True]
        assert not app.exception

    def test_should_run_the_study_and_read_its_results_step_by_step(
        self, app: AppTest, tmp_path: Path
    ):
        # Arrange
        self._go_to_step(app, "Step 1: design and run the study")
        assert len(app.table) == 1

        # Act
        self._run_study(app, tmp_path)

        # Assert: the study is kept where Validation lists it, and the step says it is ready
        assert any("The study is ready" in s.value for s in app.success)
        assert (tmp_path / "validation-runs" / STUDY_ID / "study.yaml").is_file()

        # Step 2: the medians, with the tuned NSGA-II best on ZDT1
        self._go_to_step(app, "Step 2: medians")
        assert not app.exception
        table = next(d.value for d in app.dataframe if PIVOT_NAME in d.value.columns)
        assert list(table.index) == ["WFG2", "WFG4", "ZDT1", "DTLZ2(12, 2)"]
        app.radio(key="tutorial_validation_best_quiz_EP").set_value(PIVOT_NAME).run()
        assert any("Right" in s.value for s in app.success)

        # Step 3: twelve comparisons, and the one the test cannot settle on NHV
        self._go_to_step(app, "Step 3: is the difference real?")
        assert len(next(d.value for d in app.dataframe if "A12" in d.value.columns)) == 12
        # the step opens on the indicator that has such a case, not on the first one
        assert app.radio(key="tutorial_validation_significance_indicator").value == "NHV"
        assert any("Look at WFG4, against NSGA-II" in m.value for m in app.markdown)
        quiz = app.radio(key="tutorial_validation_doubtful_quiz_NHV")
        quiz.set_value("Not with these runs: the difference is not significant").run()
        assert any("Right" in s.value for s in app.success)

        # Step 4: what the same comparison says with 5 to 15 runs
        self._go_to_step(app, "Step 4: how many runs?")
        assert app.radio(key="tutorial_validation_runs_indicator").value == "NHV"
        runs_table = next(d.value for d in app.dataframe if "Runs" in d.value.columns)
        assert list(runs_table["Runs"]) == list(range(5, RUNS + 1))

        # Step 5: seen and unseen problems
        self._go_to_step(app, "Step 5: tuned-for and unseen problems")
        groups = next(d.value for d in app.dataframe if "Significantly better" in d.value.columns)
        assert len(groups) == 4  # two indicators, two groups of problems
        assert not app.exception

    def test_should_reuse_a_finished_study_instead_of_running_it_again(
        self, app: AppTest, tmp_path: Path
    ):
        # Arrange
        self._go_to_step(app, "Step 1: design and run the study")
        self._run_study(app, tmp_path)
        started = (tmp_path / "validation-runs" / STUDY_ID / "status.yaml").stat().st_mtime

        # Act: a new session of the tutorial
        again = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
        again.switch_page("pages/tutorials.py").run()
        again.button(key=OPEN_KEY).click().run()
        self._go_to_step(again, "Step 2: medians")

        # Assert
        assert not again.exception
        assert not any("has not been run yet" in info.value for info in again.info)
        assert (tmp_path / "validation-runs" / STUDY_ID / "status.yaml").stat().st_mtime == started


class TestWithAnOlderEvolver:
    def test_should_say_that_the_study_needs_a_newer_evolver(
        self, jar: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange
        if _describes_problems():
            pytest.skip("The Evolver jar in use describes the problems")
        (tmp_path / "resources").symlink_to(RESOURCES)
        monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
        app.switch_page("pages/tutorials.py").run()
        app.button(key=OPEN_KEY).click().run()

        # Act
        while app.subheader[0].value != "Step 1: design and run the study":
            next(button for button in app.button if button.label == "Next →").click().run()

        # Assert
        assert not app.exception
        assert any("2.4 or later" in info.value for info in app.info)
