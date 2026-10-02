"""Tests for the Run algorithm page."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client
from evolver_studio.catalogue import is_at_least
from evolver_studio.evolver_client import jar_path, write_pid_file
from evolver_studio.resource_files import jar_evolver_version
from evolver_studio.runs import RunPhase, run_phase

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "solve.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"
NSGAII_DEFAULT = (
    "--algorithmResult population --createInitialSolutions default "
    "--offspringPopulationSize 100 --variation crossoverAndMutationVariation --crossover SBX "
    "--crossoverProbability 0.9 --crossoverRepairStrategy bounds --sbxDistributionIndex 20.0 "
    "--mutation polynomial --mutationProbabilityFactor 1.0 --mutationRepairStrategy bounds "
    "--polynomialMutationDistributionIndex 20.0 --selection tournament --selectionTournamentSize 2"
)
RUN_TIMEOUT_SECONDS = 60


def _jar_reports_progress() -> bool:
    """Whether the Evolver jar in use updates the status while a run is in progress (2.3 on)."""
    version = jar_evolver_version(jar_path())
    return version is None or is_at_least(version, "2.3")


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The page, with its runs kept in a temporary working directory that shares the resources."""
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    (tmp_path / "resources").symlink_to(RESOURCES)
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    return AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()


def _choose_zdt1_and_nsgaii(app: AppTest) -> AppTest:
    app.selectbox(key="solve_problem").select("ZDT1").run()
    return app.selectbox(key="solve_algorithm").select("NSGA-II").run()


class TestOpening:
    def test_should_open_without_exceptions_and_with_only_the_problem_to_choose(self, app: AppTest):
        # Assert
        assert not app.exception
        assert app.selectbox(key="solve_problem").value is None
        assert [subheader.value for subheader in app.subheader] == ["1. Problem"]
        assert len(app.button) == 0

    def test_should_preselect_the_only_reference_front_of_the_problem(self, app: AppTest):
        # Act
        app.selectbox(key="solve_problem").select("ZDT1").run()

        # Assert
        assert (
            app.selectbox(key="solve_reference_front_ZDT1").value
            == "resources/referenceFronts/ZDT1.csv"
        )

    def test_should_ask_for_the_reference_front_when_there_are_several(self, app: AppTest):
        """DTLZ2 has fronts for 2 to 8 objectives, and which fits depends on the problem."""
        # Act
        app.selectbox(key="solve_problem").select("DTLZ2").run()

        # Assert
        assert app.selectbox(key="solve_reference_front_DTLZ2").value is None
        assert [subheader.value for subheader in app.subheader] == ["1. Problem"]


class TestConfiguration:
    def test_should_show_the_default_configuration_of_the_algorithm(self, app: AppTest):
        # Act
        _choose_zdt1_and_nsgaii(app)

        # Assert
        assert not app.exception
        assert app.code[0].value == NSGAII_DEFAULT
        assert any("0 parameter(s) changed" in caption.value for caption in app.caption)

    def test_should_offer_the_three_default_variants_of_rvea(self, app: AppTest):
        # Act
        app.selectbox(key="solve_problem").select("ZDT1").run()
        app.selectbox(key="solve_algorithm").select("RVEA").run()

        # Assert
        assert list(app.radio(key="solve_default_RVEA_Double").options) == [
            "RVEA",
            "RVEA*",
            "iRVEA",
        ]

    def test_should_adjust_a_parameter_within_the_space_and_activate_its_sub_parameters(
        self, app: AppTest
    ):
        # Arrange
        _choose_zdt1_and_nsgaii(app)
        crossover = app.selectbox(key="solve_configuration_NSGA-II_Double_Default_0_crossover")

        # Act
        crossover.select("blxAlpha").run()

        # Assert: blxAlpha's own parameter replaces SBX's, and the changes are counted
        configuration = app.code[0].value
        assert "--crossover blxAlpha" in configuration
        assert "--blxAlphaCrossoverAlpha" in configuration
        assert "--sbxDistributionIndex" not in configuration
        assert not any("0 parameter(s) changed" in caption.value for caption in app.caption)

    def test_should_start_an_algorithm_without_default_from_the_parameter_space(self, app: AppTest):
        # Arrange: NSGA-II with a permutation encoding has no default configuration
        app.selectbox(key="solve_problem").select("ZDT1").run()
        app.selectbox(key="solve_algorithm").select("NSGA-II").run()

        # Act
        app.selectbox(key="solve_encoding_NSGA-II").select("Permutation").run()

        # Assert
        assert not app.exception
        assert any("no default configuration" in caption.value for caption in app.caption)


class TestProgressControls:
    def test_should_offer_the_progress_slider_only_when_the_jar_can_report_it(self, app: AppTest):
        # Act
        _choose_zdt1_and_nsgaii(app)

        # Assert
        checkbox = app.checkbox(key="solve_show_progress")
        assert checkbox.value is _jar_reports_progress()
        assert len(app.select_slider) == (1 if _jar_reports_progress() else 0)

    def test_should_say_why_there_is_no_progress_slider_with_an_older_jar(self, app: AppTest):
        # Arrange
        if _jar_reports_progress():
            pytest.skip("the Evolver jar in use reports the progress while running")

        # Act
        _choose_zdt1_and_nsgaii(app)

        # Assert
        assert app.checkbox(key="solve_show_progress").disabled
        assert any("only when each run ends" in caption.value for caption in app.caption)

    def test_should_warn_that_updating_after_every_evaluation_slows_the_run(self, app: AppTest):
        # Arrange
        if not _jar_reports_progress():
            pytest.skip("the Evolver jar in use cannot report the progress while running")
        _choose_zdt1_and_nsgaii(app)

        # Act
        app.select_slider(key="solve_status_frequency").set_value(1).run()

        # Assert
        assert any("slow the run down" in warning.value for warning in app.warning)

    def test_should_not_warn_at_the_default_frequency(self, app: AppTest):
        # Arrange
        if not _jar_reports_progress():
            pytest.skip("the Evolver jar in use cannot report the progress while running")

        # Act
        _choose_zdt1_and_nsgaii(app)

        # Assert
        assert not any("slow the run down" in warning.value for warning in app.warning)

    def test_should_go_silent_when_the_progress_is_switched_off(self, app: AppTest):
        # Arrange
        if not _jar_reports_progress():
            pytest.skip("the Evolver jar in use cannot report the progress while running")
        _choose_zdt1_and_nsgaii(app)

        # Act
        app.checkbox(key="solve_show_progress").uncheck().run()

        # Assert
        assert len(app.select_slider) == 0
        assert any("Silent" in caption.value for caption in app.caption)


class TestRunInProgress:
    def test_should_show_the_progress_and_keep_the_form_with_the_run_button_disabled(
        self, app: AppTest, tmp_path: Path
    ):
        # Arrange: a run that is still running, as a live process with a RUNNING status
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            run_dir = tmp_path / "solve-runs" / "20260101-000000"
            run_dir.mkdir(parents=True)
            (run_dir / "status.yaml").write_text(
                "status: RUNNING\nevaluationsDone: 10\nmaxEvaluations: 100\nupdatedAt: x\n"
            )
            write_pid_file(run_dir / "pid.txt", process.pid)

            # Act
            _choose_zdt1_and_nsgaii(app)

            # Assert
            assert any("Running (20260101-000000)" in info.value for info in app.info)
            assert app.selectbox(key="solve_problem").value == "ZDT1"
            assert next(button for button in app.button if button.label == "Run").disabled
        finally:
            process.kill()
            process.wait()


class TestRun:
    def test_should_run_the_algorithm_and_show_its_results(self, app: AppTest, tmp_path: Path):
        # Arrange
        _choose_zdt1_and_nsgaii(app)
        app.number_input(key="solve_evaluations").set_value(500)
        app.number_input(key="solve_runs").set_value(2)

        # Act
        next(button for button in app.button if button.label == "Run").click().run()
        run_dir = next((tmp_path / "solve-runs").iterdir())
        deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
        while run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the run did not finish"
            time.sleep(0.2)
        app.run()
        app.selectbox(key="solve_history").select(run_dir.name).run()

        # Assert
        assert not app.exception
        assert app.selectbox(key="solve_problem").value == "ZDT1"  # the form kept its choices
        assert run_phase(run_dir) is RunPhase.FINISHED
        assert [tab.label for tab in app.tabs] == ["Front", "Indicators", "Details"]
        assert len(app.dataframe) == 2  # the runs' indicators, and their summary
        assert (run_dir / "output" / "run-2" / "FUN.csv").is_file()
        request = yaml.safe_load((run_dir / "request.yaml").read_text())
        assert ("statusFrequency" in request) is _jar_reports_progress()
