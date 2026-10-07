"""Tests for the Run algorithm page."""

import functools
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from evolver_studio import catalogue, evolver_client
from evolver_studio.catalogue import is_at_least
from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path, write_pid_file
from evolver_studio.problem_catalogue import parse_problem_catalogue
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
# A problem of each encoding, for a jar that describes the problems (the encoding then follows it).
PROBLEM_OF_ENCODING = {"Double": "ZDT1", "Binary": "ZDT5", "Permutation": "KroAB100TSP"}


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


@functools.cache
def _jar_describes_problems() -> bool:
    """Whether the Evolver jar in use has a problem catalogue (newer than 2.3)."""
    result = describe(WORKING_DIRECTORY, jar_path())
    return parse_problem_catalogue(getattr(result, "value", {})) is not None


def _choose_encoding(app: AppTest, algorithm: str, encoding: str) -> AppTest:
    """Choose an algorithm for a problem of an encoding: chosen with the problem when the jar
    describes the problems, or in its own selector otherwise."""
    if _jar_describes_problems():
        app.selectbox(key="solve_problem").select(PROBLEM_OF_ENCODING[encoding]).run()
        return app.selectbox(key="solve_algorithm").select(algorithm).run()
    app.selectbox(key="solve_problem").select("ZDT1").run()
    app.selectbox(key="solve_algorithm").select(algorithm).run()
    return app.selectbox(key=f"solve_encoding_{algorithm}").select(encoding).run()


def _needs_problem_catalogue() -> None:
    if not _jar_describes_problems():
        pytest.skip("The Evolver jar in use does not describe the problems")


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

    def test_should_pick_among_several_fronts_only_when_the_objectives_are_known(
        self, app: AppTest
    ):
        """DTLZ2 has fronts for 2 to 8 objectives; the catalogue says it has 3 by default."""
        # Act
        app.selectbox(key="solve_problem").select("DTLZ2").run()

        # Assert
        if _jar_describes_problems():
            assert (
                app.selectbox(key="solve_reference_front_DTLZ2").value
                == "resources/referenceFronts/DTLZ2.3D.csv"
            )
        else:
            assert app.selectbox(key="solve_reference_front_DTLZ2").value is None
            assert [subheader.value for subheader in app.subheader] == ["1. Problem"]


class TestProblemCatalogue:
    def test_should_offer_only_the_algorithms_that_solve_the_problems_encoding(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.selectbox(key="solve_problem").select("ZDT5").run()

        # Assert: RVEA and NSGA-III solve real-valued problems only, and no encoding is asked
        options = list(app.selectbox(key="solve_algorithm").options)
        assert "NSGA-II" in options and "PAES" in options
        assert "RVEA" not in options and "NSGA-III" not in options
        app.selectbox(key="solve_algorithm").select("NSGA-II").run()
        assert not any(box.key == "solve_encoding_NSGA-II" for box in app.selectbox)
        assert any("NSGAIIBinaryDefault.txt" in caption.value for caption in app.caption)

    def test_should_warn_when_the_front_does_not_have_the_objectives_set(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()
        app.selectbox(key="solve_problem").select("DTLZ2").run()

        # Act
        app.checkbox(key="solve_problem_arguments_DTLZ2").check().run()
        app.number_input(key="solve_problem_argument_DTLZ2_numberOfObjectives").set_value(2).run()

        # Assert
        assert any("This front has 3 objectives" in warning.value for warning in app.warning)

    def test_should_ask_for_every_argument_without_default(self, app: AppTest):
        """LZ09's arguments have no default: Evolver takes all of them or none."""
        # Arrange
        _needs_problem_catalogue()
        app.selectbox(key="solve_problem").select("LZ09F1").run()

        # Act
        app.checkbox(key="solve_problem_arguments_LZ09F1").check().run()

        # Assert
        assert any("Give every argument a value" in warning.value for warning in app.warning)
        assert not any(subheader.value == "2. Algorithm" for subheader in app.subheader)


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

    @pytest.mark.parametrize(
        ("encoding", "file"),
        [("Binary", "NSGAIIBinaryDefault.txt"), ("Permutation", "NSGAIIPermutationDefault.txt")],
    )
    def test_should_start_every_encoding_of_nsgaii_from_its_default(
        self, app: AppTest, encoding: str, file: str
    ):
        # Act
        _choose_encoding(app, "NSGA-II", encoding)

        # Assert
        assert not app.exception
        assert any(f"Starting from `{file}`" in caption.value for caption in app.caption)

    def test_should_start_an_algorithm_without_default_from_the_parameter_space(
        self, app: AppTest, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange: Evolver 2.3 ships a default for every runnable encoding, so the case is built
        # from a catalogue whose NSGA-II has none (as with a jar that does not ship them)
        monkeypatch.setattr(
            catalogue,
            "BASE_ALGORITHMS",
            tuple(
                replace(algorithm, default_configurations={})
                if algorithm.name == "NSGA-II"
                else algorithm
                for algorithm in catalogue.BASE_ALGORITHMS
            ),
        )

        # Act
        _choose_encoding(app, "NSGA-II", "Permutation")

        # Assert
        assert not app.exception
        assert any("no default configuration" in caption.value for caption in app.caption)


class TestPopulationSize:
    def test_should_offer_only_the_sizes_with_a_weight_vector_file_for_moead(self, app: AppTest):
        # Act: ZDT1 has two objectives
        app.selectbox(key="solve_problem").select("ZDT1").run()
        app.selectbox(key="solve_algorithm").select("MOEA/D").run()

        # Assert
        sizes = app.selectbox(key="solve_population_choice").options
        assert list(sizes)[:3] == ["100", "300", "400"]
        assert not any(box.label == "Population size" for box in app.number_input)

    def test_should_follow_the_objectives_of_the_problem(self, app: AppTest):
        # Act: DTLZ2 with its three-objective front
        app.selectbox(key="solve_problem").select("DTLZ2").run()
        app.selectbox(key="solve_reference_front_DTLZ2").select(
            "resources/referenceFronts/DTLZ2.3D.csv"
        ).run()
        app.selectbox(key="solve_algorithm").select("RVEA").run()

        # Assert
        assert "91" in app.selectbox(key="solve_population_choice").options

    def test_should_keep_a_free_number_for_an_algorithm_without_weight_vectors(self, app: AppTest):
        # Act
        _choose_zdt1_and_nsgaii(app)

        # Assert
        assert app.number_input(key="solve_population").value == 100
        assert len(app.selectbox(key="solve_algorithm").options) > 0
        assert not any(box.key == "solve_population_choice" for box in app.selectbox)

    def test_should_warn_when_the_directory_has_no_file_for_the_objectives(self, app: AppTest):
        # Arrange
        app.selectbox(key="solve_problem").select("ZDT1").run()
        app.selectbox(key="solve_algorithm").select("MOEA/D").run()

        # Act
        app.text_input(key="solve_weight_vectors").set_value("resources/nowhere").run()

        # Assert
        assert app.number_input(key="solve_population").value == 100
        assert any("No weight vector file" in warning.value for warning in app.warning)


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

            # Assert: the blinking label, with the evaluation the run is at
            assert any(
                "Running · evaluation 10 of 100" in markdown.value for markdown in app.markdown
            )
            assert app.selectbox(key="solve_problem").value == "ZDT1"
            assert next(button for button in app.button if button.label == "Run").disabled
        finally:
            process.kill()
            process.wait()

    def test_should_say_only_running_until_the_run_reports_its_evaluations(
        self, app: AppTest, tmp_path: Path
    ):
        # Arrange: a run that has not written its status yet
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            run_dir = tmp_path / "solve-runs" / "20260101-000000"
            run_dir.mkdir(parents=True)
            (run_dir / "request.yaml").write_text("numberOfIndependentRuns: 2\n")
            write_pid_file(run_dir / "pid.txt", process.pid)

            # Act
            app.run()

            # Assert
            assert not app.exception
            assert any("⏳ Running…" in markdown.value for markdown in app.markdown)
            assert any(button.label == "Cancel" for button in app.button)
        finally:
            process.kill()
            process.wait()


class TestSolutions:
    def test_should_list_the_solutions_of_a_run_and_filter_them(self, app: AppTest, tmp_path: Path):
        # Arrange: a finished run with a front in a known place
        _choose_zdt1_and_nsgaii(app)
        app.number_input(key="solve_evaluations").set_value(500).run()
        next(button for button in app.button if button.label == "Run").click().run()
        run_dir = next((tmp_path / "solve-runs").iterdir())
        deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
        while run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the run did not finish"
            time.sleep(0.2)
        app.run()
        app.selectbox(key="solve_history").select(run_dir.name).run()
        table = next(frame for frame in app.dataframe if "x1" in frame.value.columns)
        total = len(table.value)

        # Act: restrict the first objective to its lower half
        slider = app.slider(key=f"solve_filter_{run_dir.name}_1_f1")
        lower, upper = slider.value
        slider.set_value((lower, (lower + upper) / 2)).run()

        # Assert
        shown = next(frame for frame in app.dataframe if "x1" in frame.value.columns)
        assert 0 < len(shown.value) < total
        assert shown.value["f1"].max() <= (lower + upper) / 2
        assert any(f"{len(shown.value)} of {total} solutions" in c.value for c in app.caption)
        assert len(app.get("download_button")) == 3  # the table, the request and the zip


class TestRepeatARun:
    def test_should_fill_the_form_with_a_past_run_and_offer_its_request(
        self, app: AppTest, tmp_path: Path
    ):
        # Arrange: a finished run of NSGA-II with crossover blxAlpha, 2 runs of 500 evaluations
        _choose_zdt1_and_nsgaii(app)
        app.selectbox(key="solve_configuration_NSGA-II_Double_Default_0_crossover").select(
            "blxAlpha"
        ).run()
        app.number_input(key="solve_evaluations").set_value(500)
        app.number_input(key="solve_runs").set_value(2).run()
        next(button for button in app.button if button.label == "Run").click().run()
        run_dir = next((tmp_path / "solve-runs").iterdir())
        deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
        while run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the run did not finish"
            time.sleep(0.2)
        app.run()
        app.selectbox(key="solve_history").select(run_dir.name).run()

        # Act: change the form, then restore the run
        app.number_input(key="solve_evaluations").set_value(900).run()
        app.button(key=f"solve_load_{run_dir.name}").click().run()

        # Assert
        assert not app.exception
        assert app.number_input(key="solve_evaluations").value == 500
        assert app.number_input(key="solve_runs").value == 2
        assert app.selectbox(key="solve_problem").value == "ZDT1"
        assert "--crossover blxAlpha" in app.code[0].value
        assert "--blxAlphaCrossoverAlpha" in app.code[0].value
        assert len(app.get("download_button")) == 3  # the table, the request and the zip

    def test_should_run_a_problem_with_arguments_and_restore_them(
        self, app: AppTest, tmp_path: Path
    ):
        # Arrange: DTLZ2 with two objectives, its two-objective front and NSGA-II
        _needs_problem_catalogue()
        app.selectbox(key="solve_problem").select("DTLZ2").run()
        app.checkbox(key="solve_problem_arguments_DTLZ2").check().run()
        app.number_input(key="solve_problem_argument_DTLZ2_numberOfObjectives").set_value(2).run()
        app.selectbox(key="solve_reference_front_DTLZ2").select(
            "resources/referenceFronts/DTLZ2.2D.csv"
        ).run()
        app.selectbox(key="solve_algorithm").select("NSGA-II").run()
        app.number_input(key="solve_evaluations").set_value(500).run()

        # Act
        next(button for button in app.button if button.label == "Run").click().run()
        run_dir = next((tmp_path / "solve-runs").iterdir())
        deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
        while run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the run did not finish"
            time.sleep(0.2)
        app.run()
        request = yaml.safe_load((run_dir / "request.yaml").read_text())

        # Assert: Evolver ran the problem it was given
        assert run_phase(run_dir) == RunPhase.FINISHED
        assert request["problem"] == {"class": "DTLZ2", "args": [12, 2]}
        assert "DTLZ2(12, 2)" in app.selectbox(key="solve_history").options[0]

        # Act: change the arguments, then restore the run
        app.selectbox(key="solve_history").select(run_dir.name).run()
        app.checkbox(key="solve_problem_arguments_DTLZ2").uncheck().run()
        app.button(key=f"solve_load_{run_dir.name}").click().run()

        # Assert
        assert not app.exception
        assert app.checkbox(key="solve_problem_arguments_DTLZ2").value is True
        assert app.number_input(key="solve_problem_argument_DTLZ2_numberOfObjectives").value == 2


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
        assert [tab.label for tab in app.tabs] == ["Front", "Indicators", "Solutions", "Details"]
        assert len(app.dataframe) == 3  # the indicators, their summary, and the solutions
        assert (run_dir / "output" / "run-2" / "FUN.csv").is_file()
        request = yaml.safe_load((run_dir / "request.yaml").read_text())
        assert ("statusFrequency" in request) is _jar_reports_progress()
