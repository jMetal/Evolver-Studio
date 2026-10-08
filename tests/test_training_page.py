"""Tests for the Training page's checks of the training set's problems."""

import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client
from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path, write_pid_file
from evolver_studio.problem_catalogue import parse_problem_catalogue

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "training.py"


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The page, with no run in progress (its runs would live in a temporary directory)."""
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    return AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()


def _launch_button(app: AppTest):
    return next(button for button in app.button if button.label == "Launch training")


def _needs_problem_catalogue() -> None:
    result = describe(WORKING_DIRECTORY, jar_path())
    if parse_problem_catalogue(getattr(result, "value", {})) is None:
        pytest.skip("The Evolver jar in use does not describe the problems")


class TestTrainingSetProblems:
    def test_should_open_ready_to_launch_with_zdt4_chosen(self, app: AppTest):
        # Assert: NSGA-II, Double, on ZDT4
        assert not app.exception
        assert not app.error
        assert app.multiselect(key="train_problems_Double").value == ["ZDT4"]
        assert not _launch_button(app).disabled

    def test_should_offer_a_list_with_only_the_problems_of_the_encoding(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        options = list(app.multiselect(key="train_problems_Double").options)

        # Assert
        assert "ZDT4" in options and "DTLZ2" in options
        assert "ZDT5" not in options and "KroAB100TSP" not in options

    def test_should_add_a_row_for_each_problem_chosen(self, app: AppTest):
        # Act
        app.multiselect(key="train_problems_Double").select("DTLZ2").run()

        # Assert
        table = next(d.value for d in app.dataframe if "reference_front" in d.value.columns)
        assert list(table["problem"]) == ["ZDT4", "DTLZ2"]
        assert not app.exception

    def test_should_ask_for_a_problem_when_none_is_chosen(self, app: AppTest):
        # Act
        app.multiselect(key="train_problems_Double").unselect("ZDT4").run()

        # Assert
        assert any("Choose at least one training problem" in e.value for e in app.error)
        assert _launch_button(app).disabled

    def test_should_offer_the_problems_of_another_encoding_when_it_changes(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.selectbox(key="train_encoding_NSGA-II").select("Binary").run()

        # Assert: nothing is chosen yet, and the list is that of binary problems
        multiselect = app.multiselect(key="train_problems_Binary")
        assert multiselect.value == []
        assert "ZDT5" in multiselect.options and "ZDT4" not in multiselect.options
        assert _launch_button(app).disabled

    def test_should_refuse_a_problem_of_another_encoding_than_the_algorithms(self, app: AppTest):
        # Arrange: a binary problem typed in, as a class name may be
        _needs_problem_catalogue()
        app.session_state["train_problems_Double"] = ["ZDT4", "ZDT5"]

        # Act
        app.run()

        # Assert
        assert any("ZDT5 is a Binary problem" in error.value for error in app.error)
        assert _launch_button(app).disabled


class TestProblemBrowser:
    """The listing of Explore › Problems, to add problems to the training set."""

    @staticmethod
    def _browser_table(app: AppTest):
        return next(d.value for d in app.dataframe if "Family" in d.value.columns)

    def test_should_keep_the_listing_hidden_until_asked_for(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Assert
        assert not any("Family" in d.value.columns for d in app.dataframe)
        assert app.toggle(key="train_browse_Double").value is False

    def test_should_list_the_problems_of_the_encoding_with_their_details(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.toggle(key="train_browse_Double").set_value(True).run()

        # Assert: Double problems only, with their arguments and fronts, and no encoding filter
        table = self._browser_table(app)
        assert set(table["Encoding"]) == {"Double"}
        assert "ZDT4" in set(table["Problem"])
        assert {"Objectives", "Variables", "Arguments", "Reference fronts"} <= set(table.columns)
        assert not any(box.key == "train_browser_Double_encoding" for box in app.selectbox)
        assert not app.exception

    def test_should_filter_the_listing_by_family_and_name(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()
        app.toggle(key="train_browse_Double").set_value(True).run()

        # Act
        app.selectbox(key="train_browser_Double_family").select("DTLZ").run()
        app.text_input(key="train_browser_Double_filter").input("DTLZ2").run()

        # Assert
        assert list(self._browser_table(app)["Problem"]) == ["DTLZ2"]

    def test_should_not_offer_to_add_until_rows_are_selected(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.toggle(key="train_browse_Double").set_value(True).run()

        # Assert
        add = app.button(key="train_browser_add_Double")
        assert add.disabled and add.label == "Add the selected problems"


class TestMetaOptimizerEncoding:
    """The selector of flat or tree, before the box with the meta-optimizer's operator flags."""

    @staticmethod
    def _preview(app: AppTest) -> dict:
        import yaml

        meta = next(
            c.value for c in app.code if "algorithm: " in c.value and "encoding: " in c.value
        )
        return yaml.safe_load(meta)

    def test_should_offer_flat_and_tree_for_a_meta_optimizer_that_supports_both(self, app: AppTest):
        # Assert: NSGA-II, the default meta-optimizer
        radio = app.radio(key="train_meta_encoding_NSGA-II")
        assert list(radio.options) == ["Flat", "Tree"]
        assert radio.value == "Flat"
        assert self._preview(app)["encoding"] == "flat"

    def test_should_start_the_operator_flags_from_the_example_of_the_chosen_encoding(
        self, app: AppTest
    ):
        # Act
        app.radio(key="train_meta_encoding_NSGA-II").set_value("Tree").run()

        # Assert: the tree encoding has its own operators, not SBX and polynomial mutation
        flags = app.text_area(key="meta_operator_flags_NSGA-II_tree").value
        assert "mutationDistributionIndex" in flags
        assert "sbxDistributionIndex" not in flags
        assert self._preview(app)["encoding"] == "tree"
        assert not app.exception

    def test_should_offer_only_flat_to_a_meta_optimizer_without_tree_support(self, app: AppTest):
        # Act
        app.selectbox(key="train_meta_algorithm").select("SPEA2").run()

        # Assert
        radio = app.radio(key="train_meta_encoding_SPEA2")
        assert list(radio.options) == ["Flat"]
        assert any("supports only the flat encoding" in c.value for c in app.caption)

    def test_should_not_ask_random_search_for_operator_flags_in_the_tree_encoding(
        self, app: AppTest
    ):
        # Act
        app.selectbox(key="train_meta_algorithm").select("RandomSearch").run()
        app.radio(key="train_meta_encoding_RandomSearch").set_value("Tree").run()

        # Assert: nothing but the scalars is written
        preview = self._preview(app)
        assert preview["encoding"] == "tree"
        assert set(preview) <= {"algorithm", "encoding", "metaMaxEvaluations", "numberOfCores"}
        assert not app.exception


class TestMetaStoppingCondition:
    @staticmethod
    def _meta(app: AppTest) -> dict:
        import yaml

        return yaml.safe_load(
            next(c.value for c in app.code if "algorithm: " in c.value and "encoding: " in c.value)
        )

    def test_should_stop_by_evaluations_by_default(self, app: AppTest):
        # Assert
        assert app.radio(key="train_meta_stop_by").value == "Evaluations"
        assert app.number_input(key="train_meta_evals").value == 2000
        meta = self._meta(app)
        assert meta["metaMaxEvaluations"] == 2000
        assert "metaMaxComputingTimeMinutes" not in meta

    def test_should_stop_by_computing_time_when_chosen(self, app: AppTest):
        # Act
        app.radio(key="train_meta_stop_by").set_value("Computing time").run()
        app.number_input(key="train_meta_minutes").set_value(2.5).run()

        # Assert: only the time is written, as Evolver accepts one limit or the other
        meta = self._meta(app)
        assert meta["metaMaxComputingTimeMinutes"] == 2.5
        assert "metaMaxEvaluations" not in meta
        assert not any(box.key == "train_meta_evals" for box in app.number_input)
        assert not app.exception

    def test_should_combine_with_the_tree_encoding(self, app: AppTest):
        # Act
        app.radio(key="train_meta_encoding_NSGA-II").set_value("Tree").run()
        app.radio(key="train_meta_stop_by").set_value("Computing time").run()

        # Assert
        meta = self._meta(app)
        assert meta["encoding"] == "tree"
        assert meta["metaMaxComputingTimeMinutes"] == 10.0
        assert not _launch_button(app).disabled


class TestCheckpointFrequency:
    def test_should_start_with_a_frequency_that_gives_checkpoints(self, app: AppTest):
        # Assert
        assert app.number_input(key="train_update_every_50").value == 100
        assert not _launch_button(app).disabled

    def test_should_refuse_a_frequency_that_is_not_a_multiple_of_the_meta_population(
        self, app: AppTest
    ):
        # Act
        app.number_input(key="train_update_every_50").set_value(75).run()

        # Assert
        assert any("multiple of the frequency" in e.value for e in app.error)
        assert _launch_button(app).disabled


class TestPopulationOption:
    def test_should_leave_the_population_off_unless_asked_for(self, app: AppTest):
        # Assert
        assert app.checkbox(key="train_write_population").value is False

    def test_should_offer_it_alongside_the_update_frequency(self, app: AppTest):
        # Act
        app.checkbox(key="train_write_population").check().run()

        # Assert
        assert app.checkbox(key="train_write_population").value is True
        assert not app.exception
        assert not _launch_button(app).disabled


INDICATORS = "Evaluation,SolutionId,EP,NHV\n100,0,5.0,0.9\n100,1,4.0,0.8\n200,0,2.0,0.5\n"
POPULATION = (
    "Evaluation,SolutionId,EP,NHV\n100,0,5.0,0.9\n100,1,4.0,0.8\n100,2,9.0,0.99\n"
    "200,0,2.0,0.5\n200,1,6.0,0.7\n200,2,3.0,0.6\n"
)
VAR_CONF = (
    "# Evaluation: 100\n# Time (min): 0.5\nEP=4 NHV=0.8 | --crossover SBX\n\n"
    "# Evaluation: 200\n# Time (min): 1.0\nEP=2 NHV=0.5 | --crossover PCX\n\n"
)


def _running_training(working_directory: Path, population: bool) -> subprocess.Popen:
    """A run in progress: a live process, and the files Evolver has written so far."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    run_dir = working_directory / "cli-runner-runs" / "20260101-000000"
    output = working_directory / "results" / "x" / "20260101-000000"
    output.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    request = "baseLevel: b.yaml\nmetaSearch: m.yaml\noutputDirectory: results/x/20260101-000000\n"
    (run_dir / "request.yaml").write_text(
        request + ("writePopulation: true\n" if population else "")
    )
    (run_dir / "status.yaml").write_text(
        "status: RUNNING\nevaluationsDone: 200\nmaxEvaluations: 1000\nupdatedAt: x\n"
    )
    (run_dir / "runner.log").write_text("INFO: Evaluations: 100\nINFO: Evaluations: 200\n")
    (output / "INDICATORS.csv").write_text(INDICATORS)
    (output / "VAR_CONF.txt").write_text(VAR_CONF)
    if population:
        (output / "POPULATION_INDICATORS.csv").write_text(POPULATION)
    write_pid_file(run_dir / "pid.txt", process.pid)
    return process


class TestMonitorOfARunInProgress:
    @pytest.fixture
    def process(self, app: AppTest, tmp_path: Path):
        processes = []
        yield processes
        for process in processes:
            process.kill()
            process.wait()

    def test_should_show_the_tabs_of_the_monitor(self, app: AppTest, tmp_path: Path, process):
        # Arrange
        process.append(_running_training(tmp_path, population=False))

        # Act
        app.run()

        # Assert: no Population tab, as the run was not asked to write its population
        assert not app.exception
        assert [t.label for t in app.tabs] == [
            "Overview",
            "Front",
            "Convergence",
            "Best so far",
            "Log",
        ]

    def test_should_add_a_population_tab_when_the_run_writes_its_population(
        self, app: AppTest, tmp_path: Path, process
    ):
        # Arrange
        process.append(_running_training(tmp_path, population=True))

        # Act
        app.run()

        # Assert
        assert not app.exception
        assert "Population" in [t.label for t in app.tabs]
        assert len(app.get("plotly_chart")) >= 2  # the front and the population (and convergence)

    def test_should_show_the_progress_and_the_best_values_found_so_far(
        self, app: AppTest, tmp_path: Path, process
    ):
        # Arrange
        process.append(_running_training(tmp_path, population=False))

        # Act
        app.run()

        # Assert
        labels = {metric.label: metric.value for metric in app.metric}
        assert labels["Meta-evaluations"] == "200 / 1,000"
        assert labels["EP"] == "2" and labels["NHV"] == "0.5"
        assert any("Last checkpoint: evaluation 200" in c.value for c in app.caption)

    def test_should_list_the_configurations_found_so_far_and_the_log(
        self, app: AppTest, tmp_path: Path, process
    ):
        # Arrange
        process.append(_running_training(tmp_path, population=False))

        # Act
        app.run()

        # Assert
        assert any("--crossover PCX" in code.value for code in app.code)
        assert any("INFO: Evaluations: 200" in code.value for code in app.code)

    def test_should_not_open_a_front_plot_window(self, app: AppTest, tmp_path: Path, process):
        """frontPlotFrequency would open a Swing window: the request never carries it."""
        # Arrange
        process.append(_running_training(tmp_path, population=True))

        # Act
        app.run()
        request = (tmp_path / "cli-runner-runs" / "20260101-000000" / "request.yaml").read_text()

        # Assert
        assert "frontPlotFrequency" not in request


class TestResultsOfAFinishedRun:
    def _finished(self, working_directory: Path, population: bool) -> None:
        from evolver_studio.evolver_client import RunState, RunStatus

        run_dir = working_directory / "cli-runner-runs" / "20260101-000000"
        output = working_directory / "results" / "x"
        output.mkdir(parents=True)
        run_dir.mkdir(parents=True)
        lines = [
            f"outputDirectory: {output}",
            f"metadataFile: {output}/METADATA.txt",
            f"indicatorsFile: {output}/INDICATORS.csv",
            f"configurationsFile: {output}/CONFIGURATIONS.csv",
        ]
        if population:
            lines += [
                f"populationIndicatorsFile: {output}/POPULATION_INDICATORS.csv",
                f"populationConfigurationsFile: {output}/POPULATION_CONFIGURATIONS.csv",
            ]
        (run_dir / "results.yaml").write_text("\n".join(lines) + "\n")
        (output / "METADATA.txt").write_text("a training\n")
        (output / "INDICATORS.csv").write_text(INDICATORS)
        (output / "VAR_CONF.txt").write_text(VAR_CONF)
        if population:
            (output / "POPULATION_INDICATORS.csv").write_text(POPULATION)
        (run_dir / "runner.log").write_text("INFO: done\n")
        self.state = (RunStatus(RunState.FINISHED, 1000, 1000, "x"), run_dir)

    def test_should_show_the_convergence_and_the_final_configurations(
        self, app: AppTest, tmp_path: Path
    ):
        # Arrange
        self._finished(tmp_path, population=False)
        app.session_state["last_finished_run"] = self.state

        # Act
        app.run()

        # Assert: the final front is that of the last checkpoint only
        assert not app.exception
        assert [t.label for t in app.tabs] == [
            "Front",
            "Convergence",
            "Best configurations",
            "Files",
        ]
        assert any("--crossover PCX" in code.value for code in app.code)
        assert not any("--crossover SBX" in code.value for code in app.code)

    def test_should_add_the_population_when_the_run_wrote_it(self, app: AppTest, tmp_path: Path):
        # Arrange
        self._finished(tmp_path, population=True)
        app.session_state["last_finished_run"] = self.state

        # Act
        app.run()

        # Assert
        assert not app.exception
        assert "Population" in [t.label for t in app.tabs]


class TestMetaPopulationSize:
    @staticmethod
    def _meta(app: AppTest) -> dict:
        import yaml

        return yaml.safe_load(
            next(c.value for c in app.code if "algorithm: " in c.value and "encoding: " in c.value)
        )

    def test_should_offer_the_population_size_with_evolvers_default(self, app: AppTest):
        # Assert
        assert app.number_input(key="train_meta_population_NSGA-II").value == 50
        assert self._meta(app)["metaPopulationSize"] == 50

    def test_should_write_the_population_size_chosen(self, app: AppTest):
        # Act
        app.number_input(key="train_meta_population_NSGA-II").set_value(100).run()

        # Assert
        assert self._meta(app)["metaPopulationSize"] == 100
        assert not app.exception

    def test_should_ask_for_checkpoints_that_are_a_multiple_of_the_population_size(
        self, app: AppTest
    ):
        # Act: 60 is not a divisor of the default 100, but the page suggests a multiple of it
        app.number_input(key="train_meta_population_NSGA-II").set_value(60).run()

        # Assert
        assert app.number_input(key="train_update_every_60").value == 120
        assert not any("multiple of the frequency" in e.value for e in app.error)
        assert not _launch_button(app).disabled

        # Act: and one that is not a multiple is refused, naming the population size
        app.number_input(key="train_update_every_60").set_value(100).run()
        assert any("(60)" in e.value for e in app.error)
        assert _launch_button(app).disabled

    def test_should_not_ask_random_search_for_a_population(self, app: AppTest):
        # Act
        app.selectbox(key="train_meta_algorithm").select("RandomSearch").run()

        # Assert
        assert not any(
            (box.key or "").startswith("train_meta_population") for box in app.number_input
        )
        assert any("has no population" in c.value for c in app.caption)
        assert "metaPopulationSize" not in self._meta(app)


class TestPreviewOfTheRequestFiles:
    def test_should_say_where_the_parameter_space_file_will_be_and_why(self, app: AppTest):
        # Assert: not an opaque placeholder
        base_level = next(c.value for c in app.code if "yamlParameterSpaceFile" in c.value)
        assert "<run folder>/base_parameter_space.yaml" in base_level
        assert any("only known then" in c.value for c in app.caption)


class TestMetaOptimizersWithFixedOperators:
    def test_should_explain_why_spea2_has_no_flags_to_choose_and_no_tree_encoding(
        self, app: AppTest
    ):
        # Act
        app.selectbox(key="train_meta_algorithm").select("SPEA2").run()

        # Assert
        assert any("hardcodes its operators" in i.value for i in app.info)
        assert any("mutationProbabilityFactor" in i.value for i in app.info)
        assert any(
            "supports only the flat encoding: its operators are written for real-valued" in c.value
            for c in app.caption
        )

    def test_should_say_smpso_takes_no_flags(self, app: AppTest):
        # Act
        app.selectbox(key="train_meta_algorithm").select("SMPSO").run()

        # Assert
        assert any("nothing to set here" in i.value for i in app.info)
        assert any("particle swarm" in c.value for c in app.caption)

    def test_should_not_explain_anything_for_a_meta_optimizer_with_an_operator_catalogue(
        self, app: AppTest
    ):
        # Assert: NSGA-II, the default one
        assert not any("hardcodes" in i.value for i in app.info)
