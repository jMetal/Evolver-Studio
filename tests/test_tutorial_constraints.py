"""Tests for tutorial "Problems with constraints"."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client, tutorial_constraints
from evolver_studio.evolver_client import jar_path
from evolver_studio.runs import RunPhase, run_phase
from evolver_studio.tutorial_constraints import (
    latest_tutorial_training,
    push_pull_configuration,
    training_form_state,
)

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"


@pytest.fixture
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    (tmp_path / "resources").symlink_to(RESOURCES)
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    return tmp_path


def _open() -> AppTest:
    app = AppTest.from_file(str(APP_SCRIPT), default_timeout=180).run()
    app.switch_page("app_pages/tutorials.py").run()
    app.button(key="tutorial_open_problems_with_constraints").click().run()
    return app


def _go_to_step(app: AppTest, title: str) -> AppTest:
    while app.subheader[0].value != title:
        next(button for button in app.button if button.label == "Next →").click().run()
        assert not app.exception
    return app


def _run(app: AppTest, key: str) -> None:
    app.button(key=f"tutorial_solving_button_constraints_{key}").click().run()
    assert not app.exception


class TestWalkThrough:
    def test_should_render_every_step_before_any_run(self, working_directory: Path):
        # Arrange
        app = _open()

        # Act
        titles = []
        for _ in tutorial_constraints.STEPS:
            titles.append(app.subheader[0].value)
            assert not app.exception
            button = next(button for button in app.button if button.label == "Next →")
            if not button.disabled:
                button.click().run()

        # Assert
        assert titles == [step.title for step in tutorial_constraints.STEPS]
        assert latest_tutorial_training() is None

    def test_should_show_that_without_a_technique_no_run_is_feasible(self, working_directory: Path):
        # Arrange
        app = _go_to_step(_open(), "Step 3: with and without a technique")

        # Act
        _run(app, "without_DASCMOP2")
        _run(app, "cdp_DASCMOP2")

        # Assert
        assert any(
            "**0 of 10** runs" in m.value and "**10 of 10**" in m.value for m in app.markdown
        )

    def test_should_compare_three_ways_of_handling_the_constraints(self, working_directory: Path):
        # Arrange
        app = _go_to_step(_open(), "Step 4: the technique matters")

        # Act
        for key in ("without_DASCMOP5", "cdp_DASCMOP5", "push_pull"):
            _run(app, key)

        # Assert
        table = next(d.value for d in app.dataframe if "Feasible runs" in d.value.columns)
        assert list(table["Configuration"]) == ["Without a technique", "cdp", "pushPull"]
        assert any("which technique" in m.value for m in app.markdown)

    def test_should_open_a_run_in_run_algorithm(self, working_directory: Path):
        # Arrange
        app = _go_to_step(_open(), "Step 3: with and without a technique")

        # Act
        app.button(key="tutorial_solving_open_constraints_cdp_DASCMOP2").click().run()

        # Assert: the one without a technique cannot be opened there
        assert not app.exception
        assert app.radio(key="solve_constraints").value == "With constraints"
        assert app.selectbox(key="solve_problem").value == "DASCMOP2"
        keys = [button.key for button in _open().button]
        assert "tutorial_solving_open_constraints_without_DASCMOP2" not in keys


class TestConfiguration:
    def test_should_use_push_and_pull_in_the_default_configuration(self, working_directory: Path):
        # Act
        configuration = push_pull_configuration(jar_path())

        # Assert
        assert "--constraintHandling pushPull" in configuration and "cdp" not in configuration
        assert "--crossover SBX" in configuration


class TestTraining:
    def test_should_train_and_read_what_the_training_found(self, working_directory: Path):
        # Arrange: this tutorial's training, with fewer configurations
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=60)
        for key, value in training_form_state().items():
            app.session_state[key] = value
        app.session_state["train_meta_evals"] = 150
        app.switch_page("app_pages/training.py").run()
        assert not app.exception and not app.error

        # Act
        next(b for b in app.button if b.label == "Launch training").click().run()
        run_dir = next((working_directory / "cli-runner-runs").iterdir())
        deadline = time.monotonic() + 180
        while run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the training did not finish"
            time.sleep(0.3)

        # Assert
        assert run_phase(run_dir) == RunPhase.FINISHED
        tutorial = _go_to_step(_open(), "Step 6: what the training found")
        assert any("found feasible solutions" in m.value for m in tutorial.markdown)
        _go_to_step(tutorial, "Step 7: is it better than the default?")
        assert any("--constraintHandling" in code.value for code in tutorial.code)
