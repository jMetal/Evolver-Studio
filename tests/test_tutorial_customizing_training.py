"""Tests for tutorial "Customizing a training run"."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client, tutorial_customizing_training
from evolver_studio.evolver_client import jar_path

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"
OPEN_KEY = "tutorial_open_customizing_a_training"


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    (tmp_path / "resources").symlink_to(RESOURCES)
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    app = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    app.switch_page("pages/tutorials.py").run()
    app.button(key=OPEN_KEY).click().run()
    return app


def _go_to(app: AppTest, title: str) -> None:
    while app.subheader[0].value != title:
        next(button for button in app.button if button.label == "Next →").click().run()


def _preset(app: AppTest, key: str) -> AppTest:
    app.button(key=f"tutorial_customizing_{key}").click().run()
    return app


class TestWalkThrough:
    def test_should_render_every_step_without_exceptions(self, app: AppTest):
        # Act
        subheaders = []
        for _ in tutorial_customizing_training.STEPS:
            subheaders.append(app.subheader[0].value)
            assert not app.exception, subheaders[-1]
            next_button = next(button for button in app.button if button.label == "Next →")
            if not next_button.disabled:
                next_button.click().run()

        # Assert
        assert subheaders == [step.title for step in tutorial_customizing_training.STEPS]

    def test_should_compute_the_cost_of_a_training(self, app: AppTest):
        # Arrange
        _go_to(app, "Step 3: what it costs")

        # Act: E3 on three problems with two runs per configuration
        app.number_input(key="tutorial_customizing_problems").set_value(3)
        app.number_input(key="tutorial_customizing_runs").set_value(2).run()

        # Assert: 2000 x 2 x 3 x 12000
        assert app.metric[0].value == "144 million"

    def test_should_count_fewer_parameters_with_fewer_crossovers(self, app: AppTest):
        # Arrange
        _go_to(app, "Step 4: the parameter space")
        total = int(app.metric[0].value)

        # Act
        app.multiselect(key="tutorial_customizing_crossover").set_value(["SBX"]).run()

        # Assert
        assert int(app.metric[0].value) < total


class TestPresets:
    @pytest.mark.parametrize(
        ("step", "key", "checks"),
        [
            (
                "Step 1: the training set",
                "training_set",
                {"train_problems_Double": ["ZDT1", "ZDT4", "ZDT6"]},
            ),
            (
                "Step 2: meta-objectives, and measuring each configuration",
                "measure",
                {
                    "train_indicators": ["InvertedGenerationalDistancePlus", "Epsilon"],
                    "train_independent_runs": 3,
                },
            ),
            (
                "Step 5: the meta-optimizer, its encoding and its budget",
                "meta",
                {"train_meta_stop_by": "Computing time", "train_meta_minutes": 5.0},
            ),
            (
                "Step 6: binary and permutation problems",
                "binary",
                {"train_problems_Binary": ["ZDT5"]},
            ),
        ],
    )
    def test_should_open_training_with_the_example_of_the_step(
        self, app: AppTest, step: str, key: str, checks: dict
    ):
        # Arrange
        _go_to(app, step)

        # Act
        _preset(app, key)

        # Assert: the Training page, with the example in its form and nothing wrong with it
        assert not app.exception
        assert app.title[0].value == "Training"
        assert not app.error, [e.value for e in app.error]
        for widget_key, value in checks.items():
            assert app.session_state[widget_key] == value
        assert not next(b for b in app.button if b.label == "Launch training").disabled
