"""Tests for the tutorials catalogue and tutorial S2's walk-through in the app."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import tutorial_parameter_spaces
from evolver_studio.tutorials import TUTORIALS, TutorialLevel, tutorials_by_level

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
EVOLVER_STUDIO_TUTORIAL_IDS = [f"S{number}" for number in range(1, 13)]


class TestTutorialsCatalogue:
    def test_should_list_the_twelve_studio_tutorials_of_the_catalogue(self):
        # Act
        ids = [tutorial.tutorial_id for tutorial in TUTORIALS]

        # Assert
        assert ids == EVOLVER_STUDIO_TUTORIAL_IDS

    def test_should_mark_only_s2_as_available(self):
        # Act
        available = [tutorial.tutorial_id for tutorial in TUTORIALS if tutorial.available]

        # Assert
        assert available == ["S2"]

    @pytest.mark.parametrize("level", list(TutorialLevel))
    def test_should_group_every_tutorial_under_its_level(self, level: TutorialLevel):
        # Act
        grouped = tutorials_by_level(level)

        # Assert
        assert grouped
        assert all(tutorial.level is level for tutorial in grouped)

    def test_should_give_tutorial_s2_its_steps(self):
        # Act
        titles = [step.title for step in tutorial_parameter_spaces.STEPS]

        # Assert
        assert titles[0] == "What is a parameter space?"
        assert len(titles) == 6


class TestTutorialS2WalkThrough:
    """Runs the real app (app.py, so st.page_link resolves) through every step of S2."""

    @pytest.fixture
    def app(self) -> AppTest:
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=60).run()
        app.switch_page("pages/tutorials.py").run()
        next(button for button in app.button if button.label == "Start").click().run()
        return app

    def test_should_render_every_step_without_exceptions(self, app: AppTest):
        # Act
        subheaders = []
        for _ in tutorial_parameter_spaces.STEPS:
            subheaders.append(app.subheader[0].value)
            assert not app.exception
            next_button = next(button for button in app.button if button.label == "Next →")
            if not next_button.disabled:
                next_button.click().run()

        # Assert
        assert subheaders == [step.title for step in tutorial_parameter_spaces.STEPS]

    def test_should_activate_the_archive_parameters_when_choosing_an_external_archive(
        self, app: AppTest
    ):
        # Arrange
        for _ in range(3):
            next(button for button in app.button if button.label == "Next →").click().run()

        # Act
        app.selectbox(key="tutorial_s2_choice_algorithmResult").set_value("externalArchive").run()

        # Assert
        assert app.metric[0].value == "16 of 34"
