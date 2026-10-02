"""Tests for the tutorials catalogue and tutorial S2's walk-through in the app."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import tutorial_parameter_spaces
from evolver_studio.evolver_client import jar_path
from evolver_studio.tutorials import TUTORIALS, TutorialLevel, tutorials_by_level

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
EVOLVER_STUDIO_TUTORIAL_IDS = [f"S{number}" for number in range(1, 13)]


class TestTutorialsCatalogue:
    def test_should_list_the_twelve_studio_tutorials_of_the_catalogue(self):
        # Act
        ids = [tutorial.tutorial_id for tutorial in TUTORIALS]

        # Assert
        assert ids == EVOLVER_STUDIO_TUTORIAL_IDS

    def test_should_mark_only_s2_and_s3_as_available(self):
        # Act
        available = [tutorial.tutorial_id for tutorial in TUTORIALS if tutorial.available]

        # Assert
        assert available == ["S2", "S3"]

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
        assert titles[-1] == "Explore on your own"
        assert len(titles) == 7


class TestTutorialS2WalkThrough:
    """Runs the real app (app.py, so st.page_link resolves) through every step of S2."""

    @pytest.fixture
    def app(self) -> AppTest:
        if not jar_path().is_file():
            pytest.skip(f"Evolver jar not found at {jar_path()}")
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

    @staticmethod
    def _go_to_step(app: AppTest, title: str) -> None:
        while app.subheader[0].value != title:
            next(button for button in app.button if button.label == "Next →").click().run()

    def test_should_show_the_space_of_nsgaii_as_the_table_of_the_explore_page(self, app: AppTest):
        # Act
        self._go_to_step(app, "Reading a parameter space")

        # Assert
        assert len(app.dataframe) == 1
        assert len(app.dataframe[0].value) == 34
        assert not app.exception

    def test_should_check_the_answer_about_the_top_level_parameters(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Reading a parameter space")

        # Act
        app.radio(key="tutorial_s2_top_level_quiz").set_value("5").run()

        # Assert
        assert any("Right" in success.value for success in app.success)

    def test_should_write_the_configuration_of_the_chosen_values(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Active parameters")

        # Act
        app.selectbox(key="tutorial_s2_choice_algorithmResult").set_value("externalArchive").run()

        # Assert
        configuration = app.code[0].value
        assert "--algorithmResult externalArchive" in configuration
        assert "--populationSizeWithArchive" in configuration

    def test_should_show_the_meta_optimizer_catalogues_and_the_indicators(self, app: AppTest):
        # Act
        self._go_to_step(app, "Beyond the algorithms")

        # Assert
        assert [tab.label for tab in app.tabs] == [
            "NSGA-II, flat encoding",
            "NSGA-II, tree encoding",
        ]
        assert len(app.dataframe) == 2
        assert any("Normalized hypervolume" in markdown.value for markdown in app.markdown)
        app.radio(key="tutorial_s2_tree_quiz").set_value("subtree").run()
        assert any("Right" in success.value for success in app.success)

    def test_should_go_back_to_the_list_of_tutorials_from_the_last_step(self, app: AppTest):
        """A link to the Tutorials page does nothing from inside it, so it is a button."""
        # Arrange
        self._go_to_step(app, "Explore on your own")

        # Act
        app.button(key="tutorial_s2_all_tutorials").click().run()

        # Assert
        assert not app.exception
        assert [header.value for header in app.subheader][:1] == ["Introductory"]
        assert [button.label for button in app.button if button.label == "Start"]
