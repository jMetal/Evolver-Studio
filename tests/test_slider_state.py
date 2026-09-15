"""Tests for the auto-tracking slider value decision logic."""

from evolver_studio.slider_state import next_slider_value


class TestNextSliderValue:
    def test_should_track_available_on_first_use(self):
        """With no stored value yet, the slider must default to showing everything."""
        # Act
        value, tracked = next_slider_value(current_value=None, tracked_available=None, available=2)

        # Assert
        assert value == 2
        assert tracked == 2

    def test_should_keep_tracking_available_when_untouched(self):
        """As long as the user hasn't moved it away from the tracked value, keep following it."""
        # Act
        value, tracked = next_slider_value(current_value=2, tracked_available=2, available=5)

        # Assert
        assert value == 5
        assert tracked == 5

    def test_should_keep_the_users_chosen_value_once_touched(self):
        """A value that no longer matches what was auto-tracked must be left alone."""
        # Act
        value, tracked = next_slider_value(current_value=1, tracked_available=2, available=5)

        # Assert
        assert value == 1
        assert tracked == 2

    def test_should_keep_the_users_choice_across_further_growth(self):
        """Once fixed, further growth in `available` must not move it again."""
        # Arrange
        value, tracked = next_slider_value(current_value=1, tracked_available=2, available=5)

        # Act
        value, tracked = next_slider_value(
            current_value=value, tracked_available=tracked, available=8
        )

        # Assert
        assert value == 1
        assert tracked == 2
