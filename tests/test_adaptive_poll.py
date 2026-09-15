"""Tests for the adaptive poll interval."""

from evolver_studio.adaptive_poll import AdaptivePollInterval


class TestAdaptivePollInterval:
    def test_should_start_at_the_given_initial_interval(self):
        """Before any poll is recorded, the interval is whatever was configured."""
        # Arrange
        interval = AdaptivePollInterval(initial_seconds=3.0)

        # Act
        seconds = interval.seconds()

        # Assert
        assert seconds == 3.0

    def test_should_back_off_when_a_poll_finds_nothing_new(self):
        """A miss must grow the interval by the backoff factor."""
        # Arrange
        interval = AdaptivePollInterval(
            initial_seconds=2.0, min_seconds=0.5, max_seconds=100.0, backoff_factor=2.0
        )

        # Act
        interval.record_poll(changed=False, now=0.0)

        # Assert
        assert interval.seconds() == 4.0

    def test_should_not_back_off_past_the_max_interval(self):
        """Repeated misses must not push the interval above the configured cap."""
        # Arrange
        interval = AdaptivePollInterval(
            initial_seconds=8.0, min_seconds=0.5, max_seconds=10.0, backoff_factor=2.0
        )

        # Act
        interval.record_poll(changed=False, now=0.0)

        # Assert
        assert interval.seconds() == 10.0

    def test_should_settle_toward_the_observed_gap_between_changes(self):
        """Two changes 10s apart must pull the interval toward that gap."""
        # Arrange
        interval = AdaptivePollInterval(
            initial_seconds=2.0, min_seconds=0.5, max_seconds=100.0, backoff_factor=1.5
        )
        interval.record_poll(changed=True, now=0.0)

        # Act
        interval.record_poll(changed=True, now=10.0)

        # Assert — halfway between the previous interval (2.0) and the observed gap (10.0)
        assert interval.seconds() == 6.0

    def test_should_not_change_on_the_very_first_recorded_change(self):
        """With no prior change to measure a gap against, the interval is left as-is."""
        # Arrange
        interval = AdaptivePollInterval(initial_seconds=2.0)

        # Act
        interval.record_poll(changed=True, now=123.0)

        # Assert
        assert interval.seconds() == 2.0

    def test_should_not_settle_below_the_min_interval(self):
        """A very short observed gap must not push the interval under the configured floor."""
        # Arrange
        interval = AdaptivePollInterval(
            initial_seconds=2.0, min_seconds=1.5, max_seconds=100.0, backoff_factor=1.5
        )
        interval.record_poll(changed=True, now=0.0)

        # Act — averaging 2.0 and a 0.1s gap gives 1.05, below the 1.5s floor
        interval.record_poll(changed=True, now=0.1)

        # Assert
        assert interval.seconds() == 1.5
