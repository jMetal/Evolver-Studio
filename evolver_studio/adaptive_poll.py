"""An adaptive sleep interval for polling a slowly, irregularly changing file.

Not meant to keep a live view in tight sync with the data source — only to
avoid wasted polls when nothing is happening, and to avoid lagging far behind
once data starts arriving. The interval settles toward the observed gap
between changes and backs off geometrically when a poll finds nothing new.
"""


class AdaptivePollInterval:
    """Tracks how long to sleep before the next poll of a changing file."""

    def __init__(
        self,
        initial_seconds: float = 2.0,
        min_seconds: float = 1.0,
        max_seconds: float = 15.0,
        backoff_factor: float = 1.5,
    ) -> None:
        self._interval = initial_seconds
        self._min_seconds = min_seconds
        self._max_seconds = max_seconds
        self._backoff_factor = backoff_factor
        self._last_change_at: float | None = None

    def seconds(self) -> float:
        """Return the interval to sleep before the next poll."""
        return self._interval

    def record_poll(self, changed: bool, now: float) -> None:
        """Update the interval based on the outcome of a poll just made.

        Args:
            changed: Whether the polled file had new data this time.
            now: Current time (any monotonically increasing clock, in
                seconds), used to measure the gap between changes.
        """
        if changed:
            self._settle_toward_observed_gap(now)
        else:
            self._back_off()

    def _settle_toward_observed_gap(self, now: float) -> None:
        """Move the interval halfway toward the gap since the last change."""
        if self._last_change_at is not None:
            observed_gap = now - self._last_change_at
            self._interval = self._clip((self._interval + observed_gap) / 2)
        self._last_change_at = now

    def _back_off(self) -> None:
        """Grow the interval, since the last poll found nothing new."""
        self._interval = self._clip(self._interval * self._backoff_factor)

    def _clip(self, value: float) -> float:
        return min(max(value, self._min_seconds), self._max_seconds)
