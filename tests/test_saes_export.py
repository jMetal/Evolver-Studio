"""Tests for the results of a validation study as the files SAES reads."""

import pandas as pd

from evolver_studio.saes_export import (
    METRICS_COLUMNS,
    RESULTS_COLUMNS,
    saes_metrics,
    saes_results,
)

PIVOT = "NSGA-II (tuned)"


def _runs() -> pd.DataFrame:
    """Two problems; on WFG2 the pivot is better than both, on ZDT_1 MOEA/D is better than it."""
    values = {
        ("WFG2", "NSGA-II"): [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
        ("WFG2", "MOEA/D"): [0.50, 0.51, 0.52, 0.53, 0.54, 0.55],
        ("WFG2", PIVOT): [0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
        ("ZDT_1", "NSGA-II"): [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
        ("ZDT_1", "MOEA/D"): [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        ("ZDT_1", PIVOT): [0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
    }
    rows = [
        {
            "contender": c,
            "problem": p,
            "Run": r + 1,
            "Seed": r + 1,
            "TimeMs": 1,
            "EP": v,
            "NHV": v / 2,
        }
        for (p, c), run_values in values.items()
        for r, v in enumerate(run_values)
    ]
    return pd.DataFrame(rows)


class TestSaesFiles:
    def test_should_write_a_row_per_run_and_indicator(self):
        # Act
        results = saes_results(_runs())

        # Assert
        assert list(results.columns) == list(RESULTS_COLUMNS)
        assert len(results) == 36 * 2
        first = results.iloc[0]
        assert (first["Algorithm"], first["Instance"], first["MetricName"]) == (
            "NSGA-II",
            "WFG2",
            "EP",
        )
        assert first["ExecutionId"] == 1

    def test_should_say_that_every_indicator_is_minimized(self):
        # Act
        metrics = saes_metrics(_runs())

        # Assert
        assert list(metrics.columns) == list(METRICS_COLUMNS)
        assert list(metrics["MetricName"]) == ["EP", "NHV"]
        assert not metrics["Maximize"].any()
