"""Tests for the results of a validation study for SAES, and its Wilcoxon pivot table."""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.saes_export import (
    METRICS_COLUMNS,
    RESULTS_COLUMNS,
    saes_available,
    saes_metrics,
    saes_results,
    wilcoxon_pivot_table,
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


@pytest.fixture
def saes() -> None:
    if not saes_available():
        pytest.skip("SAES is not installed")


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


class TestWilcoxonPivotTable:
    def test_should_put_the_pivot_in_the_last_column(self, saes):
        # Act
        document = wilcoxon_pivot_table(_runs(), "EP", PIVOT)

        # Assert
        header = next(line for line in document.splitlines() if "NSGA-II &" in line)
        assert header.rstrip(r" \hline").endswith("NSGA-II (tuned)")

    def test_should_count_the_results_of_the_test_with_whole_numbers_in_bold(self, saes):
        # Act
        document = wilcoxon_pivot_table(_runs(), "EP", PIVOT)

        # Assert: the pivot beats NSGA-II on both problems; MOEA/D once each way
        counts = next(line for line in document.splitlines() if "+ / - / =" in line)
        assert r"\textbf{2} / \textbf{0} / \textbf{0}" in counts
        assert r"\textbf{1} / \textbf{1} / \textbf{0}" in counts

    def test_should_escape_what_latex_reads_as_commands(self, saes):
        # Act
        document = wilcoxon_pivot_table(_runs(), "EP", PIVOT)

        # Assert
        assert r"ZDT\_1 &" in document

    @pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
    def test_should_compile(self, saes, tmp_path: Path):
        # Arrange
        (tmp_path / "table.tex").write_text(wilcoxon_pivot_table(_runs(), "EP", PIVOT))

        # Act
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        # Assert
        assert result.returncode == 0, result.stdout[-2000:]
