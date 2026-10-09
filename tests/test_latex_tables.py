"""Tests for the LaTeX tables of a validation study."""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.latex_tables import BEST_COLOR, SECOND_COLOR, median_table, wilcoxon_pivot_table

PIVOT = "NSGA-II (tuned)"


def _runs() -> pd.DataFrame:
    """Two problems; on WFG2 the pivot is the best and MOEA/D clearly worse, on ZDT_1 MOEA/D is
    clearly better and NSGA-II the same as the pivot."""
    values = {
        ("WFG2", "NSGA-II"): [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
        ("WFG2", "MOEA/D"): [0.50, 0.51, 0.52, 0.53, 0.54, 0.55],
        ("WFG2", PIVOT): [0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
        ("ZDT_1", "NSGA-II"): [0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
        ("ZDT_1", "MOEA/D"): [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        ("ZDT_1", PIVOT): [0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
    }
    rows = [
        {"contender": c, "problem": p, "Run": r, "Seed": r, "TimeMs": 1, "EP": v}
        for (p, c), run_values in values.items()
        for r, v in enumerate(run_values)
    ]
    return pd.DataFrame(rows)


def _row(document: str, start: str) -> str:
    return next(line for line in document.splitlines() if line.startswith(start))


class TestMedianTable:
    def test_should_put_the_pivot_in_the_last_column(self):
        # Act
        document = median_table(_runs(), "EP", PIVOT)

        # Assert
        assert _row(document, " & ") == r" & NSGA-II & MOEA/D & NSGA-II (tuned) \\"

    def test_should_shade_the_best_and_the_second_best_median_of_each_problem(self):
        # Act
        cells = _row(median_table(_runs(), "EP", PIVOT), "WFG2").split(" & ")

        # Assert: NSGA-II, MOEA/D, pivot
        assert SECOND_COLOR in cells[1]
        assert "cellcolor" not in cells[2]
        assert BEST_COLOR in cells[3]

    def test_should_write_the_median_with_the_interquartile_range_as_a_subscript(self):
        # Act
        pivot_cell = _row(median_table(_runs(), "EP", PIVOT), "WFG2").split(" & ")[3]

        # Assert: median 0.125, IQR 0.025
        assert "$1.25e{-}01_{2.5e{-}02}$" in pivot_cell

    def test_should_escape_what_latex_reads_as_commands(self):
        # Act
        document = median_table(_runs(), "EP", PIVOT)

        # Assert
        assert _row(document, r"ZDT\_1")


class TestWilcoxonPivotTable:
    def test_should_mark_each_algorithm_against_the_pivot(self):
        # Act
        document = wilcoxon_pivot_table(_runs(), "EP", PIVOT)

        # Assert: on WFG2 the pivot beats both; on ZDT_1 it ties NSGA-II and loses to MOEA/D
        wfg2 = _row(document, "WFG2").split(" & ")
        zdt1 = _row(document, r"ZDT\_1").split(" & ")
        assert wfg2[1].endswith("$+$") and wfg2[2].endswith("$+$")
        assert zdt1[1].endswith("$=$") and zdt1[2].endswith("$-$")
        assert not any(mark in wfg2[3] for mark in ("$+$", "$-$", "$=$"))

    def test_should_count_the_marks_of_each_algorithm_in_the_last_row(self):
        # Act
        counts = _row(wilcoxon_pivot_table(_runs(), "EP", PIVOT), "$+/-/=$")

        # Assert: + / - / =
        assert counts == r"$+/-/=$ & 1/0/1 & 1/1/0 &  \\"

    @pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
    def test_should_compile(self, tmp_path: Path):
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
