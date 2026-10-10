"""Tests for the LaTeX tables of a validation study."""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.latex_tables import (
    BEST_COLOR,
    SECOND_COLOR,
    ranking_table,
    wilcoxon_pivot_table,
)
from evolver_studio.validation_stats import average_ranks, holm_against_pivot, medians

PIVOT = "NSGA-II (tuned)"


def _runs() -> pd.DataFrame:
    """On WFG2 the pivot beats both; on ZDT_1 it ties NSGA-II and loses to MOEA/D."""
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
        for r, v in enumerate(run_values, start=1)
    ]
    return pd.DataFrame(rows)


def _cells(document: str, start: str) -> list[str]:
    line = next(line for line in document.splitlines() if line.startswith(start))
    return line.removesuffix(r" \\").split(" & ")


def _compiles(document: str, directory: Path) -> subprocess.CompletedProcess:
    (directory / "table.tex").write_text(document)
    return subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
        cwd=directory,
        capture_output=True,
        text=True,
    )


class TestWilcoxonPivotTable:
    def test_should_put_the_pivot_in_the_last_column(self):
        # Act
        document = wilcoxon_pivot_table(_runs(), "EP", PIVOT, maximize=False)

        # Assert
        header = next(line for line in document.splitlines() if line.startswith(" & "))
        assert header == r" & NSGA-II & MOEA/D & NSGA-II (tuned) \\ \hline"

    def test_should_mark_each_algorithm_against_the_pivot(self):
        # Act
        document = wilcoxon_pivot_table(_runs(), "EP", PIVOT, maximize=False)

        # Assert
        wfg2, zdt1 = _cells(document, "WFG2"), _cells(document, r"ZDT\_1")
        assert wfg2[1].endswith("+$") and wfg2[2].endswith("+$")
        assert zdt1[1].endswith("=$") and zdt1[2].endswith("-$")
        assert wfg2[3].endswith("{} } $")

    def test_should_write_median_and_iqr_as_saes_does(self):
        # Act
        pivot_cell = _cells(wilcoxon_pivot_table(_runs(), "EP", PIVOT, maximize=False), "WFG2")[3]

        # Assert: median 0.125, IQR 0.025
        assert pivot_cell.endswith(r"$\SI{1.25e-01}{}_{ \SI{2.50e-02}{} } $")

    def test_should_shade_the_best_and_the_second_best_median(self):
        # Act
        cells = _cells(wilcoxon_pivot_table(_runs(), "EP", PIVOT, maximize=False), "WFG2")

        # Assert: the pivot, then NSGA-II
        assert cells[3].startswith(rf"\cellcolor{{{BEST_COLOR}}}")
        assert cells[1].startswith(rf"\cellcolor{{{SECOND_COLOR}}}")
        assert "cellcolor" not in cells[2]

    def test_should_shade_the_highest_for_a_maximized_indicator(self):
        # Act
        cells = _cells(wilcoxon_pivot_table(_runs(), "EP", PIVOT, maximize=True), "WFG2")

        # Assert: MOEA/D has the highest median
        assert cells[2].startswith(rf"\cellcolor{{{BEST_COLOR}}}")

    def test_should_count_the_marks_in_bold_in_the_last_row(self):
        # Act
        document = wilcoxon_pivot_table(_runs(), "EP", PIVOT, maximize=False)

        # Assert: + / - / =
        counts = _cells(document, r"\hline + / - / =")
        assert counts[1] == r"\textbf{1} / \textbf{0} / \textbf{1}"
        assert counts[2] == r"\textbf{1} / \textbf{1} / \textbf{0}"

    @pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
    def test_should_compile(self, tmp_path: Path):
        # Act
        result = _compiles(wilcoxon_pivot_table(_runs(), "EP", PIVOT, maximize=False), tmp_path)

        # Assert
        assert result.returncode == 0, result.stdout[-2000:]


class TestRankingTable:
    @staticmethod
    def _document() -> str:
        ranks = average_ranks(medians(_runs(), "EP"))
        holm = holm_against_pivot(ranks, PIVOT, problems=2)
        return ranking_table(holm, ranks, PIVOT, "EP")

    def test_should_list_every_algorithm_with_its_average_rank(self):
        # Act
        document = self._document()

        # Assert
        assert _cells(document, "NSGA-II (tuned)")[1:] == ["1.75", "--", "--"]
        assert _cells(document, "MOEA/D")[1] == "2.00"

    @pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
    def test_should_compile(self, tmp_path: Path):
        # Act
        result = _compiles(self._document(), tmp_path)

        # Assert
        assert result.returncode == 0, result.stdout[-2000:]
