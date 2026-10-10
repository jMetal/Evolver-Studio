"""The results of a validation study for SAES, and its Wilcoxon pivot table made by SAES.

SAES (https://github.com/jMetal/SAES) analyzes the results of an experimental study from two CSV
files: the results, one row per independent run of an algorithm on a problem with the value of one
quality indicator (`Algorithm`, `Instance`, `MetricName`, `ExecutionId`, `MetricValue`), and the
metrics, which say whether each indicator is maximized (`MetricName`, `Maximize`). The study's runs
are laid out here as those files, so that SAES can be run on them as on any other study.

The Wilcoxon pivot table is SAES's own (`WilcoxonPivot`), as Evolver's
`scripts/wilcoxon_pivot_tables.py` makes it: the median and interquartile range of every algorithm
on every problem, the pivot (the tuned configuration) in the last column, each other algorithm
marked by a Wilcoxon rank-sum test against it and the best two medians of each problem shaded. Two
things of SAES's document are fixed, as that script does: the counts of the last row, which SAES
writes as `\\textbf20` (only the first digit bold), and its preamble, which asks xcolor for an
option (`table*`) that current LaTeX distributions reject. The table is put in a preamble that
compiles.

Nothing here depends on Streamlit.
"""

import importlib.util
import logging
import re

import pandas as pd

from evolver_studio.validation_stats import indicator_names

RESULTS_COLUMNS = ("Algorithm", "Instance", "MetricName", "ExecutionId", "MetricValue")
METRICS_COLUMNS = ("MetricName", "Maximize")
PREAMBLE = r"""\documentclass{article}
\usepackage{colortbl}
\usepackage{float}
\usepackage[table]{xcolor}
\usepackage{siunitx}
\sisetup{output-exponent-marker=\text{e}}
\xdefinecolor{gray95}{gray}{0.65}
\xdefinecolor{gray25}{gray}{0.8}
\begin{document}
"""
_SPECIAL_CHARACTERS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def saes_available() -> bool:
    """Whether SAES is installed, which the LaTeX table needs (the CSV files do not)."""
    return importlib.util.find_spec("SAES") is not None


def saes_results(runs: pd.DataFrame) -> pd.DataFrame:
    """The runs of a study as SAES's results file.

    Args:
        runs: The runs table of the study (see `validation_stats`).

    Returns:
        One row per run and indicator, with the columns `RESULTS_COLUMNS`; the execution of a run
        is its number.
    """
    results = runs.melt(
        id_vars=["contender", "problem", "Run"],
        value_vars=indicator_names(runs),
        var_name="MetricName",
        value_name="MetricValue",
    ).rename(columns={"contender": "Algorithm", "problem": "Instance", "Run": "ExecutionId"})
    return results.dropna(subset=["MetricValue"])[list(RESULTS_COLUMNS)].reset_index(drop=True)


def saes_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """The indicators of a study as SAES's metrics file: all of them are minimized.

    Args:
        runs: The runs table of the study.

    Returns:
        One row per indicator, with the columns `METRICS_COLUMNS`.
    """
    return pd.DataFrame(
        {"MetricName": indicator_names(runs), "Maximize": False}, columns=list(METRICS_COLUMNS)
    )


def wilcoxon_pivot_table(runs: pd.DataFrame, indicator: str, pivot: str) -> str:
    """SAES's Wilcoxon pivot table of an indicator, as a LaTeX document that compiles.

    Args:
        runs: The runs table of the study.
        indicator: The indicator.
        pivot: The pivot, which goes in the last column.

    Returns:
        The LaTeX document.
    """
    from SAES.latex_generation.stats_table import WilcoxonPivot

    results = saes_results(runs)
    # SAES writes the names as they are: those LaTeX reads as commands are escaped first.
    results["Algorithm"] = results["Algorithm"].map(_escape)
    results["Instance"] = results["Instance"].map(_escape)
    with _quiet_saes():
        table = WilcoxonPivot(results, saes_metrics(runs), indicator, pivot=_escape(pivot))
        table.create_latex_table()
    match = re.search(r"\\begin\{table\}.*\\end\{table\}", table.latex_doc, re.DOTALL)
    if match is None:
        raise ValueError("SAES wrote no table")
    body = re.sub(r"\\textbf(\d+)", r"\\textbf{\1}", match.group(0))
    body = "\n".join(line.strip() for line in body.splitlines())
    return f"{PREAMBLE}{body}\n\\end{{document}}\n"


class _quiet_saes:  # noqa: N801 - used as a context manager, like contextlib's
    """Silence SAES's log while it makes a table: it warns of what the page already says."""

    def __enter__(self) -> None:
        self._levels = {}
        for name in list(logging.root.manager.loggerDict):
            if name.startswith("SAES"):
                logger = logging.getLogger(name)
                self._levels[name] = logger.level
                logger.setLevel(logging.ERROR)

    def __exit__(self, *_) -> None:
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)


def _escape(text: str) -> str:
    return "".join(_SPECIAL_CHARACTERS.get(character, character) for character in str(text))
