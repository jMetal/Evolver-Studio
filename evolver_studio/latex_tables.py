"""The tables of a validation study as LaTeX, in the format of Evolver's scripts.

Evolver's `scripts/wilcoxon_pivot_tables.py` writes, through SAES, a table per quality indicator
with the median and the interquartile range of every algorithm on every problem, the pivot (the
tuned configuration) in the last column, the best and the second-best median of each problem shaded
dark and light gray, and each other cell marked with the result of the Wilcoxon rank-sum test
against the pivot. These are the same tables, written here without SAES, so that a study run in
Evolver-Studio gives the tables a paper shows.

Each table is a complete LaTeX document, to compile on its own; the `table` environment can be
pasted into another document, which then needs the packages of the preamble.

Nothing here depends on Streamlit.
"""

import pandas as pd

from evolver_studio.validation_stats import (
    ALPHA,
    NO_DIFFERENCE,
    PIVOT_BETTER,
    PIVOT_WORSE,
    compare_with_pivot,
    interquartile_ranges,
    medians,
)

PREAMBLE = r"""\documentclass{article}
\usepackage{colortbl}
\usepackage[table]{xcolor}
\xdefinecolor{gray95}{gray}{0.65}
\xdefinecolor{gray25}{gray}{0.8}
\begin{document}
"""
BEST_COLOR = "gray95"
SECOND_COLOR = "gray25"
# The marks of a cell compared with the pivot, as Evolver's tables write them.
MARKS = {PIVOT_BETTER: "$+$", PIVOT_WORSE: "$-$", NO_DIFFERENCE: "$=$"}
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


def median_table(runs: pd.DataFrame, indicator: str, pivot: str) -> str:
    """The median and the interquartile range of an indicator, per algorithm and problem.

    Args:
        runs: The runs table of the study (see `validation_stats`).
        indicator: The indicator.
        pivot: The pivot, which goes in the last column.

    Returns:
        A LaTeX document with the table.
    """
    rows = _cells(runs, indicator, pivot)
    caption = (
        f"Median and interquartile range of {_escape(indicator)} (lower is better). Dark and "
        "light gray: the best and the second-best median of each problem."
    )
    return _document(_contenders(runs, pivot), rows, caption, f"tab:median:{indicator}")


def wilcoxon_pivot_table(runs: pd.DataFrame, indicator: str, pivot: str) -> str:
    """The median table, with each algorithm marked by a Wilcoxon rank-sum test against the pivot.

    A cell is marked `+` when the pivot is significantly better than that algorithm on that problem,
    `-` when it is significantly worse, and `=` when the difference is not significant. The last row
    counts the marks of each algorithm.

    Args:
        runs: The runs table of the study (see `validation_stats`).
        indicator: The indicator.
        pivot: The pivot, which goes in the last column.

    Returns:
        A LaTeX document with the table.
    """
    contenders = _contenders(runs, pivot)
    comparison = compare_with_pivot(runs, indicator, pivot)
    verdicts = {(row.problem, row.contender): row.verdict for row in comparison.itertuples()}
    rows = _cells(runs, indicator, pivot)
    for problem, cells in rows.items():
        for contender in contenders[:-1]:
            verdict = verdicts.get((problem, contender))
            if verdict is not None:
                cells[contender] += f" {MARKS[verdict]}"
    counts = [
        "/".join(
            str(sum(1 for (_, c), v in verdicts.items() if c == contender and v == verdict))
            for verdict in (PIVOT_BETTER, PIVOT_WORSE, NO_DIFFERENCE)
        )
        for contender in contenders[:-1]
    ]
    footer = r"\hline" + "\n" + " & ".join([r"$+/-/=$", *counts, ""]) + r" \\"
    caption = (
        f"Median and interquartile range of {_escape(indicator)} (lower is better), and a "
        f"Wilcoxon rank-sum test at {ALPHA} against {_escape(pivot)}: $+$ {_escape(pivot)} is "
        "significantly better, $-$ significantly worse, $=$ no significant difference. Dark and "
        "light gray: the best and the second-best median of each problem."
    )
    return _document(contenders, rows, caption, f"tab:wilcoxon:{indicator}", footer)


def _contenders(runs: pd.DataFrame, pivot: str) -> list[str]:
    """The algorithms in their order in the study, with the pivot moved to the end."""
    others = [c for c in dict.fromkeys(runs["contender"]) if c != pivot]
    return [*others, pivot]


def _cells(runs: pd.DataFrame, indicator: str, pivot: str) -> dict[str, dict[str, str]]:
    """The cell of each algorithm on each problem: its median, its IQR and its shade."""
    contenders = _contenders(runs, pivot)
    median_values = medians(runs, indicator).reindex(columns=contenders)
    iqr_values = interquartile_ranges(runs, indicator).reindex(columns=contenders)
    rows = {}
    for problem in median_values.index:
        ranked = median_values.loc[problem].dropna().sort_values(kind="stable")
        shades = dict(zip(ranked.index[:2], (BEST_COLOR, SECOND_COLOR), strict=False))
        cells = {}
        for contender in contenders:
            median = median_values.loc[problem, contender]
            if pd.isna(median):
                cells[contender] = "--"
                continue
            shade = shades.get(contender)
            prefix = rf"\cellcolor{{{shade}}}" if shade else ""
            iqr = iqr_values.loc[problem, contender]
            cells[contender] = f"{prefix}${_number(median, 2)}_{{{_number(iqr, 1)}}}$"
        rows[problem] = cells
    return rows


def _document(
    contenders: list[str],
    rows: dict[str, dict[str, str]],
    caption: str,
    label: str,
    footer: str | None = None,
) -> str:
    header = " & ".join(["", *(_escape(c) for c in contenders)]) + r" \\"
    body = "\n".join(
        " & ".join([_escape(problem), *(cells[c] for c in contenders)]) + r" \\"
        for problem, cells in rows.items()
    )
    lines = [
        PREAMBLE.rstrip("\n"),
        r"\begin{table}",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "l" * len(contenders) + "}",
        r"\hline",
        header,
        r"\hline",
        body,
        *([footer] if footer else []),
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def _number(value: float, decimals: int) -> str:
    """A number in scientific notation, for math mode: the sign of the exponent in braces, so
    that it is not spaced as a binary operator (1.23e{-}02, not 1.23e - 02)."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return f"{mantissa}e{{{exponent[0]}}}{exponent[1:]}"


def _escape(text: str) -> str:
    return "".join(_SPECIAL_CHARACTERS.get(character, character) for character in str(text))
