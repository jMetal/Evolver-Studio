"""The tables of a validation study as LaTeX: the Wilcoxon pivot table and the ranking.

The Wilcoxon pivot table has the layout of SAES's (https://github.com/jMetal/SAES), the one
Evolver's `scripts/wilcoxon_pivot_tables.py` makes: one row per problem and one column per
algorithm, the pivot (the tuned configuration) last; each cell the median of the indicator with its
interquartile range as a subscript; the best and the second-best median of each problem shaded dark
and light gray; each other algorithm marked against the pivot (`+` the pivot is significantly
better, `-` significantly worse, `=` no significant difference); and a last row that counts the
marks. It is written here, without SAES, from the comparison of `validation_stats`, which uses the
Wilcoxon rank-sum test: the runs of two algorithms are independent samples, not pairs.

The ranking table has the average rank of each algorithm over the problems, and Holm's adjusted
p-value of its difference with the pivot.

Each table is a LaTeX document that compiles on its own; the `table` environment can be pasted
into another document, which then needs the packages of the preamble.

Nothing here depends on Streamlit.
"""

import pandas as pd

from evolver_studio.catalogue import is_maximized
from evolver_studio.validation_stats import (
    ALPHA,
    BENJAMINI_HOCHBERG,
    HOLM,
    NO_CORRECTION,
    NO_DIFFERENCE,
    PIVOT_BETTER,
    PIVOT_WORSE,
    compare_with_pivot,
    interquartile_ranges,
    medians,
)

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
BEST_COLOR = "gray95"
SECOND_COLOR = "gray25"
MARKS = {PIVOT_BETTER: "+", PIVOT_WORSE: "-", NO_DIFFERENCE: "="}
CORRECTION_NOTES = {
    HOLM: " (p-values adjusted with Holm's procedure)",
    BENJAMINI_HOCHBERG: " (p-values adjusted with the Benjamini-Hochberg procedure)",
}
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


def wilcoxon_pivot_table(
    runs: pd.DataFrame,
    indicator: str,
    pivot: str,
    maximize: bool | None = None,
    correction: str = NO_CORRECTION,
) -> str:
    """The Wilcoxon pivot table of an indicator, as a LaTeX document.

    Args:
        runs: The runs table of the study (see `validation_stats`).
        indicator: The indicator.
        pivot: The pivot, which goes in the last column.
        maximize: Whether higher values are better; None looks it up in the catalogue.
        correction: How the p-values are adjusted for being many comparisons (see
            `validation_stats.adjust_p_values`); the marks follow the adjusted ones.

    Returns:
        The LaTeX document.
    """
    if maximize is None:
        maximize = is_maximized(indicator)
    contenders = [c for c in dict.fromkeys(runs["contender"]) if c != pivot] + [pivot]
    median_values = medians(runs, indicator).reindex(columns=contenders)
    iqr_values = interquartile_ranges(runs, indicator).reindex(columns=contenders)
    comparison = compare_with_pivot(
        runs, indicator, pivot, maximize=maximize, correction=correction
    )
    verdicts = {(row.problem, row.contender): row.verdict for row in comparison.itertuples()}
    counts = {contender: [0, 0, 0] for contender in contenders[:-1]}
    rows = []
    for problem in median_values.index:
        best, second = _top_two(median_values.loc[problem], iqr_values.loc[problem], maximize)
        cells = [_escape(problem)]
        for contender in contenders:
            median = median_values.loc[problem, contender]
            if pd.isna(median):
                cells.append("--")
                continue
            mark = ""
            if contender != pivot:
                verdict = verdicts.get((problem, contender), NO_DIFFERENCE)
                mark = MARKS[verdict]
                counts[contender][(PIVOT_BETTER, PIVOT_WORSE, NO_DIFFERENCE).index(verdict)] += 1
            shade = {best: BEST_COLOR, second: SECOND_COLOR}.get(contender)
            prefix = rf"\cellcolor{{{shade}}}" if shade else ""
            iqr = iqr_values.loc[problem, contender]
            cells.append(rf"{prefix}$\SI{{{median:.2e}}}{{}}_{{ \SI{{{iqr:.2e}}}{{}} }} {mark}$")
        rows.append(" & ".join(cells) + r" \\")
    count_row = " & ".join(
        [r"\hline + / - / =", *(_count_cell(counts[c]) for c in contenders[:-1]), ""]
    )
    caption = (
        f"{_escape(indicator)}. Median and interquartile range "
        f"({'higher' if maximize else 'lower'} is better), and the Wilcoxon rank-sum test at "
        f"{ALPHA}{CORRECTION_NOTES.get(correction, '')} against {_escape(pivot)}, the last "
        "column: + it is significantly better, - "
        "significantly worse, = the difference is not significant. Dark and light gray: the best "
        "and the second-best median of each problem."
    )
    return _document(
        caption,
        "l|" + "c|" * (len(contenders) - 1) + "c",
        " & ".join(["", *(_escape(c) for c in contenders)]) + r" \\ \hline",
        rows + [count_row + r" \\"],
        f"tab:wilcoxon-pivot:{indicator}",
    )


def ranking_table(holm: pd.DataFrame, ranks: pd.Series, pivot: str, indicator: str) -> str:
    """The average ranks over the problems and Holm's comparison with the pivot, as LaTeX.

    Args:
        holm: What `validation_stats.holm_against_pivot` returns.
        ranks: What `validation_stats.average_ranks` returns.
        pivot: The pivot.
        indicator: The indicator the algorithms are ranked by.

    Returns:
        The LaTeX document.
    """
    by_contender = holm.set_index("contender") if not holm.empty else pd.DataFrame()
    rows = []
    for contender, rank in ranks.items():
        if contender == pivot or contender not in by_contender.index:
            cells = [_escape(contender), f"{rank:.2f}", "--", "--"]
        else:
            row = by_contender.loc[contender]
            adjusted = f"{row['adjusted_p_value']:.2e}"
            cells = [
                _escape(contender),
                f"{rank:.2f}",
                f"{row['p_value']:.2e}",
                rf"\textbf{{{adjusted}}}" if row["significant"] else adjusted,
            ]
        rows.append(" & ".join(cells) + r" \\")
    caption = (
        f"{_escape(indicator)}. Average rank of each algorithm over the problems (1 is the best), "
        f"and the p-value of its difference with {_escape(pivot)}, adjusted with Holm's "
        f"procedure; in bold, significant at {ALPHA}."
    )
    return _document(
        caption,
        "l|c|c|c",
        r"Algorithm & Average rank & $p$-value & Holm $p$-value \\ \hline",
        rows,
        f"tab:ranking:{indicator}",
    )


def _top_two(
    median_row: pd.Series, iqr_row: pd.Series, maximize: bool
) -> tuple[str | None, str | None]:
    """The best and the second-best median, as SAES picks them (the IQR settles a tie)."""
    values = median_row.dropna()
    if values.empty:
        return None, None
    ordered = values.sort_values(ascending=not maximize, kind="stable")
    if len(ordered) > 1 and ordered.iloc[0] == ordered.iloc[1]:
        ordered = iqr_row[ordered.index].sort_values(kind="stable")
    return ordered.index[0], ordered.index[1] if len(ordered) > 1 else None


def _count_cell(count: list[int]) -> str:
    return " / ".join(rf"\textbf{{{value}}}" for value in count)


def _document(caption: str, columns: str, header: str, rows: list[str], label: str) -> str:
    lines = [
        PREAMBLE.rstrip("\n"),
        r"\begin{table}[H]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\vspace{1mm}",
        r"\centering",
        r"\begin{scriptsize}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\hline",
        header,
        *rows,
        r"\hline",
        r"\end{tabular}",
        r"\end{scriptsize}",
        r"\end{table}",
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    return "".join(_SPECIAL_CHARACTERS.get(character, character) for character in str(text))
