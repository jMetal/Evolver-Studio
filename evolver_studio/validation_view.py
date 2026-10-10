"""The tabs of the analysis of a validation study, for the Validation analysis page.

Each answers a question for deciding whether to use the tuned configuration (the pivot), and
where: does it win (Verdict), on which problems (Wilcoxon), by how much (Effect size), over all the
problems at once (Ranking), how stable it is (Distributions), what its fronts look like (Fronts),
what it costs (Cost), and how the study was made (Runs & details). Every one follows the direction
of the indicator chosen (`catalogue.is_maximized`).
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from evolver_studio.catalogue import is_maximized
from evolver_studio.figure_export import render_chart
from evolver_studio.latex_tables import ranking_table, wilcoxon_pivot_table
from evolver_studio.saes_export import saes_metrics, saes_results
from evolver_studio.validation import (
    StudyInfo,
    read_study_front,
    read_study_reference_front,
)
from evolver_studio.validation_figures import (
    boxplot_figure,
    cost_figure,
    critical_difference_figure,
    effect_size_figure,
    fronts_figure,
    parallel_fronts_figures,
)
from evolver_studio.validation_stats import (
    ALPHA,
    BEST_RUN,
    MEDIAN_RUN,
    WORST_RUN,
    average_ranks,
    chosen_run,
    compare_with_pivot,
    critical_difference,
    friedman_test,
    holm_against_pivot,
    indicator_names,
    interquartile_ranges,
    medians,
    nonsignificant_groups,
    verdict_counts,
)

TAB_NAMES = (
    "Verdict",
    "Wilcoxon",
    "Effect size",
    "Ranking",
    "Distributions",
    "Fronts",
    "Cost",
    "Runs & details",
)
# Below this many problems, Friedman's test and the critical difference have little power.
FEW_PROBLEMS = 5
RUN_CHOICES = {
    MEDIAN_RUN: "Median run",
    BEST_RUN: "Best run",
    WORST_RUN: "Worst run",
}


def direction(indicator: str) -> str:
    """ "higher is better" or "lower is better"."""
    return "higher is better" if is_maximized(indicator) else "lower is better"


def render_verdict(runs: pd.DataFrame, pivot: str) -> None:
    """Whether the pivot wins: its wins, ties and losses against each algorithm, per indicator."""
    st.markdown(
        f"How often **{pivot}** is significantly better (+), not different (=) or worse (-) "
        f"than each algorithm, over the problems: a Wilcoxon rank-sum test at {ALPHA} on each "
        "problem."
    )
    columns = {}
    sentences = []
    for indicator in indicator_names(runs):
        counts = verdict_counts(compare_with_pivot(runs, indicator, pivot))
        if counts.empty:
            continue
        columns[f"{indicator} (+ / = / -)"] = counts.apply(
            lambda row: f"{row['better']} / {row['equal']} / {row['worse']}", axis=1
        )
        total = int(counts.to_numpy().sum())
        sentences.append(
            f"- **{indicator}** ({direction(indicator)}): better in {int(counts['better'].sum())}, "
            f"worse in {int(counts['worse'].sum())} of {total} comparisons."
        )
    if not columns:
        st.info("There is nothing to compare yet.")
        return
    st.dataframe(pd.DataFrame(columns), width="stretch")
    st.markdown("\n".join(sentences))


def render_wilcoxon(runs: pd.DataFrame, indicator: str, pivot: str, study_id: str) -> None:
    """The Wilcoxon pivot table, on screen and as LaTeX, and the detail of each comparison."""
    maximize = is_maximized(indicator)
    contenders = [c for c in dict.fromkeys(runs["contender"]) if c != pivot] + [pivot]
    median_table = medians(runs, indicator).reindex(columns=contenders)
    iqr_table = interquartile_ranges(runs, indicator).reindex(columns=contenders)
    comparison = compare_with_pivot(runs, indicator, pivot)
    if comparison.empty:
        st.info("There is nothing to compare yet.")
        return
    marks = {"pivot better": "+", "pivot worse": "-", "no significant difference": "="}
    verdicts = {(r.problem, r.contender): marks[r.verdict] for r in comparison.itertuples()}
    shown = median_table.copy().astype(object)
    for problem in median_table.index:
        for contender in contenders:
            median = median_table.loc[problem, contender]
            if pd.isna(median):
                shown.loc[problem, contender] = "—"
                continue
            mark = verdicts.get((problem, contender), "")
            shown.loc[problem, contender] = (
                f"{median:.3e} ({iqr_table.loc[problem, contender]:.1e}) {mark}".rstrip()
            )
    st.markdown(
        f"Median and interquartile range of **{indicator}** ({direction(indicator)}); "
        f"**{pivot}** in the last column, the others marked against it: + it is significantly "
        "better, - significantly worse, = the difference is not significant. The best and "
        "second-best median of each problem are shaded."
    )
    st.dataframe(
        shown.style.apply(lambda row: _top_two_shades(median_table.loc[row.name], maximize), 1),
        width="stretch",
    )
    st.download_button(
        "Download the table (LaTeX)",
        lambda: wilcoxon_pivot_table(runs, indicator, pivot),
        file_name=f"WilcoxonPivot_{indicator}.tex",
        mime="application/x-tex",
        key=f"validation_analysis_wilcoxon_latex_{study_id}_{indicator}",
        on_click="ignore",
        icon=":material/download:",
        help="The Wilcoxon pivot table in the layout of SAES's (and of Evolver's "
        "scripts/wilcoxon_pivot_tables.py), as a LaTeX document that compiles on its own.",
    )
    with st.expander("Each comparison: medians, p-value and A12"):
        detail = comparison.rename(
            columns={
                "problem": "Problem",
                "contender": "Algorithm",
                "pivot_median": "Pivot median",
                "median": "Median",
                "p_value": "p-value",
                "a12": "A12",
                "magnitude": "Effect",
                "verdict": "Verdict",
            }
        )
        st.dataframe(detail, hide_index=True, width="stretch")
        st.download_button(
            "Download the comparisons (CSV)",
            detail.to_csv(index=False),
            file_name=f"comparison_{indicator}.csv",
            mime="text/csv",
            key=f"validation_analysis_comparison_{study_id}_{indicator}",
            on_click="ignore",
        )


def render_effect_size(runs: pd.DataFrame, indicator: str, pivot: str, study_id: str) -> None:
    """How large the differences are: the A12 of the pivot against each algorithm."""
    comparison = compare_with_pivot(runs, indicator, pivot)
    if comparison.empty:
        st.info("There is nothing to compare yet.")
        return
    st.markdown(
        f"**A12** is the probability that a run of **{pivot}** is better than a run of the other "
        "algorithm on the same problem: 0.5 is no difference, 1 always better, 0 always worse. "
        "A significant difference can be small: |A12 − 0.5| of 0.06, 0.14 and 0.21 are a small, "
        "a medium and a large effect. Each cell also has the mark of the Wilcoxon test."
    )
    render_chart(
        effect_size_figure(comparison, pivot),
        f"validation_analysis_a12_{study_id}_{indicator}",
        f"a12_{indicator}",
        width="stretch",
    )


def render_ranking(runs: pd.DataFrame, indicator: str, pivot: str, study_id: str) -> None:
    """Who is best over all the problems: average ranks, Friedman, Holm and the CD plot."""
    maximize = is_maximized(indicator)
    median_table = medians(runs, indicator).dropna()
    problems, contenders = median_table.shape
    if problems < 2 or contenders < 2:
        st.info("Ranking needs at least two problems and two algorithms with results.")
        return
    ranks = average_ranks(median_table, maximize)
    st.markdown(
        f"On each problem the algorithms are ranked by their median {indicator} (1 is the "
        "best), and the ranks are averaged over the problems, as Demšar proposes (J. Mach. "
        "Learn. Res. 7, 2006)."
    )
    if problems < FEW_PROBLEMS:
        st.warning(
            f"Only {problems} problems: with so few, these tests can rarely tell the algorithms "
            "apart, and a difference they miss may still exist."
        )
    friedman = friedman_test(median_table)
    if friedman is None:
        st.caption("Friedman's test needs at least three algorithms.")
    else:
        statistic, p_value = friedman
        verdict = "differ" if p_value < ALPHA else "do not differ significantly"
        st.markdown(
            f"**Friedman's test**: χ² = {statistic:.2f}, p = {p_value:.2e}: the algorithms "
            f"{verdict} at {ALPHA}."
        )
    difference = critical_difference(contenders, problems)
    st.markdown(
        f"**Critical difference plot**: two algorithms joined by a bar are not significantly "
        f"different (Nemenyi's test at {ALPHA}: their average ranks differ by less than "
        f"CD = {difference:.2f})."
    )
    render_chart(
        critical_difference_figure(ranks, difference, nonsignificant_groups(ranks, difference)),
        f"validation_analysis_cd_{study_id}_{indicator}",
        f"cd_{indicator}",
        width="stretch",
    )
    holm = holm_against_pivot(ranks, pivot, problems)
    st.markdown(
        f"**{pivot} against each algorithm** (Holm's procedure, which adjusts the p-values for "
        "making several comparisons):"
    )
    table = holm.rename(
        columns={
            "contender": "Algorithm",
            "rank": "Average rank",
            "p_value": "p-value",
            "adjusted_p_value": "Holm p-value",
            "significant": f"Significant at {ALPHA}",
        }
    ).drop(columns="z")
    pivot_row = pd.DataFrame([{"Algorithm": f"{pivot} (pivot)", "Average rank": ranks[pivot]}])
    st.dataframe(pd.concat([pivot_row, table], ignore_index=True), hide_index=True, width="stretch")
    st.download_button(
        "Download the ranking (LaTeX)",
        lambda: ranking_table(holm, ranks, pivot, indicator),
        file_name=f"Ranking_{indicator}.tex",
        mime="application/x-tex",
        key=f"validation_analysis_ranking_latex_{study_id}_{indicator}",
        on_click="ignore",
        icon=":material/download:",
    )


def render_distributions(runs: pd.DataFrame, indicator: str, study_id: str) -> None:
    """How stable each algorithm is: the values of the indicator over the runs, per problem."""
    st.markdown(f"The values of **{indicator}** over the runs ({direction(indicator)}).")
    render_chart(
        boxplot_figure(runs, indicator),
        f"validation_analysis_boxplot_{study_id}_{indicator}",
        f"boxplots_{indicator}",
        width="stretch",
    )


def render_fronts(
    study: StudyInfo, runs: pd.DataFrame, indicator: str, working_directory: Path
) -> None:
    """The fronts of chosen runs of each algorithm on a problem, over its reference front."""
    key = f"validation_analysis_fronts_{study.study_id}"
    problems = list(dict.fromkeys(runs["problem"]))
    contenders = list(dict.fromkeys(runs["contender"]))
    columns = st.columns([2, 2, 3])
    problem = columns[0].selectbox("Problem", problems, key=f"{key}_problem")
    which = columns[1].radio(
        "Run of each algorithm",
        list(RUN_CHOICES),
        format_func=RUN_CHOICES.get,
        key=f"{key}_which",
        help=f"Chosen by {indicator}, the indicator of the page: the median run is the one a "
        "paper usually shows (with an even number of runs, the better of the two middle ones).",
    )
    shown = columns[2].multiselect("Algorithms", contenders, default=contenders, key=f"{key}_shown")
    fronts = {}
    for contender in shown:
        run = chosen_run(runs, indicator, problem, contender, which)
        if run is None:
            continue
        front = read_study_front(study.directory, study.manifest, contender, problem, run)
        if front is None:
            continue
        value = runs.loc[
            (runs["problem"] == problem) & (runs["contender"] == contender) & (runs["Run"] == run),
            indicator,
        ].iloc[0]
        fronts[f"{contender} · run {run} ({indicator} = {value:.3g})"] = front
    if not fronts:
        st.info("No front to show: the runs of these algorithms kept none.")
        return
    reference = read_study_reference_front(study.manifest, problem, working_directory)
    objectives = len(next(iter(fronts.values())).columns)
    if objectives == 3:
        st.caption("Drag to rotate; click an algorithm in the legend to hide or show it.")
    elif objectives >= 4:
        st.caption(
            f"{objectives} objectives: a chart per algorithm, its front in color over the "
            "reference front in gray, each objective normalized with the reference front's "
            "bounds (0 its lowest value, 1 its highest), on the same scale in every chart."
        )
    if objectives < 4:
        render_chart(
            fronts_figure(fronts, reference),
            f"{key}_chart_{problem}_{which}",
            f"fronts_{problem}_{which}",
            width="stretch",
        )
        return
    for index, figure in enumerate(parallel_fronts_figures(fronts, reference)):
        render_chart(
            figure,
            f"{key}_chart_{problem}_{which}_{index}",
            f"fronts_{problem}_{which}_{index + 1}",
            width="stretch",
        )


def render_cost(runs: pd.DataFrame, study_id: str) -> None:
    """What each algorithm costs: the median computing time of a run, per problem."""
    st.markdown(
        "The median computing time of a run of each algorithm on each problem: a configuration "
        "can be better and much slower. The times are those of the machine the study ran on, "
        "with several runs at a time."
    )
    render_chart(cost_figure(runs), f"validation_analysis_cost_{study_id}", "cost", width="stretch")
    table = runs.groupby(["problem", "contender"], sort=False)["TimeMs"].median().unstack()
    st.dataframe((table / 1000).style.format("{:.2f} s"), width="stretch")


def render_runs_and_details(study: StudyInfo, runs: pd.DataFrame) -> None:
    """How the study was made, and every run, to download."""
    manifest = study.manifest
    st.markdown(
        f"**Budget**: population {manifest['populationSize']}, {manifest['maxEvaluations']:,} "
        f"evaluations, {manifest['runs']} runs from the seed {manifest['seed']}, "
        f"{manifest['encoding']} problems. Folder: `{study.directory}`."
    )
    st.dataframe(pd.DataFrame(manifest["contenders"]), hide_index=True, width="stretch")
    if manifest.get("skipped"):
        st.warning(
            "Left out: "
            + "; ".join(
                f"{s['contender']} on {s['problem']} ({s['reason']})" for s in manifest["skipped"]
            )
        )
    failed = failed_jobs(study.directory)
    if failed:
        st.error(
            f"Jobs that failed: {', '.join(map(str, failed))}. See `runner.log` in their folders."
        )
    st.markdown("**Every run**")
    st.dataframe(runs, hide_index=True, width="stretch")
    downloads = st.columns(4)
    downloads[0].download_button(
        "Every run (CSV)",
        runs.to_csv(index=False),
        file_name=f"{study.study_id}_runs.csv",
        mime="text/csv",
        key=f"validation_analysis_runs_{study.study_id}",
        on_click="ignore",
    )
    downloads[1].download_button(
        "Results for SAES (CSV)",
        saes_results(runs).to_csv(index=False),
        file_name=f"{study.study_id}_saes_results.csv",
        mime="text/csv",
        key=f"validation_analysis_saes_results_{study.study_id}",
        on_click="ignore",
        help="One row per run and indicator, the file SAES reads with -ds.",
    )
    downloads[2].download_button(
        "Metrics for SAES (CSV)",
        saes_metrics(runs).to_csv(index=False),
        file_name=f"{study.study_id}_saes_metrics.csv",
        mime="text/csv",
        key=f"validation_analysis_saes_metrics_{study.study_id}",
        on_click="ignore",
        help="Whether each indicator is maximized, the file SAES reads with -ms.",
    )
    downloads[3].download_button(
        "study.yaml",
        (study.directory / "study.yaml").read_text(),
        file_name="study.yaml",
        mime="application/x-yaml",
        key=f"validation_analysis_manifest_{study.study_id}",
        on_click="ignore",
    )


def failed_jobs(study_directory: Path) -> list[int]:
    """The numbers of the jobs of a study that failed."""
    try:
        return [int(n) for n in (study_directory / "failed_jobs.txt").read_text().split()]
    except (OSError, ValueError):
        return []


def _top_two_shades(median_row: pd.Series, maximize: bool) -> list[str]:
    values = median_row.dropna().sort_values(ascending=not maximize, kind="stable")
    shades = dict(
        zip(
            values.index[:2],
            (
                "background-color: rgba(128, 128, 128, 0.45)",
                "background-color: rgba(128, 128, 128, 0.2)",
            ),
            strict=False,
        )
    )
    return [shades.get(contender, "") for contender in median_row.index]
