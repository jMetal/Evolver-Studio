"""Tutorial "Analyzing a validation": its steps.

The last step of meta-optimization. Tutorial "Validating a configuration" runs a study (the
NSGA-II that Evolver's tutorial E8 tuned for the WFG problems against three algorithms, on four
problems) and reads its first results; this one reads that same study the way the Validation
analysis page does, tab by tab, with the page's own views: the verdict and the correction for many
comparisons, the Wilcoxon pivot table, the size of the effects, the ranking over the problems, a
Bayesian test, the fronts and where in the objective space each algorithm reaches, whether the
indicators agree, what each algorithm costs, and what to put in a paper. It pairs with Evolver's
tutorial E8, whose analysis it does without scripts.

What the text says about the results is read from them, not written down.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from evolver_studio import evolver_client, validation_view
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorial_validation import (
    PIVOT_NAME,
    STUDY_ID,
    runs_or_ask_to_run,
    study_directory,
)
from evolver_studio.tutorials import TutorialStep
from evolver_studio.validation import StudyInfo, read_manifest
from evolver_studio.validation_stats import (
    BENJAMINI_HOCHBERG,
    HOLM,
    NO_CORRECTION,
    PIVOT_BETTER,
    PIVOT_WORSE,
    bayesian_signed_rank,
    compare_with_pivot,
    friedman_test,
    indicator_names,
    medians,
)

E8_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/validating_a_configuration.rst"
)
KEY_PREFIX = "tutorial_validation_analysis"
# Validation analysis's selector of the study, to open the tutorial's study there.
ANALYSIS_STUDY_KEY = "validation_analysis_study"


def _study(jar: Path) -> tuple[StudyInfo, pd.DataFrame] | None:
    """The tutorial's study and its runs, or the controls to run it when it is not there."""
    runs = runs_or_ask_to_run(jar)
    if runs is None:
        return None
    directory = study_directory()
    manifest = read_manifest(directory)
    if manifest is None:
        return None
    return StudyInfo(STUDY_ID, directory, manifest), runs


def _indicator(runs: pd.DataFrame, step: str) -> str:
    return st.selectbox(
        "Indicator",
        indicator_names(runs),
        key=f"{KEY_PREFIX}_indicator_{step}",
        help="The page has one selector for every tab; here, one per step.",
    )


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "A validation study is many numbers: every algorithm run many times on every problem, "
        "measured with several indicators. **Validation analysis** turns them into the answers "
        "a decision needs, one tab per question:\n\n"
        "- **Verdict** and **Wilcoxon**: does the tuned configuration (the *pivot*) win, and on "
        "which problems?\n"
        "- **Effect size**: by how much?\n"
        "- **Ranking** and **Bayesian**: over all the problems at once, and how sure?\n"
        "- **Distributions**, **Fronts** and **Attainment**: how stable it is, what its fronts "
        "look like, and *where* in the objective space it reaches more often.\n"
        "- **Indicators** and **Cost**: whether the indicators agree, and what each run takes.\n"
        "- **Runs & details**: the study itself, and its data for SAES.\n\n"
        "This tutorial reads, with the views of that page, the study of the tutorial "
        f"*Validating a configuration*: **{PIVOT_NAME}**, the NSGA-II that Evolver's tutorial "
        f"[E8]({E8_TUTORIAL_URL}) tuned for the WFG problems, against the default NSGA-II, "
        "MOEA/D and RVEA on four problems. If you have not run it, run it here (about half a "
        "minute)."
    )
    if _study(jar) is not None:
        st.success("The study is there: go on to the first step.")


def _render_verdict(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    _, runs = found
    st.markdown("The first question is whether the pivot wins: the **Verdict** tab.")
    validation_view.render_verdict(runs, PIVOT_NAME, NO_CORRECTION)
    counts = {}
    for correction in (NO_CORRECTION, HOLM, BENJAMINI_HOCHBERG):
        verdicts = pd.concat(
            [
                compare_with_pivot(runs, indicator, PIVOT_NAME, correction=correction)["verdict"]
                for indicator in indicator_names(runs)
            ]
        )
        counts[correction] = (
            int(verdicts.isin([PIVOT_BETTER, PIVOT_WORSE]).sum()),
            len(verdicts),
        )
    found_none, total = counts[NO_CORRECTION]
    st.markdown(
        f"That is **{total} tests** at the 5% level, and each can find a difference by chance. "
        "The page's **Correction for many comparisons** adjusts the p-values for being many: "
        f"without it, {found_none} of the {total} comparisons are significant; with "
        f"**Holm's** procedure (no false difference, with 95% confidence), {counts[HOLM][0]}; "
        f"with **Benjamini-Hochberg** (at most 5% of the differences found are false), "
        f"{counts[BENJAMINI_HOCHBERG][0]}. A difference that survives Holm is a solid one; "
        "one that only appears without correction deserves caution."
    )


def _render_wilcoxon(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    _, runs = found
    indicator = _indicator(runs, "wilcoxon")
    st.markdown(
        "The **Wilcoxon** tab is the table a paper shows: the median and interquartile range "
        "of each algorithm on each problem, the pivot in the last column, every other cell "
        "marked against it, and the best two medians of each problem shaded. Its LaTeX has "
        "the layout of SAES's tables (and of Evolver's `scripts/wilcoxon_pivot_tables.py`), and "
        "compiles on its own:"
    )
    validation_view.render_wilcoxon(runs, indicator, PIVOT_NAME, STUDY_ID, NO_CORRECTION)


def _render_effect_size(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    _, runs = found
    indicator = _indicator(runs, "effect")
    comparison = compare_with_pivot(runs, indicator, PIVOT_NAME)
    validation_view.render_effect_size(runs, indicator, PIVOT_NAME, STUDY_ID, NO_CORRECTION)
    if comparison.empty:
        return
    strongest = comparison.loc[(comparison["a12"] - 0.5).abs().idxmax()]
    significant_small = comparison[
        comparison["verdict"].isin([PIVOT_BETTER, PIVOT_WORSE])
        & comparison["magnitude"].isin(["negligible", "small"])
    ]
    st.markdown(
        f"On {indicator}, the strongest effect is on **{strongest['problem']}** against "
        f"**{strongest['contender']}**: A12 = {strongest['a12']:.2f}, a "
        f"{strongest['magnitude']} effect. "
        + (
            f"{len(significant_small)} significant difference(s) are only small or negligible: "
            "significant does not mean large, and a paper should say both."
            if len(significant_small)
            else "Every significant difference here is at least medium."
        )
        + " For a paper printed in black and white, choose **Grays**."
    )


def _render_ranking(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    _, runs = found
    indicator = _indicator(runs, "ranking")
    validation_view.render_ranking(runs, indicator, PIVOT_NAME, STUDY_ID)
    result = friedman_test(medians(runs, indicator).dropna())
    problems = runs["problem"].nunique()
    st.markdown(
        f"This study has only **{problems} problems**: the tests over the problems treat each "
        "problem as one observation, so they have little to go on. "
        + (
            f"Friedman's p-value is {result[1]:.3f}"
            + (
                ", significant even so."
                if result[1] < 0.05
                else ", not significant: the ranking may still be right, but this study cannot "
                "show it."
            )
            if result is not None
            else ""
        )
        + f" Wilcoxon's signed-rank test over {problems} problems cannot be significant at all: "
        f"the smallest p-value it can give is 2/2^{problems} = {2 / 2**problems:.3f}, even when "
        "the pivot wins on every problem. A study meant to rank algorithms needs more problems "
        "(Demšar suggests ten or more); "
        "one meant to check a configuration on a family of problems leans on the per-problem "
        "tests of the previous steps."
    )


def _render_bayesian(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    _, runs = found
    indicator = _indicator(runs, "bayesian")
    validation_view.render_bayesian(runs, indicator, PIVOT_NAME, STUDY_ID)
    table = medians(runs, indicator)
    others = [c for c in dict.fromkeys(runs["contender"]) if c != PIVOT_NAME]
    sure = [
        other
        for other in others
        if (result := bayesian_signed_rank(table, PIVOT_NAME, other, 0.01)) is not None
        and max(result.pivot_better, result.equivalent, result.pivot_worse) > 0.95
    ]
    st.markdown(
        "A p-value says how surprising the data would be if there were no difference; the "
        "Bayesian test answers the question one actually asks: *how likely is it that the pivot "
        "is better?*, and also that the two are **practically equivalent**, which a p-value "
        "cannot say. With the default ROPE (1% of the medians), "
        + (
            f"it is sure (over 95%) against {', '.join(sure)}."
            if sure
            else "it is sure against none of them."
        )
        + " A posterior spread over the triangle means the problems are too few, or disagree."
    )


def _render_fronts_and_attainment(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    study, runs = found
    indicator = _indicator(runs, "fronts")
    st.markdown(
        "Numbers do not show the shape of a front. **Fronts** draws the median run of each "
        "algorithm (by the indicator chosen; the best and the worst runs too) next to the "
        "reference front, a panel each, on the same axes:"
    )
    validation_view.render_fronts(study, runs, indicator, evolver_client.WORKING_DIRECTORY)
    st.markdown(
        "**Attainment** uses every run, not one: the *attainment surfaces* show the points "
        "reached by at least one run, by half of them and by all; and the *differences* of "
        "the attainment functions show where in the objective space the pivot's runs reach "
        "more often than another's (left) and where the other's do (right). An indicator says "
        "who is better; this says *where*: at the extremes of the front, or in the middle."
    )
    validation_view.render_attainment(study, runs, evolver_client.WORKING_DIRECTORY)


def _render_indicators_and_cost(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    _, runs = found
    validation_view.render_indicators(runs, STUDY_ID)
    st.markdown(
        "EP measures how far a front is from the reference front; NHV, how much of it the front "
        "covers. When they agree closely, one is enough to report; when they do not, report "
        "both. And a configuration that wins may cost more:"
    )
    validation_view.render_cost(runs, STUDY_ID)
    times = runs.groupby("contender")["TimeMs"].median()
    if PIVOT_NAME in times and times.drop(PIVOT_NAME).median() > 0:
        ratio = times[PIVOT_NAME] / times.drop(PIVOT_NAME).median()
        st.markdown(
            f"Here a run of {PIVOT_NAME} takes **{ratio:.1f} times** the median time of the "
            "others: the same number of evaluations, but different operators and selection."
        )


def _render_reporting(jar: Path) -> None:
    found = _study(jar)
    if found is None:
        return
    st.markdown(
        "**What to put in a paper**, all of it from Validation analysis:\n\n"
        "- the **Wilcoxon pivot table** in LaTeX (tab *Wilcoxon*), with the correction for many "
        "comparisons you used, which its caption states;\n"
        "- the **effect sizes** (tab *Effect size*, as PNG or PDF, in grays for print);\n"
        "- the **critical difference plot** and the **ranking** table (tab *Ranking*), if the "
        "problems are enough to rank;\n"
        "- the **fronts** or the **attainment** differences of a problem worth showing;\n"
        "- and, for anyone who wants to check it with SAES, the study's **results and metrics "
        "files** (tab *Runs & details*).\n\n"
        "Every chart has PNG and PDF buttons, and every table a LaTeX or CSV one. The study is "
        "listed in Validation analysis like any other:"
    )
    if st.button("Open the study in Validation analysis", icon="🔬", type="primary"):
        st.session_state[ANALYSIS_STUDY_KEY] = STUDY_ID
        st.switch_page("pages/validation_analysis.py")
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("From a study to a decision", _render_introduction),
    TutorialStep("Step 1: the verdict, and many comparisons", _render_verdict),
    TutorialStep("Step 2: the Wilcoxon pivot table", _render_wilcoxon),
    TutorialStep("Step 3: how large are the differences?", _render_effect_size),
    TutorialStep("Step 4: over all the problems", _render_ranking),
    TutorialStep("Step 5: probabilities instead of p-values", _render_bayesian),
    TutorialStep(
        "Step 6: the fronts, and where each algorithm reaches", _render_fronts_and_attainment
    ),
    TutorialStep(
        "Step 7: do the indicators agree, and what does it cost?", _render_indicators_and_cost
    ),
    TutorialStep("Reporting it", _render_reporting),
)
