"""Validation analysis: study the results of a validation, the last step of meta-optimization.

Pick a study kept under validation-runs/ and an indicator, and read it tab by tab: whether the
tuned configuration (the pivot) wins, on which problems, by how much, over all the problems at
once, how stable it is, what its fronts look like, what it costs and how the study was made.
Tables and charts are saved as LaTeX, PNG or PDF, and the runs as the files SAES reads.
"""

import streamlit as st

from evolver_studio import validation_view
from evolver_studio.app_state import require_evolver_jar
from evolver_studio.evolver_client import WORKING_DIRECTORY
from evolver_studio.runs import RunPhase, run_phase
from evolver_studio.validation import collect_runs, list_studies
from evolver_studio.validation_runner import VALIDATION_RUNS_DIRECTORY_NAME
from evolver_studio.validation_stats import indicator_names

# Set by Validation when a study finishes, to open it here.
STUDY_KEY = "validation_analysis_study"

st.title("Validation analysis")

require_evolver_jar()

studies = {
    study.study_id: study
    for study in list_studies(WORKING_DIRECTORY / VALIDATION_RUNS_DIRECTORY_NAME)
}
if not studies:
    st.info(
        "There is no validation study to analyze: the ones you run in Validation are kept and "
        "listed here."
    )
    st.page_link("pages/validation.py", label="Go to Validation", icon="✅")
    st.stop()

if st.session_state.get(STUDY_KEY) not in studies:
    st.session_state[STUDY_KEY] = next(iter(studies))
study_id = st.selectbox(
    "Study", list(studies), format_func=lambda s: studies[s].label, key=STUDY_KEY
)
study = studies[study_id]
phase = run_phase(study.directory)
if phase == RunPhase.CANCELLED:
    st.info("This study was cancelled: what follows is what had finished.")
elif phase in (RunPhase.STARTING, RunPhase.RUNNING):
    st.info("This study is still in progress: what follows is what has finished so far.")

runs = collect_runs(study.directory, WORKING_DIRECTORY)
if runs.empty:
    st.info("No job of this study has finished yet.")
    st.stop()

pivot = study.manifest["pivot"]
indicators = indicator_names(runs)
indicator = st.selectbox(
    "Indicator",
    indicators,
    key=f"validation_analysis_indicator_{study_id}",
    help="The one every tab but Verdict uses; it also chooses the run shown in Fronts.",
)
st.caption(
    f"{runs['problem'].nunique()} problems · {runs['contender'].nunique()} algorithms · "
    f"pivot **{pivot}** · {indicator}: {validation_view.direction(indicator)}"
)

tabs = st.tabs(list(validation_view.TAB_NAMES))
with tabs[0]:
    validation_view.render_verdict(runs, pivot)
with tabs[1]:
    validation_view.render_wilcoxon(runs, indicator, pivot, study_id)
with tabs[2]:
    validation_view.render_effect_size(runs, indicator, pivot, study_id)
with tabs[3]:
    validation_view.render_ranking(runs, indicator, pivot, study_id)
with tabs[4]:
    validation_view.render_distributions(runs, indicator, study_id)
with tabs[5]:
    validation_view.render_fronts(study, runs, indicator, WORKING_DIRECTORY)
with tabs[6]:
    validation_view.render_cost(runs, study_id)
with tabs[7]:
    validation_view.render_runs_and_details(study, runs)
