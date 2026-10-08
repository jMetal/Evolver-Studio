"""Analysis: study a finished training run and take what it found to where it is used.

Pick a training run kept under cli-runner-runs/ and see how it was run and what it took, how its
meta-objectives converged, how its population evolved, the configurations of its final front and
what they have in common (and where that differs from the algorithm's default configuration). A
configuration goes from here to Validation, to be compared with other algorithms, or to Run
algorithm, to be run on a problem.
"""

import streamlit as st

from evolver_studio.analysis_view import (
    monitor_state,
    render_configurations,
    render_parameters,
    render_summary,
)
from evolver_studio.app_state import require_evolver_jar
from evolver_studio.evolver_client import WORKING_DIRECTORY
from evolver_studio.monitor_view import (
    render_convergence,
    render_indicator_front,
    render_population,
)
from evolver_studio.training_runs import list_finished_trainings

SELECTION_KEY = "analysis_training"

st.title("Analysis")

jar = require_evolver_jar()

trainings = list_finished_trainings(WORKING_DIRECTORY)
if not trainings:
    st.info(
        "There is no finished training run to analyze: the ones you run in Training are kept "
        "and listed here."
    )
    st.page_link("pages/training.py", label="Go to Training", icon="🏋️")
    st.stop()

training = st.selectbox("Training run", trainings, format_func=lambda t: t.label, key=SELECTION_KEY)
render_summary(training)
state = monitor_state(training)
names = ["Convergence", "Front", *(["Population"] if training.has_population else [])]
names += ["Configurations", "Parameters"]
tabs = iter(st.tabs(names))
key = f"analysis_{training.run_id}"
with next(tabs):
    render_convergence(state, f"{key}_convergence")
with next(tabs):
    render_indicator_front(training.output_directory / "INDICATORS.csv", training.run_id)
if training.has_population:
    with next(tabs):
        render_population(state, f"{key}_population")
with next(tabs):
    render_configurations(training, jar)
with next(tabs):
    render_parameters(training, jar)
