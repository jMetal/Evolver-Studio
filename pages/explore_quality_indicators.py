"""Explore › Quality indicators: the indicators a training run can minimize.

Read from evolver_studio/catalogue.py's QUALITY_INDICATORS, which tests/test_catalogue.py checks
against Evolver's DescribeMain manifest, so the page needs no Java run.
"""

import streamlit as st

from evolver_studio.app_state import render_sidebar
from evolver_studio.catalogue import QUALITY_INDICATORS

st.title("Quality indicators")

render_sidebar()

st.markdown(
    "A training run measures every configuration it evaluates with **two** of these indicators, "
    "which are the objectives the meta-optimizer minimizes; lower values are always better. Each "
    "front is normalized with the reference front of its problem, so every training problem "
    "needs a reference front file. For `HypervolumeMinus` an approximate one is enough, since "
    "only its bounds are used (to normalize and to place the reference point); the others also "
    "use its points. A validation measures its runs with them too, all but `HypervolumeMinus`, "
    "which only exists so that a training can minimize the hypervolume, and with `Hypervolume`, "
    "which Evolver does not compute: Evolver-Studio computes it from each run's front, the same "
    "way (*In Validation*)."
)
# A markdown table, not st.dataframe: the descriptions are the point, and it wraps them instead of
# truncating them.
rows = "\n".join(
    f"| `{indicator.registry_name}` | {indicator.short_name} | {indicator.full_name} | "
    f"{'higher' if indicator.maximized else 'lower'} | "
    f"{'yes' if indicator.in_validation else 'no'} | {indicator.measures} |"
    for indicator in QUALITY_INDICATORS
)
st.markdown(
    "| Name in a request | Abbreviation | Indicator | Better | In Validation | What it measures |\n"
    "|---|---|---|---|---|---|\n" + rows
)
