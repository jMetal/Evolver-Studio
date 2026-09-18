"""Análisis: statistical comparison of training results. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

st.title("Análisis")

render_sidebar()

st.info(
    "🔍 No implementado todavía — ver ROADMAP.md, Fase 3.\n\n"
    "Previsto: parseo estructurado de CONFIGURATIONS.csv/VAR_CONF.txt, historial de runs "
    "pasados (no solo el último), y comparación estadística entre runs/configuraciones "
    "(Wilcoxon, tablas comparativas)."
)
