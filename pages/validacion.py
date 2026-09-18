"""Validación: run a tuned configuration against a validation set. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

st.title("Validación")

render_sidebar()

st.info(
    "🔍 No implementado todavía — ver ROADMAP.md, Fase 4.\n\n"
    "Previsto: lanzar la configuración ganadora de un entrenamiento contra un conjunto de "
    "validación y comparar la distribución de indicadores frente a la configuración por "
    "defecto (sin ajustar)."
)
