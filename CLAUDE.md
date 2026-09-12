# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Evolver-Studio is a **Python/Streamlit GUI and analysis tool for [Evolver](https://github.com/jMetal/Evolver)**,
a Java framework for automated meta-optimization of multi-objective metaheuristics. Evolver itself has no
GUI and no uniform CLI/API: base-level algorithms are configured via YAML parameter spaces, runs are
launched through ad hoc Java `main()` classes, and results are written as CSV/text files. Evolver-Studio's
job is to make Evolver usable without writing Java or hand-editing YAML, and to add a statistical-analysis
layer (Wilcoxon tests, plots, comparison tables) on top of the raw results — leveraging the Python
scientific stack (`pandas`, `scipy.stats`, `seaborn`/`plotly`) and, where useful, `jMetalPy`.

This is a **new, currently empty project** — this file documents the intended architecture agreed on
before implementation started, to be corrected/expanded as real code lands.

Evolver lives in a sibling repository: `/Users/ajnebro/Softw/Evolver` (or wherever it's checked out
locally — do not assume a fixed absolute path in code, make it configurable).

## Relationship to Evolver — status

Evolver does **not** yet expose a stable integration surface. Before (or alongside) building
Evolver-Studio's integration layer, Evolver needs:

- A uniform way to launch a training/validation run with structured parameters (today every example
  `main()` parses `args` differently, or ignores them and hardcodes everything).
- A single canonical output format for meta-optimization results (today `OutputResults` and
  `ConsolidatedOutputResults` coexist with different file layouts — `ConsolidatedOutputResults`
  `METADATA.txt`/`INDICATORS.csv`/`CONFIGURATIONS.csv` is the better candidate to standardize on).

Until that uniformization happens on the Evolver side, Evolver-Studio's integration code should be
treated as provisional and isolated behind a thin adapter layer (see Architecture below) so it can be
updated without touching the UI or analysis code.

## Planned Architecture

```
Streamlit UI (this repo)
  ├─> Discovery: list algorithms / parameter spaces / training sets / indicators
  │     (reads Evolver's YAML parameter spaces directly, or via a small Java helper — TBD)
  ├─> Run control: launch training / validation runs
  │     -> writes a request.json, invokes the Evolver JAR as a subprocess
  │     -> polls a status.json for progress (RUNNING/FINISHED/FAILED, evaluations done/total)
  │     -> reads a results.json (or parses INDICATORS.csv/CONFIGURATIONS.csv) once finished
  └─> Analysis: statistical comparison of runs/configurations
        (pandas + scipy.stats for Wilcoxon/other tests, seaborn/plotly for plots,
         optionally jMetalPy's own comparison utilities)
```

No REST server is planned: meta-optimization runs are long, batch-style jobs, not interactive
request/response calls, so a **file-based request/status/results contract with a subprocess** is
preferred over a persistent HTTP server (simpler to debug, nothing to keep alive).

### MVP operations (agreed scope, subject to change once implementation starts)

1. `list_algorithms()` — configurable base-level algorithms + supported encodings.
2. `get_parameter_space(algorithm, encoding)` — structured parameter tree (not raw YAML) to render forms.
3. `list_training_sets()` / `list_indicators()` — catalogue of training problems and quality indicators.
4. `start_training(request)` / `start_validation(request)` — async, returns a run id.
5. `get_run_status(run_id)` — polled progress.
6. `cancel_run(run_id)`.
7. `get_results(run_id)` — parsed configurations + indicator values, ready for the analysis layer.

## Tech Stack (decided)

- **UI**: [Streamlit](https://streamlit.io/) — chosen over Panel/Dash for fastest time-to-usable-tool;
  revisit **Panel** if/when live progress updates during long runs outgrow Streamlit's rerun model.
- **Analysis**: `pandas`, `scipy.stats`, `seaborn` and/or `plotly`; consider reusing `jMetalPy` utilities
  where they already implement what's needed (e.g. quality indicators, statistical comparison) instead
  of reimplementing them.
- **Integration with Evolver**: subprocess + JSON files, not a REST API (see Architecture above).

## Build and Test Commands

Not yet established — this is a placeholder. Once the project has real code, document here the actual
commands (expected to be a `pyproject.toml`/`requirements.txt`-based Python project, e.g. `uv`/`pip`,
`pytest` for tests, `streamlit run app.py` to launch locally).

## Conventions

Not yet established. Follow standard Python conventions (PEP 8, type hints) until a project-specific
style guide is written here.
