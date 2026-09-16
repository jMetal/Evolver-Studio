# Roadmap

Where Evolver-Studio is headed after the ZDT4 prototype (`app.py`, `evolver_studio/`), prioritized as
concrete next milestones rather than a fixed schedule. Revisit and reorder as real usage surfaces new
priorities.

## Done so far

- Request/status/results contract against Evolver's `org.uma.evolver.cli.runner.TrainingRunnerMain`
  (`evolver_studio/request.py`, `evolver_client.py`, `results.py`).
- Non-blocking run control: detached subprocess, PID tracking, cancellation, reconnect to an
  in-progress run across page reloads (`evolver_studio/runs.py`, `adaptive_poll.py`).
- Live indicator-front preview: accumulates checkpoints from `INDICATORS.csv`, deduplicates unchanged
  fronts, lets the viewer narrow the plot to the last N checkpoints
  (`evolver_studio/live_front.py`, `slider_state.py`).

This already covers most of `CLAUDE.md`'s MVP operations list (`start_training`, `get_run_status`,
`cancel_run`, `get_results`) for the single hardcoded NSGA-II/ZDT4 case.

## Phase 1 — Validated parameter forms

Today the only editable fields are free text (`output_directory`) plus a couple of `number_input`s;
algorithm, parameter space, and indicators are all hardcoded. This is `CLAUDE.md`'s originally planned
`get_parameter_space(algorithm, encoding)`:

- Parse Evolver's YAML parameter space files (`NSGAIIDouble.yaml`, etc. — the same format
  `YAMLParameterSpace.java` already reads: categorical/integer/double parameters, with nested
  conditional sub-parameters) into a structured Python model.
- Render a Streamlit form dynamically from that model: `st.selectbox` for categorical parameters,
  `st.slider`/`st.number_input` bounded by the YAML's own ranges for numeric ones, recursive rendering
  of conditional sub-parameters based on their parent's value.
- Two switchable modes: **Expert** (raw YAML text, today's behavior) and **Guided** (the generated
  form) — covering both user profiles without duplicating request-building logic.

Self-contained: the YAML parsing can be replicated in Python without touching Evolver.

## Phase 2 — Algorithm/indicator/training-set discovery

- Replace the hardcoded values (`"NSGA-II"`, `["ZDT4"]`, `["Epsilon", "NormalizedHypervolume"]`) with
  real listings: `list_algorithms()`, `list_indicators()`, `list_training_sets()`.
- Evolver's registries (`BaseAlgorithmRegistry`, `IndicatorRegistry`) only exist in Java today. Short
  term: mirror them in a Python constant, explicitly marked provisional (consistent with `CLAUDE.md`'s
  thin-adapter-layer approach). Medium term: ask Evolver to expose a small registry dump.
- Multi-problem training sets (today only single-problem ZDT4).

## Phase 3 — Analysis layer

- Structured parsing of `CONFIGURATIONS.csv`/`VAR_CONF.txt` (today only listed as files).
- Run history: browsing past runs, not just the latest one (each run is currently ephemeral from the
  UI's perspective once a new one starts).
- Statistical comparison across runs/configurations (Wilcoxon, comparison tables — `CLAUDE.md`'s
  original analysis-layer goal, not started yet).

## Phase 4 — Validation runs

- `start_validation(request)`: run the winning configuration against a validation set and compare
  indicator distributions against a baseline.

## Phase 5 — Hardening

- Tests for the dynamic form-rendering logic once Phase 1 lands.
- Track the state of the `study/uniform-training-runner` branch on Evolver (the `cli.runner` prototype
  this tool depends on is still not merged to `main`).
