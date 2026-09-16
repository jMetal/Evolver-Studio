# Roadmap

Where Evolver-Studio is headed after the ZDT4 prototype (`app.py`, `evolver_studio/`), prioritized as
concrete next milestones rather than a fixed schedule. Priorities are driven by the use cases listed
below; revisit and reorder as real usage surfaces new ones.

## Done so far

- Request/status/results contract against Evolver's `org.uma.evolver.cli.runner.TrainingRunnerMain`
  (`evolver_studio/request.py`, `evolver_client.py`, `results.py`).
- Non-blocking run control: detached subprocess, PID tracking, cancellation, reconnect to an
  in-progress run across page reloads (`evolver_studio/runs.py`, `adaptive_poll.py`).
- Live indicator-front preview: accumulates checkpoints from `INDICATORS.csv`, deduplicates unchanged
  fronts, lets the viewer narrow the plot to the last N checkpoints
  (`evolver_studio/live_front.py`, `slider_state.py`).
- Validated parameter editing (formerly "Phase 1"): parses Evolver's YAML parameter space files into a
  structured model and renders a dynamic Streamlit form from it — Guided mode (validated by
  construction) and Expert mode (raw YAML, validated on every change)
  (`evolver_studio/parameter_space.py`, `parameter_form.py`). Base-level algorithm only so far; the
  meta-optimizer's own parameter space (`NSGAIIDoubleReduced.yaml`) still uses the fixed default.

This already covers most of `CLAUDE.md`'s MVP operations list (`start_training`, `get_run_status`,
`cancel_run`, `get_results`) for the single hardcoded NSGA-II/ZDT4 case.

## Phase 2 — Discovery and a read-only parameter space explorer

Prioritized after reviewing the use cases below: users want to understand what's available and what a
given algorithm's parameters look like *before* committing to launching a run.

- Replace the hardcoded values (`"NSGA-II"`, `["ZDT4"]`, `["Epsilon", "NormalizedHypervolume"]`) with
  real listings: `list_algorithms()`, `list_indicators()`, `list_training_sets()`.
- Evolver's registries (`BaseAlgorithmRegistry`, `IndicatorRegistry`) only exist in Java today. Short
  term: mirror them in a Python constant, explicitly marked provisional (consistent with `CLAUDE.md`'s
  thin-adapter-layer approach). Medium term: ask Evolver to expose a small registry dump.
- Multi-problem training sets (today only single-problem ZDT4).
- A **read-only parameter space explorer**: browse any algorithm's tunable parameters (ranges,
  categorical choices, conditional/global sub-parameters) without launching a training run. Reuses
  `evolver_studio/parameter_space.py`'s parser directly (already built for Phase 1's form) — no new
  parsing logic needed, only a browsing view (e.g. reusing `parameter_form.py`'s recursive rendering in
  a disabled/inspection mode, or a simpler read-only tree view). Self-contained, no Evolver changes.

## Phase 3 — Analysis layer

- Structured parsing of `CONFIGURATIONS.csv`/`VAR_CONF.txt` (today only listed as files).
- Run history: browsing past runs, not just the latest one (each run is currently ephemeral from the
  UI's perspective once a new one starts).
- Statistical comparison across runs/configurations (Wilcoxon, comparison tables — `CLAUDE.md`'s
  original analysis-layer goal, not started yet).

## Phase 4 — Validation runs

- `start_validation(request)`: run the winning configuration against a validation set and compare
  indicator distributions against a baseline (default, untuned configuration).

## Phase 5 — Hardening

- Tests for the parameter-space explorer once Phase 2 lands.
- Track the state of the `study/uniform-training-runner` branch on Evolver (the `cli.runner` prototype
  this tool depends on is still not merged to `main`).

## Use cases considered

Tentative catalogue behind the phases above, grouped by the kind of goal they serve. "Covered by"
points at the phase (or already-shipped work) that addresses each one; "not yet scheduled" means no
phase currently plans for it.

**A. Tuning a single algorithm**
1. Configure and launch a training run — *shipped*.
2. Monitor progress live, cancel if needed — *shipped*.
3. Inspect the winning configuration in a readable (not raw-text) form — Phase 3.
4. Validate the tuned configuration against a problem set distinct from training (generalization) —
   Phase 4.
5. Compare the tuned configuration against the algorithm's untuned defaults — Phase 4.

**B. Comparison and statistical analysis**
6. Compare several tuned base-level algorithms (NSGA-II vs. MOEA/D, etc.) on the same problem — Phase 3.
7. Statistical comparison across runs/configurations (Wilcoxon, boxplots), needing
   `numberOfIndependentRuns > 1` and a multi-run analysis view — Phase 3.

**C. Exploring the parameter space without launching anything**
8. Browse an algorithm's parameters before deciding whether to tune it — **Phase 2 (prioritized)**.
9. Understand which parameters most influenced the configuration found (sensitivity/importance
   analysis) — not yet scheduled.

**D. Managing past experiments**
10. Browse a history of past runs, not just the latest — Phase 3.
11. Re-launch a past run with small edits ("clone and tweak") — not yet scheduled.
12. Export/share a run's results (configuration + indicators) — not yet scheduled.

**E. Batch / unattended experimentation**
13. Queue several training runs (different problems/algorithms) to run unattended — not yet scheduled.

**F. Onboarding / new users**
14. "Try Evolver" on a bundled example with sensible defaults, minimal setup — *shipped* (the ZDT4
    quick-start), informs how Phase 2's guided experience should feel.
15. Discover which algorithms/problems/indicators exist — **Phase 2 (prioritized)**.

**G. Reproducibility**
16. Recover a run's exact `request.yaml` to reproduce or cite it — *shipped* (already persisted per run
    under `cli-runner-runs/<run_id>/`; no dedicated UI for it yet).
