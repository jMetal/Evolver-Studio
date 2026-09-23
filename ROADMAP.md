# Roadmap

Where Evolver-Studio is headed after the ZDT4 prototype (`app.py`, `evolver_studio/`), prioritized as
concrete next milestones rather than a fixed schedule. Priorities are driven by the use cases listed
below; revisit and reorder as real usage surfaces new ones.

## Done so far

- Request/status/results contract against Evolver's `org.uma.evolver.cli.training.TrainingRunnerMain`
  (`evolver_studio/request.py`, `evolver_client.py`, `results.py`) — a full request is three files
  (`request.yaml` referencing a reusable `baseLevel` file and a reusable `metaSearch` file by path,
  `outputDirectory`/`writeFrequency`/`statusFrequency` inline on `request.yaml` itself), matching
  Evolver's `cli-training-prototype.md`.
- Non-blocking run control: detached subprocess, PID tracking, cancellation, reconnect to an
  in-progress run across page reloads (`evolver_studio/runs.py`, `adaptive_poll.py`).
- Live indicator-front preview: accumulates checkpoints from `INDICATORS.csv`, deduplicates unchanged
  fronts, lets the viewer narrow the plot to the last N checkpoints
  (`evolver_studio/live_front.py`, `slider_state.py`).
- Validated parameter editing: parses Evolver's YAML parameter space files into a structured model and
  renders a dynamic Streamlit form from it — Guided mode (validated by construction) and Expert mode
  (raw YAML, validated on every change) (`evolver_studio/parameter_space.py`, `parameter_form.py`).
  Base-level algorithm only; the meta-optimizer's own operators use a flat, single-level YAML editor
  instead (`evolver_studio/request.py`'s `parse_operator_flags_yaml`) — Evolver's `metaSearch` files
  are a fixed recipe, not a `ParameterSpace` to evolve, so the same guided/expert form does not apply.
- **Discovery and a read-only parameter space explorer** (formerly "Phase 2"): a four-section
  multipage app (`st.navigation`, `pages/{explore,training,analysis,validation}.py`) — the
  **Explore** page browses any base algorithm's parameter space as a compact, read-only tree
  (`parameter_form.render_parameter_space_readonly`, no widgets — editing lives only in Training's
  form) and summarizes every registered meta-optimizer's encoding support and operator parameters.
  Backed by `evolver_studio/catalogue.py`, still a hand-maintained Python mirror of Evolver's
  registries (see "Provisional catalogue" below), cross-checked against Evolver's own introspection
  manifest when the checkout is available (`tests/test_catalogue.py::TestCatalogueMatchesDescribeManifest`).
- **Provisional catalogue → Evolver's own introspection manifest**: Evolver's
  `org.uma.evolver.cli.training.DescribeMain` (`docs/proposals/cli-describe-manifest.md` in Evolver)
  now prints a YAML manifest of everything `cli.training` can resolve (registered base/meta
  algorithms, problems, indicators, available resource-directory file names, and the
  `request.yaml`/`baseLevel`/`metaSearch` schema via reflection over Evolver's own records) —
  `evolver_client.describe()` calls it. `catalogue.py` is not yet fully populated *from* the
  manifest (see "Next up" below); today the manifest is only used to catch `catalogue.py` drifting
  from Evolver's registered algorithms, not as `catalogue.py`'s data source.
- **All four wired meta-optimizers launchable, not just NSGA-II**: the Training page's meta-optimizer
  selector now lists every algorithm `catalogue.py` marks `wired_into_cli_runner` (`NSGA-II`,
  `SPEA2`, `SMPSO`, `AsyncNSGA-II`), each seeding its own operator-flags editor from its real example
  file under `metaOptimizerConfigurations/` (`MetaAlgorithm.example_config_file`). Verified against
  the real jar: all four reach `FINISHED`.
- **Arbitrary multi-problem training sets**: the single hardcoded ZDT4 problem is gone — Training's
  launch form has an editable table (`evolver_studio/training_set.py`, `st.data_editor`), one row per
  problem with its reference front file and evaluation budget (`BaseLevelConfig`'s three parallel
  fields); problem names are free text (a curated `DescribeMain` name, or any fully-qualified jMetal
  class resolved reflectively — e.g. TSP problems, which aren't `Problem<DoubleSolution>` so are
  never in the curated catalogue), with curated names shown as a hint when available. Verified
  against the real jar with a two-problem set (RE31+RE32). Indicators still stay fixed.
- **Permutation-encoded base algorithms**: Training's Base algorithm selector gained an Encoding
  selectbox (`catalogue.BaseAlgorithm.runnable_encodings`, plural — NSGA-II now supports both
  "Double" and "Permutation"), threaded through to `BaseLevelConfig.encoding` (default "Double").
  Verified end-to-end against the real jar: `PermutationNSGAII` tuned on two multi-objective TSP
  instances (`KroAB100TSP`/`KroAC100TSP`) reaches `FINISHED`.

This already covers most of `CLAUDE.md`'s MVP operations list (`start_training`, `get_run_status`,
`cancel_run`, `get_results`).

## Phase 3 — Analysis layer

Landing page now exists (`pages/analysis.py`), currently a placeholder pointing back here.

- Structured parsing of `CONFIGURATIONS.csv`/`VAR_CONF.txt` (today only listed as files).
- Run history: browsing past runs, not just the latest one (each run is currently ephemeral from the
  UI's perspective once a new one starts).
- Statistical comparison across runs/configurations (Wilcoxon, comparison tables — `CLAUDE.md`'s
  original analysis-layer goal, not started yet).

## Phase 4 — Validation runs

Landing page now exists (`pages/validation.py`), currently a placeholder pointing back here.

- `start_validation(request)`: run the winning configuration against a validation set and compare
  indicator distributions against a baseline (default, untuned configuration).

## Phase 5 — Hardening

- Tests for the parameter-space explorer once Phase 2 lands. *(Explorer itself shipped; still no
  dedicated Streamlit `AppTest` coverage beyond the manual smoke-test discipline described in
  `CLAUDE.md`.)*
- Track the state of the `study/uniform-training-runner` branch on Evolver (the `cli.training`
  prototype this tool depends on is still not merged to `main`). The rename from `cli.runner` to
  `cli.training` and the `request.yaml` schema change already broke this integration once; the
  two-sided drift-detection mechanism (`tests/test_catalogue.py` here,
  `BaseAlgorithmRegistryCompletenessTest`/`TrainingRunnerMetaBuilderCompletenessTest` in Evolver) and
  Evolver's `DescribeMain` manifest (see "Done so far") both exist to catch the next one faster.
- Fully populate `catalogue.py` from `DescribeMain`'s manifest instead of hand-maintained literals,
  for the subset it actually covers (registered/runnable algorithms) — the broader
  browsable-but-unregistered set (SMS-EMOA, RDE-MOEA, Async Genetic Algorithm, Random Search, ...)
  has no Evolver registry to introspect and stays hand-maintained regardless (see `catalogue.py`'s
  module docstring).

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
8. Browse an algorithm's parameters before deciding whether to tune it — *shipped* (Explore page).
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
    quick-start).
15. Discover which algorithms/problems/indicators exist — *shipped* (Explore page).

**G. Reproducibility**
16. Recover a run's exact `request.yaml` to reproduce or cite it — *shipped* (already persisted per run
    under `cli-runner-runs/<run_id>/`; no dedicated UI for it yet).
