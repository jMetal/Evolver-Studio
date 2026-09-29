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
- **Discovery and a read-only parameter space explorer** (formerly "Phase 2"): a multipage app
  (`st.navigation`, one script per page under `pages/`) — the
  **Explore** page browses any base algorithm's parameter space as a compact, read-only tree
  (`parameter_form.render_parameter_space_readonly`, no widgets — editing lives only in Training's
  form) and summarizes every registered meta-optimizer's encoding support and operator parameters.
  Backed by `evolver_studio/catalogue.py`, still a hand-maintained Python mirror of Evolver's
  registries (see "Provisional catalogue" below), cross-checked against Evolver's own introspection
  manifest when Evolver's jar is available (`tests/test_catalogue.py::TestCatalogueMatchesDescribeManifest`).
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

- **Catalogue synced with Evolver's September 2026 changes**: AGE-MOEA as a meta-optimizer (flat and
  tree); tree support for NSGA-II, AGE-MOEA and RandomSearch, whose tree operator catalogues
  (`*MetaTree.yaml`) the Explore page now renders; NSGA-III, PAES and SSMOEA browsable as base
  algorithms; offspring size no longer listed as configurable (Evolver fixes it to the population
  size). The Training page leaves the meta population size to Evolver's own default (50). A new
  manifest test checks `supports_tree` against `DescribeMain`'s `supportsTree`.

- **Navigation grouped by purpose**: the menu has four sections — Explore; Solve (Run algorithm,
  `pages/solve.py`); Meta-optimization (Training, Analysis, Validation); Learn (Tutorials,
  `pages/tutorials.py`). Run algorithm is a placeholder for the solving track (see Next up), like
  Analysis and Validation.
- **Home page**: the app opens on `pages/home.py`, which presents Evolver-Studio's two uses and
  shows a card for each page (its section, what it does, whether it is available or coming soon,
  and a link to it), plus a getting-started note while Evolver's jar is missing. The menu and the
  cards are built from the same list of pages (`evolver_studio/sections.py`).
- **Tutorials page with the first interactive tutorial**: the page lists Evolver-Studio's tutorials
  by level (`evolver_studio/tutorials.py`, mirroring Evolver's `docs/proposals/tutorials.md`) and
  guides the selected one step by step. **S2, "Exploring a parameter space"**
  (`evolver_studio/tutorial_parameter_spaces.py`), the companion of Evolver's tutorial E1, is the
  first available: reading the tree view, global and conditional sub-parameters, an interactive
  view of the parameters a configuration activates, and two encodings side by side.
  `tests/test_tutorials.py` walks through it with Streamlit's `AppTest`.
- **No Evolver checkout needed to use the app**: the sidebar downloads Evolver 2.1's jar from Maven
  Central (checksum-verified) into `lib/`; parameter spaces and meta-optimizer configurations are
  read from the jar (`evolver_studio/resource_files.py`), and the reference fronts, weight vectors
  and TSP instances are copied into `resources/` (MaF's fronts left out) from the release's GitHub
  source archive by `make sync-resources`, with checksums `tests/test_resources.py` checks. Training runs use this repo's root as the JVM's working directory.
  `EVOLVER_JAR` selects a locally built jar instead. Verified against the real jar: NSGA-II and
  MOEA/D on ZDT4 and permutation NSGA-II on KroAB100TSP reach `FINISHED`.

This already covers most of `CLAUDE.md`'s MVP operations list (`start_training`, `get_run_status`,
`cancel_run`, `get_results`).

## Next up

- **Tree-encoding training in the Training page**: Evolver's `cli.training` now accepts tree
  `metaSearch` files for NSGA-II, AGE-MOEA and RandomSearch (operator flags, like the flat ones, plus
  the selection); the Training page only builds flat requests today.
- **Solving track** (see `CLAUDE.md`): configure and run a configurable algorithm on a concrete problem,
  jMetal-runner style — choose problem, algorithm, encoding and configuration (default, tuned or edited
  in the guided form), run it, inspect the front and indicators, export `VAR`/`FUN`. Needs an
  Evolver-side entry point for single algorithm runs, analogous to `cli.training` (to be proposed in
  Evolver's `docs/proposals/`), and a new page reusing the guided parameter form and the live front.
- **Tutorials**: the remaining interactive tutorials for both tracks, catalogued (with their Evolver
  documentation counterparts) in Evolver's `docs/proposals/tutorials.md` and developed one at a time
  (S2 is done).

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

- Tests for the parameter-space explorer once Phase 2 lands. *(Explorer itself shipped; Streamlit
  `AppTest` coverage exists only for the Tutorials page so far, see `tests/test_tutorials.py`.)*
- Track Evolver's releases: the app is built against Evolver 2.1 (`evolver_client.EVOLVER_VERSION`,
  the first release that ships `cli.training`); moving to a newer release means bumping that
  constant, running `make sync-resources` and re-running the drift-detection tests against it
  (or, ahead of a release, against a `develop` jar set by `EVOLVER_JAR`). The rename from `cli.runner` to
  `cli.training` and the `request.yaml` schema change already broke this integration once; the
  two-sided drift-detection mechanism (`tests/test_catalogue.py` here,
  `BaseAlgorithmRegistryCompletenessTest`/`TrainingRunnerMetaBuilderCompletenessTest` in Evolver) and
  Evolver's `DescribeMain` manifest (see "Done so far") both exist to catch the next one faster.
- Fully populate `catalogue.py` from `DescribeMain`'s manifest instead of hand-maintained literals,
  for the subset it actually covers (registered/runnable algorithms) — the broader
  browsable-but-unregistered set (SMS-EMOA, RDE-MOEA, NSGA-III, PAES, SSMOEA, Async Genetic
  Algorithm, ...)
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

**H. Solving problems (no meta-optimization)**
17. Configure and run an algorithm on a concrete problem, jMetal-runner style — Next up (solving track).
18. Run a configuration found in a training run on a new problem — Next up (solving track).
19. Compare several configurations (default, tuned, custom) on a problem over independent runs — not
    yet scheduled.
20. Run an algorithm on a user-defined jMetal problem class — not yet scheduled.
