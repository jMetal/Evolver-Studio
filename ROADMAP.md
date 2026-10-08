# Roadmap

Where Evolver-Studio is headed after the ZDT4 prototype (`app.py`, `evolver_studio/`), prioritized as
concrete next milestones rather than a fixed schedule. Priorities are driven by the use cases listed
below; revisit and reorder as real usage surfaces new ones.

## Done so far

- Request/status/results contract against Evolver's `org.uma.evolver.cli.training.TrainingRunnerMain`
  (`evolver_studio/request.py`, `evolver_client.py`, `results.py`) — a full request is three files
  (`request.yaml` referencing a reusable `baseLevel` file and a reusable `metaSearch` file by path,
  `outputDirectory`/`writeFrequency`/`statusFrequency` inline on `request.yaml` itself), matching
  the contract documented in Evolver's `docs/utilities/cli_tools.rst`.
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
  **Explore** page browses any base algorithm's parameter space as a read-only, filterable table,
  one row per parameter with the condition that activates it
  (`parameter_form.render_parameter_space_table`, over `parameter_space.parameter_rows`; editing
  lives only in Training's form) and summarizes every registered meta-optimizer's encoding support and operator parameters.
  Backed by `evolver_studio/catalogue.py`, still a hand-maintained Python mirror of Evolver's
  registries (see "Provisional catalogue" below), cross-checked against Evolver's own introspection
  manifest when Evolver's jar is available (`tests/test_catalogue.py::TestCatalogueMatchesDescribeManifest`).
- **Provisional catalogue → Evolver's own introspection manifest**: Evolver's
  `org.uma.evolver.cli.training.DescribeMain` (documented in Evolver's `docs/utilities/cli_tools.rst`)
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

- **Parameter spaces as tables, catalogue synced with Evolver `develop` (October 2026)**: Explore
  shows each parameter space (base algorithm and meta-optimizer operator catalogues, flat and tree
  in tabs) as a table with a one-line size summary and a text filter that keeps the ancestors of
  each match, instead of a long nested list. RVEA is runnable as a base algorithm (it needs the
  weight vectors directory, like MOEA/D: `BaseAlgorithm.required_extra_config_keys`, checked
  against the manifest's `requiredExtraConfigKeys`), and AsyncNSGA-II supports the tree encoding
  (`AsyncNSGAIIMetaTree.yaml`); the `.irace` files, gone from Evolver, are no longer triaged. Both
  need an Evolver jar newer than 2.1 (`EVOLVER_JAR`): the catalogue records the Evolver version it
  mirrors (`CATALOGUE_EVOLVER_VERSION`), and Explore and Training warn when the jar in use (its
  `pom.properties` version) is older.

- **Catalogue synced with Evolver 2.4**: every base
  algorithm `DescribeMain` registers is runnable (NSGA-III, SMS-EMOA, RDE-MOEA, AGE-MOEA, PAES and
  SSMOEA join NSGA-II, MOEA/D and RVEA, MOEA/D gaining its Binary and Permutation encodings), with
  the default configurations the jar ships; MOPSO stays browsable only. Done against the
  `develop` jar before the release, and confirmed with 2.4.

- **Problem catalogue**: `DescribeMain`'s `problemCatalogue`
  (`evolver_studio/problem_catalogue.py`) gives each problem's encoding, dimensions and constructor
  arguments. Explore › Problems lists them; Run algorithm offers only the algorithms of the
  problem's encoding and lets its arguments be set (a `{class, args}` problem in the request); the
  Training set takes arguments per problem and checks the encodings. With a 2.3 jar, which has no
  catalogue, the pages fall back to the plain list of names.

- **A running label instead of a live view**: Run algorithm no longer offers the
  progress bar and live front modes, which a benchmark run, over in a second or two, rarely gave
  time to see. A run in progress shows a blinking label with the evaluation it is at (its status is
  updated every 1000 evaluations, `pages/solve.py`'s `STATUS_FREQUENCY`) and the Cancel button.
  `SolveRequest` still mirrors Evolver's `frontFrequency`, `writePopulation` and
  `frontDelayMillis`, unused by the page.

- **Moved to Evolver 2.4 (October 2026)**: the app downloads Evolver 2.4 and its resources come from
  tag `v2.4` (unchanged since 2.3). Everything of the sync with `develop` above holds with the
  release: the 18 registered base algorithms, the problem catalogue (Explore › Problems, Run
  algorithm, Training, Validation) and the tutorial S3 need Evolver 2.4 or later, and the two
  drift-detection tests against `DescribeMain` pass with the release jar.

- **Moved to Evolver 2.3 (October 2026)**: the app downloads Evolver 2.3 and its resources come from
  tag `v2.3` (the exact front of ZDT5). The catalogue gains NSGA-II for binary problems, with the
  default configurations of Evolver 2.3 for the binary and permutation encodings (every runnable
  encoding now has one), the Spread and GeneralizedSpread indicators, and classifies the space of
  the GECCO 2019 study used by Evolver's tutorial E5 (`NSGAIIDoubleGECCO2019.yaml`).

- **Moved to Evolver 2.2 (October 2026)**: the app downloads Evolver 2.2 and its resources come from
  tag `v2.2` (DTLZ2-4 reference fronts regenerated as in jMetal, and DTLZ1Minus-DTLZ4Minus fronts
  per number of objectives). Studio follows stable releases only, so the catalogue mirrors
  `EVOLVER_VERSION`: the separate `CATALOGUE_EVOLVER_VERSION` is gone, and the "older jar" warning
  only shows for an older jar set with `EVOLVER_JAR`.

- **Explore split into four pages (October 2026)**: Base algorithms, Meta-optimizers, Quality
  indicators and Problems (a placeholder), instead of one long page. The algorithm pages open with
  nothing selected and show a space only once an algorithm is chosen; meta-optimizers use the same
  selector as base algorithms (they were a list of expanders). Quality indicators come from
  `catalogue.QUALITY_INDICATORS`, checked against the manifest's `indicators`.

- **Run algorithm (solving track, first version)**: `pages/solve.py` builds a `cli.solving` request
  in five sections that appear as the previous one is chosen: problem (and its reference front,
  found by `problems.reference_front_candidates`), algorithm and encoding, configuration, budget
  (population, evaluations, independent runs, seed, indicators) and Run. The configuration starts
  from the default one of Evolver's jar (`BaseAlgorithm.default_configurations`; RVEA offers its
  three variants) and is adjusted with `parameter_form.render_configuration_form`, a widget per
  active parameter limited to what the parameter space allows (`evolver_studio/configuration.py`
  reads, completes and checks the `--parameter value` strings). The run is a detached subprocess
  (`evolver_client.start_solve`, whose output goes to a log file) kept under `solve-runs/<id>/`,
  with its phase told by `runs.run_phase`; the fronts are plotted over the reference front (2D, 3D
  or parallel coordinates), with the indicators, a summary over the runs and a zip to download.
  Previous runs are reopened from a list. See `cli.solving` in Evolver's
  `docs/utilities/cli_tools.rst`.
  While a run is in progress, a fragment polls its status every second and shows the evaluation it
  is at (`progress.running_label`, from `statusFrequency` with Evolver 2.3 or later).

- **Navigation grouped by purpose**: the menu has four sections — Explore; Solve (Run algorithm,
  `pages/solve.py`); Meta-optimization (Training, Analysis, Validation); Learn (Tutorials,
  `pages/tutorials.py`). Analysis is a placeholder.
- **Home page**: the app opens on `pages/home.py`, which presents Evolver-Studio's two uses and
  shows a card for each page (its section, what it does, whether it is available or coming soon,
  and a link to it), plus a getting-started note while Evolver's jar is missing. The menu and the
  cards are built from the same list of pages (`evolver_studio/sections.py`).
- **Tutorials page with the first interactive tutorial**: the page lists Evolver-Studio's tutorials
  by level (`evolver_studio/tutorials.py`, each paired with tutorials of Evolver's documentation) and
  guides the selected one step by step. **S1, "Exploring a parameter space"**
  (`evolver_studio/tutorial_parameter_spaces.py`), the companion of Evolver's tutorial E1, is the
  first available: reading the tree view, global and conditional sub-parameters, an interactive
  view of the parameters a configuration activates, and two encodings side by side; its version 1.1
  reads the spaces in the table of the Explore pages and ends with the meta-optimizers and the
  quality indicators. **S2, "Solving a problem with a configurable algorithm"**
  (`evolver_studio/tutorial_solving.py`), the companion of E2, is the first of the solving track:
  each step has a prepared run (`PreparedRun`: fixed seed, the values the text announces) that runs
  in the tutorial or opens in Run algorithm. **S3, "Validating a configuration"**
  (`evolver_studio/tutorial_validation.py`), the companion of E8, runs one validation study in the
  background (kept as `validation-runs/tutorial-validation`, reused while it is current) and reads
  its results; the text about them is computed from the data. `tests/test_tutorials.py`,
  `tests/test_tutorial_solving.py` and `tests/test_tutorial_validation.py` walk through them with
  Streamlit's `AppTest`.
- **No Evolver checkout needed to use the app**: the sidebar downloads the Evolver release's jar
  from Maven Central (checksum-verified) into `lib/`; parameter spaces and meta-optimizer
  configurations are read from the jar (`evolver_studio/resource_files.py`), and the reference
  fronts, weight vectors and TSP instances are copied into `resources/` (MaF's fronts left out) from
  the release's GitHub source archive by `make sync-resources`, with checksums
  `tests/test_resources.py` checks. Training runs use this repo's root as the JVM's working
  directory. `EVOLVER_JAR` selects a locally built jar instead. Verified against the real jar:
  NSGA-II and MOEA/D on ZDT4 and permutation NSGA-II on KroAB100TSP reach `FINISHED`.

This already covers most of `CLAUDE.md`'s MVP operations list (`start_training`, `get_run_status`,
`cancel_run`, `get_results`).

## Next up

- **Tree-encoding training in the Training page**: Evolver's `cli.training` now accepts tree
  `metaSearch` files for NSGA-II, AGE-MOEA and RandomSearch (operator flags, like the flat ones, plus
  the selection); the Training page only builds flat requests today.
- **Solving track, next steps** (the first version of Run algorithm is done): start from a
  configuration a Training run found (`VAR_CONF.txt`/`CONFIGURATIONS.csv`) and compare several
  configurations on a problem.
- **Tutorials**: the remaining interactive tutorials for both tracks, catalogued (with their Evolver
  documentation counterparts) in `evolver_studio/tutorials.py` and developed one at a time (S1, S2
  and S3 are done). As in Evolver, only the written ones are numbered, consecutively by level: writing
  a new one renumbers those after it.

## Phase 3 — Analysis layer

Landing page now exists (`pages/analysis.py`), currently a placeholder pointing back here.

- Structured parsing of `CONFIGURATIONS.csv`/`VAR_CONF.txt` (today only listed as files).
- Run history: browsing past runs, not just the latest one (each run is currently ephemeral from the
  UI's perspective once a new one starts).
- Statistical comparison across runs/configurations (Wilcoxon, comparison tables — `CLAUDE.md`'s
  original analysis-layer goal, not started yet).

## Phase 4 — Validation runs

First version shipped (`pages/validation.py`, with the Evolver 2.4 sync): a study compares a
tuned configuration, the pivot (from a finished training run's `VAR_CONF.txt`, or pasted), with the
default configurations of Evolver's algorithms for the encoding of the problems. It needs no change
in Evolver: a study is a `cli.solving` request per algorithm and problem
(`evolver_studio/validation.py`), run by a detached worker with a pool of JVMs
(`validation_worker.py`) that keeps a `status.yaml` in Evolver's format, and its statistics follow
Evolver's own scripts (`validation_stats.py`: medians and IQR, Wilcoxon rank-sum, A12). Still to do:

- Friedman test with Holm's procedure over the problems, critical difference plots and the Bayesian
  sign test (Evolver's scripts use SAES for them), and LaTeX tables.
- Adjusting an algorithm's configuration for the study (the form of Run algorithm, per contender).
- An estimate of how long a study takes: an algorithm such as SMS-EMOA with three objectives is
  orders of magnitude slower than NSGA-II, which one only learns by running it.

## Phase 5 — Hardening

- Tests for the parameter-space explorer once Phase 2 lands. *(Explorer itself shipped; Streamlit
  `AppTest` coverage exists for the Tutorials and Explore pages, see `tests/test_tutorials.py` and
  `tests/test_explore_*.py`.)*
- Track Evolver's releases: the app is built against Evolver 2.4 (`evolver_client.EVOLVER_VERSION`)
  and follows stable releases only; moving to a newer release is the `/bump-evolver` command
  (bumping that constant, `make sync-resources`, `/sync-catalogue` and the drift-detection tests).
  Work against a `develop` jar set by `EVOLVER_JAR` goes on an `experiment/*` branch. The rename
  from `cli.runner` to `cli.training` and the `request.yaml` schema change already broke this
  integration once; the two-sided drift-detection mechanism (`tests/test_catalogue.py` here,
  `BaseAlgorithmRegistryCompletenessTest`/`TrainingRunnerMetaBuilderCompletenessTest` in Evolver)
  and Evolver's `DescribeMain` manifest (see "Done so far") both exist to catch the next one faster.
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
15. Discover which algorithms/problems/indicators exist — *shipped* (Explore pages).

**G. Reproducibility**
16. Recover a run's exact `request.yaml` to reproduce or cite it — *shipped* (already persisted per run
    under `cli-runner-runs/<run_id>/`; no dedicated UI for it yet).

**H. Solving problems (no meta-optimization)**
17. Configure and run an algorithm on a concrete problem, jMetal-runner style — *shipped* (Run
    algorithm page).
18. Run a configuration found in a training run on a new problem — Next up (solving track).
19. Compare several configurations (default, tuned, custom) on a problem over independent runs — not
    yet scheduled.
20. Run an algorithm on a user-defined jMetal problem class — not yet scheduled.
