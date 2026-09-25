# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Evolver-Studio is a **Python/Streamlit GUI and analysis tool for [Evolver](https://github.com/jMetal/Evolver)**,
a Java framework for automated meta-optimization of multi-objective metaheuristics. Evolver itself has no
GUI: base-level algorithms are configured via YAML parameter spaces, training runs are launched through its
`cli.training` command-line entry point (or Java `main()` examples), and results are written as CSV/text
files. Evolver-Studio's job is to make Evolver usable without writing Java or hand-editing YAML, and to add
a statistical-analysis layer (Wilcoxon tests, plots, comparison tables) on top of the raw results —
leveraging the Python scientific stack (`pandas`, `scipy.stats`, `seaborn`/`plotly`) and, where useful, `jMetalPy`.

Evolver-Studio serves two purposes, not only meta-optimization:

- **Solving problems**: configure and run Evolver's configurable algorithms on concrete problems, in the
  style of jMetal's runners (pick a problem, an algorithm and a configuration, run it, inspect the front).
  Evolver's configurable core is usable on its own, as an alternative to jMetal.
- **Meta-optimization**: training runs, analysis and validation of the configurations found.

Tutorials for both (interactive, in the app) are catalogued in Evolver's `docs/proposals/tutorials.md`.

This file documents the intended architecture agreed on before implementation started. A first,
deliberately minimal prototype now lives in the repo (`app.py`, `evolver_studio/`): it drives Evolver's
`org.uma.evolver.cli.training.TrainingRunnerMain` (see Evolver's
`docs/proposals/cli-training-prototype.md`) to run the single ZDT4 training case, as a smoke test of the
request/status/results contract before building the full MVP surface below. Expect this file's
architecture sections to be corrected/expanded as more of the real code lands.

Evolver lives in its own repository, <https://github.com/jMetal/Evolver>. The app only needs
Evolver's release jar, never a checkout: no local paths (to an Evolver checkout or anything else) may
appear in this repository's code, tests or docs.

## Relationship to Evolver — status

Evolver's integration surface is still partial:

- Training runs have a uniform, structured entry point: `cli.training`'s `TrainingRunnerMain`
  (request/status/result files) and `DescribeMain` (a manifest of what it can resolve). There is no
  equivalent yet for **single algorithm runs** (the solving purpose above) or for **validation runs**;
  both still need an Evolver-side entry point.
- Meta-optimization results have a single canonical output format: `ConsolidatedOutputResults`
  (`METADATA.txt`/`INDICATORS.csv`/`CONFIGURATIONS.csv`/`VAR_CONF.txt`); the older `OutputResults` was
  removed.

Until that uniformization happens on the Evolver side, Evolver-Studio's integration code should be
treated as provisional and isolated behind a thin adapter layer (see Architecture below) so it can be
updated without touching the UI or analysis code.

**Evolver is not only a read-only external dependency.** When Evolver-Studio's needs require it, work
may extend into Evolver itself (its `develop` branch) to propose or implement the missing pieces — e.g. expanding `BaseAlgorithmRegistry` beyond its current
NSGA-II/MOEA-D scope, or wiring additional `Meta*Builder` classes (`MetaSPEA2Builder`,
`MetaSMPSOBuilder`, `MetaAsyncNSGAIIBuilder`, ...) into `TrainingRunner` so more of them are
selectable as meta-optimizers, not just usable from `example.training`. Prefer proposing such changes
as a `docs/proposals/*.md` document in Evolver's repo first (matching the existing
`cli-training-prototype.md`) before implementing them, consistent with how that design was itself
introduced.

### Keeping Evolver-Studio's catalogue in sync with Evolver — the drift-detection mechanism

Evolver changes with some regularity (new base algorithms, new parameter-space components, new
meta-optimizer builders). Nothing forces Evolver-Studio's provisional catalogue
(`evolver_studio/catalogue.py`) to be updated when that happens, so both sides carry a matching
**drift-detection test**: a directory/source scan that fails loudly on anything new and untriaged,
instead of silently going stale.

- **Evolver-Studio side** (`tests/test_catalogue.py::TestCatalogueMatchesEvolverJar`): scans
  the `parameterSpaces/` directory packaged in Evolver's jar and fails if any `.yaml`/`.irace` file is
  neither referenced by `BASE_ALGORITHMS` nor listed in
  `KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES` (both in `evolver_studio/catalogue.py`).
- **Evolver side** (`src/test/java/org/uma/evolver/cli/training/`): two JUnit tests with the same
  shape —
  `BaseAlgorithmRegistryCompletenessTest` scans `src/main/java/org/uma/evolver/algorithm/**` for
  concrete algorithm classes not accounted for in its `KNOWN_ALGORITHM_CLASSES` map;
  `TrainingRunnerMetaBuilderCompletenessTest` does the same for `Meta*Builder` classes under
  `src/main/java/org/uma/evolver/meta/builder/`.

When either test fails, the fix is always the same shape: add the new class/file to the relevant
map/set with a one-line reason (`"not yet registered"`, `"not yet wired into TrainingRunner"`, a
structural-incompatibility note, etc.), and — separately, as a deliberate choice, not a side effect of
silencing the test — decide whether to also wire it in for real (`BaseAlgorithmRegistry`,
`TrainingRunner`, or `evolver_studio/catalogue.py`'s `runnable_today`/`wired_into_cli_runner` flags).

## Planned Architecture

```
Streamlit UI (this repo)
  ├─> Discovery: list algorithms / parameter spaces / training sets / indicators
  │     (reads Evolver's YAML parameter spaces directly, or via a small Java helper — TBD)
  ├─> Run control: launch training / validation runs
  │     -> writes a request.yaml, invokes the Evolver fat jar as a subprocess
  │        (java -cp ... org.uma.evolver.cli.training.TrainingRunnerMain request.yaml status.yaml)
  │     -> polls a status.yaml for progress (RUNNING/FINISHED/FAILED, evaluations done/total)
  │     -> reads a results.yaml (pointing at METADATA.txt/INDICATORS.csv/CONFIGURATIONS.csv) once finished
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

Python dependencies are managed with a dedicated **Conda environment** named `evolver-studio`
(`environment.yml`, backed by `pyproject.toml` with a `setuptools` build backend — see
`CODING_GUIDELINES.md`). To set it up:

```bash
conda env create -n evolver-studio -f environment.yml   # first time
conda env update -n evolver-studio -f environment.yml   # after dependency changes
```

Common commands (also available as `make` targets, which run inside the env via `conda run -n
evolver-studio`):

```bash
make lint    # ruff check .
make format  # ruff format .
make test    # pytest tests/ -x
make run     # streamlit run app.py
```

At runtime the app needs Java ≥ 21 and Evolver's release jar (`EVOLVER_VERSION` in
`evolver_studio/evolver_client.py`, currently 2.1), which the sidebar downloads from Maven Central
into `lib/`. No Evolver checkout is needed: parameter spaces and meta-optimizer configurations are
read from the jar, and the reference fronts/weight vectors training runs need are copied into
`resources/` (the JVM runs with this repo's root as working directory). Set `EVOLVER_JAR` to run
against a locally built Evolver jar instead — e.g. one built from Evolver's `develop` branch, to
run the drift-detection tests against changes not yet released. Tests that need the jar are skipped
without it. `make sync-resources` refreshes `resources/` (and its `SHA256SUMS` manifest) from the
GitHub source archive of the `EVOLVER_VERSION` release; run it whenever that version changes.

## Conventions

Follow `CODING_GUIDELINES.md` (Python style, typing, testing) and `GIT_GUIDELINES.md` (Conventional
Commits, atomic commits) in this repo's root.
