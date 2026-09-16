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

This file documents the intended architecture agreed on before implementation started. A first,
deliberately minimal prototype now lives in the repo (`app.py`, `evolver_studio/`): it drives Evolver's
`org.uma.evolver.cli.runner.TrainingRunnerMain` (see `/Users/ajnebro/Softw/Evolver/docs/proposals/
cli-runner-prototype.md`) to run the single ZDT4 training case, as a smoke test of the
request/status/results contract before building the full MVP surface below. Expect this file's
architecture sections to be corrected/expanded as more of the real code lands.

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

**Evolver is not only a read-only external dependency.** When Evolver-Studio's needs require it, work
may extend into the Evolver checkout itself (`/Users/ajnebro/Softw/Evolver`, branch
`study/uniform-training-runner` — the branch `cli.runner` and this tool's integration code live on) to
propose or implement the missing pieces — e.g. expanding `BaseAlgorithmRegistry` beyond its current
NSGA-II/MOEA-D scope, or wiring additional `Meta*Builder` classes (`MetaSPEA2Builder`,
`MetaSMPSOBuilder`, `MetaAsyncNSGAIIBuilder`, ...) into `TrainingRunner` so more of them are
selectable as meta-optimizers, not just usable from `example.training`. Prefer proposing such changes
as a `docs/proposals/*.md` document in Evolver's repo first (matching the existing
`cli-runner-prototype.md`) before implementing them, consistent with how that branch's design was
itself introduced.

### Keeping Evolver-Studio's catalogue in sync with Evolver — the drift-detection mechanism

Evolver changes with some regularity (new base algorithms, new parameter-space components, new
meta-optimizer builders). Nothing forces Evolver-Studio's provisional catalogue
(`evolver_studio/catalogue.py`) to be updated when that happens, so both sides carry a matching
**drift-detection test**: a directory/source scan that fails loudly on anything new and untriaged,
instead of silently going stale.

- **Evolver-Studio side** (`tests/test_catalogue.py::TestCatalogueMatchesEvolverCheckout`): scans
  Evolver's `src/main/resources/parameterSpaces/` directory and fails if any `.yaml`/`.irace` file is
  neither referenced by `BASE_ALGORITHMS` nor listed in
  `KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES` (both in `evolver_studio/catalogue.py`).
- **Evolver side** (on `study/uniform-training-runner`,
  `src/test/java/org/uma/evolver/cli/runner/`): two JUnit tests with the same shape —
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
  │        (java -cp ... org.uma.evolver.cli.runner.TrainingRunnerMain request.yaml status.yaml)
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

The prototype also needs a working Evolver checkout to build/run against (JDK ≥ 21, Maven) — the
checkout path is provided at runtime via the Streamlit sidebar, default `/Users/ajnebro/Softw/Evolver`.

## Conventions

Follow `CODING_GUIDELINES.md` (Python style, typing, testing) and `GIT_GUIDELINES.md` (Conventional
Commits, atomic commits) in this repo's root.
