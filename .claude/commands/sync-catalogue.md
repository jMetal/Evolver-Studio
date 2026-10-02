Bring Evolver-Studio's catalogue (`evolver_studio/catalogue.py`) in line with an Evolver release:
its base algorithms, meta-optimizers and parameter space files, as the Explore and Training pages
show them.

The Evolver version to sync with is: $ARGUMENTS (empty: `EVOLVER_VERSION` in
`evolver_studio/evolver_client.py`).

Evolver-Studio follows **stable Evolver releases only**. Sync `main` with a release jar, never with
a SNAPSHOT. Work against Evolver's `develop` (a jar built locally and set with `EVOLVER_JAR`) goes on
an `experiment/*` branch and reaches `main` with the release, through `/bump-evolver`.

## Steps

### 1. Get the jar

- The release the app downloads: `lib/Evolver-<version>-jar-with-dependencies.jar`. If it is not
  there, download it from Maven Central
  (`https://repo1.maven.org/maven2/org/uma/jmetal/Evolver/<version>/`), the URL
  `evolver_client.MAVEN_CENTRAL_JAR_URL` builds.
- Check the version it records: `unzip -p <jar> META-INF/maven/org.uma.jmetal/Evolver/pom.properties`.
  Stop if it is a SNAPSHOT and the current branch is `main`.

Below, `JAR=<path of that jar>` and `PY=/opt/anaconda3/envs/evolver-studio/bin/python` (or
`conda run -n evolver-studio python`).

### 2. Collect what the jar offers

- The manifest of what `cli.training` can run:
  `java -cp $JAR org.uma.evolver.cli.training.DescribeMain` (`baseAlgorithms` with `encoding` and
  `requiredExtraConfigKeys`; `metaAlgorithms` with `supportsTree` and `operatorParameterSpaceFile`).
- The parameter spaces: `unzip -l $JAR 'parameterSpaces/*'`, and the meta-optimizer configurations:
  `unzip -l $JAR 'metaOptimizerConfigurations/*'`.
- For a parameter space whose content may have changed (new parameters, new values), compare it with
  the previous release's jar (`unzip -p <jar> parameterSpaces/<file> | diff - ...`).
- The Evolver changelog section of that release (`docs/changelog.rst` in the release's tag on
  GitHub), to see which algorithms and encodings were added, registered or removed.

### 3. Report the differences before editing

Present a table to the user with, per row, what changed and the edit it implies:

| Jar | Catalogue edit |
|---|---|
| base algorithm/encoding in the manifest, not `runnable_today` | `runnable_today=True`, `registry_name` (manifest name, e.g. `MOEAD`), `runnable_encodings`, `required_extra_config_keys` |
| meta-algorithm in the manifest, not `wired_into_cli_runner` | `wired_into_cli_runner=True`, `example_config_file`, `operator_parameter_space_file` |
| `supportsTree: true`, catalogue says no | `supports_tree=True`, `tree_operator_parameter_space_file` (`*MetaTree.yaml`), `tree_parameters` (the fields of its `Meta*TreeConfiguration.yaml`) |
| new `parameterSpaces/` file | a new encoding of a `BaseAlgorithm`, or an entry with its reason in `KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES` |
| file referenced by the catalogue, gone from the jar | remove the reference |
| new algorithm class without a parameter space or registry entry | nothing to add yet; mention it |

Wait for the user's confirmation if anything is ambiguous (e.g. a file that could be either a base
algorithm's space or a meta-level one).

### 4. Edit

- `evolver_studio/catalogue.py`: the entries above, with the one-line comment naming the Java
  class(es) each entry mirrors, as the existing ones do; the module docstring's list of runnable
  algorithms.
- `pages/training.py`: if a newly runnable algorithm needs extra configuration
  (`required_extra_config_keys`), check that the page asks for it (it does for
  `weightVectorFilesDirectory`; anything else needs a new input).
- `tests/test_catalogue.py`: the tests that pin exact sets (runnable base algorithms, wired
  meta-algorithms, tree support, weight vectors). Do not weaken the manifest and jar tests
  (`TestCatalogueMatchesDescribeManifest`, `TestCatalogueMatchesEvolverJar`): they are the check.
- Tutorials that name the available algorithms (`evolver_studio/tutorial_*.py`), if they no longer
  hold.
- `README.md`: the table of algorithms launchable from the app, and the list of the browsable-only
  ones. `ROADMAP.md`: a bullet under the shipped work.

### 5. Verify

```bash
EVOLVER_JAR=$JAR $PY -m ruff check . && $PY -m ruff format --check .
EVOLVER_JAR=$JAR $PY -m pytest -q
```

With the release jar that `EVOLVER_VERSION` names, no `EVOLVER_JAR` is needed. Every test must pass,
none skipped for lack of the jar.

Then look at the page: `EVOLVER_JAR=$JAR make run`, open Explore through the navigation, and check
each changed algorithm's table (its parameters, the *Active if* column) and, for meta-optimizers, the
Flat/Tree tabs. Without a browser at hand, a headless Chrome screenshot through the DevTools
protocol works (load `/`, then click the link whose `href` ends in `/explore`; opening `/explore`
directly bypasses `st.navigation`).

### 6. Commit

Atomic Conventional Commits with explicit paths (`GIT_GUIDELINES.md`), e.g.
`feat(catalogue): sync with Evolver <version>` for the catalogue, the Training page and their tests,
and `docs: ...` for README/ROADMAP. Do not push unless asked.
