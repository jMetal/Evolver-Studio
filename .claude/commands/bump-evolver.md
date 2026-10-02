Move Evolver-Studio to a new stable Evolver release.

The new Evolver version is: $ARGUMENTS (e.g. `2.2`).

Run it only after the release is public: Evolver's jar must be on Maven Central
(`https://repo1.maven.org/maven2/org/uma/jmetal/Evolver/<version>/`) and its tag `v<version>` on
GitHub (`https://github.com/jMetal/Evolver/releases`). Check both first (`curl -sI` on the jar and on
`https://codeload.github.com/jMetal/Evolver/tar.gz/refs/tags/v<version>`) and stop if either is
missing. Never bump to a SNAPSHOT: Evolver-Studio follows stable releases only.

## Steps

### 1. The version

- `evolver_studio/evolver_client.py`: `EVOLVER_VERSION` and the comment above it (the `v<version>`
  tag).
- If `evolver_studio/catalogue.py` still has a separate `CATALOGUE_EVOLVER_VERSION`, remove it:
  with stable releases only, the catalogue mirrors `EVOLVER_VERSION`. Make
  `is_older_than_catalogue` compare against `EVOLVER_VERSION` (and update its docstring, the warning
  text in `evolver_studio/app_state.py` and `tests/test_catalogue.py`'s parametrized versions).

### 2. The jar

Download it into `lib/` (the sidebar's **Download Evolver <version>** button does it, or `curl` the
`MAVEN_CENTRAL_JAR_URL` and its `.sha1`, and compare). Keep the previous jar until the tests pass,
to compare parameter spaces in the next step.

### 3. The resources

`make sync-resources`: it replaces `resources/` (reference fronts, weight vectors, TSP instances)
with those of tag `v<version>` and rewrites `resources/SHA256SUMS`. Review `git status resources/`:
new fronts are expected, deleted ones must be explained by Evolver's changelog. Update the tag in
`resources/README.md`.

### 4. The catalogue

Run `/sync-catalogue <version>` against the new jar. It reports and applies the changes to
`evolver_studio/catalogue.py`, the Training page, the tests, README and ROADMAP.

### 5. The documentation

Replace the old version where it names the release the app runs (not in history entries):
`README.md` (the `java -cp Evolver-<version>-jar-with-dependencies.jar` diagram, "downloads the jar
of **Evolver <version>**", "**Download Evolver <version>**", footnotes saying a feature needs a
newer jar), `CLAUDE.md` ("currently <version>"), `ROADMAP.md` (the "Track Evolver's releases" item,
and a shipped bullet: "Moved to Evolver <version>"). `git grep -n "<old version>"` finds them.

### 6. Verify

```bash
make lint
make test          # no EVOLVER_JAR: the release jar in lib/
```

Every test must pass with the release jar, none skipped. Then `make run` and check the home page
names the new version, the sidebar says "Evolver <version>", and Explore shows no "older jar"
warning. Remove the previous jar from `lib/` (it is not tracked).

### 7. Commit

Atomic Conventional Commits with explicit paths, e.g.:
- `build: move to Evolver <version>` (`EVOLVER_VERSION`, the version check, its tests),
- `chore(resources): sync resources with Evolver <version>`,
- the `feat(catalogue)` commit(s) from `/sync-catalogue`,
- `docs: describe Evolver <version> as the release the app runs`.

Work on `develop` (`main` holds Studio's releases), and add the entry to `CHANGELOG.md`'s unreleased
section ("Works with Evolver <version>"; the README's compatibility table too). Do not push unless
asked.
