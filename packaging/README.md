# Desktop packaging (experiment)

A spike on shipping Evolver-Studio as a desktop application: PyInstaller bundles the Streamlit app,
and a native window (pywebview) shows it. Not part of a release yet.

```bash
python -m venv build/spike-venv
build/spike-venv/bin/pip install -e . pyinstaller pywebview
build/spike-venv/bin/pyinstaller packaging/evolver-studio.spec --noconfirm \
    --distpath build/dist --workpath build/work
build/dist/evolver-studio/evolver-studio          # the window
build/dist/evolver-studio/evolver-studio --serve 8765   # only the server, to open in a browser
```

`launcher.py` starts the Streamlit server as a child process of the same executable (Streamlit
installs signal handlers, which need a main thread, and so does the native window) and waits for
its health endpoint before opening the window.

The launcher points `EVOLVER_STUDIO_DATA` at the platform's user-data directory (for example
`~/Library/Application Support/Evolver-Studio`), so that the jar, the runs and the results are not
written inside the bundle; the app copies `resources/` there.

## What the spike found

- Packaging the app as it is works: Explore, Run algorithm (Java run, front, figures) in the
  bundle (about 440 MB, without a JRE or the Evolver jar). No framework change is needed.
- `kaleido.mocker` parses `sys.argv` when imported, which aborts PyInstaller's analysis: the spec
  leaves it out.
- In a frozen app `sys.executable` is the app itself. Kaleido starts Chrome with
  `[sys.executable, script.py, ...]` and the validation worker with `[sys.executable, "-m", ...]`,
  so the launcher runs those as a Python interpreter would; otherwise each one opens another window
  and leaves an orphan server.
- Kaleido runs Chrome through a script of `choreographer`, run as `[sys.executable, script.py]`, so
  that package must be collected as files (in the PYZ archive the script is not on disk and Kaleido
  fails, silently hiding the PNG/PDF buttons).
- The first run of a freshly built, unsigned bundle is slow (tens of seconds in imports) while macOS
  scans its libraries; later runs take a second.
- Without Chrome the app works and only the PNG/PDF buttons of the charts are missing
  (`figure_export.py` probes Kaleido once and hides them). Kaleido's `get_chrome_sync()` could
  download one on demand.

## Not done yet

- A JRE: an app opened from the Finder does not inherit the shell's `PATH`, and macOS's
  `/usr/bin/java` is only a stub.
- Untested: Training, Validation, the tutorials, code signing and notarization, Windows and Linux.
