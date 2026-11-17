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

## What the spike found

- Packaging the app as it is works: Explore, Run algorithm (Java run, front, figures) in the
  bundle (about 440 MB, without a JRE or the Evolver jar). No framework change is needed.
- `kaleido.mocker` parses `sys.argv` when imported, which aborts PyInstaller's analysis: the spec
  leaves it out.
- In a frozen app `sys.executable` is the app itself. Kaleido starts Chrome with
  `[sys.executable, script.py, ...]` and the validation worker with `[sys.executable, "-m", ...]`,
  so the launcher runs those as a Python interpreter would; otherwise each one opens another window
  and leaves an orphan server.
- Without Chrome the app works and only the PNG/PDF buttons of the charts are missing
  (`figure_export.py` probes Kaleido once and hides them). Kaleido's `get_chrome_sync()` could
  download one on demand.

## Not done yet

- The jar, the runs and the results are written inside the bundle (`STUDIO_HOME` comes from
  `__file__`): they need a user-data directory.
- A JRE: an app opened from the Finder does not inherit the shell's `PATH`, and macOS's
  `/usr/bin/java` is only a stub.
- Untested: Training, Validation, the tutorials, code signing and notarization, Windows and Linux.
