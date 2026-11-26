# PyInstaller spec for the desktop-packaging spike. Run from the repository root:
#   build/spike-venv/bin/pyinstaller packaging/evolver-studio.spec --noconfirm
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    (os.path.join(ROOT, "app.py"), "."),
    (os.path.join(ROOT, "app_pages"), "app_pages"),
    (os.path.join(ROOT, "assets"), "assets"),
    (os.path.join(ROOT, "resources"), "resources"),
    (os.path.join(ROOT, "evolver_studio", "tutorial_data"), "evolver_studio/tutorial_data"),
]
# The Java runtime made by packaging/build_jre.sh, if it was built.
if os.path.isdir(os.path.join(ROOT, "build", "jre")):
    datas.append((os.path.join(ROOT, "build", "jre"), "jre"))
binaries = []
# The pages are run by Streamlit as scripts, so PyInstaller cannot see what they import.
hiddenimports = collect_submodules("evolver_studio")
for package in (
    "streamlit",
    "plotly",
    "kaleido",
    # Kaleido runs Chrome through a script of this package, so it must be a file on disk.
    "choreographer",
    "moocore",
    "scipy",
    "pandas",
    "webview",
):
    # kaleido.mocker parses sys.argv when imported, which aborts the analysis.
    package_datas, package_binaries, package_hidden = collect_all(
        package, filter_submodules=lambda name: "mocker" not in name
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="evolver-studio", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="evolver-studio")
