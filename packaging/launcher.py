"""Desktop launcher (spike): Streamlit server in a child process, shown in a native window.

Streamlit installs signal handlers, which only works in a process's main thread, and the
native window wants the main thread too, so the server runs as a child process: the same
executable, started with `--serve PORT`.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HEALTH_TIMEOUT_SECONDS = 60
# The directory the app writes to (see evolver_studio.evolver_client.DATA_DIRECTORY).
DATA_DIRECTORY_VARIABLE = "EVOLVER_STUDIO_DATA"


def bundle_root() -> Path:
    """The directory holding app.py, pages/, resources/..., frozen or not."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def user_data_directory() -> Path:
    """Where the packaged app keeps the jar, the runs and the results: the platform's own place."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Evolver-Studio"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "Evolver-Studio"
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "evolver-studio"


def serve(port: int) -> None:
    from streamlit.web import bootstrap

    root = bundle_root()
    os.chdir(root)
    flag_options = {
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(str(root / "app.py"), False, [], flag_options)


def wait_until_healthy(port: int) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.3)
    raise TimeoutError("The Streamlit server did not start")


def main() -> None:
    import webview

    # The charts' PNG/PDF buttons are download links, which the window only follows otherwise.
    webview.settings["ALLOW_DOWNLOADS"] = True
    port = free_port()
    data_directory = os.environ.get(DATA_DIRECTORY_VARIABLE) or str(user_data_directory())
    command = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, __file__]
    server = subprocess.Popen(
        [*command, "--serve", str(port)],
        env={**os.environ, DATA_DIRECTORY_VARIABLE: data_directory},
    )
    try:
        wait_until_healthy(port)
        webview.create_window("Evolver-Studio", f"http://127.0.0.1:{port}", width=1400, height=900)
        webview.start()
    finally:
        server.terminate()
        server.wait(timeout=10)


def run_as_interpreter(arguments: list[str]) -> None:
    """Behave like `python <script>` or `python -m <module>`.

    Libraries start helpers with `[sys.executable, script, ...]` (kaleido starts Chrome that
    way) and the app does `[sys.executable, "-m", module]`. In a frozen app `sys.executable`
    is this program, which must run those instead of opening another window.
    """
    import runpy

    if arguments[0] == "-m":
        sys.argv = arguments[1:]
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = arguments
        runpy.run_path(arguments[0], run_name="__main__")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) == 2 and arguments[0] == "--serve":
        serve(int(arguments[1]))
    elif arguments and (arguments[0] == "-m" or arguments[0].endswith(".py")):
        run_as_interpreter(arguments)
    else:
        main()
