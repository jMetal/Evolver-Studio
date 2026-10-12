"""Take the screenshots of the README from a running Evolver-Studio, with headless Chrome.

Run the app first (`make run`), then:

    python scripts/screenshots.py [--url http://localhost:8501] [--with-training]

The images go to docs/images/. They show what is in the app's folders, so take them where:

- Training analysis lists a finished training (the one of tutorial *Your first guided training*
  is a good one);
- Validation analysis lists a study with several problems and algorithms (`--study` chooses it by
  part of its name; the first one listed otherwise).

`--with-training` also launches the Training page's default training, waits for its monitor to
fill and captures it, then cancels the training: its folder stays in cli-runner-runs/, marked as
cancelled.

The captures are cropped to the content of a page or of a tab, which shows no local paths. Chrome
is driven through its DevTools protocol, with the `websockets` package Streamlit already needs.
"""

import argparse
import asyncio
import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets
from PIL import Image

OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "docs" / "images"
WINDOW = (1400, 2200)
WIDTH = 1300
DEBUGGING_PORT = 9341
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
)
# The bounding box of the visible tab's content, or of the page's main column.
TAB_PANEL = """(() => {
  const panel = [...document.querySelectorAll('[role=tabpanel]')]
    .find(p => p.offsetParent !== null);
  const r = panel.getBoundingClientRect();
  return [r.x, r.y + window.scrollY, r.width, r.height];
})()"""
# The page down to the end of the second row of cards of the home page.
HOME_TOP = """(() => {
  const cards = [...document.querySelectorAll('[data-testid=stVerticalBlockBorderWrapper]')];
  const bottom = cards.length > 4 ? cards[4].getBoundingClientRect().bottom : 900;
  const main = document.querySelector('[data-testid=stMainBlockContainer]');
  return [0, 0, main.getBoundingClientRect().right + 24, bottom + window.scrollY];
})()"""
# Hide what is not the app's: Streamlit's toolbar (Deploy, the menu).
HIDE_TOOLBAR = """(() => {
  const style = document.createElement('style');
  style.textContent = '[data-testid=stToolbar], [data-testid=stDecoration] {display: none;}';
  document.head.appendChild(style);
})()"""


class Browser:
    """A headless Chrome tab, driven through the DevTools protocol."""

    def __init__(self, socket) -> None:
        self.socket = socket
        self.counter = 0

    async def call(self, method: str, **params) -> dict:
        self.counter += 1
        number = self.counter
        await self.socket.send(json.dumps({"id": number, "method": method, "params": params}))
        while True:
            message = json.loads(await self.socket.recv())
            if message.get("id") == number:
                return message.get("result", message)

    async def evaluate(self, expression: str):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return result["result"].get("value")

    async def open(self, url: str, wait: float = 7) -> None:
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(wait)
        await self.evaluate(HIDE_TOOLBAR)

    async def choose(self, index: int, text: str, wait: float = 6) -> None:
        """Choose, in the page's selectbox number `index`, the option that `text` filters to.

        As a person would: a click of the mouse on it, `text` typed, and Enter.
        """
        x, y = await self.evaluate(
            f"""(() => {{
              const r = document.querySelectorAll('[data-testid=stSelectbox] input')[{index}]
                .getBoundingClientRect();
              return [r.x + r.width / 2, r.y + r.height / 2];
            }})()"""
        )
        for kind in ("mousePressed", "mouseReleased"):
            await self.call(
                "Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1
            )
        await asyncio.sleep(1)
        await self.call("Input.insertText", text=text)
        await asyncio.sleep(1)
        for kind in ("keyDown", "keyUp"):
            await self.call(
                "Input.dispatchKeyEvent",
                type=kind,
                key="Enter",
                code="Enter",
                windowsVirtualKeyCode=13,
            )
        await asyncio.sleep(wait)

    async def click(self, text: str, selector: str = "button", wait: float = 4) -> bool:
        found = await self.evaluate(
            f"""(() => {{
              const e = [...document.querySelectorAll({json.dumps(selector)})]
                .find(e => e.innerText.trim().endsWith({json.dumps(text)}));
              if (!e) return false;
              e.click();
              return true;
            }})()"""
        )
        await asyncio.sleep(wait)
        return bool(found)

    async def tab(self, name: str, wait: float = 6) -> bool:
        return await self.click(name, "[role=tab]", wait)

    async def capture(self, box_script: str, name: str, margin: int = 16) -> Path:
        x, y, width, height = await self.evaluate(box_script)
        shot = await self.call(
            "Page.captureScreenshot",
            format="png",
            captureBeyondViewport=True,
            clip={
                "x": max(x - margin, 0),
                "y": y,
                "width": width + 2 * margin,
                "height": height + margin,
                "scale": 1,
            },
        )
        return _save(base64.b64decode(shot["data"]), name)


def _save(png: bytes, name: str) -> Path:
    """Scale the image to the README's width and store it, as small as it stays sharp."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.width > WIDTH:
        image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIRECTORY / f"{name}.png"
    image.quantize(colors=256, dither=Image.Dither.NONE).save(path, optimize=True)
    print(f"{path.name}: {image.width}x{image.height}, {path.stat().st_size // 1024} KB")
    return path


def _chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file() or shutil.which(candidate):
            return candidate
    raise SystemExit("Chrome or Chromium was not found")


async def _take(url: str, with_training: bool, study: str | None) -> None:
    targets = None
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEBUGGING_PORT}/json") as response:
                targets = json.load(response)
            break
        except OSError:
            time.sleep(0.2)
    if not targets:
        raise SystemExit("Chrome did not start")
    page = next(target for target in targets if target["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as socket:
        browser = Browser(socket)
        await browser.call("Page.enable")
        await browser.call(
            "Emulation.setDeviceMetricsOverride",
            width=WINDOW[0],
            height=WINDOW[1],
            deviceScaleFactor=1,
            mobile=False,
        )

        await browser.open(url, wait=9)
        await browser.capture(HOME_TOP, "home", margin=0)

        await browser.open(f"{url}/training_analysis")
        await browser.tab("Convergence")
        await browser.capture(TAB_PANEL, "training-analysis")

        await browser.open(f"{url}/validation_analysis", wait=10)
        if study is not None:
            await browser.choose(0, study, wait=10)
        await browser.tab("Effect size")
        await browser.capture(TAB_PANEL, "validation-effect-size")
        await browser.tab("Fronts")
        await browser.capture(TAB_PANEL, "validation-fronts")

        if with_training:
            await browser.open(f"{url}/training")
            await browser.click("Show the meta-optimizer's population while it runs", "label")
            await browser.click("Launch training", wait=90)
            await browser.tab("Population", wait=8)
            await browser.capture(TAB_PANEL, "training-monitor")
            await browser.click("Cancel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--with-training", action="store_true")
    parser.add_argument(
        "--study", help="Part of the name of the validation study to show (default: the first)"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome = subprocess.Popen(
            [
                _chrome(),
                "--headless=new",
                f"--remote-debugging-port={DEBUGGING_PORT}",
                f"--window-size={WINDOW[0]},{WINDOW[1]}",
                "--hide-scrollbars",
                f"--user-data-dir={profile}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            asyncio.run(_take(args.url.rstrip("/"), args.with_training, args.study))
        finally:
            chrome.terminate()
            chrome.wait(timeout=10)


if __name__ == "__main__":
    main()
