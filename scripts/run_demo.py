"""One-command demo: starts the gateway, then traffic + dashboard pointers."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def wait_healthy(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"gateway at {url} did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser(description="DietGate one-command demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scenario", default="fast")
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "dietgate.main:app", "--host", "127.0.0.1",
         "--port", str(args.port)],
        cwd=str(REPO),
    )
    try:
        wait_healthy(base)
        print(f"DietGate gateway running: {base}")
        print(f"Dashboard: {base}/  (admin key: see config/app.yaml)")
        sim = subprocess.Popen(
            [PYTHON, "-m", "sim.traffic", "--scenario", args.scenario, "--url", base],
            cwd=str(REPO),
        )
        if args.open:
            webbrowser.open(base)
        print(f"simulator running scenario '{args.scenario}' - watch the dashboard fill up. Ctrl+C to stop.")
        try:
            sim.wait()
        except KeyboardInterrupt:
            sim.terminate()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
