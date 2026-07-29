from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:  # noqa: S310
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_for(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_up(url):
            return True
        time.sleep(0.5)
    return False


def resolve_npm(npm: str) -> str:
    """Resolve npm on Windows where the executable is npm.cmd."""
    if Path(npm).is_file():
        return str(Path(npm))
    found = shutil.which(npm)
    if found:
        return found
    if os.name == "nt":
        for candidate in (f"{npm}.cmd", f"{npm}.exe", f"{npm}.bat"):
            found = shutil.which(candidate)
            if found:
                return found
    raise FileNotFoundError(
        f"Could not find '{npm}' on PATH. Install Node.js or pass --npm <path>."
    )


def spawn(command: Sequence[str], cwd: Path) -> subprocess.Popen[bytes]:
    cmd = list(command)
    kwargs: dict = {"cwd": str(cwd)}
    if os.name == "nt":
        # CreateProcess cannot run .cmd/.bat wrappers; route through cmd.exe.
        cmd = ["cmd.exe", "/c", *cmd]
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, **kwargs)


def terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=2)
            return
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        proc.terminate()
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch thin client module servers")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--npm", default="npm")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    npm = resolve_npm(args.npm)
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []

    try:
        stt_cmd = [
            args.python,
            str(ROOT / "audio" / "RealtimeSTT" / "example_fastapi_server" / "server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            "8010",
        ]
        sam_cmd = [
            args.python,
            str(ROOT / "vision" / "perception" / "mobilesam_server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            "7860",
        ]
        track_cmd = [
            args.python,
            str(ROOT / "vision" / "perception" / "track_server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            "7861",
        ]
        glb_cmd = [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]
        client_cmd = [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5174"]

        print(f"Using npm: {npm}")
        processes.append(("stt", spawn(stt_cmd, ROOT)))
        processes.append(("sam", spawn(sam_cmd, ROOT)))
        processes.append(("track", spawn(track_cmd, ROOT)))
        processes.append(("glb", spawn(glb_cmd, ROOT / "vision" / "tools" / "glb-viewer")))
        processes.append(("client", spawn(client_cmd, ROOT / "client")))

        checks = {
            "stt": "http://127.0.0.1:8010/health",
            "sam": "http://127.0.0.1:7860/health",
            "track": "http://127.0.0.1:7861/health",
            "glb": "http://127.0.0.1:5173/health",
            "client": "http://127.0.0.1:5174",
        }

        for name, url in checks.items():
            if wait_for(url, timeout_s=args.timeout):
                print(f"[ok] {name} ready -> {url}")
            else:
                print(f"[warn] {name} not ready -> {url}")

        print("\nThin client: http://127.0.0.1:5174")
        print("Press Ctrl+C to stop all services.")

        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"[exit] {name} exited with code {proc.returncode}")
                    return 1

    except KeyboardInterrupt:
        print("\nStopping services...")
        return 0
    finally:
        for _, proc in reversed(processes):
            terminate(proc)


if __name__ == "__main__":
    raise SystemExit(main())
