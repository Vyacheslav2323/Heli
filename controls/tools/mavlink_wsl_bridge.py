#!/usr/bin/env python3
"""
Bridge PX4 SITL (WSL2 NAT) <-> QGroundControl (Windows) on Windows 10.

WSL2 NAT blocks UDP from Linux -> Windows, and Win10 lacks mirrored networking.
This process shuttles MAVLink over WSL stdio so QGC on Windows localhost works.

Usage (QGC AutoConnect UDP 14550 enabled; PX4 SITL running with default mavlink):
  python controls/tools/mavlink_wsl_bridge.py
"""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import sys
import threading
import time


def read_exact(stream, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def windows_bridge(qgc_port: int = 14550, px4_port: int = 18570, win_sport: int = 15570) -> None:
    # Avoid binding Windows ports that WSL localhost-forwards into the distro (e.g. 18570).
    qgc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Do not SO_REUSEADDR — on Windows that lets stale bridges steal replies.
    qgc_sock.bind(("127.0.0.1", win_sport))
    qgc_sock.settimeout(0.05)
    print(f"[win] QGC face {qgc_sock.getsockname()} <-> 127.0.0.1:{qgc_port}", flush=True)

    script_wsl = "/mnt/c/Users/yj.park/Repo/helicopter/controls/tools/mavlink_wsl_side.py"
    proc = subprocess.Popen(
        [
            "wsl",
            "-d",
            "Ubuntu-24.04",
            "--",
            "python3",
            "-u",
            script_wsl,
            str(px4_port),
            str(qgc_port),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0,
    )
    assert proc.stdin and proc.stdout
    print("[win] WSL helper started", flush=True)

    stop = threading.Event()
    # Queue packets from WSL stdout reader thread -> main loop
    from collections import deque

    from_wsl: deque[bytes] = deque()
    sock_lock = threading.Lock()
    stats = {"to_qgc": 0, "from_qgc": 0}

    def wsl_stdout_reader() -> None:
        while not stop.is_set():
            hdr = read_exact(proc.stdout, 4)
            if not hdr or len(hdr) < 4:
                stop.set()
                break
            (n,) = struct.unpack("!I", hdr)
            data = read_exact(proc.stdout, n)
            if not data or len(data) < n:
                stop.set()
                break
            from_wsl.append(data)

    t = threading.Thread(target=wsl_stdout_reader, daemon=True)
    t.start()
    print("[win] bridge running — keep QGC AutoConnect UDP 14550 on", flush=True)

    last_report = time.time()
    try:
        while not stop.is_set():
            # Drain WSL -> QGC
            while from_wsl:
                data = from_wsl.popleft()
                with sock_lock:
                    try:
                        qgc_sock.sendto(data, ("127.0.0.1", qgc_port))
                        stats["to_qgc"] += 1
                    except OSError as e:
                        print(f"[win] send to QGC failed: {e}", flush=True)
                        stop.set()
                        break

            # Poll QGC -> WSL
            try:
                data, _addr = qgc_sock.recvfrom(65535)
            except socket.timeout:
                data = None
            except OSError:
                data = None
            if data:
                try:
                    proc.stdin.write(struct.pack("!I", len(data)) + data)
                    proc.stdin.flush()
                    stats["from_qgc"] += 1
                except BrokenPipeError:
                    stop.set()
                    break

            now = time.time()
            if now - last_report >= 2.0:
                print(
                    f"[win] pkts to_qgc={stats['to_qgc']} from_qgc={stats['from_qgc']}",
                    flush=True,
                )
                last_report = now
    except KeyboardInterrupt:
        print("\n[win] stopping", flush=True)
    finally:
        stop.set()
        try:
            proc.terminate()
        except Exception:
            pass
        qgc_sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qgc-port", type=int, default=14550)
    parser.add_argument("--px4-port", type=int, default=18570)
    parser.add_argument("--win-sport", type=int, default=15570)
    args = parser.parse_args()
    windows_bridge(args.qgc_port, args.px4_port, args.win_sport)


if __name__ == "__main__":
    main()
