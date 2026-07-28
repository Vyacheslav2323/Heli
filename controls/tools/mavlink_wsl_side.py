#!/usr/bin/env python3
"""WSL side of mavlink_wsl_bridge.py — stdio framed UDP shuttle.

Uses ONE socket bound to 127.0.0.1:14550 for both:
  - receiving PX4 → QGC traffic (PX4 sends to :14550)
  - sending QGC → PX4 traffic (so PX4 keeps partner = :14550)
"""

from __future__ import annotations

import select
import socket
import struct
import sys
import threading


def read_exact(stream, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def main() -> None:
    px4_port = int(sys.argv[1]) if len(sys.argv) > 1 else 18570
    listen_port = int(sys.argv[2]) if len(sys.argv) > 2 else 14550

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", listen_port))
    sock.setblocking(False)

    sys.stderr.write(
        f"[wsl] bound 127.0.0.1:{listen_port} (rx from px4 + tx to px4:{px4_port})\n"
    )
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    stop = threading.Event()
    stats = {"to_win": 0, "to_px4": 0}

    def stdin_to_px4() -> None:
        while not stop.is_set():
            hdr = read_exact(stdin, 4)
            if not hdr:
                stop.set()
                break
            (n,) = struct.unpack("!I", hdr)
            data = read_exact(stdin, n)
            if not data:
                stop.set()
                break
            try:
                sock.sendto(data, ("127.0.0.1", px4_port))
                stats["to_px4"] += 1
            except OSError as e:
                sys.stderr.write(f"[wsl] send to px4 failed: {e}\n")
                stop.set()
                break

    t = threading.Thread(target=stdin_to_px4, daemon=True)
    t.start()

    last_report = 0
    try:
        while not stop.is_set():
            r, _, _ = select.select([sock], [], [], 0.2)
            if r:
                try:
                    data, _addr = sock.recvfrom(65535)
                except OSError:
                    data = None
                if data:
                    try:
                        stdout.write(struct.pack("!I", len(data)) + data)
                        stdout.flush()
                        stats["to_win"] += 1
                    except BrokenPipeError:
                        break
            # periodic stats on stderr
            import time

            now = time.time()
            if now - last_report >= 2.0:
                sys.stderr.write(
                    f"[wsl] pkts to_win={stats['to_win']} to_px4={stats['to_px4']}\n"
                )
                sys.stderr.flush()
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sock.close()


if __name__ == "__main__":
    main()
