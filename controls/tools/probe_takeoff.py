#!/usr/bin/env python3
"""Quick takeoff/land probe against local PX4 SITL onboard mavlink."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controls"))

from flight.px4.adapter import Px4MavlinkAdapter  # noqa: E402


def main() -> int:
    a = Px4MavlinkAdapter("udpin:0.0.0.0:14540")
    conn = a.connect()
    print(conn.message, flush=True)
    if not conn.ok:
        return 1
    time.sleep(1.5)
    print("tel", a.telemetry(), flush=True)
    r = a.takeoff(3.0)
    print("takeoff", r.ok, r.message, flush=True)
    print("tel2", a.telemetry(), flush=True)
    if r.ok:
        print(a.land().message, flush=True)
    else:
        a.disarm(force=True)
    a.disconnect()
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
