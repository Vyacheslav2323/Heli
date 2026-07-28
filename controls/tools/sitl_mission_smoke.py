#!/usr/bin/env python3
"""
Run an ordered PX4 SITL mission from explicit steps (or a MissionSpec JSON).

Takeoff is skipped when already flying. Land runs only if you include a land step.

NED: +north is +X. ``north:N`` / ``east:E`` are **runtime-relative** to the
vehicle's position when that step runs (not absolute from NED origin).
Use ``goto:X,Y,Z`` for absolute local NED.

Examples (your intended mission)::

  python controls/tools/sitl_mission_smoke.py --alt 3 \\
    --step takeoff --step north:200 --step hold:10 --step turn:90 --step north:200

  python controls/tools/sitl_mission_smoke.py --mission examples/missions/north_legs.json --dry-run

Prerequisites: PX4 SITL in WSL; onboard mavlink ``udpin:0.0.0.0:14540``.
On Windows this script re-execs inside WSL automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ROOT / "controls"
if str(CONTROLS) not in sys.path:
    sys.path.insert(0, str(CONTROLS))

DEFAULT_URL = "udpin:0.0.0.0:14540"
WSL_DISTRO = os.environ.get("HELI_WSL_DISTRO", "Ubuntu-24.04")


def build_spec_from_steps(steps: list[str], *, alt_m: float, acceptance_m: float) -> dict[str, Any]:
    """Parse ordered --step strings into a MissionSpec.

    Step forms:
      takeoff[:ALT]
      north:DELTA          move DELTA m in +X from **current** position (runtime-relative)
      east:DELTA           move DELTA m in +Y from **current** position (runtime-relative)
      goto:X,Y,Z           absolute NED (z positive down)
      hold:SECONDS
      yaw:DEG / turn:DEG   relative turn (degrees; +CW looking down). yaw:180 ≈ turn around
      heading:DEG          absolute heading (0=North, 90=East)
      land
    """
    goals: list[dict[str, Any]] = []
    cruise_z = -abs(alt_m)
    span = 0.0

    for raw in steps:
        step = raw.strip()
        if not step:
            continue
        if ":" in step:
            kind, _, rest = step.partition(":")
            kind = kind.strip().lower()
            rest = rest.strip()
        else:
            kind, rest = step.lower(), ""

        if kind == "takeoff":
            takeoff_alt = float(rest) if rest else float(alt_m)
            cruise_z = -abs(takeoff_alt)
            goals.append({"type": "takeoff", "alt_m": takeoff_alt})
        elif kind == "north":
            dx = float(rest)
            span += abs(dx)
            goals.append(_goto_rel(dx=dx, dy=0.0, z=cruise_z, acceptance_m=acceptance_m))
        elif kind == "east":
            dy = float(rest)
            span += abs(dy)
            goals.append(_goto_rel(dx=0.0, dy=dy, z=cruise_z, acceptance_m=acceptance_m))
        elif kind == "goto":
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) != 3:
                raise ValueError(f"goto expects X,Y,Z got {step!r}")
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            cruise_z = z
            span += abs(x) + abs(y)
            goals.append(_goto_abs(x, y, z, acceptance_m))
        elif kind == "hold":
            goals.append({"type": "hold", "duration_s": float(rest)})
        elif kind in ("yaw", "turn", "yaw_rel"):
            goals.append({"type": "yaw_to", "yaw_deg": float(rest), "relative": True})
        elif kind == "heading":
            goals.append({"type": "yaw_to", "yaw_deg": float(rest) % 360.0, "relative": False})
        elif kind == "land":
            goals.append({"type": "land"})
        else:
            raise ValueError(
                f"unknown step {step!r}; "
                "use takeoff, north:, east:, goto:, hold:, yaw:/turn: (relative), heading: (absolute), land"
            )
    if not goals:
        raise ValueError("no steps provided")

    return {
        "version": 1,
        "name": "sitl-steps",
        "description": "CLI --step mission (north/east are runtime-relative)",
        "abort_policy": "rtl",
        "constraints": {
            "max_alt_m": max(abs(alt_m) * 2, 30.0, abs(cruise_z) + 5),
            "min_alt_m": 0.0,
        },
        "goals": goals,
        "_hint_span_m": span,
    }


def _goto_abs(x: float, y: float, z: float, acceptance_m: float) -> dict[str, Any]:
    return {
        "type": "goto_ned",
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "acceptance_radius_m": float(acceptance_m),
    }


def _goto_rel(*, dx: float, dy: float, z: float, acceptance_m: float) -> dict[str, Any]:
    """Relative horizontal move; absolute cruise altitude z (NED down)."""
    return {
        "type": "goto_ned",
        "dx": float(dx),
        "dy": float(dy),
        "z": float(z),
        "acceptance_radius_m": float(acceptance_m),
    }

def load_mission_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mission JSON must be an object")
    return data


def _win_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    s = str(resolved).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def _reexec_in_wsl(argv: list[str]) -> int:
    wsl_root = _win_path_to_wsl(ROOT)
    forwarded = [a for a in argv[1:] if a != "--no-wsl"]
    inner = (
        f"export HELI_IN_WSL=1 PYTHONPATH={shlex.quote(wsl_root + '/controls')}:$PYTHONPATH; "
        f"cd {shlex.quote(wsl_root)} && "
        f"python3 controls/tools/sitl_mission_smoke.py {shlex.join(forwarded)}"
    )
    cmd = ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", inner]
    print(f"[win] re-exec in WSL ({WSL_DISTRO}) for onboard MAVLink...", flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="PX4 SITL ordered mission runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s --alt 3 --step takeoff --step north:200 --step hold:10 --step turn:90 --step north:200
  %(prog)s --mission examples/missions/north_legs.json
  %(prog)s --alt 3 --step takeoff --step north:200 --step land --dry-run
""",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="pymavlink connection URL")
    parser.add_argument("--alt", type=float, default=5.0, help="default takeoff/cruise AGL (m)")
    parser.add_argument(
        "--step",
        action="append",
        default=[],
        metavar="STEP",
        help="ordered step (repeatable): takeoff, north:N, east:E, goto:X,Y,Z, hold:S, yaw:/turn:DEG (relative), heading:DEG (absolute), land",
    )
    parser.add_argument("--mission", type=Path, help="MissionSpec JSON file (alternative to --step)")
    parser.add_argument("--acceptance", type=float, default=2.0, help="goto acceptance radius (m)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print NED position/velocity during gotos (also logs pos at each command boundary)",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan+validate only, no MAVLink")
    parser.add_argument(
        "--no-wsl",
        action="store_true",
        help="do not re-exec into WSL (use when already on the PX4 host)",
    )
    args = parser.parse_args(argv[1:])

    if (
        os.name == "nt"
        and not args.dry_run
        and not args.no_wsl
        and os.environ.get("HELI_IN_WSL") != "1"
    ):
        return _reexec_in_wsl(argv)

    from flight.adapter import AbortPolicy
    from flight.px4.adapter import Px4MavlinkAdapter
    from mission.executor import ExecutorState, MissionExecutor
    from mission.planner import plan_mission
    from mission.validator import validate_plan

    if args.mission and args.step:
        print("[FAIL] use either --mission or --step, not both", flush=True)
        return 2
    if args.mission:
        spec = load_mission_json(args.mission)
    elif args.step:
        try:
            spec = build_spec_from_steps(args.step, alt_m=args.alt, acceptance_m=args.acceptance)
        except ValueError as exc:
            print(f"[FAIL] {exc}", flush=True)
            return 2
    else:
        print(
            "[FAIL] provide --step ... or --mission file.json\n"
            "  example: --alt 3 --step takeoff --step north:200 --step hold:10 "
            "--step turn:90 --step north:200",
            flush=True,
        )
        return 2

    hint_span = float(spec.pop("_hint_span_m", 0) or 0)
    commands = plan_mission(spec)
    validation = validate_plan(spec, commands)
    if not validation.ok:
        for issue in validation.issues:
            print(f"[FAIL] validate: {issue.code}: {issue.message}", flush=True)
        return 2

    print(f"[ok] plan ({len(commands)} cmds):", flush=True)
    for i, c in enumerate(commands):
        print(f"  [{i}] {c}", flush=True)

    if args.dry_run:
        print("[ok] dry-run complete", flush=True)
        return 0

    # Scale goto timeouts for long legs (adapter default is 60s).
    goto_timeout = max(60.0, hint_span * 1.5 + 30.0)

    adapter = Px4MavlinkAdapter(args.url)
    adapter.debug = bool(args.debug)
    conn = adapter.connect()
    if not conn.ok:
        print(f"[FAIL] connect: {conn.message}", flush=True)
        return 1
    print(f"[ok] {conn.message}", flush=True)
    time.sleep(1.0)
    tel0 = adapter.telemetry()
    print(
        f"[ok] start NED=({tel0.x:.1f},{tel0.y:.1f},{tel0.z:.1f}) m  "
        f"yaw={tel0.yaw_deg:.1f}°  (units are meters)",
        flush=True,
    )
    if abs(tel0.x) > 10 or abs(tel0.y) > 10:
        print(
            "[warn] vehicle is not near NED origin — north:/east: steps are relative to "
            f"THIS position, so two north:-200 legs end near "
            f"({tel0.x - 400:.0f}, {tel0.y:.0f}) if both succeed",
            flush=True,
        )

    # Patch execute path timeouts for long goto_ned by wrapping adapter.execute
    _orig_execute = adapter.execute

    def execute_with_timeout(command: dict) -> Any:
        if command.get("type") == "goto_ned":
            # Resolve relative deltas here so timeout scaling still applies.
            if any(k in command for k in ("dx", "dy", "dz")):
                tel = adapter.telemetry()
                x = tel.x + float(command.get("dx", 0.0))
                y = tel.y + float(command.get("dy", 0.0))
                if "dz" in command:
                    z = tel.z + float(command["dz"])
                elif "z" in command:
                    z = float(command["z"])
                else:
                    z = tel.z
                print(
                    f"[goto] relative dx={command.get('dx', 0)} dy={command.get('dy', 0)} "
                    f"from ({tel.x:.1f},{tel.y:.1f},{tel.z:.1f}) -> ({x:.1f},{y:.1f},{z:.1f})",
                    flush=True,
                )
            else:
                x = float(command["x"])
                y = float(command["y"])
                z = float(command["z"])
            return adapter.goto_ned(
                x,
                y,
                z,
                command.get("yaw_deg"),
                acceptance_radius_m=float(command.get("acceptance_radius_m", args.acceptance)),
                timeout_s=goto_timeout,
            )
        return _orig_execute(command)

    adapter.execute = execute_with_timeout  # type: ignore[method-assign]

    def on_event(ev):
        tel = adapter.telemetry()
        pos = f"NED=({tel.x:.1f},{tel.y:.1f},{tel.z:.1f}) vxy={tel.speed_xy_m_s:.1f}"
        print(f"[{ev.kind}] idx={ev.index} {ev.message} {ev.command or ''} | {pos}", flush=True)

    abort = AbortPolicy(spec.get("abort_policy", "rtl"))
    executor = MissionExecutor(adapter, abort_policy=abort, on_event=on_event)
    t0 = time.monotonic()
    result = executor.run(commands)
    dt = time.monotonic() - t0
    adapter.disconnect()

    if result.state != ExecutorState.SUCCEEDED:
        print(f"[FAIL] state={result.state.value} error={result.error} ({dt:.1f}s)", flush=True)
        return 1

    print(f"[PASS] completed {result.completed} commands in {dt:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
