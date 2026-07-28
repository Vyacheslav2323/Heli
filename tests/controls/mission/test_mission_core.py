"""Unit tests for mission planner / validator / executor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flight.adapter import AbortPolicy, validate_command_sequence
from flight.px4.mock import MockFlightAdapter
from mission.executor import ExecutorState, MissionExecutor
from mission.planner import plan_mission
from mission.validator import VehicleSnapshot, validate_plan

ROOT = Path(__file__).resolve().parents[3]
SCHEMAS = ROOT / "controls" / "schemas"


def _sample_spec(**overrides):
    spec = {
        "version": 1,
        "name": "phase2-smoke",
        "abort_policy": "land",
        "constraints": {
            "max_alt_m": 50,
            "min_alt_m": 1,
            "min_battery_pct": 20,
            "geofence": {
                "polygon": [
                    {"lat": 35.17, "lon": 128.08},
                    {"lat": 35.17, "lon": 128.10},
                    {"lat": 35.19, "lon": 128.10},
                    {"lat": 35.19, "lon": 128.08},
                ]
            },
        },
        "goals": [
            {"type": "takeoff", "alt_m": 10},
            {"type": "goto_ned", "x": 5, "y": 0, "z": -10, "yaw_deg": 90},
            {"type": "yaw_to", "yaw_deg": 180},
            {"type": "hold", "duration_s": 0.05},
            {"type": "land"},
        ],
    }
    spec.update(overrides)
    return spec


def test_plan_preserves_goal_order_without_injected_arm():
    cmds = plan_mission(_sample_spec())
    assert [c["type"] for c in cmds] == [
        "takeoff",
        "goto_ned",
        "yaw_to",
        "hold",
        "land",
    ]
    assert cmds[1]["x"] == 5
    validate_command_sequence(cmds)


def test_validator_rejects_high_takeoff():
    spec = _sample_spec()
    cmds = plan_mission(spec)
    cmds[0]["alt_m"] = 100
    result = validate_plan(spec, cmds)
    assert not result.ok
    assert any(i.code == "max_alt" for i in result.issues)


def test_validator_rejects_geofence_breach():
    spec = _sample_spec(
        goals=[
            {"type": "takeoff", "alt_m": 10},
            {"type": "goto_global", "lat": 0.0, "lon": 0.0, "alt_m": 20},
            {"type": "land"},
        ]
    )
    cmds = plan_mission(spec)
    result = validate_plan(spec, cmds)
    assert not result.ok
    assert any(i.code == "geofence" for i in result.issues)


def test_validator_battery_gate():
    spec = _sample_spec()
    cmds = plan_mission(spec)
    result = validate_plan(spec, cmds, vehicle=VehicleSnapshot(battery_pct=5))
    assert not result.ok
    assert any(i.code == "battery" for i in result.issues)


def test_executor_happy_path_with_mock():
    spec = _sample_spec()
    cmds = plan_mission(spec)
    assert validate_plan(spec, cmds).ok

    adapter = MockFlightAdapter()
    assert adapter.connect().ok
    executor = MissionExecutor(adapter, abort_policy=AbortPolicy.LAND)
    result = executor.run(cmds)
    assert result.state == ExecutorState.SUCCEEDED
    assert result.completed == len(cmds)
    assert not adapter.telemetry().armed
    assert adapter.telemetry().z == 0.0


def test_relative_goto_ned_from_current_position():
    adapter = MockFlightAdapter()
    adapter.connect()
    adapter.takeoff(5)
    # Pretend we are already at x=-400 from a prior flight.
    assert adapter.goto_ned(-400, 0, -5).ok
    result = adapter.execute({"type": "goto_ned", "dx": -200, "dy": 0, "z": -5})
    assert result.ok
    assert adapter.telemetry().x == pytest.approx(-600.0)
    result2 = adapter.execute({"type": "goto_ned", "dx": -200, "dy": 0, "z": -5})
    assert result2.ok
    assert adapter.telemetry().x == pytest.approx(-800.0)


def test_takeoff_skipped_when_already_flying():
    adapter = MockFlightAdapter()
    adapter.connect()
    assert adapter.takeoff(5).ok
    assert adapter.telemetry().in_air
    skip = adapter.takeoff(10)
    assert skip.ok
    assert "skipped" in skip.message
    assert adapter.telemetry().z == pytest.approx(-5)


def test_plan_north_steps_are_relative():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "controls" / "tools" / "sitl_mission_smoke.py"
    spec_mod = importlib.util.spec_from_file_location("sitl_mission_smoke", path)
    mod = importlib.util.module_from_spec(spec_mod)
    assert spec_mod.loader is not None
    spec_mod.loader.exec_module(mod)
    spec = mod.build_spec_from_steps(
        ["takeoff", "north:-200", "hold:10", "turn:90", "north:-200"],
        alt_m=3,
        acceptance_m=2,
    )
    cmds = plan_mission(spec)
    assert cmds[1]["dx"] == -200
    assert "x" not in cmds[1]
    assert cmds[4]["dx"] == -200
    assert cmds[3]["relative"] is True


def test_yaw_step_is_relative_turn():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "controls" / "tools" / "sitl_mission_smoke.py"
    spec_mod = importlib.util.spec_from_file_location("sitl_mission_smoke", path)
    mod = importlib.util.module_from_spec(spec_mod)
    assert spec_mod.loader is not None
    spec_mod.loader.exec_module(mod)
    spec = mod.build_spec_from_steps(["yaw:170"], alt_m=3, acceptance_m=2)
    cmds = plan_mission(spec)
    assert cmds[0] == {"type": "yaw_to", "yaw_deg": 170.0, "relative": True}
    spec2 = mod.build_spec_from_steps(["heading:170"], alt_m=3, acceptance_m=2)
    cmds2 = plan_mission(spec2)
    assert cmds2[0]["relative"] is False
    assert cmds2[0]["yaw_deg"] == 170.0


def test_executor_failure_does_not_auto_land():
    adapter = MockFlightAdapter()
    adapter.connect()
    adapter.takeoff(5)
    # Force a failing command via unknown type through execute path is hard;
    # use yaw_to then a bad command by monkeypatching.
    original = adapter.goto_ned

    def boom(*_a, **_k):
        from flight.adapter import CommandResult

        return CommandResult(False, "injected failure")

    adapter.goto_ned = boom  # type: ignore[method-assign]
    executor = MissionExecutor(adapter, abort_policy=AbortPolicy.LAND)
    result = executor.run(
        [
            {"type": "goto_ned", "x": 1, "y": 0, "z": -5},
            {"type": "land"},
        ]
    )
    assert result.state == ExecutorState.FAILED
    assert adapter.telemetry().in_air  # still flying — no auto land
    adapter.goto_ned = original  # type: ignore[method-assign]


def test_executor_abort_mid_mission():
    spec = _sample_spec(
        goals=[
            {"type": "takeoff", "alt_m": 5},
            {"type": "hold", "duration_s": 2.0},
            {"type": "land"},
        ]
    )
    cmds = plan_mission(spec)
    adapter = MockFlightAdapter()
    adapter.connect()
    executor = MissionExecutor(adapter, abort_policy=AbortPolicy.RTL, hold_poll_s=0.05)

    def on_event(ev):
        if ev.kind == "command_start" and ev.command and ev.command.get("type") == "hold":
            executor.request_abort()

    executor.on_event = on_event
    result = executor.run(cmds)
    assert result.state == ExecutorState.ABORTED


def test_schemas_exist_and_parse():
    for name in ("mission_spec.schema.json", "flight_commands.schema.json"):
        path = SCHEMAS / name
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "$schema" in data


def test_jsonschema_mission_spec_if_available():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / "mission_spec.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(_sample_spec(), schema)
