"""Flight adapter unit tests (mock + mode encoding)."""

from __future__ import annotations

import pytest

from flight.adapter import validate_command_sequence
from flight.px4.mock import MockFlightAdapter
from flight.px4.modes import MODE_TABLE, encode_custom_mode


def test_encode_custom_mode_bits():
    main, sub = MODE_TABLE["auto:takeoff"]
    custom = encode_custom_mode(main, sub)
    assert (custom >> 16) & 0xFF == main
    assert (custom >> 24) & 0xFF == sub


def test_mock_primitives_takeoff_goto_yaw_land():
    a = MockFlightAdapter()
    assert a.connect().ok
    assert a.takeoff(10, yaw_deg=0).ok
    assert a.telemetry().z == pytest.approx(-10)
    assert a.goto_ned(2, 3, -10, yaw_deg=45).ok
    tel = a.telemetry()
    assert tel.x == 2 and tel.y == 3
    assert a.yaw_to(90).ok
    assert a.telemetry().yaw_deg == 90
    assert a.hold().ok
    assert a.land().ok
    assert not a.telemetry().armed


def test_execute_dispatch():
    a = MockFlightAdapter()
    a.connect()
    cmds = [
        {"type": "arm"},
        {"type": "takeoff", "alt_m": 5},
        {"type": "goto_ned", "x": 1, "y": 0, "z": -5},
        {"type": "yaw_to", "yaw_deg": 10},
        {"type": "land"},
    ]
    validate_command_sequence(cmds)
    for cmd in cmds:
        assert a.execute(cmd).ok
