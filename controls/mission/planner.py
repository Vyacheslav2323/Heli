"""Mission planner: MissionSpec → ordered abstract flight commands."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping, Sequence


def plan_mission(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a MissionSpec into abstract commands **in goal order**.

    Does not inject arm/land/rtl. Takeoff arms only when executed and not
    already flying; land runs only when a land goal is present.
    """
    if int(spec.get("version", 0)) != 1:
        raise ValueError("unsupported MissionSpec version")
    goals = spec.get("goals")
    if not isinstance(goals, Sequence) or not goals:
        raise ValueError("MissionSpec.goals must be a non-empty list")

    commands: list[dict[str, Any]] = []
    for goal in goals:
        if not isinstance(goal, Mapping):
            raise ValueError("each goal must be an object")
        commands.append(_goal_to_command(goal))
    return commands


def _goal_to_command(goal: Mapping[str, Any]) -> dict[str, Any]:
    gtype = goal.get("type")
    cmd: MutableMapping[str, Any] = {"type": gtype}
    if gtype == "takeoff":
        cmd["alt_m"] = float(goal["alt_m"])
        if goal.get("yaw_deg") is not None:
            cmd["yaw_deg"] = float(goal["yaw_deg"])
    elif gtype == "goto_ned":
        if any(k in goal for k in ("dx", "dy", "dz")):
            if goal.get("dx") is not None:
                cmd["dx"] = float(goal["dx"])
            if goal.get("dy") is not None:
                cmd["dy"] = float(goal["dy"])
            if goal.get("dz") is not None:
                cmd["dz"] = float(goal["dz"])
            if goal.get("z") is not None:
                cmd["z"] = float(goal["z"])
        else:
            cmd["x"] = float(goal["x"])
            cmd["y"] = float(goal["y"])
            cmd["z"] = float(goal["z"])
        if goal.get("yaw_deg") is not None:
            cmd["yaw_deg"] = float(goal["yaw_deg"])
        if goal.get("acceptance_radius_m") is not None:
            cmd["acceptance_radius_m"] = float(goal["acceptance_radius_m"])
    elif gtype == "goto_global":
        cmd["lat"] = float(goal["lat"])
        cmd["lon"] = float(goal["lon"])
        cmd["alt_m"] = float(goal["alt_m"])
        if goal.get("yaw_deg") is not None:
            cmd["yaw_deg"] = float(goal["yaw_deg"])
        if goal.get("acceptance_radius_m") is not None:
            cmd["acceptance_radius_m"] = float(goal["acceptance_radius_m"])
    elif gtype == "yaw_to":
        cmd["yaw_deg"] = float(goal["yaw_deg"])
        if goal.get("rate_deg_s") is not None:
            cmd["rate_deg_s"] = float(goal["rate_deg_s"])
        if "relative" in goal:
            cmd["relative"] = bool(goal["relative"])
    elif gtype == "hold":
        if goal.get("duration_s") is not None:
            cmd["duration_s"] = float(goal["duration_s"])
    elif gtype == "land":
        for key in ("lat", "lon", "yaw_deg"):
            if goal.get(key) is not None:
                cmd[key] = float(goal[key])
    elif gtype == "rtl":
        pass
    elif gtype == "arm":
        if goal.get("force") is not None:
            cmd["force"] = bool(goal["force"])
    elif gtype == "disarm":
        if goal.get("force") is not None:
            cmd["force"] = bool(goal["force"])
    else:
        raise ValueError(f"unsupported goal type: {gtype}")
    return dict(cmd)


def load_spec(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy for immutable-ish downstream use."""
    return deepcopy(dict(data))
