"""Mission validator / safety gate.

Every plan must pass through validate_plan regardless of origin (LLM, script, human).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return list(self.issues)


@dataclass(frozen=True)
class VehicleSnapshot:
    """Optional live state for runtime re-validation."""

    battery_pct: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_amsl_m: Optional[float] = None
    armed: bool = False


def validate_plan(
    spec: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    *,
    vehicle: Optional[VehicleSnapshot] = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    constraints = spec.get("constraints") or {}

    if not commands:
        issues.append(ValidationIssue("empty_plan", "plan has no commands"))

    max_alt = constraints.get("max_alt_m")
    min_alt = constraints.get("min_alt_m")
    min_battery = constraints.get("min_battery_pct")
    geofence = constraints.get("geofence")

    # Battery gate
    if vehicle and min_battery is not None and vehicle.battery_pct is not None:
        if vehicle.battery_pct < float(min_battery):
            issues.append(
                ValidationIssue(
                    "battery",
                    f"battery {vehicle.battery_pct}% below minimum {min_battery}%",
                    "constraints.min_battery_pct",
                )
            )

    for i, cmd in enumerate(commands):
        path = f"commands[{i}]"
        ctype = cmd.get("type")
        if ctype == "takeoff":
            alt = float(cmd["alt_m"])
            if max_alt is not None and alt > float(max_alt):
                issues.append(
                    ValidationIssue("max_alt", f"takeoff alt {alt} > max_alt_m {max_alt}", path)
                )
            if min_alt is not None and alt < float(min_alt):
                issues.append(
                    ValidationIssue("min_alt", f"takeoff alt {alt} < min_alt_m {min_alt}", path)
                )
        elif ctype == "goto_ned":
            if "z" in cmd and "dx" not in cmd and "dy" not in cmd and "dz" not in cmd:
                agl = -float(cmd["z"])
            elif "z" in cmd:
                agl = -float(cmd["z"])
            else:
                agl = None
            if agl is not None:
                if max_alt is not None and agl > float(max_alt):
                    issues.append(
                        ValidationIssue("max_alt", f"goto_ned AGL {agl} > max_alt_m {max_alt}", path)
                    )
                if min_alt is not None and agl < float(min_alt):
                    issues.append(
                        ValidationIssue("min_alt", f"goto_ned AGL {agl} < min_alt_m {min_alt}", path)
                    )
        elif ctype == "goto_global":
            alt = float(cmd["alt_m"])
            if max_alt is not None and alt > float(max_alt):
                # Treat as AMSL cap when no home; still useful as absolute ceiling.
                issues.append(
                    ValidationIssue("max_alt", f"goto_global alt {alt} > max_alt_m {max_alt}", path)
                )
            if geofence and not _point_in_polygon(
                float(cmd["lat"]), float(cmd["lon"]), geofence.get("polygon") or []
            ):
                issues.append(
                    ValidationIssue(
                        "geofence",
                        f"goto_global ({cmd['lat']},{cmd['lon']}) outside geofence",
                        path,
                    )
                )

    return ValidationResult(ok=len(issues) == 0, issues=issues)


def _point_in_polygon(lat: float, lon: float, polygon: Sequence[Mapping[str, Any]]) -> bool:
    """Ray-casting point-in-polygon in lat/lon degrees (adequate for small fences)."""
    if len(polygon) < 3:
        return True
    inside = False
    n = len(polygon)
    for i in range(n):
        j = (i - 1) % n
        yi, xi = float(polygon[i]["lat"]), float(polygon[i]["lon"])
        yj, xj = float(polygon[j]["lat"]), float(polygon[j]["lon"])
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi
        ):
            inside = not inside
    return inside


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
