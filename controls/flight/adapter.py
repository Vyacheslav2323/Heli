"""Abstract flight-control adapter.

Only implementations of this interface may talk to a flight controller.
Swapping stacks = implementing one adapter (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class AbortPolicy(str, Enum):
    RTL = "rtl"
    LAND = "land"


@dataclass(frozen=True)
class Telemetry:
    """Minimal vehicle state the mission executor needs."""

    armed: bool = False
    in_air: bool = False
    mode: str = ""
    # Local NED position (m); z positive down.
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # Local NED velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    # Global if available
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_amsl_m: Optional[float] = None
    yaw_deg: float = 0.0
    battery_pct: Optional[float] = None
    connected: bool = False

    @property
    def speed_xy_m_s(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


class FlightAdapter(ABC):
    """Narrow interface from mission/executor → vehicle."""

    @abstractmethod
    def connect(self) -> CommandResult:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def telemetry(self) -> Telemetry:
        ...

    @abstractmethod
    def arm(self, *, force: bool = False) -> CommandResult:
        ...

    @abstractmethod
    def disarm(self, *, force: bool = False) -> CommandResult:
        ...

    @abstractmethod
    def takeoff(self, alt_m: float, yaw_deg: Optional[float] = None) -> CommandResult:
        ...

    @abstractmethod
    def land(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        yaw_deg: Optional[float] = None,
    ) -> CommandResult:
        ...

    @abstractmethod
    def hold(self) -> CommandResult:
        """Stabilize / position-hold (PX4 posctl or Offboard hold)."""

    @abstractmethod
    def goto_ned(
        self,
        x: float,
        y: float,
        z: float,
        yaw_deg: Optional[float] = None,
        *,
        acceptance_radius_m: float = 1.0,
        timeout_s: float = 60.0,
    ) -> CommandResult:
        """Fly to local NED setpoint (z positive down). Blocks until reached or timeout."""

    @abstractmethod
    def goto_global(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        yaw_deg: Optional[float] = None,
        *,
        acceptance_radius_m: float = 2.0,
        timeout_s: float = 120.0,
    ) -> CommandResult:
        ...

    @abstractmethod
    def yaw_to(
        self,
        yaw_deg: float,
        *,
        rate_deg_s: Optional[float] = None,
        relative: bool = False,
        tolerance_deg: float = 5.0,
        timeout_s: float = 30.0,
    ) -> CommandResult:
        ...

    @abstractmethod
    def loiter(self) -> CommandResult:
        ...

    @abstractmethod
    def rtl(self) -> CommandResult:
        ...

    @abstractmethod
    def abort(self, policy: AbortPolicy = AbortPolicy.RTL) -> CommandResult:
        ...

    def execute(self, command: Mapping[str, Any]) -> CommandResult:
        """Dispatch a schema-shaped abstract command dict."""
        cmd_type = command.get("type")
        if cmd_type == "arm":
            return self.arm(force=bool(command.get("force", False)))
        if cmd_type == "disarm":
            return self.disarm(force=bool(command.get("force", False)))
        if cmd_type == "takeoff":
            return self.takeoff(float(command["alt_m"]), command.get("yaw_deg"))
        if cmd_type == "land":
            return self.land(command.get("lat"), command.get("lon"), command.get("yaw_deg"))
        if cmd_type == "hold":
            return self.hold()
        if cmd_type == "goto_ned":
            acceptance = float(command.get("acceptance_radius_m", 1.0))
            if any(k in command for k in ("dx", "dy", "dz")):
                tel = self.telemetry()
                x = tel.x + float(command.get("dx", 0.0))
                y = tel.y + float(command.get("dy", 0.0))
                if "dz" in command:
                    z = tel.z + float(command["dz"])
                elif "z" in command:
                    z = float(command["z"])
                else:
                    z = tel.z
                return self.goto_ned(
                    x,
                    y,
                    z,
                    command.get("yaw_deg"),
                    acceptance_radius_m=acceptance,
                )
            return self.goto_ned(
                float(command["x"]),
                float(command["y"]),
                float(command["z"]),
                command.get("yaw_deg"),
                acceptance_radius_m=acceptance,
            )
        if cmd_type == "goto_global":
            return self.goto_global(
                float(command["lat"]),
                float(command["lon"]),
                float(command["alt_m"]),
                command.get("yaw_deg"),
                acceptance_radius_m=float(command.get("acceptance_radius_m", 2.0)),
            )
        if cmd_type == "yaw_to":
            return self.yaw_to(
                float(command["yaw_deg"]),
                rate_deg_s=command.get("rate_deg_s"),
                relative=bool(command.get("relative", False)),
                tolerance_deg=float(command.get("tolerance_deg", 5.0)),
            )
        if cmd_type == "loiter":
            return self.loiter()
        if cmd_type == "rtl":
            return self.rtl()
        if cmd_type == "abort":
            policy = AbortPolicy(command.get("policy", "rtl"))
            return self.abort(policy)
        return CommandResult(False, f"unknown command type: {cmd_type}")


def validate_command_sequence(commands: Sequence[Mapping[str, Any]]) -> None:
    """Lightweight structural check (full JSON Schema optional at call site)."""
    allowed = {
        "arm",
        "disarm",
        "takeoff",
        "land",
        "hold",
        "goto_ned",
        "goto_global",
        "yaw_to",
        "loiter",
        "rtl",
        "abort",
    }
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, Mapping) or "type" not in cmd:
            raise ValueError(f"command[{i}] missing type")
        if cmd["type"] not in allowed:
            raise ValueError(f"command[{i}] unsupported type: {cmd['type']}")
