"""In-memory flight adapter for unit tests (no PX4 / MAVLink)."""

from __future__ import annotations

import math
import time
from typing import Optional

from flight.adapter import AbortPolicy, CommandResult, FlightAdapter, Telemetry


class MockFlightAdapter(FlightAdapter):
    """Simulates instantaneous/near-instant success for mission tests."""

    def __init__(self, *, settle_s: float = 0.0) -> None:
        self.settle_s = settle_s
        self._tel = Telemetry(connected=False)
        self._home_alt = 0.0

    def connect(self) -> CommandResult:
        self._tel = Telemetry(connected=True, mode="manual")
        return CommandResult(True, "mock connected")

    def disconnect(self) -> None:
        self._tel = Telemetry(connected=False)

    def telemetry(self) -> Telemetry:
        return self._tel

    def arm(self, *, force: bool = False) -> CommandResult:
        _ = force
        self._tel = _replace(self._tel, armed=True, mode=self._tel.mode or "posctl")
        return CommandResult(True, "armed")

    def disarm(self, *, force: bool = False) -> CommandResult:
        _ = force
        self._tel = _replace(self._tel, armed=False, in_air=False)
        return CommandResult(True, "disarmed")

    def takeoff(self, alt_m: float, yaw_deg: Optional[float] = None) -> CommandResult:
        if self._tel.in_air or (self._tel.armed and abs(self._tel.z) > 0.5):
            return CommandResult(True, "takeoff skipped (already flying)")
        self._settle()
        yaw = self._tel.yaw_deg if yaw_deg is None else float(yaw_deg)
        self._tel = _replace(
            self._tel,
            armed=True,
            in_air=True,
            mode="auto:takeoff",
            z=-abs(alt_m),
            yaw_deg=yaw,
            alt_amsl_m=self._home_alt + abs(alt_m),
        )
        return CommandResult(True, "takeoff")

    def land(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        yaw_deg: Optional[float] = None,
    ) -> CommandResult:
        _ = (lat, lon)
        self._settle()
        yaw = self._tel.yaw_deg if yaw_deg is None else float(yaw_deg)
        self._tel = _replace(
            self._tel,
            armed=False,
            in_air=False,
            mode="auto:land",
            z=0.0,
            yaw_deg=yaw,
            alt_amsl_m=self._home_alt,
        )
        return CommandResult(True, "land")

    def hold(self) -> CommandResult:
        self._tel = _replace(self._tel, mode="posctl")
        return CommandResult(True, "hold")

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
        _ = (acceptance_radius_m, timeout_s)
        self._settle()
        yaw = self._tel.yaw_deg if yaw_deg is None else float(yaw_deg)
        self._tel = _replace(
            self._tel,
            mode="offboard",
            in_air=True,
            armed=True,
            x=float(x),
            y=float(y),
            z=float(z),
            yaw_deg=yaw,
            alt_amsl_m=self._home_alt - float(z),
        )
        return CommandResult(True, "goto_ned")

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
        _ = (acceptance_radius_m, timeout_s)
        self._settle()
        yaw = self._tel.yaw_deg if yaw_deg is None else float(yaw_deg)
        self._tel = _replace(
            self._tel,
            mode="auto:loiter",
            in_air=True,
            armed=True,
            lat=float(lat),
            lon=float(lon),
            alt_amsl_m=float(alt_m),
            z=-(float(alt_m) - self._home_alt),
            yaw_deg=yaw,
        )
        return CommandResult(True, "goto_global")

    def yaw_to(
        self,
        yaw_deg: float,
        *,
        rate_deg_s: Optional[float] = None,
        relative: bool = False,
        tolerance_deg: float = 5.0,
        timeout_s: float = 30.0,
    ) -> CommandResult:
        _ = (rate_deg_s, tolerance_deg, timeout_s)
        self._settle()
        target = (self._tel.yaw_deg + yaw_deg) % 360.0 if relative else float(yaw_deg) % 360.0
        self._tel = _replace(self._tel, yaw_deg=target, mode="offboard")
        return CommandResult(True, "yaw_to")

    def loiter(self) -> CommandResult:
        self._tel = _replace(self._tel, mode="auto:loiter")
        return CommandResult(True, "loiter")

    def rtl(self) -> CommandResult:
        self._settle()
        self._tel = _replace(
            self._tel,
            mode="auto:rtl",
            x=0.0,
            y=0.0,
            z=0.0,
            armed=False,
            in_air=False,
            alt_amsl_m=self._home_alt,
        )
        return CommandResult(True, "rtl")

    def abort(self, policy: AbortPolicy = AbortPolicy.RTL) -> CommandResult:
        if policy == AbortPolicy.LAND:
            return self.land()
        return self.rtl()

    def _settle(self) -> None:
        if self.settle_s > 0:
            time.sleep(self.settle_s)


def _replace(tel: Telemetry, **kwargs) -> Telemetry:
    data = {
        "armed": tel.armed,
        "in_air": tel.in_air,
        "mode": tel.mode,
        "x": tel.x,
        "y": tel.y,
        "z": tel.z,
        "vx": tel.vx,
        "vy": tel.vy,
        "vz": tel.vz,
        "lat": tel.lat,
        "lon": tel.lon,
        "alt_amsl_m": tel.alt_amsl_m,
        "yaw_deg": tel.yaw_deg,
        "battery_pct": tel.battery_pct,
        "connected": tel.connected,
    }
    data.update(kwargs)
    return Telemetry(**data)


def horizontal_distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    dlat = (a_lat - b_lat) * 111320.0
    dlon = (a_lon - b_lon) * 111320.0 * math.cos(math.radians(a_lat))
    return math.sqrt(dlat * dlat + dlon * dlon)
