"""PX4 MAVLink flight adapter (pymavlink).

Connects to PX4 SITL or hardware. Prefer Offboard for XYZ/yaw; use AUTO modes
for takeoff/land/RTL. See docs/research/px4-commander-mavlink.md.

SITL onboard channel (from PX4 logs)::

    mavlink ... on udp port 14580 remote port 14540

So the companion should **listen** on 14540 (same host as PX4)::

    udpin:0.0.0.0:14540

On Windows + PX4-in-WSL, run the client inside WSL (see tools/sitl_mission_smoke.py).
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any, Optional

from flight.adapter import AbortPolicy, CommandResult, FlightAdapter, Telemetry
from flight.px4.modes import (
    MODE_TABLE,
    POSITION_TARGET_TYPEMASK_POS_ONLY,
    POSITION_TARGET_TYPEMASK_POS_YAW,
)

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - optional until installed
    mavutil = None  # type: ignore


MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128

# Defaults match PX4 SITL "Onboard" mavlink instance
DEFAULT_CONNECTION_URL = "udpin:0.0.0.0:14540"


class Px4MavlinkAdapter(FlightAdapter):
    def __init__(
        self,
        connection_url: str = DEFAULT_CONNECTION_URL,
        *,
        system: int = 1,
        component: int = 1,
        offboard_hz: float = 10.0,
        command_timeout_s: float = 5.0,
        source_system: int = 255,
    ) -> None:
        if mavutil is None:
            raise ImportError("pymavlink is required for Px4MavlinkAdapter; pip install pymavlink")
        self.connection_url = connection_url
        self.target_system = system
        self.target_component = component
        self.offboard_hz = offboard_hz
        self.command_timeout_s = command_timeout_s
        self.source_system = source_system
        self._master: Any = None
        self._lock = threading.Lock()
        self._tel = Telemetry()
        self._ack_queue: queue.Queue[Any] = queue.Queue()
        self._reader_stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._offboard_stop = threading.Event()
        self._offboard_thread: Optional[threading.Thread] = None
        self._setpoint: Optional[tuple[float, float, float, Optional[float]]] = None
        self.debug: bool = False

    def _update_tel(self, **kwargs: Any) -> None:
        with self._lock:
            t = self._tel
            self._tel = Telemetry(
                armed=kwargs.get("armed", t.armed),
                in_air=kwargs.get("in_air", t.in_air),
                mode=kwargs.get("mode", t.mode),
                x=kwargs.get("x", t.x),
                y=kwargs.get("y", t.y),
                z=kwargs.get("z", t.z),
                vx=kwargs.get("vx", t.vx),
                vy=kwargs.get("vy", t.vy),
                vz=kwargs.get("vz", t.vz),
                lat=kwargs.get("lat", t.lat),
                lon=kwargs.get("lon", t.lon),
                alt_amsl_m=kwargs.get("alt_amsl_m", t.alt_amsl_m),
                yaw_deg=kwargs.get("yaw_deg", t.yaw_deg),
                battery_pct=kwargs.get("battery_pct", t.battery_pct),
                connected=kwargs.get("connected", t.connected),
            )

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[px4] {msg}", flush=True)

    # --- lifecycle ---------------------------------------------------------

    def connect(self) -> CommandResult:
        try:
            self._master = mavutil.mavlink_connection(
                self.connection_url,
                source_system=self.source_system,
            )
            hb = self._wait_autopilot_heartbeat(timeout_s=15.0)
            if hb is None:
                return CommandResult(
                    False,
                    "no PX4 autopilot HEARTBEAT (wrong URL/host? use udpin:0.0.0.0:14540 inside WSL)",
                )
            self.target_system = int(self._master.target_system)
            self.target_component = int(self._master.target_component or 1)
            if self.target_system == 0:
                return CommandResult(False, "HEARTBEAT had system id 0 — not a vehicle link")

            # Drain any ACKs before starting the exclusive reader.
            while self._master.recv_match(type="COMMAND_ACK", blocking=False):
                pass
            self._drain_ack_queue()

            self._reader_stop.clear()
            self._reader = threading.Thread(target=self._read_loop, name="px4-mav-reader", daemon=True)
            self._reader.start()
            # Prefer message intervals (PX4); REQUEST_DATA_STREAM is legacy.
            self._set_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10)
            self._set_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5)
            self._set_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 10)
            self._set_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 2)
            self._request_data_stream(mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10)
            self._request_data_stream(mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10)
            self._request_data_stream(mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 2)
            with self._lock:
                armed = bool(hb.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
            self._update_tel(armed=armed, connected=True)
            return CommandResult(
                True,
                f"connected {self.connection_url} sys={self.target_system} comp={self.target_component}",
            )
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, f"connect failed: {exc}")

    def disconnect(self) -> None:
        self._stop_offboard_stream()
        self._reader_stop.set()
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=2.0)
        if self._master is not None:
            try:
                self._master.close()
            except Exception:  # noqa: BLE001
                pass
            self._master = None

    def telemetry(self) -> Telemetry:
        with self._lock:
            return self._tel

    # --- primitives --------------------------------------------------------

    def arm(self, *, force: bool = False) -> CommandResult:
        param2 = 21196.0 if force else 0.0
        return self._command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            1.0,
            param2,
            confirmation_name="arm",
        )

    def disarm(self, *, force: bool = False) -> CommandResult:
        param2 = 21196.0 if force else 0.0
        return self._command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0.0,
            param2,
            confirmation_name="disarm",
        )

    def takeoff(self, alt_m: float, yaw_deg: Optional[float] = None) -> CommandResult:
        """Takeoff to relative AGL altitude (meters).

        If already flying, skip takeoff and succeed immediately (caller may
        still ``goto_ned`` / climb via later commands).
        """
        tel = self.telemetry()
        if tel.in_air or (tel.armed and abs(tel.z) > 0.5):
            return CommandResult(True, "takeoff skipped (already flying)")

        # PX4 NAV_TAKEOFF param7 is AMSL — convert from relative AGL.
        amsl = self._current_amsl_m(timeout_s=3.0)
        if amsl is None:
            mode = self._set_mode("auto:takeoff")
            if not mode.ok:
                return mode
            arm_result = self.arm()
            if not arm_result.ok:
                return arm_result
            return self._wait_altitude_agl(min(alt_m, 2.5), timeout_s=60.0)

        target_amsl = float(amsl) + abs(float(alt_m))
        yaw = float(yaw_deg) if yaw_deg is not None else float("nan")
        result = self._command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0.0,
            0.0,
            0.0,
            yaw,
            0.0,
            0.0,
            target_amsl,
            confirmation_name="takeoff",
        )
        if not result.ok:
            return result
        if not self.telemetry().armed:
            arm_result = self.arm()
            if not arm_result.ok:
                return arm_result
        return self._wait_altitude_agl(alt_m, timeout_s=60.0)

    def land(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        yaw_deg: Optional[float] = None,
    ) -> CommandResult:
        yaw = float(yaw_deg) if yaw_deg is not None else float("nan")
        lat_v = float(lat) if lat is not None else 0.0
        lon_v = float(lon) if lon is not None else 0.0
        # Engage land *before* dropping Offboard setpoints.
        result = self._command_long(
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0.0,
            0.0,
            0.0,
            yaw,
            lat_v,
            lon_v,
            0.0,
            confirmation_name="land",
        )
        self._stop_offboard_stream()
        if not result.ok:
            return result
        return self._wait_disarmed(timeout_s=120.0)

    def hold(self) -> CommandResult:
        """Hold current position.

        Prefer staying in Offboard (freeze setpoint) so we never open a gap with
        no setpoints — stopping the Offboard stream before the mode switch was
        causing overshoot / flyaway between legs.
        """
        tel = self.telemetry()
        if self._offboard_thread and self._offboard_thread.is_alive():
            self._setpoint = (tel.x, tel.y, tel.z, tel.yaw_deg)
            self._dbg(f"hold offboard freeze at NED=({tel.x:.1f},{tel.y:.1f},{tel.z:.1f})")
            return CommandResult(True, "hold (offboard freeze)")
        # Not in offboard — use PX4 position hold, but only after a mode that
        # does not depend on the Offboard stream.
        result = self._set_mode("posctl")
        self._stop_offboard_stream()
        return result

    def goto_ned(
        self,
        x: float,
        y: float,
        z: float,
        yaw_deg: Optional[float] = None,
        *,
        acceptance_radius_m: float = 1.0,
        timeout_s: float = 60.0,
        settle_speed_m_s: float = 0.6,
        stable_samples: int = 8,
    ) -> CommandResult:
        target = (float(x), float(y), float(z), yaw_deg)
        tel0 = self.telemetry()
        self._dbg(
            f"goto_ned start from ({tel0.x:.1f},{tel0.y:.1f},{tel0.z:.1f}) "
            f"vxy={tel0.speed_xy_m_s:.1f} -> ({x:.1f},{y:.1f},{z:.1f})"
        )
        # Publish target *before* engaging Offboard so the first stream is correct.
        self._setpoint = target
        prep = self._ensure_offboard()
        if not prep.ok:
            return prep
        self._setpoint = target

        deadline = time.monotonic() + timeout_s
        inside = 0
        last_log = 0.0
        while time.monotonic() < deadline:
            tel = self.telemetry()
            dx, dy, dz = tel.x - x, tel.y - y, tel.z - z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            now = time.monotonic()
            if self.debug and now - last_log >= 1.0:
                self._dbg(
                    f"goto_ned NED=({tel.x:.1f},{tel.y:.1f},{tel.z:.1f}) "
                    f"dist={dist:.1f}m vxy={tel.speed_xy_m_s:.1f} inside={inside}/{stable_samples}"
                )
                last_log = now
            # Require proximity AND low horizontal speed so we don't declare
            # "reached" while still barreling through the waypoint (overshoot).
            if dist <= acceptance_radius_m and tel.speed_xy_m_s <= settle_speed_m_s:
                inside += 1
                if inside >= stable_samples:
                    # Freeze at the commanded target briefly so the next command
                    # does not inherit residual velocity.
                    self._setpoint = (float(x), float(y), float(z), tel.yaw_deg if yaw_deg is None else yaw_deg)
                    time.sleep(0.5)
                    tel1 = self.telemetry()
                    msg = (
                        f"goto_ned reached at ({tel1.x:.1f},{tel1.y:.1f},{tel1.z:.1f}) "
                        f"err={math.sqrt((tel1.x-x)**2+(tel1.y-y)**2+(tel1.z-z)**2):.1f}m"
                    )
                    self._dbg(msg)
                    return CommandResult(True, msg, details={"x": tel1.x, "y": tel1.y, "z": tel1.z})
            else:
                inside = 0
            time.sleep(0.1)
        tel = self.telemetry()
        return CommandResult(
            False,
            f"goto_ned timeout at ({tel.x:.1f},{tel.y:.1f},{tel.z:.1f}) target=({x:.1f},{y:.1f},{z:.1f})",
        )
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
        self._stop_offboard_stream()
        yaw = float(yaw_deg) if yaw_deg is not None else float("nan")
        result = self._command_long(
            mavutil.mavlink.MAV_CMD_DO_REPOSITION,
            0.0,
            0.0,
            0.0,
            yaw,
            float(lat),
            float(lon),
            float(alt_m),
            confirmation_name="reposition",
        )
        if not result.ok:
            return result
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            tel = self.telemetry()
            if tel.lat is None or tel.lon is None:
                time.sleep(0.1)
                continue
            dlat = (tel.lat - lat) * 111320.0
            dlon = (tel.lon - lon) * 111320.0 * math.cos(math.radians(lat))
            dalt = (tel.alt_amsl_m or alt_m) - alt_m
            if math.sqrt(dlat * dlat + dlon * dlon + dalt * dalt) <= acceptance_radius_m:
                return CommandResult(True, "goto_global reached")
            time.sleep(0.1)
        return CommandResult(False, "goto_global timeout")

    def yaw_to(
        self,
        yaw_deg: float,
        *,
        rate_deg_s: Optional[float] = None,
        relative: bool = False,
        tolerance_deg: float = 5.0,
        timeout_s: float = 30.0,
    ) -> CommandResult:
        tel = self.telemetry()
        target = (tel.yaw_deg + yaw_deg) % 360.0 if relative else float(yaw_deg) % 360.0
        # Freeze XY at current (or last setpoint) so a yaw step cannot slide.
        hold_x, hold_y, hold_z = tel.x, tel.y, tel.z
        if self._setpoint is not None:
            hold_x, hold_y, hold_z = self._setpoint[0], self._setpoint[1], self._setpoint[2]
        self._setpoint = (hold_x, hold_y, hold_z, target)
        self._dbg(f"yaw_to {tel.yaw_deg:.1f} -> {target:.1f} (relative={relative}) hold XY=({hold_x:.1f},{hold_y:.1f})")
        prep = self._ensure_offboard()
        if not prep.ok:
            return prep
        self._setpoint = (hold_x, hold_y, hold_z, target)
        _ = rate_deg_s
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            now = self.telemetry()
            err = abs(_angle_diff_deg(now.yaw_deg, target))
            if err <= tolerance_deg and now.speed_xy_m_s <= 0.6:
                return CommandResult(True, f"yaw_to reached (yaw={now.yaw_deg:.1f})")
            time.sleep(0.1)
        return CommandResult(False, "yaw_to timeout")

    def loiter(self) -> CommandResult:
        result = self._set_mode("auto:loiter")
        self._stop_offboard_stream()
        return result

    def rtl(self) -> CommandResult:
        result = self._set_mode("auto:rtl")
        self._stop_offboard_stream()
        return result

    def abort(self, policy: AbortPolicy = AbortPolicy.RTL) -> CommandResult:
        if policy == AbortPolicy.LAND:
            return self.land()
        return self.rtl()
    # --- internals ---------------------------------------------------------

    def _wait_autopilot_heartbeat(self, timeout_s: float) -> Any:
        """Wait for a non-GCS vehicle HEARTBEAT (prefer PX4)."""
        assert self._master is not None
        deadline = time.monotonic() + timeout_s
        gcs = mavutil.mavlink.MAV_TYPE_GCS
        invalid = mavutil.mavlink.MAV_AUTOPILOT_INVALID
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            msg = self._master.recv_match(type="HEARTBEAT", blocking=True, timeout=min(1.0, remaining))
            if msg is None:
                continue
            sysid = msg.get_srcSystem()
            if sysid == 0:
                continue
            if msg.type == gcs or msg.autopilot == invalid:
                continue
            self._master.target_system = sysid
            self._master.target_component = msg.get_srcComponent()
            return msg
        return None

    def _set_message_interval(self, msg_id: int, rate_hz: float) -> None:
        assert self._master is not None
        interval_us = int(1_000_000 / max(rate_hz, 0.1))
        self._master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(msg_id),
            float(interval_us),
            0,
            0,
            0,
            0,
            0,
        )

    def _request_data_stream(self, stream_id: int, rate_hz: int) -> None:
        assert self._master is not None
        self._master.mav.request_data_stream_send(
            self.target_system,
            self.target_component,
            stream_id,
            rate_hz,
            1,
        )

    def _current_amsl_m(self, timeout_s: float) -> Optional[float]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            alt = self.telemetry().alt_amsl_m
            if alt is not None:
                return float(alt)
            time.sleep(0.1)
        return None

    def _read_loop(self) -> None:
        assert self._master is not None
        while not self._reader_stop.is_set():
            msg = self._master.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            self._handle_message(msg)

    def _handle_message(self, msg: Any) -> None:
        mtype = msg.get_type()
        if mtype == "COMMAND_ACK":
            self._ack_queue.put(msg)
            return
        if mtype == "HEARTBEAT":
            if msg.type == mavutil.mavlink.MAV_TYPE_GCS:
                return
            self._update_tel(
                armed=bool(msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED),
                connected=True,
            )
        elif mtype == "LOCAL_POSITION_NED":
            self._update_tel(
                x=float(msg.x),
                y=float(msg.y),
                z=float(msg.z),
                vx=float(msg.vx),
                vy=float(msg.vy),
                vz=float(msg.vz),
                in_air=abs(msg.z) > 0.3,
                connected=True,
            )
        elif mtype == "GLOBAL_POSITION_INT":
            self._update_tel(
                lat=msg.lat / 1e7,
                lon=msg.lon / 1e7,
                alt_amsl_m=msg.alt / 1000.0,
                connected=True,
            )
        elif mtype == "ATTITUDE":
            self._update_tel(yaw_deg=math.degrees(float(msg.yaw)) % 360.0, connected=True)
        elif mtype == "SYS_STATUS":
            pct = msg.battery_remaining if msg.battery_remaining >= 0 else self.telemetry().battery_pct
            self._update_tel(
                battery_pct=float(pct) if pct is not None else None,
                connected=True,
            )
        elif mtype == "EXTENDED_SYS_STATE":
            self._update_tel(in_air=int(msg.landed_state) == 2, connected=True)

    def _drain_ack_queue(self) -> None:
        while True:
            try:
                self._ack_queue.get_nowait()
            except queue.Empty:
                break

    def _command_long(
        self,
        command: int,
        *params: float,
        confirmation_name: str = "command",
    ) -> CommandResult:
        if self._master is None:
            return CommandResult(False, "not connected")
        padded = list(params) + [0.0] * (7 - len(params))
        padded = padded[:7]
        self._drain_ack_queue()
        self._master.mav.command_long_send(
            self.target_system,
            self.target_component,
            command,
            0,
            *padded,
        )
        deadline = time.monotonic() + self.command_timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                ack = self._ack_queue.get(timeout=min(0.5, max(0.05, remaining)))
            except queue.Empty:
                continue
            if ack.command != command:
                continue
            if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return CommandResult(False, f"{confirmation_name}: result={ack.result}")
            return CommandResult(True, f"{confirmation_name}: accepted")
        return CommandResult(False, f"{confirmation_name}: no ACK")

    def _set_mode(self, mode_name: str) -> CommandResult:
        if mode_name not in MODE_TABLE:
            return CommandResult(False, f"unknown mode {mode_name}")
        main, sub = MODE_TABLE[mode_name]
        return self._command_long(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            float(MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(main),
            float(sub),
            confirmation_name=f"mode:{mode_name}",
        )

    def _ensure_offboard(self) -> CommandResult:
        if self._master is None:
            return CommandResult(False, "not connected")
        tel = self.telemetry()
        if self._setpoint is None:
            self._setpoint = (tel.x, tel.y, tel.z, tel.yaw_deg)
        already = bool(self._offboard_thread and self._offboard_thread.is_alive())
        self._start_offboard_stream()
        if already:
            time.sleep(0.2)
        else:
            # PX4 requires a stream before accepting Offboard.
            time.sleep(1.1)
        return self._set_mode("offboard")

    def _start_offboard_stream(self) -> None:
        if self._offboard_thread and self._offboard_thread.is_alive():
            return
        self._offboard_stop.clear()
        self._offboard_thread = threading.Thread(
            target=self._offboard_loop, name="px4-offboard", daemon=True
        )
        self._offboard_thread.start()

    def _stop_offboard_stream(self) -> None:
        self._offboard_stop.set()
        if self._offboard_thread and self._offboard_thread.is_alive():
            self._offboard_thread.join(timeout=2.0)
        self._offboard_thread = None

    def _offboard_loop(self) -> None:
        assert self._master is not None
        period = 1.0 / max(self.offboard_hz, 2.0)
        while not self._offboard_stop.is_set():
            sp = self._setpoint
            if sp is not None:
                x, y, z, yaw = sp
                type_mask = POSITION_TARGET_TYPEMASK_POS_YAW
                yaw_rad = 0.0
                if yaw is None:
                    type_mask = POSITION_TARGET_TYPEMASK_POS_ONLY
                else:
                    yaw_rad = math.radians(float(yaw))
                self._master.mav.set_position_target_local_ned_send(
                    0,
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    type_mask,
                    float(x),
                    float(y),
                    float(z),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    yaw_rad,
                    0.0,
                )
            time.sleep(period)

    def _wait_altitude_agl(self, alt_m: float, timeout_s: float) -> CommandResult:
        target_z = -abs(alt_m)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            tel = self.telemetry()
            # LOCAL_POSITION_NED z positive down; also accept relative AMSL climb.
            if tel.z <= target_z + 0.5:
                return CommandResult(True, "takeoff altitude reached")
            if tel.alt_amsl_m is not None and tel.in_air and abs(tel.z) >= abs(alt_m) * 0.7:
                return CommandResult(True, "takeoff altitude reached (approx)")
            time.sleep(0.2)
        return CommandResult(
            False,
            f"takeoff altitude timeout (z={self.telemetry().z:.2f}, armed={self.telemetry().armed})",
        )

    def _wait_disarmed(self, timeout_s: float) -> CommandResult:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.telemetry().armed:
                return CommandResult(True, "disarmed after land")
            time.sleep(0.2)
        return CommandResult(False, "land wait disarm timeout")


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0
