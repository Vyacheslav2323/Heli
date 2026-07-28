# PX4 Commander & MAVLink Mapping

Reference for the companion `flight/px4` adapter. Sources: PX4 `commander`
module usage, `VehicleCommand.msg`, Offboard mode docs.

## Architecture reminder

Companion Python owns mission rules. PX4 executes modes/setpoints over MAVLink.
QGC monitors telemetry and can author static missions; it is not the rule engine.

## Full `commander` CLI surface

| Command | Purpose |
|---|---|
| `start [-h]` | Start commander (optional HIL) |
| `stop` | Stop module |
| `status` | Print arming/mode/failsafe state |
| `calibrate <type>` | `mag\|baro\|accel\|gyro\|level\|esc\|airspeed` (+ `quick` for mag/accel) |
| `check` | Run preflight/prearm checks |
| `safety on\|off` | Prearm safety switch state |
| `arm [-f]` | Arm (`-f` force / skip checks) |
| `disarm [-f]` | Disarm (`-f` allow in-air) |
| `takeoff` | `NAV_TAKEOFF` then arm |
| `land` | `NAV_LAND` |
| `transition` | VTOL transition |
| `mode <name>` | Change flight mode (see below) |
| `pair` | RC receiver pairing |
| `termination on\|off` | Flight termination / lockdown |
| `set_ekf_origin <lat> <lon> <alt>` | Set EKF global origin |
| `set_heading <deg>` | Set current heading estimate |
| `poweroff` | Board power off if supported |

### `commander mode` values

`manual` · `acro` · `offboard` · `stabilized` · `altctl` · `posctl` ·
`altitude_cruise` · `position:slow` · `auto:mission` · `auto:loiter` ·
`auto:rtl` · `auto:takeoff` · `auto:land` · `auto:precland` · `ext1`

CLI verbs wrap `VEHICLE_CMD_*` / MAVLink `COMMAND_LONG`. The adapter speaks
MAVLink, not NSH.

## Primitives → PX4 / MAVLink

| Abstract need | Preferred PX4 path | MAVLink / messages | Notes |
|---|---|---|---|
| **Takeoff** | `auto:takeoff` or `NAV_TAKEOFF` (+ arm) | `MAV_CMD_NAV_TAKEOFF` (22); mode via `MAV_CMD_DO_SET_MODE` (176) | Param4 = yaw; alt AMSL. CLI `commander takeoff` does takeoff then arm. |
| **Land** | `auto:land` or `NAV_LAND` | `MAV_CMD_NAV_LAND` (21) | Optional lat/lon/yaw. Leave Offboard first. |
| **Stabilization / hold** | Hold modes | `stabilized`, `altctl`, `posctl`, `auto:loiter` via `DO_SET_MODE` | Autonomy hover: **`posctl`** or Offboard hold. |
| **Move to XYZ** | Offboard position setpoints | `SET_POSITION_TARGET_LOCAL_NED`; or `MAV_CMD_DO_REPOSITION` (192); mission `NAV_WAYPOINT` (16) | Offboard needs ≥~2 Hz stream. |
| **Turn (yaw)** | Setpoint yaw or condition yaw | `SET_POSITION_TARGET_*` yaw bits; `MAV_CMD_CONDITION_YAW` (115) | Relative vs absolute via CONDITION_YAW param4. |

### Offboard constraint

Offboard is for continuous setpoints. Takeoff / land / RTL are better as **mode
switches** (`auto:takeoff`, `auto:land`, `auto:rtl`), then return to Offboard
for scripted XYZ/yaw.

## PX4 custom mode encoding (`DO_SET_MODE`)

`MAV_CMD_DO_SET_MODE` with `base_mode` bit `MAV_MODE_FLAG_CUSTOM_MODE_ENABLED`
and PX4 custom main/sub modes:

| Mode string | Main | Sub |
|---|---|---|
| `manual` | MANUAL | — |
| `acro` | ACRO | — |
| `stabilized` | STABILIZED | — |
| `altctl` | ALTCTL | — |
| `posctl` | POSCTL | — |
| `offboard` | OFFBOARD | — |
| `auto:mission` | AUTO | MISSION |
| `auto:loiter` | AUTO | LOITER |
| `auto:rtl` | AUTO | RTL |
| `auto:takeoff` | AUTO | TAKEOFF |
| `auto:land` | AUTO | LAND |
| `auto:precland` | AUTO | PRECLAND |

Numeric values match PX4 `px4_custom_mode` (see adapter constants).

## Abstract adapter mapping (v1)

| Abstract command | Adapter action |
|---|---|
| `takeoff` | `NAV_TAKEOFF` or AUTO+TAKEOFF + arm; wait for altitude |
| `hold` | `DO_SET_MODE` POSCTL (or Offboard hold) |
| `goto_ned` / `goto_global` / `yaw_to` | Enter Offboard; stream position (+ yaw); wait acceptance |
| `land` | Exit Offboard → `NAV_LAND` / AUTO+LAND; wait disarmed |
| `abort` | `auto:rtl` or `auto:land` per policy |

## QGC role

Plan view can author `NAV_TAKEOFF` → `NAV_WAYPOINT`(s) → `NAV_LAND` /
`CONDITION_YAW` for fixed demos. The mission executor generates the same
semantics programmatically and may later upload missions for `auto:mission`.

## SITL connection (companion)

PX4 SITL starts two relevant mavlink instances:

| Instance | Listen | Remote (PX4 → client) |
|---|---|---|
| Normal (GCS / QGC) | `:18570` | `:14550` |
| Onboard (offboard / companion) | `:14580` | `:14540` |

Companion clients should use **`udpin:0.0.0.0:14540`** on the **same host as PX4** (WSL).
Windows cannot reach WSL loopback; `tools/sitl_mission_smoke.py` re-execs into WSL automatically.
QGC keeps using `tools/mavlink_wsl_bridge.py` on `:14550`.
