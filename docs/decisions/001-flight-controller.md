# ADR 001 — Flight Controller Selection

**Status:** Open (deliberately deferred)
**Date:** 2026-07-23

## Context

The physical flight controller (FC) for the RC helicopter is not yet chosen.
The architecture isolates this choice behind the `flight/adapter` interface, so
simulation and mission work proceed on PX4 SITL while the hardware decision stays open.

## Options

### Option A — PX4 (Pixhawk-class hardware)

- **What:** Full open-source autopilot; already used in our SITL setup (`tools/`).
- **Helicopter support:** Native single-rotor helicopter airframe support (swashplate mixing, autorotation logic since v1.14+).
- **Offboard API:** Excellent — MAVLink offboard mode, MAVSDK/pymavlink, ROS 2 (uXRCE-DDS).
- **Hardware:** Pixhawk 6C/6X (~$100–250), or smaller FMU boards.
- **Pros:** Sim-to-real continuity (we already simulate PX4); strong offboard control for autonomy; active development.
- **Cons:** Helicopter airframes are less battle-tested than multirotors; setup complexity.

### Option B — ArduPilot (Pixhawk-class or cheaper boards)

- **What:** Open-source autopilot with the **most mature traditional-helicopter support** (ArduCopter "Heli" branch, decades of tuning knowledge, autorotation, governor support).
- **Offboard API:** MAVLink guided mode, DroneKit/pymavlink, ROS 2 support.
- **Hardware:** Same Pixhawk-class boards, plus cheap boards (Matek, SpeedyBee, ~$40–80).
- **Pros:** Best-in-class heli firmware; huge community for traditional helis; runs on cheaper hardware.
- **Cons:** Our current sim tooling is PX4-based (though ArduPilot SITL exists and is good); codebase older/cruftier.

### Option C — Betaflight / Rotorflight

- **What:** Rotorflight is a Betaflight fork specifically for RC helicopters (flybarless control, rescue mode).
- **Pros:** Excellent raw stabilization for hobby helis; cheap FC boards (~$30–60); large RC-heli hobbyist community.
- **Cons:** **No real autonomy stack** — no waypoints, no offboard API, minimal MAVLink. Would need a companion computer doing all guidance and injecting RC-style commands. Poor fit for our autonomy roadmap unless paired with Option D thinking.

### Option D — Arduino / custom FC + companion computer

- **What:** Build our own stabilization on a microcontroller (Arduino/Teensy/ESP32 + IMU), with a companion computer (Raspberry Pi / Jetson) for autonomy.
- **Pros:** Maximum learning and control; perfectly matches the "software-centered" ethos; the `flight/arduino` adapter slot exists for this.
- **Cons:** Reimplementing solved problems (EKF, mixing, failsafes) is months of work and the biggest crash risk; hard to certify safety.

### Option E — Hybrid (recommended direction, not yet decided)

- Proven FC (PX4 or ArduPilot) handles **stabilization + failsafes**;
  a companion computer (Jetson Orin Nano / Raspberry Pi 5) runs **perception,
  mapping, mission, and reasoning** and commands the FC via MAVLink.
- This mirrors the architecture's flight/mission separation exactly and is the
  industry-standard pattern for autonomous UAVs.

## Comparison

| Criterion | PX4 | ArduPilot | Rotorflight | Custom |
|---|---|---|---|---|
| Traditional-heli firmware maturity | Good | **Best** | Good (stabilization only) | None |
| Offboard/autonomy API | **Excellent** | Excellent | Poor | DIY |
| Sim continuity with current setup | **Yes (SITL in use)** | SITL available | Limited | heli-gym only |
| Cost | $$ | $–$$ | $ | $ |
| Effort to first autonomous flight | Medium | Medium | High | Very high |

## Decision Triggers

Revisit and close this ADR when:

1. The physical helicopter airframe is chosen (size/payload determines FC options), and
2. Phase 2 (basic autonomous commands) is demonstrated in SITL.

Until then: all code targets the `flight/adapter` interface; PX4 SITL remains the development backend.
