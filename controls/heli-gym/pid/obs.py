"""Observation index constants for gym vs full-dynamics views.

Attitude controllers read gym observations (HeliStabilization._get_obs base slice).
Position cascade reads full dynamics via env.heli_dyn.observation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FullObs:
    """Indices into heli_dyn.observation (OBS_NAMES in skills.training.envs)."""

    nvel: int = 4
    evel: int = 5
    roll: int = 7
    pitch: int = 8
    yaw: int = 9
    rollrate: int = 10
    pitchrate: int = 11
    yawrate: int = 12
    npos: int = 13
    epos: int = 14
    alt: int = 15
    alt_gr: int = 16


@dataclass(frozen=True)
class GymObs:
    """Indices into HeliStabilization gym observation base slice."""

    roll: int = 3
    pitch: int = 4
    yaw: int = 5
    rollrate: int = 6
    pitchrate: int = 7
    yawrate: int = 8


FULL_OBS = FullObs()
GYM_OBS = GymObs()
