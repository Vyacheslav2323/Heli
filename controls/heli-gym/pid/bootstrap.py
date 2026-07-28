"""Ensure heli-gym root is on sys.path for script-style invocations."""

from __future__ import annotations

import sys
from pathlib import Path

HELI_GYM_ROOT = Path(__file__).resolve().parents[1]


def ensure_heli_gym_path() -> None:
    if str(HELI_GYM_ROOT) not in sys.path:
        sys.path.insert(0, str(HELI_GYM_ROOT))
