"""Factory for HeliStabilization environments used by pid scripts."""

from __future__ import annotations

from pid.wind_config import configure_wind
from skills.training.envs import DEFAULT_TRIM_COND, HeliStabilization


def make_stabilization_env(
    *,
    action_limit: float,
    max_time: float,
    init_attitude_range: float,
    render_enabled: bool = False,
    trim_cond: dict | None = None,
    target_alt: float | None = None,
    init_pos_range: float | None = None,
    wind_spd: float | None = None,
    wind_dir: float | None = None,
    turb_lvl: int | None = None,
) -> HeliStabilization:
    resolved_trim = dict(trim_cond or DEFAULT_TRIM_COND)
    if target_alt is not None:
        resolved_trim["gr_alt"] = target_alt

    env = HeliStabilization(
        heli_name="aw109",
        render_enabled=render_enabled,
        max_time=max_time,
        action_limit=action_limit,
        init_attitude_range=init_attitude_range,
        trim_cond=resolved_trim,
    )
    if init_pos_range is not None:
        env._init_pos_range = init_pos_range  # type: ignore[attr-defined]
    if any(value is not None for value in (wind_spd, wind_dir, turb_lvl)):
        configure_wind(
            env,
            wind_spd=wind_spd,
            wind_dir=wind_dir,
            turb_lvl=turb_lvl,
        )
    return env
