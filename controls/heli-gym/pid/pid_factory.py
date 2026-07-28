"""simple-pid helpers shared by controllers."""

from __future__ import annotations

from simple_pid import PID


def require_simple_pid() -> None:
    """Import guard; call after bootstrap if PID type is needed at module load."""
    try:
        from simple_pid import PID as _PID  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "simple-pid is required for this script.\n"
            "Install with: pip install simple-pid"
        ) from exc


def make_axis_pids(
    kp: float,
    ki: float,
    kd: float,
    *,
    sample_time: float,
    output_limits: tuple[float, float] | None,
    axes: tuple[str, ...],
) -> dict[str, PID]:
    """Create PIDs with setpoint 0.

    Setpoints may be reassigned each control step (cascade controllers).
    Integral state persists across steps while setpoints track moving targets.
    """
    return {
        axis: PID(
            kp,
            ki,
            kd,
            setpoint=0.0,
            sample_time=sample_time,
            output_limits=output_limits,
        )
        for axis in axes
    }


def reset_pids(*pid_groups: dict[str, PID]) -> None:
    for group in pid_groups:
        for pid in group.values():
            pid.reset()
