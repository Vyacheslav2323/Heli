"""Mission executor: run approved commands strictly in order."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from flight.adapter import AbortPolicy, CommandResult, FlightAdapter


class ExecutorState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ExecutorEvent:
    kind: str
    index: int = -1
    message: str = ""
    command: Optional[Mapping[str, Any]] = None


@dataclass
class ExecutorResult:
    state: ExecutorState
    completed: int
    events: list[ExecutorEvent] = field(default_factory=list)
    error: str = ""


ProgressCallback = Callable[[ExecutorEvent], None]


class MissionExecutor:
    """Linear executor: one command after another, no injected takeoff/land.

    - ``takeoff`` is skipped by the adapter when already flying.
    - ``land`` runs only when present in the command list.
    - Command failure stops the mission; it does **not** auto-land.
    - Explicit ``request_abort()`` uses ``abort_policy`` (rtl/land).
    """

    def __init__(
        self,
        adapter: FlightAdapter,
        *,
        abort_policy: AbortPolicy = AbortPolicy.RTL,
        hold_poll_s: float = 0.1,
        on_event: Optional[ProgressCallback] = None,
    ) -> None:
        self.adapter = adapter
        self.abort_policy = abort_policy
        self.hold_poll_s = hold_poll_s
        self.on_event = on_event
        self.state = ExecutorState.IDLE
        self._abort_requested = False

    def request_abort(self) -> None:
        self._abort_requested = True

    def run(self, commands: Sequence[Mapping[str, Any]]) -> ExecutorResult:
        self.state = ExecutorState.RUNNING
        self._abort_requested = False
        events: list[ExecutorEvent] = []
        completed = 0

        def emit(event: ExecutorEvent) -> None:
            events.append(event)
            if self.on_event:
                self.on_event(event)

        emit(ExecutorEvent("started", message=f"{len(commands)} commands"))

        for i, cmd in enumerate(commands):
            if self._abort_requested:
                return self._finish_abort(events, completed, i)

            emit(ExecutorEvent("command_start", index=i, command=dict(cmd)))
            result = self._execute_one(cmd)

            if self._abort_requested or result.details.get("aborted"):
                return self._finish_abort(
                    events,
                    completed,
                    i,
                    message=result.message if result.details.get("aborted") else "",
                )

            if not result.ok:
                self.state = ExecutorState.FAILED
                emit(
                    ExecutorEvent(
                        "command_failed",
                        index=i,
                        message=result.message,
                        command=dict(cmd),
                    )
                )
                # Stop in place — do not land/rtl unless the plan said so.
                return ExecutorResult(self.state, completed, events, error=result.message)

            emit(ExecutorEvent("command_done", index=i, message=result.message, command=dict(cmd)))
            completed += 1

        self.state = ExecutorState.SUCCEEDED
        emit(ExecutorEvent("succeeded", message=f"completed {completed} commands"))
        return ExecutorResult(self.state, completed, events)

    def _finish_abort(
        self,
        events: list[ExecutorEvent],
        completed: int,
        index: int,
        message: str = "",
    ) -> ExecutorResult:
        abort_res = self.adapter.abort(self.abort_policy)
        self.state = ExecutorState.ABORTED
        event = ExecutorEvent(
            "aborted",
            index=index,
            message=message or abort_res.message,
            command={"type": "abort", "policy": self.abort_policy.value},
        )
        events.append(event)
        if self.on_event:
            self.on_event(event)
        return ExecutorResult(self.state, completed, events, error="abort requested")

    def _execute_one(self, cmd: Mapping[str, Any]) -> CommandResult:
        ctype = cmd.get("type")
        if ctype == "hold" and cmd.get("duration_s") is not None:
            hold = self.adapter.hold()
            if not hold.ok:
                return hold
            deadline = time.monotonic() + float(cmd["duration_s"])
            while time.monotonic() < deadline:
                if self._abort_requested:
                    return CommandResult(
                        False,
                        "hold interrupted by abort",
                        details={"aborted": True},
                    )
                time.sleep(self.hold_poll_s)
            return CommandResult(True, f"held {cmd['duration_s']}s")
        return self.adapter.execute(cmd)
