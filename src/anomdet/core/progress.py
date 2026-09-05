"""Durable, console-visible progress events for long-running platform jobs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from anomdet.core.io import utc_now, write_json

LOGGER = logging.getLogger("anomdet")


class DurableProgress:
    """Persist a bounded event timeline that a dashboard can read during a CLI job.

    The command-line log remains the authoritative operational trace.  This file
    is deliberately a small, human-readable mirror for a dashboard or another
    host to monitor without keeping a Python process/session object alive.
    """

    def __init__(self, path: Path, kind: str, history_limit: int = 160) -> None:
        self.path = path
        self.kind = kind
        self.history_limit = history_limit
        now = utc_now()
        self.payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "kind": kind,
            "status": "running",
            "started_at": now,
            "updated_at": now,
            "current": {},
            "events": [],
        }
        self._write()

    def emit(
        self,
        event: str,
        *,
        stage: str,
        message: str,
        progress_ratio: float | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        """Record one meaningful job event and emit the same fact to the console log."""
        now = utc_now()
        entry: dict[str, Any] = {
            "at": now,
            "event": event,
            "stage": stage,
            "message": message,
            **details,
        }
        if progress_ratio is not None:
            entry["progress_ratio"] = min(max(float(progress_ratio), 0.0), 1.0)
        self.payload["updated_at"] = now
        self.payload["current"] = entry
        self.payload["events"] = [*self.payload["events"], entry][-self.history_limit :]
        self._write()
        LOGGER.info(
            "%s_PROGRESS stage=%s event=%s %s",
            self.kind.upper(),
            stage,
            event,
            message,
        )
        return entry

    def complete(self, message: str, **details: Any) -> None:
        self.payload["status"] = "completed"
        self.emit("completed", stage="complete", message=message, progress_ratio=1.0, **details)

    def fail(self, message: str, **details: Any) -> None:
        self.payload["status"] = "failed"
        self.emit("failed", stage="failed", message=message, **details)

    def _write(self) -> None:
        write_json(self.payload, self.path)
