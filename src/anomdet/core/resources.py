"""Runtime-resource inspection and conservative CPU parallelism selection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    """A point-in-time resource report used in the CLI and dashboard."""

    logical_cpus: int
    physical_cpus: int | None
    cpu_percent: float
    total_memory_gb: float
    available_memory_gb: float
    memory_percent: float

    def as_dict(self) -> dict[str, float | int | None]:
        """Convert the immutable snapshot into serialisable dashboard data."""
        return asdict(self)


def snapshot() -> ResourceSnapshot:
    """Return the host resources available to the current Python process."""
    memory = psutil.virtual_memory()
    return ResourceSnapshot(
        logical_cpus=os.cpu_count() or 1,
        physical_cpus=psutil.cpu_count(logical=False),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        total_memory_gb=round(memory.total / 1024**3, 2),
        available_memory_gb=round(memory.available / 1024**3, 2),
        memory_percent=memory.percent,
    )


def effective_workers(configured_workers: int) -> int:
    """Resolve zero to a safe default that leaves one logical CPU available."""
    if configured_workers > 0:
        return configured_workers
    return max(1, (os.cpu_count() or 2) - 1)
