"""Background worker for durable Experiment Studio training progress.

The Streamlit process is intentionally not responsible for training models.  A
separate Python process writes an atomically updated JSON progress record that
the dashboard can poll without blocking its browser session.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from anomdet.core.io import utc_now
from anomdet.modelling.training import run_feature_experiments, run_lstm_sweep

MAX_PROGRESS_EVENTS = 200


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace the progress file so a polling dashboard never reads a partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _event_record(event: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    """Add durable time fields while retaining the useful training callback details."""
    return {
        "at": utc_now(),
        "elapsed_s": round(elapsed_seconds, 1),
        **event,
    }


def _paths(values: list[Any]) -> list[Path]:
    """Convert JSON request paths back to Path objects at the worker boundary."""
    return [Path(str(value)) for value in values]


def run_request(request: dict[str, Any]) -> int:
    """Execute one requested experiment and persist every observable state transition."""
    progress_path = Path(str(request["progress_path"]))
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "events": [],
    }

    def persist() -> None:
        payload["updated_at"] = utc_now()
        _write_progress(progress_path, payload)

    def report(event: dict[str, Any]) -> None:
        payload["events"].append(_event_record(event, time.perf_counter() - started))
        if len(payload["events"]) > MAX_PROGRESS_EVENTS:
            del payload["events"][:-MAX_PROGRESS_EVENTS]
        persist()

    report(
        {
            "event": "worker_started",
            "feature_path": str(request["feature_path"]),
            "detectors": request.get("candidates", []),
        }
    )
    try:
        comparison, summary = run_feature_experiments(
            feature_path=Path(str(request["feature_path"])),
            config=dict(request["config"]),
            strategy=str(request["strategy"]),
            group=str(request["group"]),
            profiles=_paths(list(request.get("profiles", []))),
            output_dir=Path(str(request["output_dir"])),
            labels_path=(Path(str(request["labels_path"])) if request.get("labels_path") else None),
            candidates=[str(item) for item in request.get("candidates", [])],
            model_overrides=dict(request.get("model_overrides", {})),
            progress_callback=report,
        )
        result: dict[str, Any] = {
            "detector_runs": len(comparison),
            "comparison": summary.get("comparison"),
            "successful_profiles": summary.get("successful_profiles"),
        }
        if request.get("run_sweep"):
            sweep, sweep_summary = run_lstm_sweep(
                feature_path=Path(str(request["feature_path"])),
                config=dict(request["config"]),
                strategy=str(request["strategy"]),
                group=str(request["group"]),
                profiles=_paths(list(request.get("profiles", []))),
                parameter_sets=list(request.get("sweep_variants", [])),
                output_dir=Path(str(request["sweep_output_dir"])),
                labels_path=(
                    Path(str(request["labels_path"])) if request.get("labels_path") else None
                ),
                progress_callback=report,
            )
            result["sweep_runs"] = len(sweep)
            result["sweep_comparison"] = sweep_summary.get("comparison")
        payload["result"] = result
        payload["status"] = "complete"
        report({"event": "worker_completed", **result})
    except Exception as error:  # The dashboard must remain usable after a worker failure.
        payload["status"] = "error"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc(limit=12)
        report({"event": "worker_failed", "error": payload["error"]})
        return 1
    return 0


def main() -> int:
    """Run the worker from a JSON request produced by the Streamlit dashboard."""
    parser = argparse.ArgumentParser(description="Run one anomaly dashboard experiment.")
    parser.add_argument(
        "--request", required=True, type=Path, help="Path to the JSON request file."
    )
    arguments = parser.parse_args()
    with arguments.request.open("r", encoding="utf-8") as handle:
        request = json.load(handle)
    if not isinstance(request, dict):
        raise ValueError("Experiment request must be a JSON object.")
    return run_request(request)


if __name__ == "__main__":  # pragma: no cover - subprocess entry point.
    raise SystemExit(main())
