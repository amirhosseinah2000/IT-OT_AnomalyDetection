"""Contracts for the durable background training monitor."""

from __future__ import annotations

import json

import pandas as pd

from anomdet.dashboard import experiment_worker


def test_worker_persists_epoch_events_and_completion(tmp_path, monkeypatch) -> None:
    """The monitor file must keep observable progress after the Streamlit request ends."""
    progress_path = tmp_path / "training-progress.json"

    def fake_experiment(**kwargs):
        callback = kwargs["progress_callback"]
        callback({"event": "model_started", "scope": "modbus", "model": "lstm_autoencoder"})
        callback(
            {
                "event": "epoch",
                "scope": "modbus",
                "model": "lstm_autoencoder",
                "epoch": 1,
                "epochs": 3,
                "train_loss": 0.8,
                "validation_loss": 0.9,
                "best_loss": 0.9,
            }
        )
        return pd.DataFrame({"model": ["lstm_autoencoder"]}), {
            "comparison": str(tmp_path / "comparison.parquet"),
            "successful_profiles": 1,
        }

    monkeypatch.setattr(experiment_worker, "run_feature_experiments", fake_experiment)
    request = {
        "progress_path": str(progress_path),
        "feature_path": str(tmp_path / "modbus.parquet"),
        "config": {},
        "strategy": "per_protocol",
        "group": "all",
        "profiles": [],
        "output_dir": str(tmp_path / "experiment"),
        "labels_path": None,
        "candidates": ["lstm_autoencoder"],
        "model_overrides": {},
        "run_sweep": False,
        "sweep_variants": [],
    }

    assert experiment_worker.run_request(request) == 0
    with progress_path.open(encoding="utf-8") as handle:
        progress = json.load(handle)

    assert progress["status"] == "complete"
    assert progress["result"]["detector_runs"] == 1
    assert [event["event"] for event in progress["events"]] == [
        "worker_started",
        "model_started",
        "epoch",
        "worker_completed",
    ]
