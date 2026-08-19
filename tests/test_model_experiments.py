"""Focused contracts for profile experiments and the optional LSTM detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomdet.modelling.training import run_feature_experiments
from anomdet.selection.profiles import create_profile


def _config(tmp_path) -> dict:
    """Return a deliberately small CPU configuration for model contract tests."""
    return {
        "project": {"artifact_dir": str(tmp_path / "artifacts"), "random_seed": 42},
        "runtime": {"cpu_workers": 1},
        "features": {"high_cardinality_max_categories": 20},
        "preprocessing": {
            "min_non_null_ratio": 0.05,
            "numeric_scaler": "robust",
            "categorical_encoder": "onehot",
        },
        "models": {
            "contamination": 0.1,
            "train_split": 0.75,
            "split_strategy": "temporal",
            "candidates": ["isolation_forest", "lstm_autoencoder"],
            "isolation_forest": {"n_estimators": 10, "max_samples": "auto"},
            "local_outlier_factor": {"n_neighbors": 5},
            "one_class_svm": {"nu": 0.1, "gamma": "scale"},
            "lstm_autoencoder": {
                "sequence_length": 4,
                "sequence_stride": 4,
                "hidden_size": 8,
                "latent_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "learning_rate": 0.01,
                "batch_size": 8,
                "epochs": 2,
                "validation_fraction": 0.2,
                "patience": 2,
                "max_train_windows": 20,
                "device": "cpu",
            },
        },
    }


def test_profile_experiment_writes_baseline_and_selected_comparison(tmp_path) -> None:
    """The aggregate report makes all-feature versus selected-profile evidence explicit."""
    config = _config(tmp_path)
    rows = 40
    source = pd.DataFrame(
        {
            "flow_id": [f"flow-{index // 2}" for index in range(rows)],
            "protocol": ["http"] * rows,
            "packet_length": np.linspace(60, 600, rows),
            "payload_entropy": np.linspace(0.1, 7.5, rows),
            "http_method": ["GET", "POST"] * (rows // 2),
        }
    )
    feature_path = tmp_path / "features.parquet"
    source.to_parquet(feature_path, index=False)
    profile = create_profile(
        "compact-http", ["packet_length", "payload_entropy"], config, protocols=["http"]
    )

    comparison, summary = run_feature_experiments(
        feature_path=feature_path,
        config=config,
        strategy="per_protocol",
        group="all",
        profiles=[profile],
        output_dir=tmp_path / "experiment",
        candidates=["isolation_forest"],
    )

    assert set(comparison["feature_profile"]) == {"all_features", str(profile)}
    assert set(comparison["model"]) == {"isolation_forest"}
    assert (tmp_path / "experiment" / "comparison.parquet").exists()
    assert len(list((tmp_path / "experiment").rglob("feature-importance.parquet"))) == 4
    assert summary["successful_profiles"] == 2


def test_lstm_autoencoder_scores_and_persists_history_when_torch_is_available(tmp_path) -> None:
    """The sequence detector must retain one score per record and an epoch history."""
    pytest.importorskip("torch")
    from anomdet.modelling.lstm_autoencoder import LSTMAutoencoder

    values = np.random.default_rng(42).normal(size=(48, 3)).astype(np.float32)
    events: list[dict[str, object]] = []
    model = LSTMAutoencoder(
        sequence_length=6,
        sequence_stride=6,
        hidden_size=8,
        latent_size=4,
        batch_size=8,
        epochs=2,
        patience=2,
        max_train_windows=20,
    ).fit(values, progress_callback=events.append)

    assert len(model.score_samples(values)) == len(values)
    assert len(model.training_history_) >= 1
    epoch_events = [event for event in events if event["event"] == "epoch"]
    assert len(epoch_events) == len(model.training_history_)
    assert events[0]["event"] == "lstm_started"
    assert events[-1]["event"] == "lstm_completed"
    assert model.save(tmp_path / "model.pt").exists()
