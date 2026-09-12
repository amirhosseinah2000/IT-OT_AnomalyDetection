"""Small end-to-end contract test for the deployable two-stage model."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from anomdet.core.config import load_config
from anomdet.core.io import write_table
from anomdet.modelling.two_stage import (
    _attack_capture_roles,
    _fit_stage_one_binary_gate,
    score_two_stage,
    train_two_stage,
)


def _records(count: int, *, attack: bool, attack_type: str = "unknown") -> pd.DataFrame:
    values = np.arange(count, dtype=float)
    offset = 40.0 if attack else 0.0
    return pd.DataFrame(
        {
            "capture": ["attack.pcap" if attack else "benign.pcap"] * count,
            "timestamp": pd.date_range("2026-01-01", periods=count, freq="s", tz="UTC"),
            "protocol": ["dns"] * count,
            "flow_id": [f"{'a' if attack else 'b'}-{index}" for index in range(count)],
            "src_ip": ["10.0.0.1"] * count,
            "src_port": 50000 + values,
            "dst_ip": ["10.0.0.2"] * count,
            "dst_port": [53] * count,
            "packet_length": 100.0 + values + offset,
            "payload_size": 20.0 + values / 3 + offset,
            "payload_entropy": 1.0 + values / 50 + offset / 100,
            "flow_duration": values / 10 + offset / 10,
            "label": [attack_type if attack else "benign"] * count,
            "is_attack": [attack] * count,
            "mapping_accepted": [attack] * count,
        }
    )


def test_trains_and_scores_frozen_two_stage_contract(tmp_path) -> None:
    """The saved model preserves selected features and gates type predictions behind stage one."""
    config = copy.deepcopy(load_config())
    config["models"]["max_source_rows"] = None
    config["runtime"]["cpu_workers"] = 1
    config["evaluation"].update(
        {
            "diagnostic_cpu_workers": 1,
            "permutation_repeats": 1,
            "seed_stability_seeds": [17],
        }
    )
    config["models"]["isolation_forest"]["n_estimators"] = 8
    config["models"]["lstm_autoencoder"].update(
        {
            "sequence_length": 4,
            "sequence_stride": 4,
            "epochs": 1,
            "patience": 1,
            "batch_size": 8,
            "max_train_windows": 12,
            "validation_fraction": 0.0,
        }
    )
    config["stage_two"].update({"n_estimators": 8, "min_samples_leaf": 1, "test_fraction": 0.25})
    benign_path = tmp_path / "benign.parquet"
    attack_one_path = tmp_path / "attack-one.parquet"
    attack_two_path = tmp_path / "attack-two.parquet"
    write_table(_records(36, attack=False), benign_path)
    write_table(_records(12, attack=True, attack_type="type-one"), attack_one_path)
    write_table(_records(12, attack=True, attack_type="type-two"), attack_two_path)

    summary = train_two_stage(
        [benign_path],
        [attack_one_path, attack_two_path],
        config,
        tmp_path / "model",
        protocol="dns",
    )

    contract = tmp_path / "model" / "model-contract.json"
    predictions = score_two_stage(_records(4, attack=True, attack_type="type-one"), contract)
    assert summary["stage1"]["input_features"] > 0
    assert contract.exists()
    assert (tmp_path / "model" / "onnx" / "lstm-autoencoder.onnx").exists()
    assert (tmp_path / "model" / "onnx" / "attack-classifier.onnx").exists()
    for name in [
        "ablation-comparison.parquet",
        "inference-latency.parquet",
        "threshold-sensitivity.parquet",
        "end-to-end-probabilities.parquet",
        "seed-stability.parquet",
    ]:
        assert (tmp_path / "model" / "pipeline" / name).exists()
    assert {"stage1_anomaly", "predicted_attack_type"}.issubset(predictions.columns)


def test_stage_one_remains_deployable_when_stage_two_labels_are_unavailable(tmp_path) -> None:
    """A protocol with no accepted attack labels must not lose its anomaly detector."""
    config = copy.deepcopy(load_config())
    config["models"]["max_source_rows"] = None
    config["runtime"]["cpu_workers"] = 1
    config["models"]["isolation_forest"]["n_estimators"] = 8
    config["models"]["lstm_autoencoder"].update(
        {
            "sequence_length": 4,
            "sequence_stride": 4,
            "epochs": 1,
            "patience": 1,
            "batch_size": 8,
            "max_train_windows": 12,
            "validation_fraction": 0.0,
        }
    )
    benign_path = tmp_path / "benign.parquet"
    write_table(_records(36, attack=False), benign_path)

    summary = train_two_stage([benign_path], [], config, tmp_path / "stage1-only", protocol="dns")
    contract = tmp_path / "stage1-only" / "model-contract.json"
    predictions = score_two_stage(_records(4, attack=True), contract)

    assert summary["stage2"]["status"] == "skipped"
    assert summary["stage1"]["held_out_attack_test"] is None
    assert contract.exists()
    assert (tmp_path / "stage1-only" / "stage2" / "readiness.json").exists()
    assert set(predictions["stage2_status"]) == {"skipped"}


def test_binary_gate_requires_and_uses_capture_disjoint_attack_training() -> None:
    """Known attacks only train the small Stage 1 gate from separate captures."""
    config = copy.deepcopy(load_config())
    config["stage_one"]["supervised_gate"].update(
        {"min_labelled_attack_records": 2, "max_iter": 5, "min_samples_leaf": 2}
    )
    attacks = pd.DataFrame(
        {
            "capture": ["attack-a"] * 4 + ["attack-b"] * 4 + ["attack-c"] * 4,
            "label": ["scan"] * 12,
        }
    )
    attack_truth = np.ones(len(attacks), dtype=int)
    roles, split = _attack_capture_roles(attacks, attack_truth, 0.2)
    normal_values = np.zeros((12, 2), dtype=float)
    attack_values = np.full((12, 2), 4.0, dtype=float)

    gate, details = _fit_stage_one_binary_gate(
        normal_values,
        np.arange(len(normal_values)),
        attack_values,
        attack_truth,
        roles,
        config,
    )

    assert split["training_captures"] == ["attack-a"]
    assert split["evaluation_captures"] == ["attack-c"]
    assert details["status"] == "trained"
    assert gate is not None
