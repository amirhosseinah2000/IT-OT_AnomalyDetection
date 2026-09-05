"""Small end-to-end contract test for the deployable two-stage model."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from anomdet.core.config import load_config
from anomdet.core.io import write_table
from anomdet.modelling.two_stage import score_two_stage, train_two_stage


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
    assert {"stage1_anomaly", "predicted_attack_type"}.issubset(predictions.columns)
