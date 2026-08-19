"""Contract tests for profile manifests and reproducible model matrices."""

from __future__ import annotations

import pandas as pd

from anomdet.preprocessing.pipeline import prepare_features
from anomdet.selection.profiles import create_profile, load_profile


def test_profile_and_preparation_round_trip(tmp_path) -> None:
    """A profile selects valid features and generates numeric matrix plus pipeline artefact."""
    config = {
        "project": {"artifact_dir": str(tmp_path / "artifacts")},
        "features": {"high_cardinality_max_categories": 50},
        "preprocessing": {
            "min_non_null_ratio": 0.05,
            "numeric_scaler": "robust",
            "categorical_encoder": "onehot",
        },
    }
    profile_path = create_profile(
        "compact", ["packet_length", "payload_entropy", "http_method"], config
    )
    profile = load_profile(profile_path, config)
    assert profile["feature_count"] == 3

    source = pd.DataFrame(
        {
            "flow_id": ["a", "b", "c", "d"],
            "protocol": ["http", "http", "http", "http"],
            "packet_length": [10, 11, 12, 100],
            "payload_entropy": [0.2, 0.3, 0.4, 7.0],
            "http_method": ["GET", "GET", "POST", "GET"],
        }
    )
    source_path = tmp_path / "features.parquet"
    source.to_parquet(source_path, index=False)
    output_path = tmp_path / "prepared.parquet"

    prepared, manifest, pipeline = prepare_features(
        source_path, output_path, config, str(profile_path)
    )

    assert len(prepared) == 4
    assert manifest["transformed_feature_count"] >= 3
    assert pipeline.exists()
    assert output_path.exists()


def test_robust_preparation_scales_sparse_zero_feature(tmp_path) -> None:
    """A mostly-zero protocol field must not leak its raw magnitude into an autoencoder."""
    config = {
        "project": {"artifact_dir": str(tmp_path / "artifacts")},
        "features": {"high_cardinality_max_categories": 50},
        "preprocessing": {
            "min_non_null_ratio": 0.05,
            "numeric_scaler": "robust",
            "categorical_encoder": "onehot",
        },
    }
    source = pd.DataFrame(
        {
            "flow_id": [f"flow-{index}" for index in range(100)],
            "protocol": ["modbus"] * 100,
            "flow_total_bytes": [0.0] * 98 + [-1_200_000.0, -800_000.0],
        }
    )
    source_path = tmp_path / "sparse-features.parquet"
    output_path = tmp_path / "sparse-prepared.parquet"
    source.to_parquet(source_path, index=False)

    prepared, _manifest, _pipeline = prepare_features(source_path, output_path, config)

    assert prepared["flow_total_bytes"].abs().max() < 20


def test_per_protocol_default_excludes_foreign_schema_features(tmp_path) -> None:
    """The all-features baseline must respect the selected protocol's catalogue."""
    config = {
        "project": {"artifact_dir": str(tmp_path / "artifacts")},
        "features": {"high_cardinality_max_categories": 50},
        "preprocessing": {
            "min_non_null_ratio": 0.05,
            "numeric_scaler": "robust",
            "categorical_encoder": "onehot",
        },
    }
    source = pd.DataFrame(
        {
            "flow_id": ["a", "b", "c", "d"],
            "protocol": ["modbus"] * 4,
            "packet_length": [60, 72, 64, 80],
            "modbus_quantity": [1, 2, 3, 4],
            "dns_rcode": [0, 0, 0, 0],
            "http_status_code": [0, 0, 0, 0],
        }
    )
    source_path = tmp_path / "mixed-schema.parquet"
    output_path = tmp_path / "modbus-prepared.parquet"
    source.to_parquet(source_path, index=False)

    _prepared, manifest, _pipeline = prepare_features(
        source_path, output_path, config, protocols=["modbus"]
    )

    assert "modbus_quantity" in manifest["selected_input_features"]
    assert "dns_rcode" not in manifest["selected_input_features"]
    assert "http_status_code" not in manifest["selected_input_features"]
