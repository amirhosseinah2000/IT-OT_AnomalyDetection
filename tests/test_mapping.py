"""Contract tests for evidence-preserving flow-label matching."""

from __future__ import annotations

import pandas as pd

from anomdet.mapping.mapper import map_features_to_labels


def test_maps_exact_five_tuple_and_time() -> None:
    """A compatible 5-tuple within tolerance receives the CSV label and confidence."""
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    features = pd.DataFrame(
        {
            "flow_id": ["tcp|10.0.0.1:50000|10.0.0.2:80"],
            "timestamp": [timestamp],
            "src_ip": ["10.0.0.1"],
            "src_port": [50000],
            "dst_ip": ["10.0.0.2"],
            "dst_port": [80],
            "protocol": ["http"],
            "packet_length": [120],
        }
    )
    labels = pd.DataFrame(
        {
            "csv_row": [4],
            "label": ["attack"],
            "label_timestamp": [timestamp],
            "label_end_timestamp": [pd.NaT],
            "src_ip": ["10.0.0.1"],
            "src_port": [50000],
            "dst_ip": ["10.0.0.2"],
            "dst_port": [80],
            "source_file": ["labels.csv"],
        }
    )
    config = {
        "mapping": {
            "timestamp_tolerance_seconds": 5.0,
            "allow_reverse_flow_match": True,
            "default_label": "unknown",
        }
    }

    result = map_features_to_labels(features, labels, config)

    assert result.loc[0, "label"] == "attack"
    assert result.loc[0, "match_status"] == "matched"
    assert result.loc[0, "match_confidence"] == 0.95


def test_does_not_assign_label_when_time_is_outside_tolerance() -> None:
    """A 5-tuple alone is insufficient when a conflicting known timestamp is far away."""
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    features = pd.DataFrame(
        {
            "flow_id": ["tcp|10.0.0.1:50000|10.0.0.2:80"],
            "timestamp": [timestamp],
            "src_ip": ["10.0.0.1"],
            "src_port": [50000],
            "dst_ip": ["10.0.0.2"],
            "dst_port": [80],
            "protocol": ["http"],
            "packet_length": [120],
        }
    )
    labels = pd.DataFrame(
        {
            "csv_row": [4],
            "label": ["attack"],
            "label_timestamp": [timestamp + pd.Timedelta(minutes=10)],
            "label_end_timestamp": [pd.NaT],
            "src_ip": ["10.0.0.1"],
            "src_port": [50000],
            "dst_ip": ["10.0.0.2"],
            "dst_port": [80],
            "source_file": ["labels.csv"],
        }
    )
    config = {
        "mapping": {
            "timestamp_tolerance_seconds": 5.0,
            "allow_reverse_flow_match": True,
            "default_label": "unknown",
        }
    }

    result = map_features_to_labels(features, labels, config)

    assert result.loc[0, "label"] == "unknown"
    assert result.loc[0, "match_status"] == "unmatched"
