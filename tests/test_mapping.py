"""Contract tests for evidence-preserving flow-label matching."""

from __future__ import annotations

import pandas as pd

from anomdet.core.io import write_table
from anomdet.mapping.mapper import (
    apply_cached_packet_labels_parquet,
    attach_packet_labels,
    load_mapping_cache,
    mapping_cache_plan,
    map_features_to_labels,
    save_mapping_cache,
)


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


def test_packet_mapping_preserves_a_label_transition_inside_one_long_flow() -> None:
    """A later attack interval must not inherit the benign label at flow start."""
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    features = pd.DataFrame(
        {
            "flow_id": ["tcp|10.0.0.1:50000|10.0.0.2:80"] * 4,
            "timestamp": [
                timestamp,
                timestamp + pd.Timedelta(seconds=2),
                timestamp + pd.Timedelta(seconds=4),
                timestamp + pd.Timedelta(seconds=6),
            ],
            "src_ip": ["10.0.0.1"] * 4,
            "src_port": [50000] * 4,
            "dst_ip": ["10.0.0.2"] * 4,
            "dst_port": [80] * 4,
            "protocol": ["http"] * 4,
        }
    )
    labels = pd.DataFrame(
        {
            "csv_row": [1, 2],
            "label": ["benign", "attack"],
            "label_timestamp": [timestamp, timestamp + pd.Timedelta(seconds=3)],
            "label_end_timestamp": [
                timestamp + pd.Timedelta(seconds=2.9),
                timestamp + pd.Timedelta(seconds=10),
            ],
            "src_ip": ["10.0.0.1"] * 2,
            "src_port": [50000] * 2,
            "dst_ip": ["10.0.0.2"] * 2,
            "dst_port": [80] * 2,
            "source_file": ["labels.csv"] * 2,
        }
    )
    config = {
        "mapping": {
            "timestamp_tolerance_seconds": 5.0,
            "allow_reverse_flow_match": True,
            "default_label": "unknown",
        }
    }

    result = attach_packet_labels(features, labels, config)

    assert result["label"].tolist() == ["benign", "benign", "attack", "attack"]
    assert result["is_attack"].tolist() == [False, False, True, True]


def test_packet_csv_cache_reuses_every_packet_link_and_invalidates_changed_csv(tmp_path) -> None:
    """A cache relation is reusable only for the same PCAP/CSV snapshot and packet identities."""
    capture = tmp_path / "attack.pcap"
    labels_csv = tmp_path / "labels.csv"
    capture.write_bytes(b"pcap-source")
    labels_csv.write_text("Label\nattack\n", encoding="utf-8")
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    features = pd.DataFrame(
        {
            "packet_uid": ["attack.pcap:1", "attack.pcap:2", "attack.pcap:3"],
            "capture": ["attack.pcap"] * 3,
            "packet_index": [1, 2, 3],
            "timestamp": [timestamp, timestamp + pd.Timedelta(seconds=1), timestamp + pd.Timedelta(seconds=2)],
            "flow_id": ["flow-1"] * 3,
            "packet_length": [60, 61, 62],
        }
    )
    labelled = features.assign(
        label=["benign", "attack", "attack"],
        is_attack=[False, True, True],
        match_status=["matched_interval"] * 3,
        match_confidence=[0.98] * 3,
        csv_row=[1.0, 2.0, 2.0],
        label_source_file=["labels.csv"] * 3,
        candidate_count=[1.0] * 3,
        mapping_accepted=True,
        dataset_split="attack",
    )
    feature_path = tmp_path / "features.parquet"
    labelled_path = tmp_path / "labelled.parquet"
    write_table(features, feature_path)
    write_table(labelled, labelled_path)
    config = {
        "project": {"artifact_dir": str(tmp_path / "artifacts")},
        "mapping": {
            "cache_enabled": True,
            "cache_directory": "mapping_cache",
            "cache_schema_version": "1.0.0",
            "timestamp_tolerance_seconds": 5.0,
        },
    }
    plan = mapping_cache_plan(
        config,
        protocol="http",
        capture_path=capture,
        label_paths=[labels_csv],
        max_packets=None,
    )
    save_mapping_cache(
        plan,
        flow_evidence=pd.DataFrame({"flow_id": ["flow-1"], "label": ["attack"]}),
        labelled_path=labelled_path,
        metadata={
            "selected_csv": str(labels_csv),
            "mapping_audit": {"match_rate": 1.0},
            "candidate_evaluations": [],
            "label_fallback": {},
            "accepted": True,
            "mapping_evidence_accepted": True,
            "rejection_reason": None,
            "packet_label_coverage": {"records": 3, "attack_records": 2},
        },
    )
    cached = load_mapping_cache(plan)

    assert cached is not None
    reapplied_path = tmp_path / "reapplied.parquet"
    result = apply_cached_packet_labels_parquet(
        feature_path, cached["links_path"], reapplied_path, batch_rows=2
    )
    reapplied = pd.read_parquet(reapplied_path)
    assert result["rows"] == 3
    assert result["attack_records"] == 2
    assert reapplied["packet_uid"].tolist() == features["packet_uid"].tolist()
    assert reapplied["csv_row"].tolist() == [1.0, 2.0, 2.0]

    labels_csv.write_text("Label\nattack\nbenign\n", encoding="utf-8")
    changed_plan = mapping_cache_plan(
        config,
        protocol="http",
        capture_path=capture,
        label_paths=[labels_csv],
        max_packets=None,
    )
    assert changed_plan["signature"] != plan["signature"]
    assert load_mapping_cache(changed_plan) is None
