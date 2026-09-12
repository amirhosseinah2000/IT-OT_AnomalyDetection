"""End-to-end contract for the reusable, exhaustive packet-to-CSV mapping cache."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
from scapy.all import Ether, IP, TCP, Raw, wrpcap

from anomdet.core.config import load_config
from anomdet.orchestration.dataset_pipeline import (
    run_dataset_extraction,
    run_dataset_mapping,
    run_dataset_pipeline,
)
from anomdet.mapping.mapper import mapping_cache_plan


def _http_packet(timestamp: float, payload: bytes) -> Ether:
    packet = (
        Ether()
        / IP(src="198.51.100.10", dst="198.51.100.20")
        / TCP(sport=51000, dport=80)
        / Raw(payload)
    )
    packet.time = timestamp
    return packet


def test_dataset_run_reuses_complete_packet_csv_relation(tmp_path: Path) -> None:
    """The second identical run applies cached links rather than invoking CSV matching."""
    root = tmp_path / "raw"
    benign = root / "benign" / "pcap" / "http" / "benign.pcap"
    attack = root / "attack" / "pcap" / "http" / "attack.pcap"
    labels = root / "attack" / "labels" / "http" / "attack.csv"
    benign.parent.mkdir(parents=True)
    attack.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    stamp = 1_735_689_600.0  # 2025-01-01T00:00:00Z
    wrpcap(str(benign), [_http_packet(stamp, b"GET / HTTP/1.1\r\nHost: benign.test\r\n\r\n")])
    wrpcap(
        str(attack),
        [
            _http_packet(stamp, b"GET /attack HTTP/1.1\r\nHost: attack.test\r\n\r\n"),
            _http_packet(stamp + 1, b"payload"),
        ],
    )
    pd.DataFrame(
        {
            "Timestamp": ["2025-01-01T00:00:00Z"],
            "Src IP": ["198.51.100.10"],
            "Dst IP": ["198.51.100.20"],
            "Src Port": [51000],
            "Dst Port": [80],
            "Label": ["attack"],
        }
    ).to_csv(labels, index=False)

    config = copy.deepcopy(load_config())
    config["project"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["data"].update(
        {
            "dataset_root": str(root),
            "benign_pcap_dir": "benign/pcap",
            "attack_pcap_dir": "attack/pcap",
            "attack_label_dir": "attack/labels",
        }
    )
    config["capture"].update(
        {
            "supported_protocols": ["http"],
            "dataset_default_packet_cap": 2,
            "streaming_threshold_packets": 1,
            "streaming_chunk_rows": 1_000,
            "streaming_return_sample_rows": 1_000,
        }
    )
    config["mapping"].update(
        {"minimum_match_rate": 0.1, "minimum_match_confidence": 0.1}
    )

    first_root = tmp_path / "first"
    run_dataset_pipeline(config, output=first_root, max_packets=2)
    first_audit = json.loads(
        (first_root / "mappings" / "http" / "attack.audit.json").read_text(encoding="utf-8")
    )
    first_progress = json.loads((first_root / "dataset-progress.json").read_text(encoding="utf-8"))
    assert first_audit["mapping_cache"]["status"] == "created"
    assert first_audit["packet_label_coverage"]["records"] == 2
    assert {event["event"] for event in first_progress["events"]}.issuperset(
        {"mapping_cache_miss", "packet_mapping_progress"}
    )

    second_root = tmp_path / "second"
    run_dataset_pipeline(config, output=second_root, max_packets=2)
    second_audit = json.loads(
        (second_root / "mappings" / "http" / "attack.audit.json").read_text(encoding="utf-8")
    )
    second_progress = json.loads((second_root / "dataset-progress.json").read_text(encoding="utf-8"))
    labelled = pd.read_parquet(second_root / "labelled" / "attack" / "http" / "attack" / "records.parquet")

    assert second_audit["mapping_cache"]["status"] == "reused"
    assert labelled["packet_uid"].tolist() == ["attack.pcap:1", "attack.pcap:2"]
    assert labelled["csv_row"].notna().all()
    assert labelled["is_attack"].tolist() == [True, True]
    assert {event["event"] for event in second_progress["events"]}.issuperset(
        {"mapping_cache_hit", "mapping_cache_apply_progress"}
    )


def test_saved_feature_mapping_is_an_explicit_reusable_stage(tmp_path: Path) -> None:
    """`dataset map` must label saved features without running feature extraction again."""
    root = tmp_path / "raw"
    benign = root / "benign" / "pcap" / "http" / "benign.pcap"
    attack = root / "attack" / "pcap" / "http" / "attack.pcap"
    labels = root / "attack" / "labels" / "http" / "attack.csv"
    benign.parent.mkdir(parents=True)
    attack.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    stamp = 1_735_689_600.0
    wrpcap(str(benign), [_http_packet(stamp, b"GET / HTTP/1.1\r\nHost: benign.test\r\n\r\n")])
    wrpcap(
        str(attack),
        [
            _http_packet(stamp, b"GET /attack HTTP/1.1\r\nHost: attack.test\r\n\r\n"),
            _http_packet(stamp + 1, b"payload"),
        ],
    )
    pd.DataFrame(
        {
            "Timestamp": ["2025-01-01T00:00:00Z"],
            "Src IP": ["198.51.100.10"],
            "Dst IP": ["198.51.100.20"],
            "Src Port": [51000],
            "Dst Port": [80],
            "Label": ["attack"],
        }
    ).to_csv(labels, index=False)

    config = copy.deepcopy(load_config())
    config["project"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["data"].update(
        {
            "dataset_root": str(root),
            "benign_pcap_dir": "benign/pcap",
            "attack_pcap_dir": "attack/pcap",
            "attack_label_dir": "attack/labels",
        }
    )
    config["capture"].update(
        {
            "supported_protocols": ["http"],
            "dataset_default_packet_cap": 2,
            "streaming_threshold_packets": 1,
            "streaming_chunk_rows": 1_000,
            "streaming_return_sample_rows": 1_000,
        }
    )
    config["mapping"].update(
        {
            "cache_directory": "separate-stage-cache",
            "minimum_match_rate": 0.1,
            "minimum_match_confidence": 0.1,
        }
    )

    run_root = tmp_path / "extract-only"
    run_dataset_extraction(config, output=run_root, max_packets=2)
    feature_path = run_root / "features" / "attack" / "http" / "attack" / "records.parquet"
    labelled_path = run_root / "labelled" / "attack" / "http" / "attack" / "records.parquet"
    assert feature_path.is_file()
    assert feature_path.with_name("mapping-evidence.parquet").is_file()
    assert not labelled_path.exists()
    assert json.loads((run_root / "run-summary.json").read_text(encoding="utf-8"))["mapping_status"] == (
        "not_started"
    )

    first_summary, _ = run_dataset_mapping(config, run_root)
    assert first_summary["mapping_cache_created"] == 1
    assert labelled_path.is_file()
    assert pd.read_parquet(labelled_path)["is_attack"].tolist() == [True, True]

    second_summary, _ = run_dataset_mapping(config, run_root)
    assert second_summary["mapping_cache_reused"] == 1


def test_mapping_cache_packet_identity_is_order_independent_and_rebuilds_on_mismatch(
    tmp_path: Path,
) -> None:
    """Packet order is irrelevant; an invalid identity is a safe cache miss."""
    root = tmp_path / "raw"
    benign = root / "benign" / "pcap" / "http" / "benign.pcap"
    attack = root / "attack" / "pcap" / "http" / "attack.pcap"
    labels = root / "attack" / "labels" / "http" / "attack.csv"
    benign.parent.mkdir(parents=True)
    attack.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    stamp = 1_735_689_600.0
    wrpcap(str(benign), [_http_packet(stamp, b"GET / HTTP/1.1\r\nHost: benign.test\r\n\r\n")])
    wrpcap(
        str(attack),
        [
            _http_packet(stamp, b"GET /first HTTP/1.1\r\nHost: attack.test\r\n\r\n"),
            _http_packet(stamp + 1, b"GET /second HTTP/1.1\r\nHost: attack.test\r\n\r\n"),
        ],
    )
    pd.DataFrame(
        {
            "Timestamp": ["2025-01-01T00:00:00Z"],
            "Src IP": ["198.51.100.10"],
            "Dst IP": ["198.51.100.20"],
            "Src Port": [51000],
            "Dst Port": [80],
            "Label": ["attack"],
        }
    ).to_csv(labels, index=False)

    config = copy.deepcopy(load_config())
    config["project"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["data"].update(
        {
            "dataset_root": str(root),
            "benign_pcap_dir": "benign/pcap",
            "attack_pcap_dir": "attack/pcap",
            "attack_label_dir": "attack/labels",
        }
    )
    config["capture"].update(
        {
            "supported_protocols": ["http"],
            "dataset_default_packet_cap": 2,
            "streaming_threshold_packets": 1,
            "streaming_chunk_rows": 1_000,
            "streaming_return_sample_rows": 1_000,
        }
    )
    config["mapping"].update({"minimum_match_rate": 0.1, "minimum_match_confidence": 0.1})

    run_root = tmp_path / "run"
    run_dataset_extraction(config, output=run_root, max_packets=2)
    run_dataset_mapping(config, run_root)
    cache_plan = mapping_cache_plan(
        config,
        protocol="http",
        capture_path=attack,
        label_paths=[labels],
        max_packets=2,
    )
    stale_links = pd.read_parquet(cache_plan["links_path"]).iloc[::-1].reset_index(drop=True)
    stale_links.to_parquet(cache_plan["links_path"], index=False)

    reused_summary, _ = run_dataset_mapping(config, run_root)
    reused_audit = json.loads(
        (run_root / "mappings" / "http" / "attack.audit.json").read_text(encoding="utf-8")
    )
    assert reused_summary["mapping_cache_reused"] == 1
    assert reused_audit["mapping_cache"]["status"] == "reused"

    stale_links.loc[0, "packet_uid"] = "wrong-capture.pcap:999"
    stale_links.to_parquet(cache_plan["links_path"], index=False)
    summary, _ = run_dataset_mapping(config, run_root)
    audit = json.loads(
        (run_root / "mappings" / "http" / "attack.audit.json").read_text(encoding="utf-8")
    )
    rebuilt_links = pd.read_parquet(cache_plan["links_path"])

    assert summary["mapping_cache_created"] == 1
    assert audit["mapping_cache"]["status"] == "created"
    assert rebuilt_links["packet_uid"].tolist() == ["attack.pcap:1", "attack.pcap:2"]
