"""Contracts for protocol-folder ingestion and extraction-validity evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scapy.all import IP, TCP, Ether, Raw, wrpcap

from anomdet.features.extractor import extract_pcap_features
from anomdet.orchestration.batch import _inventory
from anomdet.selection.profiles import feature_quality_report


def _config(raw_pcap_dir: Path) -> dict[str, object]:
    return {
        "project": {"artifact_dir": str(raw_pcap_dir / "artifacts")},
        "data": {"raw_pcap_dir": str(raw_pcap_dir), "datasets": [], "protocol_folders": []},
        "capture": {
            "supported_protocols": ["ssh", "dns", "http", "modbus", "s7comm"],
            "behavior_window_seconds": 60,
        },
    }


def _modbus_packet(transaction_id: int) -> Ether:
    """Create one small Modbus TCP request suitable for a deterministic parser test."""
    payload = transaction_id.to_bytes(2, "big") + b"\x00\x00\x00\x06\x01\x03\x00\x00\x00\x01"
    return (
        Ether()
        / IP(src="192.0.2.10", dst="192.0.2.20")
        / TCP(sport=50000, dport=502)
        / Raw(payload)
    )


def _http_packet(payload: bytes) -> Ether:
    """Create one TCP payload packet in a stable HTTP request direction."""
    return (
        Ether()
        / IP(src="198.51.100.10", dst="198.51.100.20")
        / TCP(sport=51000, dport=80)
        / Raw(payload)
    )


def test_protocol_folder_extracts_all_direct_pcaps(tmp_path) -> None:
    """One protocol folder combines each direct PCAP while retaining capture provenance."""
    folder = tmp_path / "pcap" / "modbus"
    folder.mkdir(parents=True)
    wrpcap(str(folder / "first.pcap"), [_modbus_packet(1)])
    wrpcap(str(folder / "second.pcap"), [_modbus_packet(2)])
    output = tmp_path / "features.parquet"

    features, manifest = extract_pcap_features(
        folder, output, _config(folder.parent), expected_protocol="modbus"
    )

    assert manifest["capture_count"] == 2
    assert manifest["protocol_counts"] == {"modbus": 2}
    assert features["capture"].nunique() == 2
    assert features["flow_id"].nunique() == 2
    assert features["modbus_transaction_id"].tolist() == [1, 2]


def test_http_context_is_available_to_packets_in_the_same_direction(tmp_path) -> None:
    """Header fields remain usable after the first HTTP payload packet in a flow."""
    capture = tmp_path / "http.pcap"
    wrpcap(
        str(capture),
        [
            _http_packet(b"GET /health HTTP/1.1\r\nHost: example.test\r\n\r\n"),
            _http_packet(b"body-payload"),
        ],
    )
    features, _manifest = extract_pcap_features(
        capture, tmp_path / "http-features.parquet", _config(tmp_path), expected_protocol="http"
    )

    assert features["http_method"].tolist() == ["GET", "GET"]
    assert features["http_host"].tolist() == ["example.test", "example.test"]


def test_protocol_folder_inventory_is_discovered_without_per_file_manifest(tmp_path) -> None:
    """The default inventory reads protocol-named folders without listing every capture."""
    folder = tmp_path / "pcap" / "modbus"
    folder.mkdir(parents=True)
    (folder / "sample.pcap").touch()

    inventory = _inventory(_config(folder.parent))

    assert inventory == [
        {
            "id": "modbus",
            "pcap": "modbus",
            "domain": "ot",
            "labels": [],
            "expected_protocol": "modbus",
            "capture_count": 1,
        }
    ]


def test_feature_quality_report_distinguishes_usable_constant_and_not_observed(tmp_path) -> None:
    """A schema field is not labelled model-usable until its protocol values vary."""
    source = pd.DataFrame(
        {
            "protocol": ["modbus"] * 20,
            "packet_length": list(range(60, 80)),
            "modbus_function_code": [3] * 20,
            "modbus_starting_address": [None] * 20,
        }
    )
    source_path = tmp_path / "features.parquet"
    report_path = tmp_path / "quality.parquet"
    source.to_parquet(source_path, index=False)

    report = feature_quality_report(source_path, report_path)
    statuses = report.set_index("name")["extraction_status"].to_dict()

    assert statuses["packet_length"] == "usable"
    assert statuses["modbus_function_code"] == "constant"
    assert statuses["modbus_starting_address"] == "not_observed"
