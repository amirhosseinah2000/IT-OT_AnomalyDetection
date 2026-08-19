"""Regression coverage for protocol-first dashboard artefact discovery."""

from __future__ import annotations

import json

import pandas as pd

from anomdet.dashboard.app import (
    _discover_runs,
    _load_dashboard_table,
    _protocol_display_columns,
    _protocol_feature_sources,
    _run_picker_label,
    _safe_numeric,
)


def test_dashboard_prefers_protocol_table_and_bounds_large_source_sample(tmp_path) -> None:
    """A dashboard run must expose a protocol table without loading a combined artefact."""
    run = tmp_path / "artifacts" / "runs" / "modbus-review"
    feature_dir = run / "features"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "modbus-observation.parquet"
    pd.DataFrame(
        {
            "protocol": ["modbus"] * 20,
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="s", tz="UTC"),
            "packet_length": range(20),
        }
    ).to_parquet(feature_path, index=False)
    feature_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "expected_protocol": "modbus",
                "capture_count": 2,
                "rows": 20,
                "captures": [{"capture": "first.pcap"}, {"capture": "second.pcap"}],
            }
        ),
        encoding="utf-8",
    )

    assert _discover_runs(tmp_path / "artifacts") == [run]
    assert _protocol_feature_sources(run) == {"modbus": feature_path}
    assert "2 PCAP" in _run_picker_label(run, tmp_path / "artifacts", is_latest=True)
    assert "20 records" in _run_picker_label(run, tmp_path / "artifacts", is_latest=True)

    sample = _load_dashboard_table(str(feature_path), feature_path.stat().st_mtime_ns, 6)

    assert len(sample) == 6
    assert sample.attrs["source_rows"] == 20
    assert sample.attrs["sampled_for_dashboard"] is True


def test_dashboard_hides_foreign_protocol_schema_features() -> None:
    """Modbus dashboards must not analyse empty DNS/HTTP/S7 schema columns."""
    frame = pd.DataFrame(
        {
            "protocol": ["modbus", "modbus"],
            "capture": ["modbus.pcap", "modbus.pcap"],
            "flow_id": ["flow-1", "flow-2"],
            "packet_length": [60, 72],
            "modbus_quantity": [1, 2],
            "modbus_response_matched": [1, 0],
            "dns_rcode": [0, 0],
            "http_status_code": [0, 0],
            "s7_error_code": [0, 0],
        }
    )

    assert _safe_numeric(frame, "modbus") == [
        "packet_length",
        "modbus_quantity",
        "modbus_response_matched",
    ]
    assert "dns_rcode" not in _protocol_display_columns(frame, "modbus")
    assert "http_status_code" not in _protocol_display_columns(frame, "modbus")
