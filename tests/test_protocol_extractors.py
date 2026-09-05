"""Contract tests for the explicit per-protocol feature extractor entrypoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anomdet.features.protocols import EXTRACTORS, extractor_for


def test_every_protocol_has_a_separate_registered_extractor() -> None:
    """The public registry exposes one independent entrypoint per supported protocol."""
    assert set(EXTRACTORS) == {"dns", "http", "modbus", "s7comm", "ssh"}
    assert extractor_for("DNS") is EXTRACTORS["dns"]


def test_protocol_entrypoint_forwards_its_fixed_protocol(monkeypatch, tmp_path) -> None:
    """A DNS entrypoint cannot silently emit HTTP/OT rows from a mixed capture."""
    observed: dict[str, object] = {}

    def fake_extract(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return pd.DataFrame(), {"rows": 0, "flow_count": 0}

    monkeypatch.setattr("anomdet.features.extractor.extract_pcap_features", fake_extract)
    EXTRACTORS["dns"](Path("capture.pcap"), tmp_path / "dns.parquet", {}, 20)

    assert observed["kwargs"] == {"max_packets": 20, "expected_protocol": "dns"}
