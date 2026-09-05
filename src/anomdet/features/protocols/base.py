"""Shared mechanics for one-protocol extraction scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def extract_one_protocol(
    capture: Path,
    output: Path,
    config: dict[str, Any],
    protocol: str,
    max_packets: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract a single declared protocol; other packet protocols are excluded."""
    # Local import avoids a cycle: the shared PCAP engine imports the decoder
    # functions housed next to these one-protocol entrypoints.
    from anomdet.features.extractor import extract_pcap_features

    return extract_pcap_features(
        capture, output, config, max_packets=max_packets, expected_protocol=protocol
    )
