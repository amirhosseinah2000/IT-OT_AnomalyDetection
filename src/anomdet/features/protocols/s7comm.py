"""S7comm-only PCAP feature extraction entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import extract_one_protocol

PROTOCOL = "s7comm"


def extract_s7comm_features(
    capture: Path, output: Path, config: dict[str, Any], max_packets: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return extract_one_protocol(capture, output, config, PROTOCOL, max_packets)


def decode_fields(raw: bytes) -> dict[str, Any]:
    """Decode visible S7comm TPKT/COTP header metadata from one payload."""
    marker = raw.find(b"\x32")
    if marker < 0 or marker + 10 > len(raw):
        return {"s7_is_plus": int(b"S7comm-Plus" in raw)}
    parameter_length = int.from_bytes(raw[marker + 4 : marker + 6], "big")
    data_length = int.from_bytes(raw[marker + 6 : marker + 8], "big")
    parameter_start = marker + 10
    parameter = raw[parameter_start : parameter_start + parameter_length]
    values: dict[str, Any] = {
        "s7_rosctr": raw[marker + 1],
        "s7_pdu_reference": int.from_bytes(raw[marker + 2 : marker + 4], "big"),
        "s7_parameter_length": parameter_length,
        "s7_data_length": data_length,
        "s7_is_plus": int(b"S7comm-Plus" in raw),
        "s7_error_class": raw[marker + 8],
        "s7_error_code": raw[marker + 9],
    }
    if parameter:
        values["s7_function_code"] = parameter[0]
        values["s7_item_count"] = parameter[1] if len(parameter) > 1 else 0
        values["s7_control_command"] = int(parameter[0] in {0x28, 0x29})
        values["s7_block_transfer"] = int(parameter[0] in {0x1A, 0x1B})
    if len(parameter) >= 10 and parameter[0] in {0x04, 0x05}:
        values["s7_transfer_size"] = parameter[4]
        values["s7_db_number"] = int.from_bytes(parameter[6:8], "big")
        values["s7_area_code"] = parameter[8]
    return values
