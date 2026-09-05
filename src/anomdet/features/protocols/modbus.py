"""Modbus-TCP-only PCAP feature extraction entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import extract_one_protocol

PROTOCOL = "modbus"
STANDARD_FUNCTIONS = set(range(1, 25)) | {43}


def extract_modbus_features(
    capture: Path, output: Path, config: dict[str, Any], max_packets: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return extract_one_protocol(capture, output, config, PROTOCOL, max_packets)


def _function_category(function_code: int) -> str:
    if function_code in {1, 2, 3, 4, 7, 11, 12, 17, 20, 24, 43}:
        return "read"
    if function_code in {5, 6, 15, 16, 21, 22, 23}:
        return "write"
    if function_code == 8:
        return "diagnostic"
    return "other"


def decode_fields(raw: bytes) -> dict[str, Any]:
    """Decode Modbus TCP MBAP/PDU fields from one packet payload."""
    if len(raw) < 8:
        return {}
    transaction_id = int.from_bytes(raw[0:2], "big")
    protocol_id = int.from_bytes(raw[2:4], "big")
    mbap_length = int.from_bytes(raw[4:6], "big")
    function_code = raw[7]
    is_exception = bool(function_code & 0x80)
    base_function = function_code & 0x7F
    values: dict[str, Any] = {
        "modbus_transaction_id": transaction_id,
        "modbus_unit_id": raw[6],
        "modbus_function_code": base_function,
        "modbus_function_category": _function_category(base_function),
        "modbus_protocol_id_valid": int(protocol_id == 0),
        "modbus_length_valid": int(mbap_length == len(raw) - 6),
        "modbus_nonstandard_function": int(base_function not in STANDARD_FUNCTIONS),
        "modbus_is_exception": int(is_exception),
    }
    if is_exception and len(raw) > 8:
        values["modbus_exception_code"] = raw[8]
    elif len(raw) >= 12 and base_function in {1, 2, 3, 4, 5, 6, 15, 16, 22, 23}:
        values["modbus_starting_address"] = int.from_bytes(raw[8:10], "big")
        values["modbus_quantity"] = int.from_bytes(raw[10:12], "big")
    if len(raw) > 8:
        values["modbus_byte_count"] = raw[8]
    if len(raw) >= 10 and base_function in {3, 4, 6, 16}:
        values["modbus_value"] = int.from_bytes(raw[-2:], "big")
    return values
