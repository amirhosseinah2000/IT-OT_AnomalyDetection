"""SSH-only PCAP feature extraction entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import extract_one_protocol
from .common import safe_text

PROTOCOL = "ssh"


def extract_ssh_features(
    capture: Path, output: Path, config: dict[str, Any], max_packets: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return extract_one_protocol(capture, output, config, PROTOCOL, max_packets)


def _read_namelist(raw: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(raw):
        return "", len(raw)
    length = int.from_bytes(raw[offset : offset + 4], "big")
    start, end = offset + 4, offset + 4 + length
    if end > len(raw):
        return "", len(raw)
    return safe_text(raw[start:end], 1024), end


def decode_fields(raw: bytes, dst_port: int) -> dict[str, Any]:
    """Decode visible SSH banner and KEXINIT fields before encryption begins."""
    values: dict[str, Any] = {}
    if raw.startswith(b"SSH-"):
        banner = safe_text(raw).splitlines()[0][:255]
        values["ssh_client_banner" if dst_port == 22 else "ssh_server_banner"] = banner
    kex_index = raw.find(b"\x14")
    if kex_index < 0 or kex_index + 17 >= len(raw):
        return values
    offset = kex_index + 17
    algorithms: list[str] = []
    for _ in range(10):
        item, offset = _read_namelist(raw, offset)
        algorithms.append(item)
        if offset >= len(raw):
            break
    if not algorithms:
        return values
    import hashlib

    joined = ";".join(algorithms)
    values.update(
        {
            "ssh_kex_algorithms": algorithms[0] if algorithms else None,
            "ssh_cipher_count": len(algorithms[1].split(","))
            if len(algorithms) > 1 and algorithms[1]
            else 0,
            "ssh_mac_count": len(algorithms[3].split(","))
            if len(algorithms) > 3 and algorithms[3]
            else 0,
            "ssh_compression_count": len(algorithms[5].split(","))
            if len(algorithms) > 5 and algorithms[5]
            else 0,
            "hassh": hashlib.md5(joined.encode("utf-8")).hexdigest(),  # noqa: S324
        }
    )
    return values
