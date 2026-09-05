"""Protocol-specific feature-extraction entrypoints with a shared PCAP contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .dns import extract_dns_features
from .http import extract_http_features
from .modbus import extract_modbus_features
from .s7comm import extract_s7comm_features
from .ssh import extract_ssh_features

ProtocolExtractor = Callable[[Path, Path, dict[str, Any], int | None], tuple[Any, dict[str, Any]]]

EXTRACTORS: dict[str, ProtocolExtractor] = {
    "dns": extract_dns_features,
    "http": extract_http_features,
    "modbus": extract_modbus_features,
    "s7comm": extract_s7comm_features,
    "ssh": extract_ssh_features,
}


def extractor_for(protocol: str) -> ProtocolExtractor:
    """Resolve one explicit protocol extractor instead of mixing protocol output."""
    try:
        return EXTRACTORS[protocol.casefold()]
    except KeyError as error:
        supported = ", ".join(sorted(EXTRACTORS))
        raise ValueError(
            f"Unsupported protocol '{protocol}'. Choose one of: {supported}."
        ) from error


__all__ = ["EXTRACTORS", "extractor_for"]
