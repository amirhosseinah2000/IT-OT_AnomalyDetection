"""HTTP-only PCAP feature extraction entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import extract_one_protocol
from .common import entropy, safe_text

PROTOCOL = "http"
SUSPICIOUS_TOKENS = ("../", "%2e%2e", "<script", "%3cscript", "union select", " or 1=1", ";--")


def extract_http_features(
    capture: Path, output: Path, config: dict[str, Any], max_packets: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return extract_one_protocol(capture, output, config, PROTOCOL, max_packets)


def _headers(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.split("\r\n")[1:]:
        if not line:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip().lower()] = value.strip()
    return result


def decode_fields(raw: bytes) -> dict[str, Any]:
    """Decode HTTP request/response fields without TCP reassembly assumptions."""
    text = safe_text(raw)
    first_line = text.split("\r\n", 1)[0]
    headers = _headers(text)
    values: dict[str, Any] = {
        "http_header_count": len(headers),
        "http_host": headers.get("host"),
        "http_user_agent": headers.get("user-agent"),
        "http_content_length": pd.to_numeric(headers.get("content-length"), errors="coerce"),
    }
    cookie = headers.get("cookie", "")
    values["http_cookie_count"] = len([part for part in cookie.split(";") if "=" in part])
    import re

    request_match = re.match(
        r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|CONNECT)\s+(\S+)", first_line
    )
    if request_match:
        method, target = request_match.groups()
        lower_target = target.lower()
        values.update(
            {
                "http_method": method,
                "http_url": target[:2048],
                "http_url_length": len(target),
                "http_url_entropy": entropy(target),
                "http_query_parameter_count": target.split("?", 1)[-1].count("=")
                if "?" in target
                else 0,
                "http_suspicious_token_count": sum(
                    token in lower_target for token in SUSPICIOUS_TOKENS
                ),
            }
        )
    else:
        response_match = re.match(r"^HTTP/\d(?:\.\d)?\s+(\d{3})", first_line)
        if response_match:
            values["http_status_code"] = int(response_match.group(1))
            values["http_response_size"] = len(raw)
    return values
