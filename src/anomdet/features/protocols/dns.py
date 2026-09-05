"""DNS-only PCAP feature extraction entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scapy.all import DNS

from .base import extract_one_protocol
from .common import entropy, safe_text

PROTOCOL = "dns"


def extract_dns_features(
    capture: Path, output: Path, config: dict[str, Any], max_packets: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return extract_one_protocol(capture, output, config, PROTOCOL, max_packets)


def decode_fields(packet: Any, raw: bytes) -> dict[str, Any]:
    """Decode DNS query, response, and header features from one packet."""
    dns = packet.getlayer(DNS)
    if dns is None:
        return {}
    values: dict[str, Any] = {
        "dns_rcode": int(getattr(dns, "rcode", 0)),
        "dns_authoritative": int(getattr(dns, "aa", 0)),
        "dns_recursion_available": int(getattr(dns, "ra", 0)),
        "dns_truncated": int(getattr(dns, "tc", 0)),
        "dns_answer_count": int(getattr(dns, "ancount", 0)),
    }
    question = getattr(dns, "qd", None)
    qname_bytes = getattr(question, "qname", b"") if question else b""
    if qname_bytes:
        qname = safe_text(qname_bytes).rstrip(".")
        values.update(
            {
                "dns_qname": qname,
                "dns_qname_length": len(qname),
                "dns_qname_entropy": entropy(qname),
                "dns_label_count": qname.count(".") + 1,
                "dns_qtype": int(getattr(question, "qtype", 0)),
                "dns_digit_ratio": sum(char.isdigit() for char in qname) / max(len(qname), 1),
                "dns_hyphen_ratio": qname.count("-") / max(len(qname), 1),
                "dns_ngram_score": round(
                    sum(char.isalpha() for char in qname) / max(len(qname), 1), 4
                ),
            }
        )
    try:
        values["dns_ttl"] = int(dns.an.ttl)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    if int(getattr(dns, "qr", 0)):
        values["dns_response_size"] = len(raw)
    return values
