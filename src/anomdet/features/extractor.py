"""Protocol-aware, PCAP-first feature extraction for IT and OT traffic."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from scapy.all import DNS, IP, TCP, UDP, IPv6, PcapNgReader, PcapReader, Raw

from anomdet.core.io import utc_now, write_json
from anomdet.core.paths import resolve_capture_paths
from anomdet.features.catalog import feature_names
from anomdet.features.protocols.common import entropy as _entropy
from anomdet.features.protocols.dns import decode_fields as _decode_dns
from anomdet.features.protocols.http import decode_fields as _decode_http
from anomdet.features.protocols.modbus import decode_fields as _decode_modbus
from anomdet.features.protocols.s7comm import decode_fields as _decode_s7comm
from anomdet.features.protocols.ssh import decode_fields as _decode_ssh

LOGGER = logging.getLogger("anomdet")
HTTP_PORTS = {80, 8000, 8080, 8081, 8888}
SERVICE_PORTS = {22: "ssh", 53: "dns", 80: "http", 502: "modbus", 102: "s7comm"}
# These catalogue fields require decrypted/session-aware SSH parsing or a
# fuller S7 data-item decoder. They remain explicit in validation reports so
# a schema placeholder is never mistaken for extracted evidence.
NOT_YET_IMPLEMENTED_FEATURES = {
    "ssh_auth_method",
    "ssh_failed_auth_count",
    "ssh_open_channel_count",
    "ssh_keepalive_interval",
    "s7_return_code",
}
HTTP_CONTEXT_COLUMNS = [
    "http_method",
    "http_url_length",
    "http_url_entropy",
    "http_query_parameter_count",
    "http_suspicious_token_count",
    "http_host",
    "http_user_agent",
    "http_header_count",
    "http_content_length",
    "http_cookie_count",
    "http_status_code",
    "http_response_size",
    "http_error_ratio",
    "http_post_get_ratio",
    "http_request_repeat_count",
]
BASE_COLUMNS = [
    "capture",
    "timestamp",
    "protocol",
    "transport",
    "flow_id",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "direction",
    "is_request_direction",
    "packet_length",
    "payload_size",
    "payload_entropy",
    "tcp_flag_count",
]


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]
) -> None:
    """Publish best-effort extraction progress without allowing observers to stop extraction."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # pragma: no cover - an observer is never part of extraction correctness.
        LOGGER.exception("Feature-extraction progress observer failed")


def _payload(packet: Any) -> bytes:
    """Return a transport payload where Scapy retained a raw layer."""
    layer = packet.getlayer(Raw)
    return bytes(getattr(layer, "load", b"")) if layer is not None else b""


def _normalized_flow(
    src_ip: str, src_port: int, dst_ip: str, dst_port: int, transport: str
) -> tuple[str, int]:
    """Build a bidirectional 5-tuple key and a stable per-packet direction flag."""
    left, right = (src_ip, src_port), (dst_ip, dst_port)
    ordered = sorted((left, right), key=lambda item: (item[0], item[1]))
    direction = int(left == ordered[0])
    flow_id = f"{transport}|{ordered[0][0]}:{ordered[0][1]}|{ordered[1][0]}:{ordered[1][1]}"
    return flow_id, direction


def _infer_protocol(packet: Any, src_port: int, dst_port: int, raw: bytes) -> str | None:
    """Infer one of the supported protocols from parsed layers, ports, and signatures."""
    if packet.haslayer(DNS) or 53 in {src_port, dst_port}:
        return "dns"
    if 502 in {src_port, dst_port}:
        return "modbus"
    if 102 in {src_port, dst_port} and (b"\x32" in raw or len(raw) > 7):
        return "s7comm"
    if 22 in {src_port, dst_port} or raw.startswith(b"SSH-"):
        return "ssh"
    http_start = raw[:16].upper()
    if {src_port, dst_port}.intersection(HTTP_PORTS) or http_start.startswith(
        (b"GET ", b"POST ", b"PUT ", b"DELETE ", b"HEAD ", b"HTTP/")
    ):
        return "http"
    return None


def _packet_row(packet: Any, source_name: str) -> dict[str, Any] | None:
    """Convert one supported IP/TCP-or-UDP packet into an unaggregated feature row."""
    ip_layer = packet.getlayer(IP) or packet.getlayer(IPv6)
    transport_layer = packet.getlayer(TCP) or packet.getlayer(UDP)
    if ip_layer is None or transport_layer is None:
        return None
    src_ip, dst_ip = str(ip_layer.src), str(ip_layer.dst)
    src_port, dst_port = int(transport_layer.sport), int(transport_layer.dport)
    transport = "tcp" if packet.haslayer(TCP) else "udp"
    raw = _payload(packet)
    protocol = _infer_protocol(packet, src_port, dst_port, raw)
    if protocol is None:
        return None
    flow_id, direction = _normalized_flow(src_ip, src_port, dst_ip, dst_port, transport)
    flow_id = f"{source_name}::{flow_id}"
    timestamp = pd.Timestamp(float(packet.time), unit="s", tz="UTC")
    service_direction = int(SERVICE_PORTS.get(dst_port) == protocol or dst_port in HTTP_PORTS)
    row: dict[str, Any] = {
        "capture": source_name,
        "timestamp": timestamp,
        "protocol": protocol,
        "transport": transport,
        "flow_id": flow_id,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "direction": direction,
        "is_request_direction": service_direction,
        "packet_length": len(packet),
        "payload_size": len(raw),
        "payload_entropy": _entropy(raw),
        "tcp_flag_count": int(bin(int(transport_layer.flags)).count("1"))
        if transport == "tcp"
        else 0,
    }
    if protocol == "dns":
        row.update(_decode_dns(packet, raw))
    elif protocol == "http":
        row.update(_decode_http(raw))
    elif protocol == "ssh":
        row.update(_decode_ssh(raw, dst_port))
    elif protocol == "modbus":
        row.update(_decode_modbus(raw))
    elif protocol == "s7comm":
        row.update(_decode_s7comm(raw))
    return row


def _reader(path: Path) -> Iterator[Any]:
    """Open PCAP or PCAPNG captures through the matching streaming Scapy reader."""
    return PcapNgReader(str(path)) if path.suffix.lower() == ".pcapng" else PcapReader(str(path))


def _augment_features(frame: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Add flow, timing, request-response, and host-behaviour features to packet rows."""
    if frame.empty:
        return frame
    result = frame.sort_values(["capture", "flow_id", "timestamp"], kind="stable").copy()
    flow_keys = ["capture", "flow_id"]
    source_keys = ["capture", "src_ip"]
    flow = result.groupby(flow_keys, sort=False)
    first_timestamp = flow["timestamp"].transform("min")
    result["flow_duration"] = (
        (result["timestamp"] - first_timestamp).dt.total_seconds().clip(lower=0)
    )
    result["flow_total_packets"] = flow["packet_length"].transform("size")
    result["flow_total_bytes"] = flow["packet_length"].transform("sum")
    result["packet_length_mean"] = flow["packet_length"].transform("mean")
    result["packet_length_std"] = flow["packet_length"].transform("std").fillna(0.0)
    result["inter_arrival_time"] = (
        flow["timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0)
    )
    result["jitter"] = flow["inter_arrival_time"].diff().abs().fillna(0.0)
    result["packet_rate"] = (flow.cumcount() + 1) / result["flow_duration"].clip(lower=1.0)

    same_direction_bytes = result.groupby([*flow_keys, "direction"], sort=False)[
        "packet_length"
    ].transform("sum")
    reverse_direction_bytes = (result["flow_total_bytes"] - same_direction_bytes).clip(lower=1.0)
    result["flow_byte_ratio"] = same_direction_bytes / reverse_direction_bytes

    result["hour_sin"] = np.sin(2 * np.pi * result["timestamp"].dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["timestamp"].dt.hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * result["timestamp"].dt.dayofweek / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result["timestamp"].dt.dayofweek / 7)

    source = result.groupby(source_keys, sort=False)
    source_elapsed = (
        (result["timestamp"] - source["timestamp"].transform("min"))
        .dt.total_seconds()
        .clip(lower=1.0)
    )
    result["source_packet_rate"] = (source.cumcount() + 1) / source_elapsed
    result["source_destination_count"] = source["dst_ip"].transform("nunique")
    result["destination_entropy"] = source["dst_ip"].transform(
        lambda column: _entropy("|".join(sorted(column.astype(str).unique())))
    )
    capture = result.groupby("capture", sort=False)
    capture_duration = (
        (capture["timestamp"].transform("max") - capture["timestamp"].transform("min"))
        .dt.total_seconds()
        .clip(lower=1.0)
    )
    capture_rate = capture["packet_length"].transform("size") / capture_duration
    result["burstiness"] = result["source_packet_rate"] / capture_rate.clip(lower=1e-6)

    result = _add_protocol_behaviour(result)
    result = _propagate_protocol_context(result)
    result["behavior_window_seconds"] = window_seconds
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _add_protocol_behaviour(frame: pd.DataFrame) -> pd.DataFrame:
    """Add protocol-specific behavioural counts after all packet rows are available."""
    result = frame.copy()
    dns = result["protocol"].eq("dns")
    if dns.any():
        result.loc[dns, "dns_nxdomain_rate"] = (
            result.loc[dns]
            .groupby(["capture", "src_ip"])["dns_rcode"]
            .transform(lambda values: (values.fillna(0) == 3).mean())
        )
        result.loc[dns, "dns_query_repeat_count"] = (
            result.loc[dns].groupby(["capture", "src_ip", "dns_qname"])["flow_id"].transform("size")
        )

    http = result["protocol"].eq("http")
    if http.any():
        http_rows = result.loc[http]
        result.loc[http, "http_error_ratio"] = http_rows.groupby(["capture", "src_ip"])[
            "http_status_code"
        ].transform(
            lambda values: (
                ((values.dropna() >= 400) & (values.dropna() < 600)).mean()
                if values.notna().any()
                else np.nan
            )
        )
        methods = http_rows["http_method"].fillna("")
        counts = pd.crosstab([http_rows["capture"], http_rows["src_ip"]], methods)
        post_counts = counts["POST"] if "POST" in counts else pd.Series(0, index=counts.index)
        get_counts = counts["GET"] if "GET" in counts else pd.Series(0, index=counts.index)
        ratio = (post_counts / get_counts.clip(lower=1)).to_dict()
        result.loc[http, "http_post_get_ratio"] = [
            ratio.get((capture, source), 0.0)
            for capture, source in zip(http_rows["capture"], http_rows["src_ip"], strict=True)
        ]
        result.loc[http, "http_request_repeat_count"] = http_rows.groupby(
            ["capture", "src_ip", "http_url"]
        )["flow_id"].transform("size")

    modbus = result["protocol"].eq("modbus")
    if modbus.any():
        modbus_rows = result.loc[modbus]
        result.loc[modbus, "modbus_exception_rate"] = modbus_rows.groupby(["capture", "flow_id"])[
            "modbus_is_exception"
        ].transform("mean")
        medians = modbus_rows.groupby(["capture", "flow_id"])["modbus_starting_address"].transform(
            "median"
        )
        result.loc[modbus, "modbus_address_deviation"] = (
            modbus_rows["modbus_starting_address"] - medians
        ).abs()
        repeated = modbus_rows.groupby(["capture", "flow_id", "modbus_transaction_id"])[
            "direction"
        ].transform("nunique")
        result.loc[modbus, "modbus_response_matched"] = (repeated > 1).astype(int)

    s7 = result["protocol"].eq("s7comm")
    if s7.any():
        s7_rows = result.loc[s7]
        repeated = s7_rows.groupby(["capture", "flow_id", "s7_pdu_reference"])[
            "direction"
        ].transform("nunique")
        result.loc[s7, "s7_response_matched"] = (repeated > 1).astype(int)
    return result


def _propagate_protocol_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Carry parsed application metadata across packets in its TCP direction.

    HTTP headers are usually observable in only one or two packets while a
    TCP flow can contain thousands. Once a header has been parsed, its stable
    request/response context is valid for the neighbouring payload packets in
    that same capture, flow, and direction. This avoids imputing a parser
    value away solely because it is packet-sparse.
    """
    result = frame.copy()
    http = result["protocol"].eq("http")
    if not http.any():
        return result
    scoped = result.loc[http]
    keys = ["capture", "flow_id", "direction"]
    for column in [name for name in HTTP_CONTEXT_COLUMNS if name in result.columns]:
        result.loc[http, column] = scoped.groupby(keys, sort=False)[column].transform(
            lambda values: values.ffill().bfill()
        )
    return result


def _capture_rows(
    capture_path: Path,
    max_packets: int | None,
    expected_protocol: str | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one capture and retain only its expected protocol when requested."""
    LOGGER.info("Reading capture %s", capture_path)
    started = time.perf_counter()
    _emit_progress(
        progress_callback,
        {
            "event": "capture_started",
            "capture": str(capture_path),
            "capture_name": capture_path.name,
            "packet_limit": max_packets,
            "expected_protocol": expected_protocol,
        },
    )
    rows: list[dict[str, Any]] = []
    observed_protocols: Counter[str] = Counter()
    filtered_protocols: Counter[str] = Counter()
    packet_count = 0
    report_every = max(1, max_packets // 10) if max_packets else 5_000
    next_report = report_every
    reader = _reader(capture_path)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn()
    )
    with progress:
        task = progress.add_task(f"Extracting {capture_path.name}", total=max_packets)
        try:
            for packet_index, packet in enumerate(reader, start=1):
                if max_packets is not None and packet_index > max_packets:
                    break
                packet_count += 1
                row = _packet_row(packet, capture_path.name)
                if row is not None:
                    protocol = str(row["protocol"])
                    observed_protocols[protocol] += 1
                    if expected_protocol is None or protocol == expected_protocol:
                        rows.append(row)
                    else:
                        filtered_protocols[protocol] += 1
                if packet_count >= next_report:
                    elapsed = time.perf_counter() - started
                    LOGGER.info(
                        "EXTRACT_PROGRESS capture=%s packets=%s%s retained=%s elapsed=%.1fs",
                        capture_path.name,
                        packet_count,
                        f"/{max_packets}" if max_packets else "",
                        len(rows),
                        elapsed,
                    )
                    _emit_progress(
                        progress_callback,
                        {
                            "event": "capture_progress",
                            "capture": str(capture_path),
                            "capture_name": capture_path.name,
                            "packets_read": packet_count,
                            "packet_limit": max_packets,
                            "retained_rows": len(rows),
                            "protocol_counts": dict(observed_protocols),
                            "elapsed_seconds": round(elapsed, 2),
                            "progress_ratio": (
                                min(packet_count / max_packets, 1.0) if max_packets else None
                            ),
                        },
                    )
                    next_report += report_every
                progress.advance(task)
        finally:
            reader.close()
    summary = {
        "capture": str(capture_path),
        "packets_read": packet_count,
        "supported_protocol_counts": dict(observed_protocols),
        "retained_rows": len(rows),
        "filtered_protocol_counts": dict(filtered_protocols),
    }
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "EXTRACT_COMPLETE capture=%s packets=%s retained=%s protocols=%s elapsed=%.1fs",
        capture_path.name,
        packet_count,
        len(rows),
        dict(observed_protocols),
        elapsed,
    )
    _emit_progress(
        progress_callback,
        {
            "event": "capture_completed",
            "capture": str(capture_path),
            "capture_name": capture_path.name,
            "packets_read": packet_count,
            "packet_limit": max_packets,
            "retained_rows": len(rows),
            "protocol_counts": dict(observed_protocols),
            "elapsed_seconds": round(elapsed, 2),
            "progress_ratio": 1.0,
        },
    )
    return rows, summary


def _stable_feature_schema(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep metadata and catalogue fields visible even when a parser observed none."""
    frame = pd.DataFrame(rows)
    for column in [*BASE_COLUMNS, *sorted(feature_names())]:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def extract_pcap_features(
    capture_path: Path,
    output_path: Path,
    config: dict[str, Any],
    max_packets: int | None = None,
    expected_protocol: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract one capture or every direct PCAP in a protocol folder into one feature table."""
    captures = resolve_capture_paths(capture_path)
    supported = set(config["capture"]["supported_protocols"])
    if expected_protocol is not None and expected_protocol not in supported:
        raise ValueError(
            f"Expected protocol '{expected_protocol}' is not configured as supported. "
            f"Choose one of: {', '.join(sorted(supported))}."
        )
    rows: list[dict[str, Any]] = []
    capture_summaries: list[dict[str, Any]] = []
    LOGGER.info(
        "EXTRACT_START source=%s captures=%s protocol=%s packet_limit=%s output=%s",
        capture_path,
        len(captures),
        expected_protocol or "auto",
        max_packets if max_packets is not None else "unlimited",
        output_path,
    )
    for path in captures:
        capture_rows, summary = _capture_rows(
            path, max_packets, expected_protocol, progress_callback=progress_callback
        )
        rows.extend(capture_rows)
        capture_summaries.append(summary)

    _emit_progress(
        progress_callback,
        {
            "event": "feature_aggregation_started",
            "capture_count": len(captures),
            "packet_rows": len(rows),
            "expected_protocol": expected_protocol,
        },
    )
    packet_rows = _stable_feature_schema(rows)
    features = _augment_features(packet_rows, config["capture"]["behavior_window_seconds"])
    from anomdet.core.io import write_table  # Local import keeps simple CLI startup lightweight.

    write_table(features, output_path)
    manifest = {
        "created_at": utc_now(),
        "capture": str(capture_path),
        "captures": capture_summaries,
        "capture_count": len(captures),
        "expected_protocol": expected_protocol,
        "max_packets_per_capture": max_packets,
        "output": str(output_path),
        "rows": len(features),
        "flow_count": int(features["flow_id"].nunique(dropna=True)) if not features.empty else 0,
        "protocol_counts": features["protocol"].value_counts(dropna=False).to_dict()
        if not features.empty
        else {},
        "columns": features.columns.tolist(),
        "feature_schema_version": "1.1.0",
    }
    write_json(manifest, output_path.with_suffix(".manifest.json"))
    LOGGER.info(
        "Extracted %s feature rows from %s capture(s) to %s",
        f"{len(features):,}",
        len(captures),
        output_path,
    )
    _emit_progress(
        progress_callback,
        {
            "event": "feature_extraction_completed",
            "output": str(output_path),
            "rows": len(features),
            "flow_count": manifest["flow_count"],
            "protocol_counts": manifest["protocol_counts"],
            "progress_ratio": 1.0,
        },
    )
    return features, manifest
