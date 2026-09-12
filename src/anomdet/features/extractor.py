"""Protocol-aware, PCAP-first feature extraction for IT and OT traffic."""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from scapy.all import (
    DNS,
    ICMP,
    IP,
    TCP,
    UDP,
    IPv6,
    PcapNgReader,
    PcapReader,
    Raw,
    RawPcapNgReader,
    RawPcapReader,
    conf,
)

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
    "packet_uid",
    "packet_index",
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

# Streaming extraction has to use one stable schema for every Parquet row group.
# Most catalogue fields are numeric; these are the small set whose raw protocol
# representation is categorical text.  ``http_url`` is parser context rather
# than a selectable feature, but is retained for the HTTP repeat-count feature
# and downstream audits.
_STRING_COLUMNS = {
    "capture",
    "packet_uid",
    "protocol",
    "detected_protocol",
    "transport",
    "flow_id",
    "src_ip",
    "dst_ip",
    "ssh_client_banner",
    "ssh_server_banner",
    "ssh_kex_algorithms",
    "hassh",
    "ssh_auth_method",
    "dns_qname",
    "http_method",
    "http_url",
    "http_host",
    "http_user_agent",
    "modbus_function_category",
}
_STREAM_EXTRA_COLUMNS = ["detected_protocol", "http_url", "behavior_window_seconds"]


@dataclass
class _CausalFeatureState:
    """Bounded packet-history state used by the disk-backed extractor.

    Batch aggregation used to calculate several values with packets that occur
    *after* the current record.  Besides requiring a full capture in RAM, that
    leaks future context into model training.  The streaming path deliberately
    uses causal, inference-compatible values instead.  Old flow/source entries
    are evicted once their bounded state reaches the configured ceiling.
    """

    max_active_flows: int
    max_active_sources: int
    max_destinations_per_source: int
    flows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    sources: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    captures: dict[str, dict[str, Any]] = field(default_factory=dict)
    dns_sources: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    http_sources: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    http_context: dict[tuple[str, str, float], dict[str, Any]] = field(default_factory=dict)
    modbus_flows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    s7_flows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def _bounded(mapping: dict[Any, Any], key: Any, limit: int) -> dict[str, Any]:
        current = mapping.get(key)
        if current is not None:
            return current
        if len(mapping) >= max(limit, 1):
            # dict preserves insertion order; evicting the oldest state keeps
            # memory bounded even for a hostile high-cardinality capture.
            mapping.pop(next(iter(mapping)))
        current = {}
        mapping[key] = current
        return current

    def flow(self, key: tuple[str, str]) -> dict[str, Any]:
        return self._bounded(self.flows, key, self.max_active_flows)

    def source(self, key: tuple[str, str]) -> dict[str, Any]:
        return self._bounded(self.sources, key, self.max_active_sources)

    def protocol_flow(self, mapping: dict[Any, Any], key: tuple[str, str]) -> dict[str, Any]:
        return self._bounded(mapping, key, self.max_active_flows)


def _canonical_stream_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every streaming batch to one Arrow-compatible feature schema."""

    columns = list(dict.fromkeys([*BASE_COLUMNS, *_STREAM_EXTRA_COLUMNS, *sorted(feature_names())]))
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA if column in _STRING_COLUMNS else np.nan
    result = result[columns]
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    for column in _STRING_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string")
    for column in columns:
        if column not in _STRING_COLUMNS and column != "timestamp":
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    return result


def _entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=float) / float(total)
    return float(-(probabilities * np.log2(probabilities)).sum())


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


def _packet_row(
    packet: Any,
    source_name: str,
    protocol_scope: str | None = None,
    *,
    packet_index: int | None = None,
) -> dict[str, Any] | None:
    """Convert one IP/TCP-or-UDP packet into a scoped feature row.

    A protocol folder is the dataset owner's declaration of the scenario.  An
    OT capture can therefore include a TCP flood or scan that does not carry a
    Modbus payload, but must still be retained and labelled as part of the
    Modbus scenario. ``detected_protocol`` preserves packet-level inference;
    ``protocol`` is the immutable dataset scope used by feature profiles.
    """
    ip_layer = packet.getlayer(IP) or packet.getlayer(IPv6)
    transport_layer = packet.getlayer(TCP) or packet.getlayer(UDP)
    is_icmp = packet.haslayer(ICMP)
    if ip_layer is None or (transport_layer is None and not is_icmp):
        return None
    src_ip, dst_ip = str(ip_layer.src), str(ip_layer.dst)
    src_port = int(transport_layer.sport) if transport_layer is not None else 0
    dst_port = int(transport_layer.dport) if transport_layer is not None else 0
    transport = "tcp" if packet.haslayer(TCP) else "udp" if packet.haslayer(UDP) else "icmp"
    raw = _payload(packet)
    detected_protocol = _infer_protocol(packet, src_port, dst_port, raw)
    protocol = protocol_scope or detected_protocol
    if protocol is None:
        return None
    flow_id, direction = _normalized_flow(src_ip, src_port, dst_ip, dst_port, transport)
    flow_id = f"{source_name}::{flow_id}"
    timestamp = pd.Timestamp(float(packet.time), unit="s", tz="UTC")
    service_direction = int(SERVICE_PORTS.get(dst_port) == protocol or dst_port in HTTP_PORTS)
    row: dict[str, Any] = {
        "capture": source_name,
        # This identity is intentionally independent of generated feature row
        # order. It lets a saved PCAP↔CSV mapping be re-applied to a later
        # extraction of the same source, and lets external systems point to
        # one original capture packet unambiguously.
        "packet_uid": f"{source_name}:{packet_index}" if packet_index is not None else pd.NA,
        "packet_index": packet_index if packet_index is not None else np.nan,
        "timestamp": timestamp,
        "protocol": protocol,
        "detected_protocol": detected_protocol or "other",
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
        if transport == "tcp" and transport_layer is not None
        else 0,
    }
    # Decode application fields only when that protocol was detected in the
    # packet itself. A folder-scoped TCP flood must not be mis-parsed as a
    # synthetic Modbus request merely because it belongs to a Modbus scenario.
    if detected_protocol == "dns":
        row.update(_decode_dns(packet, raw))
    elif detected_protocol == "http":
        row.update(_decode_http(raw))
    elif detected_protocol == "ssh":
        row.update(_decode_ssh(raw, dst_port))
    elif detected_protocol == "modbus":
        row.update(_decode_modbus(raw))
    elif detected_protocol == "s7comm":
        row.update(_decode_s7comm(raw))
    return row


def _reader(path: Path) -> Iterator[Any]:
    """Open PCAP or PCAPNG captures through the matching streaming Scapy reader."""
    return PcapNgReader(str(path)) if path.suffix.lower() == ".pcapng" else PcapReader(str(path))


def _raw_reader(path: Path) -> Iterator[Any]:
    """Open a low-overhead reader used only by bounded smoke-test sampling."""
    return (
        RawPcapNgReader(str(path)) if path.suffix.lower() == ".pcapng" else RawPcapReader(str(path))
    )


def _raw_timestamp(metadata: Any) -> float:
    """Recover a Unix timestamp from Scapy PCAP or PCAPNG raw metadata."""
    if hasattr(metadata, "sec"):
        return float(metadata.sec) + float(getattr(metadata, "usec", 0)) / 1_000_000
    ticks = (int(getattr(metadata, "tshigh", 0)) << 32) + int(getattr(metadata, "tslow", 0))
    return ticks / max(float(getattr(metadata, "tsresol", 1_000_000)), 1.0)


def _reservoir_sample_packets(
    capture_path: Path,
    max_packets: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[tuple[int, Any]], int]:
    """Read a deterministic uniform packet reservoir without full dissection.

    ``--max-packets`` is explicitly a smoke-test path. Reservoir sampling
    visits the raw bytes only once, then dissects at most the requested number
    of packets. This keeps a 200-packet validation practical even for a
    multi-gigabyte PCAPNG and avoids the former prefix-only bias.
    """
    seed_bytes = hashlib.blake2b(str(capture_path).encode("utf-8"), digest_size=8).digest()
    generator = random.Random(int.from_bytes(seed_bytes, "big"))
    reservoir: list[tuple[int, bytes, Any, int]] = []
    reader = _raw_reader(capture_path)
    packet_count = 0
    report_every = 100_000
    try:
        for packet_count, (raw, metadata) in enumerate(reader, start=1):
            linktype = int(getattr(metadata, "linktype", getattr(reader, "linktype", 1)) or 1)
            item = (packet_count, bytes(raw), metadata, linktype)
            if len(reservoir) < max_packets:
                reservoir.append(item)
            else:
                replacement = generator.randrange(packet_count)
                if replacement < max_packets:
                    reservoir[replacement] = item
            if packet_count % report_every == 0:
                LOGGER.info(
                    "SAMPLING_PROGRESS capture=%s source_packets=%s selected=%s",
                    capture_path.name,
                    packet_count,
                    len(reservoir),
                )
                _emit_progress(
                    progress_callback,
                    {
                        "event": "capture_sampling_progress",
                        "capture": str(capture_path),
                        "capture_name": capture_path.name,
                        "packets_read": packet_count,
                        "packets_selected": len(reservoir),
                        "packet_limit": max_packets,
                    },
                )
    finally:
        reader.close()

    packets: list[tuple[int, Any]] = []
    for packet_index, raw, metadata, linktype in sorted(reservoir, key=lambda item: item[0]):
        decoder = conf.l2types.get(linktype, Raw)
        try:
            packet = decoder(raw)
        except Exception:  # pragma: no cover - malformed raw frames are skipped later.
            packet = Raw(raw)
        packet.time = _raw_timestamp(metadata)
        packets.append((packet_index, packet))
    return packets, packet_count


def _uniform_packet_indices(capture_path: Path, max_packets: int) -> tuple[set[int] | None, int]:
    """Return a bounded, time-spread sample instead of only a capture prefix.

    The first packets of an attack capture are often a benign setup phase.
    Counting once and extracting a uniform second-pass sample keeps trial runs
    small without silently discarding a later labelled attack interval.
    """
    reader = _reader(capture_path)
    try:
        total = sum(1 for _ in reader)
    finally:
        reader.close()
    if total <= max_packets:
        return None, total
    return set(np.linspace(1, total, num=max_packets, dtype=int).tolist()), total


def _large_uniform_packet_stream(
    capture_path: Path,
    max_packets: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Iterator[tuple[int, Any]], int]:
    """Return a disk-streamed uniform sample without retaining raw packets.

    A reservoir is excellent for a small smoke run, but retaining three million
    raw frames is already enough to exhaust an ordinary server.  This two-pass
    path first counts cheap raw frames, then decodes only evenly distributed
    packet positions.  Its memory cost is O(selected-position array), not O(raw
    packet bytes), and the following extraction path flushes decoded rows in
    fixed-size Parquet row groups.
    """

    raw_reader = _raw_reader(capture_path)
    source_total = 0
    report_every = 100_000
    try:
        for source_total, _item in enumerate(raw_reader, start=1):
            if source_total % report_every == 0:
                LOGGER.info(
                    "STREAM_COUNT_PROGRESS capture=%s source_packets=%s",
                    capture_path.name,
                    source_total,
                )
                _emit_progress(
                    progress_callback,
                    {
                        "event": "capture_sampling_progress",
                        "capture": str(capture_path),
                        "capture_name": capture_path.name,
                        "packets_read": source_total,
                        "packets_selected": min(source_total, max_packets),
                        "packet_limit": max_packets,
                    },
                )
    finally:
        raw_reader.close()

    if source_total <= max_packets:
        positions: np.ndarray | None = None
        selected_total = source_total
    else:
        # ``linspace`` is sorted, compact (8 bytes per selected position), and
        # does not require a large Python set.  A 3M-packet selection therefore
        # uses roughly 24 MiB for positions rather than gigabytes of frame data.
        positions = np.linspace(1, source_total, num=max_packets, dtype=np.int64)
        selected_total = max_packets

    def selected_packets() -> Iterator[tuple[int, Any]]:
        reader = _reader(capture_path)
        position_index = 0
        try:
            for packet_index, packet in enumerate(reader, start=1):
                if positions is not None:
                    while (
                        position_index < len(positions)
                        and int(positions[position_index]) < packet_index
                    ):
                        position_index += 1
                    if (
                        position_index >= len(positions)
                        or int(positions[position_index]) != packet_index
                    ):
                        continue
                    position_index += 1
                yield packet_index, packet
        finally:
            reader.close()

    LOGGER.info(
        "STREAM_SAMPLE_PLAN capture=%s source_packets=%s selected=%s",
        capture_path.name,
        source_total,
        selected_total,
    )
    return selected_packets(), source_total


def _present(value: Any) -> bool:
    """Return whether a scalar packet field has an observed value."""

    if value is None:
        return False
    missing = pd.isna(value)
    return bool(not missing) if isinstance(missing, (bool, np.bool_)) else True


def _number(value: Any, default: float = 0.0) -> float:
    if not _present(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _causal_augment_features(
    frame: pd.DataFrame,
    state: _CausalFeatureState,
    window_seconds: int,
) -> pd.DataFrame:
    """Create bounded, causal behaviour features for one extracted row group."""

    if frame.empty:
        return _canonical_stream_frame(frame)
    records: list[dict[str, Any]] = []
    for record in frame.sort_values("timestamp", kind="stable").to_dict("records"):
        timestamp = pd.Timestamp(record["timestamp"])
        capture = str(record["capture"])
        flow_id = str(record["flow_id"])
        source_ip = str(record["src_ip"])
        direction = int(_number(record.get("direction")))
        protocol = str(record.get("protocol") or "unknown")
        flow_key = (capture, flow_id)
        source_key = (capture, source_ip)

        capture_state = state.captures.setdefault(capture, {"first": timestamp, "count": 0})
        capture_state["count"] += 1
        capture_elapsed = max((timestamp - capture_state["first"]).total_seconds(), 1.0)
        capture_rate = capture_state["count"] / capture_elapsed

        packet_length = _number(record.get("packet_length"))
        flow = state.flow(flow_key)
        flow_count = int(flow.get("count", 0)) + 1
        flow_bytes = _number(flow.get("bytes")) + packet_length
        first = flow.setdefault("first", timestamp)
        previous = flow.get("last")
        inter_arrival = (
            max((timestamp - previous).total_seconds(), 0.0) if previous is not None else 0.0
        )
        jitter = abs(inter_arrival - _number(flow.get("last_iat"))) if previous is not None else 0.0
        length_sum = _number(flow.get("length_sum")) + packet_length
        length_sum_sq = _number(flow.get("length_sum_sq")) + packet_length**2
        mean_length = length_sum / flow_count
        std_length = float(max(length_sum_sq / flow_count - mean_length**2, 0.0) ** 0.5)
        direction_bytes = dict(flow.get("direction_bytes", {}))
        direction_bytes[direction] = _number(direction_bytes.get(direction)) + packet_length
        reverse_bytes = max(flow_bytes - direction_bytes[direction], 1.0)
        flow.update(
            {
                "count": flow_count,
                "bytes": flow_bytes,
                "last": timestamp,
                "last_iat": inter_arrival,
                "length_sum": length_sum,
                "length_sum_sq": length_sum_sq,
                "direction_bytes": direction_bytes,
            }
        )
        flow_duration = max((timestamp - first).total_seconds(), 0.0)
        record.update(
            {
                "flow_duration": flow_duration,
                "flow_total_packets": flow_count,
                "flow_total_bytes": flow_bytes,
                "packet_length_mean": mean_length,
                "packet_length_std": std_length,
                "inter_arrival_time": inter_arrival,
                "jitter": jitter,
                "packet_rate": flow_count / max(flow_duration, 1.0),
                "flow_byte_ratio": direction_bytes[direction] / reverse_bytes,
            }
        )

        source = state.source(source_key)
        source_count = int(source.get("count", 0)) + 1
        source_first = source.setdefault("first", timestamp)
        destinations: Counter[str] = source.setdefault("destinations", Counter())
        destination = str(record.get("dst_ip") or "")
        if (
            destination not in destinations
            and len(destinations) >= state.max_destinations_per_source
        ):
            destinations["__other__"] += 1
        else:
            destinations[destination] += 1
        source.update({"count": source_count})
        source_elapsed = max((timestamp - source_first).total_seconds(), 1.0)
        source_rate = source_count / source_elapsed
        record.update(
            {
                "source_packet_rate": source_rate,
                "source_destination_count": len(destinations),
                "destination_entropy": _entropy_from_counts(destinations),
                "burstiness": source_rate / max(capture_rate, 1e-6),
                "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
                "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
                "weekday_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7),
                "weekday_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7),
            }
        )

        if protocol == "dns":
            dns = state.protocol_flow(state.dns_sources, source_key)
            dns["count"] = int(dns.get("count", 0)) + 1
            dns["nxdomain"] = int(dns.get("nxdomain", 0)) + int(
                _number(record.get("dns_rcode")) == 3
            )
            qnames: Counter[str] = dns.setdefault("qnames", Counter())
            if _present(record.get("dns_qname")):
                qnames[str(record["dns_qname"])] += 1
            record["dns_nxdomain_rate"] = dns["nxdomain"] / max(dns["count"], 1)
            record["dns_query_repeat_count"] = qnames.get(str(record.get("dns_qname")), 0)

        if protocol == "http":
            context_key = (capture, flow_id, direction)
            context = state._bounded(state.http_context, context_key, state.max_active_flows)
            for column in HTTP_CONTEXT_COLUMNS:
                value = record.get(column)
                if _present(value):
                    context[column] = value
                elif column in context:
                    record[column] = context[column]
            http = state.protocol_flow(state.http_sources, source_key)
            method = str(record.get("http_method") or "").upper()
            if method == "POST":
                http["post"] = int(http.get("post", 0)) + 1
            elif method == "GET":
                http["get"] = int(http.get("get", 0)) + 1
            status = record.get("http_status_code")
            if _present(status):
                http["responses"] = int(http.get("responses", 0)) + 1
                if 400 <= _number(status) < 600:
                    http["errors"] = int(http.get("errors", 0)) + 1
            urls: Counter[str] = http.setdefault("urls", Counter())
            if _present(record.get("http_url")):
                urls[str(record["http_url"])] += 1
            record["http_error_ratio"] = _number(http.get("errors")) / max(
                _number(http.get("responses")), 1.0
            )
            record["http_post_get_ratio"] = _number(http.get("post")) / max(
                _number(http.get("get")), 1.0
            )
            record["http_request_repeat_count"] = urls.get(str(record.get("http_url")), 0)

        if protocol == "modbus":
            modbus = state.protocol_flow(state.modbus_flows, flow_key)
            modbus["count"] = int(modbus.get("count", 0)) + 1
            modbus["exceptions"] = int(modbus.get("exceptions", 0)) + int(
                _number(record.get("modbus_is_exception")) > 0
            )
            address = record.get("modbus_starting_address")
            if _present(address):
                modbus["address_count"] = int(modbus.get("address_count", 0)) + 1
                modbus["address_sum"] = _number(modbus.get("address_sum")) + _number(address)
                record["modbus_address_deviation"] = abs(
                    _number(address) - modbus["address_sum"] / max(modbus["address_count"], 1)
                )
            transaction = record.get("modbus_transaction_id")
            if _present(transaction):
                transactions: dict[float, set[int]] = modbus.setdefault("transactions", {})
                tx_key = _number(transaction)
                directions = transactions.setdefault(tx_key, set())
                record["modbus_response_matched"] = int(bool(directions - {direction}))
                directions.add(direction)
                if len(transactions) > state.max_destinations_per_source:
                    transactions.pop(next(iter(transactions)))
            record["modbus_exception_rate"] = modbus["exceptions"] / max(modbus["count"], 1)

        if protocol == "s7comm" and _present(record.get("s7_pdu_reference")):
            s7 = state.protocol_flow(state.s7_flows, flow_key)
            references: dict[float, set[int]] = s7.setdefault("references", {})
            reference = _number(record["s7_pdu_reference"])
            directions = references.setdefault(reference, set())
            record["s7_response_matched"] = int(bool(directions - {direction}))
            directions.add(direction)
            if len(references) > state.max_destinations_per_source:
                references.pop(next(iter(references)))

        record["behavior_window_seconds"] = window_seconds
        records.append(record)
    return _canonical_stream_frame(pd.DataFrame(records))


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
        # Folder-scoped TCP floods have no MBAP payload. Their shared features
        # remain valid, while Modbus-only behaviour stays explicitly missing.
        if "modbus_is_exception" in modbus_rows:
            result.loc[modbus, "modbus_exception_rate"] = modbus_rows.groupby(
                ["capture", "flow_id"]
            )["modbus_is_exception"].transform("mean")
        if "modbus_starting_address" in modbus_rows:
            medians = modbus_rows.groupby(["capture", "flow_id"])[
                "modbus_starting_address"
            ].transform("median")
            result.loc[modbus, "modbus_address_deviation"] = (
                modbus_rows["modbus_starting_address"] - medians
            ).abs()
        if "modbus_transaction_id" in modbus_rows:
            repeated = modbus_rows.groupby(["capture", "flow_id", "modbus_transaction_id"])[
                "direction"
            ].transform("nunique")
            result.loc[modbus, "modbus_response_matched"] = (repeated > 1).astype(int)

    s7 = result["protocol"].eq("s7comm")
    if s7.any() and "s7_pdu_reference" in result:
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
    detected_protocols: Counter[str] = Counter()
    scope_overrides = 0
    sampled_packets: list[tuple[int, Any]] | None = None
    source_packets_total: int | None = None
    reader: Iterator[Any] | None = None
    if max_packets is not None:
        sampled_packets, source_packets_total = _reservoir_sample_packets(
            capture_path, max_packets, progress_callback
        )
        packet_stream: Iterator[tuple[int, Any]] = iter(sampled_packets)
    else:
        reader = _reader(capture_path)
        packet_stream = enumerate(reader, start=1)
    packets_seen = 0
    selected_packets = 0
    report_every = max(1, int(source_packets_total or max_packets) // 10) if max_packets else 5_000
    next_report = report_every
    progress = Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn()
    )
    with progress:
        task = progress.add_task(
            f"Extracting {capture_path.name}",
            total=min(max_packets, source_packets_total) if max_packets else None,
        )
        try:
            for packet_index, packet in packet_stream:
                packets_seen = packet_index
                selected_packets += 1
                row = _packet_row(
                    packet,
                    capture_path.name,
                    protocol_scope=expected_protocol,
                    packet_index=packet_index,
                )
                if row is not None:
                    detected_protocol = str(row["detected_protocol"])
                    detected_protocols[detected_protocol] += 1
                    if expected_protocol is not None and detected_protocol != expected_protocol:
                        scope_overrides += 1
                    rows.append(row)
                if packets_seen >= next_report:
                    elapsed = time.perf_counter() - started
                    LOGGER.info(
                        "EXTRACT_PROGRESS capture=%s source_packets=%s selected=%s%s "
                        "retained=%s elapsed=%.1fs",
                        capture_path.name,
                        packets_seen,
                        selected_packets,
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
                            "packets_read": packets_seen,
                            "packet_limit": max_packets,
                            "packets_selected": selected_packets,
                            "source_packets_total": source_packets_total,
                            "retained_rows": len(rows),
                            "detected_protocol_counts": dict(detected_protocols),
                            "scope_override_rows": scope_overrides,
                            "elapsed_seconds": round(elapsed, 2),
                            "progress_ratio": (
                                min(selected_packets / max_packets, 1.0) if max_packets else None
                            ),
                        },
                    )
                    next_report += report_every
                progress.advance(task)
        finally:
            if reader is not None:
                reader.close()
    if source_packets_total is not None:
        packets_seen = source_packets_total
    summary = {
        "capture": str(capture_path),
        "packets_read": packets_seen,
        "packets_selected": selected_packets,
        "source_packets_total": source_packets_total,
        "sampling_strategy": "deterministic_reservoir"
        if sampled_packets is not None
        else "all_packets",
        "detected_protocol_counts": dict(detected_protocols),
        "scope_override_rows": scope_overrides,
        "retained_rows": len(rows),
    }
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "EXTRACT_COMPLETE capture=%s packets=%s selected=%s retained=%s protocols=%s elapsed=%.1fs",
        capture_path.name,
        packets_seen,
        selected_packets,
        len(rows),
        dict(detected_protocols),
        elapsed,
    )
    _emit_progress(
        progress_callback,
        {
            "event": "capture_completed",
            "capture": str(capture_path),
            "capture_name": capture_path.name,
            "packets_read": packets_seen,
            "packet_limit": max_packets,
            "packets_selected": selected_packets,
            "source_packets_total": source_packets_total,
            "retained_rows": len(rows),
            "detected_protocol_counts": dict(detected_protocols),
            "scope_override_rows": scope_overrides,
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


def _streaming_enabled(config: dict[str, Any], max_packets: int | None) -> bool:
    """Choose disk-backed extraction before a large packet selection reaches RAM."""

    settings = config.get("capture", {})
    if not bool(settings.get("streaming_enabled", True)):
        return False
    threshold = max(1, int(settings.get("streaming_threshold_packets", 200_000)))
    return bool(settings.get("streaming_unbounded", False)) or (
        max_packets is not None and max_packets >= threshold
    )


def _stream_extract_pcap_features(
    captures: list[Path],
    output_path: Path,
    config: dict[str, Any],
    max_packets: int | None,
    expected_protocol: str | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    batch_callback: Callable[[pd.DataFrame], None] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract large captures to a single row-grouped Parquet file with bounded RAM.

    The returned frame is intentionally a deterministic reservoir for charts and
    mapping evidence.  The Parquet artifact remains the complete PCAP-derived
    source for model sampling, batch scoring, and external systems.
    """

    settings = config.get("capture", {})
    chunk_rows = max(1_000, int(settings.get("streaming_chunk_rows", 50_000)))
    sample_rows = max(1_000, int(settings.get("streaming_return_sample_rows", 100_000)))
    active_flows = max(1_000, int(settings.get("streaming_max_active_flows", 200_000)))
    active_sources = max(100, int(settings.get("streaming_max_active_sources", 50_000)))
    destinations = max(32, int(settings.get("streaming_max_destinations_per_source", 4_096)))
    # Keep this configurable down to one packet for deterministic tests and
    # constrained deployments. Production defaults to 200k, where retaining
    # raw frames is no longer a safe sampling strategy.
    reservoir_threshold = max(
        1, int(settings.get("streaming_reservoir_threshold_packets", 200_000))
    )
    state = _CausalFeatureState(active_flows, active_sources, destinations)
    seed_bytes = hashlib.blake2b(str(output_path).encode("utf-8"), digest_size=8).digest()
    generator = random.Random(int.from_bytes(seed_bytes, "big"))
    dashboard_reservoir: list[dict[str, Any]] = []
    capture_summaries: list[dict[str, Any]] = []
    protocol_counts: Counter[str] = Counter()
    total_rows = 0
    writer: pq.ParquetWriter | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def retain_sample(frame: pd.DataFrame) -> None:
        nonlocal total_rows
        for record in frame.to_dict("records"):
            total_rows += 1
            if len(dashboard_reservoir) < sample_rows:
                dashboard_reservoir.append(record)
                continue
            replacement = generator.randrange(total_rows)
            if replacement < sample_rows:
                dashboard_reservoir[replacement] = record

    def flush(rows: list[dict[str, Any]]) -> int:
        nonlocal writer
        if not rows:
            return 0
        packet_rows = _stable_feature_schema(rows)
        features = _causal_augment_features(
            packet_rows, state, int(settings.get("behavior_window_seconds", 60))
        )
        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        writer.write_table(table, row_group_size=chunk_rows)
        if batch_callback is not None:
            batch_callback(features)
        retain_sample(features)
        protocol_counts.update(features["protocol"].dropna().astype(str).tolist())
        return len(features)

    LOGGER.info(
        "STREAM_EXTRACT_START captures=%s packet_limit=%s chunk_rows=%s output=%s",
        len(captures),
        max_packets if max_packets is not None else "unlimited",
        chunk_rows,
        output_path,
    )
    try:
        for capture_path in captures:
            started = time.perf_counter()
            _emit_progress(
                progress_callback,
                {
                    "event": "capture_started",
                    "capture": str(capture_path),
                    "capture_name": capture_path.name,
                    "packet_limit": max_packets,
                    "expected_protocol": expected_protocol,
                    "extraction_mode": "streaming_causal",
                },
            )
            if max_packets is not None and max_packets >= reservoir_threshold:
                packet_stream, source_total = _large_uniform_packet_stream(
                    capture_path, max_packets, progress_callback
                )
                selected_total = min(max_packets, source_total)
            elif max_packets is not None:
                packets, source_total = _reservoir_sample_packets(
                    capture_path, max_packets, progress_callback
                )
                packet_stream = iter(packets)
                selected_total = len(packets)
            else:
                reader = _reader(capture_path)
                source_total = None
                selected_total = None

                def all_packets(active_reader: Iterator[Any]) -> Iterator[tuple[int, Any]]:
                    try:
                        yield from enumerate(active_reader, start=1)
                    finally:
                        active_reader.close()

                packet_stream = all_packets(reader)

            rows: list[dict[str, Any]] = []
            selected_packets = 0
            retained_rows = 0
            detected_protocols: Counter[str] = Counter()
            scope_overrides = 0
            report_every = max(1_000, min(chunk_rows, max(selected_total or chunk_rows, 1) // 10))
            try:
                for packet_index, packet in packet_stream:
                    selected_packets += 1
                    row = _packet_row(
                        packet,
                        capture_path.name,
                        protocol_scope=expected_protocol,
                        packet_index=packet_index,
                    )
                    if row is not None:
                        detected = str(row["detected_protocol"])
                        detected_protocols[detected] += 1
                        if expected_protocol is not None and detected != expected_protocol:
                            scope_overrides += 1
                        rows.append(row)
                    if len(rows) >= chunk_rows:
                        retained_rows += flush(rows)
                        rows = []
                    if selected_packets % report_every == 0:
                        _emit_progress(
                            progress_callback,
                            {
                                "event": "capture_progress",
                                "capture": str(capture_path),
                                "capture_name": capture_path.name,
                                "packets_read": source_total or packet_index,
                                "packet_limit": max_packets,
                                "packets_selected": selected_packets,
                                "source_packets_total": source_total,
                                "retained_rows": retained_rows + len(rows),
                                "detected_protocol_counts": dict(detected_protocols),
                                "scope_override_rows": scope_overrides,
                                "elapsed_seconds": round(time.perf_counter() - started, 2),
                                "progress_ratio": (
                                    selected_packets / max(selected_total or selected_packets, 1)
                                ),
                            },
                        )
                retained_rows += flush(rows)
            finally:
                # Generators own a PcapReader in the unbounded path. Explicitly
                # closing makes Ctrl+C and failed mapping runs release the file.
                close = getattr(packet_stream, "close", None)
                if callable(close):
                    close()
            capture_summaries.append(
                {
                    "capture": str(capture_path),
                    "packets_read": source_total if source_total is not None else selected_packets,
                    "packets_selected": selected_packets,
                    "source_packets_total": source_total,
                    "sampling_strategy": (
                        "two_pass_uniform"
                        if max_packets is not None and max_packets >= reservoir_threshold
                        else "deterministic_reservoir"
                        if max_packets is not None
                        else "all_packets_stream"
                    ),
                    "detected_protocol_counts": dict(detected_protocols),
                    "scope_override_rows": scope_overrides,
                    "retained_rows": retained_rows,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                }
            )
            LOGGER.info(
                "STREAM_EXTRACT_CAPTURE_COMPLETE capture=%s selected=%s retained=%s elapsed=%.1fs",
                capture_path.name,
                selected_packets,
                retained_rows,
                time.perf_counter() - started,
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty = _canonical_stream_frame(pd.DataFrame())
        pq.write_table(
            pa.Table.from_pandas(empty, preserve_index=False), output_path, compression="zstd"
        )
    returned = _canonical_stream_frame(pd.DataFrame(dashboard_reservoir))
    manifest = {
        "created_at": utc_now(),
        "capture": str(captures[0].parent if len(captures) > 1 else captures[0]),
        "captures": capture_summaries,
        "capture_count": len(captures),
        "expected_protocol": expected_protocol,
        "max_packets_per_capture": max_packets,
        "output": str(output_path),
        "rows": total_rows,
        "flow_count": int(returned["flow_id"].nunique(dropna=True)) if not returned.empty else 0,
        "flow_count_scope": "returned_stream_sample",
        "protocol_counts": dict(protocol_counts),
        "columns": returned.columns.tolist(),
        "feature_schema_version": "1.2.0",
        "extraction_mode": "streaming_causal",
        "row_group_rows": chunk_rows,
        "returned_sample_rows": len(returned),
        "state_bounds": {
            "active_flows": active_flows,
            "active_sources": active_sources,
            "destinations_per_source": destinations,
        },
    }
    write_json(manifest, output_path.with_suffix(".manifest.json"))
    _emit_progress(
        progress_callback,
        {
            "event": "feature_extraction_completed",
            "output": str(output_path),
            "rows": total_rows,
            "flow_count": manifest["flow_count"],
            "protocol_counts": manifest["protocol_counts"],
            "extraction_mode": "streaming_causal",
            "progress_ratio": 1.0,
        },
    )
    LOGGER.info(
        "STREAM_EXTRACT_COMPLETE rows=%s sample_rows=%s row_group_rows=%s output=%s",
        total_rows,
        len(returned),
        chunk_rows,
        output_path,
    )
    return returned, manifest


def extract_pcap_features(
    capture_path: Path,
    output_path: Path,
    config: dict[str, Any],
    max_packets: int | None = None,
    expected_protocol: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    capture_paths: list[Path] | None = None,
    batch_callback: Callable[[pd.DataFrame], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract one capture, a folder, or an explicit capture selection into one table.

    ``capture_paths`` is used by the dataset-run allowlist.  It deliberately
    bypasses folder re-discovery, so a targeted smoke test cannot accidentally
    scan a multi-gigabyte sibling capture from the same protocol folder.
    """
    captures = capture_paths if capture_paths is not None else resolve_capture_paths(capture_path)
    if not captures:
        raise FileNotFoundError(f"No PCAP/PCAPNG captures selected beneath {capture_path}.")
    missing = [str(path) for path in captures if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selected capture paths do not exist: {missing}")
    supported = set(config["capture"]["supported_protocols"])
    if expected_protocol is not None and expected_protocol not in supported:
        raise ValueError(
            f"Expected protocol '{expected_protocol}' is not configured as supported. "
            f"Choose one of: {', '.join(sorted(supported))}."
        )
    if _streaming_enabled(config, max_packets):
        return _stream_extract_pcap_features(
            captures,
            output_path,
            config,
            max_packets,
            expected_protocol,
            progress_callback,
            batch_callback,
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
