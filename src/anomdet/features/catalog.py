"""A curated, human-readable feature catalogue for dashboard and selection use."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    """Describes a feature's meaning, applicability, and approximate CPU cost."""

    name: str
    display_name: str
    protocols: tuple[str, ...]
    category: str
    description: str
    cost: str


def _feature(
    name: str,
    protocols: tuple[str, ...],
    category: str,
    description: str,
    cost: str = "low",
) -> FeatureDefinition:
    """Create a catalogue entry while keeping the static list concise."""
    return FeatureDefinition(name, name.replace("_", " ").title(), protocols, category, description, cost)


IT = ("ssh", "dns", "http")
OT = ("modbus", "s7comm")
ALL = IT + OT

FEATURE_CATALOG: tuple[FeatureDefinition, ...] = (
    # Shared packet, flow, and behavioural features.
    _feature("packet_length", ALL, "packet", "Full frame length observed in the capture."),
    _feature("payload_size", ALL, "packet", "Application payload byte count."),
    _feature("payload_entropy", ALL, "packet", "Shannon entropy of the transport payload.", "medium"),
    _feature("tcp_flag_count", ALL, "packet", "Number of active TCP control flags."),
    _feature("direction", ALL, "packet", "Packet direction inferred from the normalized flow key."),
    _feature("flow_duration", ALL, "flow", "Elapsed time from the first to current packet in a flow."),
    _feature("flow_total_packets", ALL, "flow", "Packets observed in the normalized bidirectional flow."),
    _feature("flow_total_bytes", ALL, "flow", "Bytes observed in the normalized bidirectional flow."),
    _feature("flow_byte_ratio", ALL, "flow", "Current directional bytes divided by reverse-direction bytes."),
    _feature("packet_length_mean", ALL, "flow", "Mean packet length within the flow."),
    _feature("packet_length_std", ALL, "flow", "Packet-length standard deviation within the flow."),
    _feature("inter_arrival_time", ALL, "timing", "Time since the previous packet in the same flow."),
    _feature("jitter", ALL, "timing", "Absolute change between consecutive inter-arrival times."),
    _feature("packet_rate", ALL, "timing", "Packets per second for the flow so far."),
    _feature("hour_sin", ALL, "timing", "Cyclical encoding of the hour of day."),
    _feature("hour_cos", ALL, "timing", "Cyclical encoding of the hour of day."),
    _feature("weekday_sin", ALL, "timing", "Cyclical encoding of the weekday."),
    _feature("weekday_cos", ALL, "timing", "Cyclical encoding of the weekday."),
    _feature("source_packet_rate", ALL, "host_behavior", "Packets emitted by the source in the configured window."),
    _feature("source_destination_count", ALL, "host_behavior", "Distinct destinations used by the source in the configured window."),
    _feature("destination_entropy", ALL, "host_behavior", "Entropy of destinations contacted by the source.", "medium"),
    _feature("burstiness", ALL, "host_behavior", "Windowed source packet rate relative to global activity."),
    # SSH features.
    _feature("ssh_client_banner", ("ssh",), "ssh_banner", "Client version/banner string."),
    _feature("ssh_server_banner", ("ssh",), "ssh_banner", "Server version/banner string."),
    _feature("ssh_kex_algorithms", ("ssh",), "ssh_negotiation", "KEX algorithms exposed in an observable SSH KEXINIT."),
    _feature("ssh_cipher_count", ("ssh",), "ssh_negotiation", "Number of advertised encryption algorithms."),
    _feature("ssh_mac_count", ("ssh",), "ssh_negotiation", "Number of advertised MAC algorithms."),
    _feature("ssh_compression_count", ("ssh",), "ssh_negotiation", "Number of advertised compression algorithms."),
    _feature("hassh", ("ssh",), "ssh_negotiation", "Hash of observable SSH KEXINIT algorithm lists.", "medium"),
    _feature("ssh_auth_method", ("ssh",), "ssh_authentication", "Authentication method if a protocol-aware parser can observe it."),
    _feature("ssh_failed_auth_count", ("ssh",), "ssh_authentication", "Observable failed authentication events per flow."),
    _feature("ssh_open_channel_count", ("ssh",), "ssh_session", "Observable SSH channel open events."),
    _feature("ssh_keepalive_interval", ("ssh",), "ssh_session", "Interval between observable keepalive packets."),
    # DNS features.
    _feature("dns_qname", ("dns",), "dns_query", "Requested DNS name."),
    _feature("dns_qname_length", ("dns",), "dns_query", "Length of the requested DNS name."),
    _feature("dns_qname_entropy", ("dns",), "dns_query", "Shannon entropy of the requested DNS name."),
    _feature("dns_label_count", ("dns",), "dns_query", "Number of labels in the requested DNS name."),
    _feature("dns_qtype", ("dns",), "dns_query", "Requested DNS record type."),
    _feature("dns_digit_ratio", ("dns",), "dns_query", "Fraction of digits in the DNS name."),
    _feature("dns_hyphen_ratio", ("dns",), "dns_query", "Fraction of hyphens in the DNS name."),
    _feature("dns_ngram_score", ("dns",), "dns_query", "Simple lexical score for domain-label naturalness.", "medium"),
    _feature("dns_rcode", ("dns",), "dns_response", "DNS response code."),
    _feature("dns_ttl", ("dns",), "dns_response", "First-answer TTL when present."),
    _feature("dns_answer_count", ("dns",), "dns_response", "Number of answer records."),
    _feature("dns_response_size", ("dns",), "dns_response", "Response packet length."),
    _feature("dns_authoritative", ("dns",), "dns_response", "Authoritative-answer header flag."),
    _feature("dns_recursion_available", ("dns",), "dns_response", "Recursion-available header flag."),
    _feature("dns_truncated", ("dns",), "dns_response", "Truncated-response header flag."),
    _feature("dns_nxdomain_rate", ("dns",), "dns_behavior", "NXDOMAIN ratio for the source during the capture."),
    _feature("dns_query_repeat_count", ("dns",), "dns_behavior", "Repeat count for a source and DNS name."),
    # HTTP features.
    _feature("http_method", ("http",), "http_request", "HTTP request method."),
    _feature("http_url_length", ("http",), "http_request", "Length of request target."),
    _feature("http_url_entropy", ("http",), "http_request", "Shannon entropy of request target."),
    _feature("http_query_parameter_count", ("http",), "http_request", "Number of URL query parameters."),
    _feature("http_suspicious_token_count", ("http",), "http_request", "SQLi, XSS, encoding, and traversal token count."),
    _feature("http_host", ("http",), "http_request", "Host header value."),
    _feature("http_user_agent", ("http",), "http_request", "User-Agent header value."),
    _feature("http_header_count", ("http",), "http_request", "Number of HTTP headers."),
    _feature("http_content_length", ("http",), "http_request", "Content-Length header value."),
    _feature("http_cookie_count", ("http",), "http_request", "Cookie-pair count."),
    _feature("http_status_code", ("http",), "http_response", "HTTP response status code."),
    _feature("http_response_size", ("http",), "http_response", "HTTP response packet length."),
    _feature("http_error_ratio", ("http",), "http_behavior", "4xx/5xx response ratio for the source."),
    _feature("http_post_get_ratio", ("http",), "http_behavior", "POST-to-GET request ratio for the source."),
    _feature("http_request_repeat_count", ("http",), "http_behavior", "Repeat count for source and request target."),
    # Modbus features.
    _feature("modbus_function_code", ("modbus",), "modbus", "Modbus application PDU function code."),
    _feature("modbus_exception_code", ("modbus",), "modbus", "Modbus exception code, if applicable."),
    _feature("modbus_starting_address", ("modbus",), "modbus", "Requested register or coil start address."),
    _feature("modbus_quantity", ("modbus",), "modbus", "Requested register or coil quantity."),
    _feature("modbus_byte_count", ("modbus",), "modbus", "PDU byte-count field."),
    _feature("modbus_unit_id", ("modbus",), "modbus", "Modbus slave/unit identifier."),
    _feature("modbus_transaction_id", ("modbus",), "modbus", "MBAP transaction identifier."),
    _feature("modbus_protocol_id_valid", ("modbus",), "modbus", "Whether MBAP protocol identifier equals zero."),
    _feature("modbus_length_valid", ("modbus",), "modbus", "Whether MBAP length agrees with observed payload."),
    _feature("modbus_function_category", ("modbus",), "modbus", "Read, write, diagnostic, or other function category."),
    _feature("modbus_nonstandard_function", ("modbus",), "modbus", "Whether the function code is nonstandard or reserved."),
    _feature("modbus_value", ("modbus",), "modbus", "First observable register value or written value."),
    _feature("modbus_exception_rate", ("modbus",), "modbus_behavior", "Exception ratio in the current capture."),
    _feature("modbus_address_deviation", ("modbus",), "modbus_behavior", "Distance from the flow's median requested address."),
    # S7comm features.
    _feature("s7_rosctr", ("s7comm",), "s7comm", "S7 message ROSCTR type."),
    _feature("s7_function_code", ("s7comm",), "s7comm", "S7 parameter function code."),
    _feature("s7_pdu_reference", ("s7comm",), "s7comm", "S7 PDU reference value."),
    _feature("s7_error_class", ("s7comm",), "s7comm", "S7 error class."),
    _feature("s7_error_code", ("s7comm",), "s7comm", "S7 error code."),
    _feature("s7_parameter_length", ("s7comm",), "s7comm", "S7 parameter section length."),
    _feature("s7_data_length", ("s7comm",), "s7comm", "S7 data section length."),
    _feature("s7_item_count", ("s7comm",), "s7comm", "Requested variable-item count where available."),
    _feature("s7_area_code", ("s7comm",), "s7comm", "Memory area code from a variable specification."),
    _feature("s7_db_number", ("s7comm",), "s7comm", "Data-block number from a variable specification."),
    _feature("s7_transfer_size", ("s7comm",), "s7comm", "Declared transfer size/data type."),
    _feature("s7_is_plus", ("s7comm",), "s7comm", "Whether traffic appears to be S7comm-Plus."),
    _feature("s7_control_command", ("s7comm",), "s7comm", "PLC control command indicator."),
    _feature("s7_block_transfer", ("s7comm",), "s7comm", "Block download/upload indicator."),
    _feature("s7_return_code", ("s7comm",), "s7comm", "First per-item return code if present."),
    _feature("s7_response_matched", ("s7comm",), "s7comm_behavior", "Whether an opposite-direction PDU reference is observed."),
)


def available_features(protocols: tuple[str, ...] | None = None) -> list[dict[str, object]]:
    """Return serialisable catalogue entries filtered to any selected protocol."""
    entries = FEATURE_CATALOG
    if protocols:
        requested = set(protocols)
        entries = tuple(item for item in entries if requested.intersection(item.protocols))
    return [asdict(item) for item in entries]


def feature_names(protocols: tuple[str, ...] | None = None) -> set[str]:
    """Return valid selectable feature names, optionally for specific protocols."""
    return {item["name"] for item in available_features(protocols)}
