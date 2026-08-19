"""Evidence-preserving matching between PCAP flows and heterogeneous label CSV files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from anomdet.core.io import read_table, utc_now, write_json, write_table
from anomdet.features.extractor import extract_pcap_features

LOGGER = logging.getLogger("anomdet")
FLOW_ID_PATTERN = re.compile(
    r"(?P<src_ip>(?:\d{1,3}\.){3}\d{1,3})[-:](?P<src_port>\d+)[-:](?P<dst_ip>(?:\d{1,3}\.){3}\d{1,3})[-:](?P<dst_port>\d+)"
)


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first configured column that is present in the CSV."""
    return next((name for name in candidates if name in frame.columns), None)


def _as_string(series: pd.Series | None) -> pd.Series:
    """Create normalized nullable strings safe for identifier comparison."""
    if series is None:
        return pd.Series(pd.NA, index=range(0), dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _take_ip(value: object) -> object:
    """Select a usable IPv4 address when OT CSV fields contain address lists."""
    if pd.isna(value):
        return pd.NA
    found = re.search(r"(?:\d{1,3}\.){3}\d{1,3}", str(value))
    return found.group(0) if found else pd.NA


def _parse_flow_id(series: pd.Series) -> pd.DataFrame:
    """Extract 5-tuple fields from common CIC-style flow identifier strings."""
    extracted = series.astype("string").str.extract(FLOW_ID_PATTERN)
    for port_column in ("src_port", "dst_port"):
        extracted[port_column] = pd.to_numeric(extracted[port_column], errors="coerce").astype(
            "Int64"
        )
    return extracted


def _parse_timestamp(series: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Parse heterogeneous dataset timestamps with an explicit regional date-order setting."""
    dayfirst = bool(config["mapping"].get("timestamp_dayfirst", True))
    try:
        return pd.to_datetime(series, format="mixed", dayfirst=dayfirst, errors="coerce", utc=True)
    except (TypeError, ValueError):
        # Compatibility path for older pandas versions without `format=mixed`.
        return pd.to_datetime(series, dayfirst=dayfirst, errors="coerce", utc=True)


def normalize_label_csv(path: Path, domain: str, config: dict[str, Any]) -> pd.DataFrame:
    """Normalize IT or OT labels to a minimal common schema without changing source data."""
    raw = read_table(path)
    if raw.empty:
        raise ValueError(f"Label CSV has no rows: {path}")
    mapping = config["mapping"][domain]
    normalized = pd.DataFrame(index=raw.index)
    normalized["csv_row"] = raw.index

    label_columns = [column for column in mapping["label_columns"] if column in raw.columns]
    if label_columns:
        normalized["label"] = (
            raw[label_columns].replace(r"^\s*$", pd.NA, regex=True).bfill(axis=1).iloc[:, 0]
        )
    else:
        normalized["label"] = config["mapping"]["default_label"]

    timestamp_column = _first_present(
        raw, mapping.get("timestamp_columns", mapping.get("start_columns", []))
    )
    normalized["label_timestamp"] = (
        _parse_timestamp(raw[timestamp_column], config) if timestamp_column else pd.NaT
    )
    end_column = _first_present(raw, mapping.get("end_columns", []))
    normalized["label_end_timestamp"] = (
        _parse_timestamp(raw[end_column], config) if end_column else pd.NaT
    )

    source_column = _first_present(raw, mapping["source_ip_columns"])
    destination_column = _first_present(raw, mapping["destination_ip_columns"])
    normalized["src_ip"] = raw[source_column].map(_take_ip) if source_column else pd.NA
    normalized["dst_ip"] = raw[destination_column].map(_take_ip) if destination_column else pd.NA

    source_port_column = _first_present(raw, mapping.get("source_port_columns", []))
    destination_port_column = _first_present(raw, mapping.get("destination_port_columns", []))
    normalized["src_port"] = (
        pd.to_numeric(raw[source_port_column], errors="coerce").astype("Int64")
        if source_port_column
        else pd.NA
    )
    normalized["dst_port"] = (
        pd.to_numeric(raw[destination_port_column], errors="coerce").astype("Int64")
        if destination_port_column
        else pd.NA
    )

    flow_id_column = _first_present(raw, mapping.get("flow_id_columns", []))
    if flow_id_column:
        parsed = _parse_flow_id(raw[flow_id_column])
        for column in ("src_ip", "dst_ip", "src_port", "dst_port"):
            normalized[column] = normalized[column].fillna(parsed[column])
    normalized["source_file"] = path.name
    normalized["source_domain"] = domain
    return normalized


def _flow_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Reduce packet-level extraction output to an individual flow label target."""
    required = {"flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing mapping fields: {sorted(missing)}")
    ordered = features.sort_values("timestamp", kind="stable")
    flows = ordered.groupby("flow_id", as_index=False).agg(
        src_ip=("src_ip", "first"),
        src_port=("src_port", "first"),
        dst_ip=("dst_ip", "first"),
        dst_port=("dst_port", "first"),
        protocol=("protocol", "first"),
        flow_start=("timestamp", "min"),
        flow_end=("timestamp", "max"),
        packet_count=("packet_length", "size"),
    )
    return flows


def _candidate_mask(flow: pd.Series, labels: pd.DataFrame, reverse: bool) -> pd.Series:
    """Return candidate labels that agree with all label fields available for this flow."""
    src_equal = labels["src_ip"].isna() | labels["src_ip"].eq(flow.src_ip)
    dst_equal = labels["dst_ip"].isna() | labels["dst_ip"].eq(flow.dst_ip)
    src_port_equal = labels["src_port"].isna() | labels["src_port"].eq(flow.src_port)
    dst_port_equal = labels["dst_port"].isna() | labels["dst_port"].eq(flow.dst_port)
    direct = src_equal & dst_equal & src_port_equal & dst_port_equal
    if not reverse:
        return direct
    inverse = (
        (labels["src_ip"].isna() | labels["src_ip"].eq(flow.dst_ip))
        & (labels["dst_ip"].isna() | labels["dst_ip"].eq(flow.src_ip))
        & (labels["src_port"].isna() | labels["src_port"].eq(flow.dst_port))
        & (labels["dst_port"].isna() | labels["dst_port"].eq(flow.src_port))
    )
    return direct | inverse


def _best_candidate(
    candidates: pd.DataFrame, flow_start: pd.Timestamp, tolerance: float
) -> tuple[pd.Series | None, float]:
    """Pick the temporally nearest label candidate, respecting known label intervals."""
    if candidates.empty:
        return None, 0.0
    timed = candidates.dropna(subset=["label_timestamp"]).copy()
    if timed.empty:
        return candidates.iloc[0], 0.7
    # Label files (and all-NaT columns in particular) may reach this function
    # as timezone-naive values, while extracted PCAP timestamps are UTC-aware.
    # Normalising both sides prevents pandas from rejecting valid interval and
    # nearest-time comparisons.
    timed["label_timestamp"] = pd.to_datetime(timed["label_timestamp"], errors="coerce", utc=True)
    timed["label_end_timestamp"] = pd.to_datetime(
        timed["label_end_timestamp"], errors="coerce", utc=True
    )
    flow_start = pd.to_datetime(flow_start, errors="coerce", utc=True)
    if pd.isna(flow_start):
        return None, 0.0
    seconds = (timed["label_timestamp"] - flow_start).abs().dt.total_seconds()
    within_interval = (
        timed["label_end_timestamp"].notna()
        & (timed["label_timestamp"] <= flow_start)
        & (timed["label_end_timestamp"] >= flow_start)
    )
    if within_interval.any():
        return timed.loc[within_interval].iloc[0], 0.98
    best_position = seconds.idxmin()
    distance = float(seconds.loc[best_position])
    if distance > tolerance:
        return None, 0.0
    return timed.loc[best_position], round(max(0.6, 0.95 - distance / max(tolerance, 1) * 0.2), 3)


def map_features_to_labels(
    features: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Map normalized labels to PCAP flows and preserve match evidence for auditability."""
    flows = _flow_summary(features)
    tolerance = float(config["mapping"]["timestamp_tolerance_seconds"])
    allow_reverse = bool(config["mapping"]["allow_reverse_flow_match"])
    records: list[dict[str, Any]] = []
    for flow in flows.itertuples(index=False):
        mask = _candidate_mask(flow, labels, allow_reverse)
        candidates = labels.loc[mask]
        match, confidence = _best_candidate(candidates, flow.flow_start, tolerance)
        record = flow._asdict()
        record["candidate_count"] = int(len(candidates))
        if match is None:
            record.update(
                {
                    "label": config["mapping"]["default_label"],
                    "match_status": "unmatched",
                    "match_confidence": 0.0,
                    "csv_row": pd.NA,
                    "label_source_file": pd.NA,
                }
            )
        else:
            record.update(
                {
                    "label": match["label"],
                    "match_status": "matched"
                    if len(candidates) == 1
                    else "matched_from_candidates",
                    "match_confidence": confidence,
                    "csv_row": match["csv_row"],
                    "label_source_file": match["source_file"],
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def map_pcap_to_labels(
    capture_path: Path,
    label_path: Path,
    output_path: Path,
    domain: str,
    config: dict[str, Any],
    max_packets: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract PCAP flow evidence, map a heterogeneous labelled CSV, and persist an audit result."""
    if domain not in {"it", "ot"}:
        raise ValueError("Mapping domain must be either 'it' or 'ot'.")
    intermediate_path = output_path.with_name(f"{output_path.stem}.pcap-features.parquet")
    LOGGER.info("Extracting PCAP evidence for mapping")
    features, _ = extract_pcap_features(capture_path, intermediate_path, config, max_packets)
    LOGGER.info("Normalizing label source %s", label_path)
    labels = normalize_label_csv(label_path, domain, config)
    mapped = map_features_to_labels(features, labels, config)
    write_table(mapped, output_path)
    summary = {
        "created_at": utc_now(),
        "capture": str(capture_path),
        "labels": str(label_path),
        "domain": domain,
        "output": str(output_path),
        "pcap_feature_evidence": str(intermediate_path),
        "flow_count": len(mapped),
        "match_status_counts": mapped["match_status"].value_counts(dropna=False).to_dict(),
        "label_counts": mapped["label"].value_counts(dropna=False).to_dict(),
    }
    write_json(summary, output_path.with_suffix(".summary.json"))
    LOGGER.info(
        "Mapped %s flows; %s matched", len(mapped), int((mapped.match_status != "unmatched").sum())
    )
    return mapped, summary


def map_feature_file_to_labels(
    feature_path: Path,
    label_path: Path,
    output_path: Path,
    domain: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map labels to an existing extraction output without reading the PCAP a second time."""
    if domain not in {"it", "ot"}:
        raise ValueError("Mapping domain must be either 'it' or 'ot'.")
    features = read_table(feature_path)
    LOGGER.info(
        "Normalizing label source %s against extracted features %s", label_path, feature_path
    )
    labels = normalize_label_csv(label_path, domain, config)
    mapped = map_features_to_labels(features, labels, config)
    write_table(mapped, output_path)
    summary = {
        "created_at": utc_now(),
        "feature_evidence": str(feature_path),
        "labels": str(label_path),
        "domain": domain,
        "output": str(output_path),
        "flow_count": len(mapped),
        "match_status_counts": mapped["match_status"].value_counts(dropna=False).to_dict(),
        "label_counts": mapped["label"].value_counts(dropna=False).to_dict(),
    }
    write_json(summary, output_path.with_suffix(".summary.json"))
    LOGGER.info("Mapped %s flows using existing feature evidence", len(mapped))
    return mapped, summary
