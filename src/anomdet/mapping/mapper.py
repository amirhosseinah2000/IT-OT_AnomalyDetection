"""Evidence-preserving matching between PCAP flows and heterogeneous label CSV files."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from anomdet.core.io import read_table, utc_now, write_json, write_table
from anomdet.features.extractor import extract_pcap_features

LOGGER = logging.getLogger("anomdet")
MAPPING_CACHE_SCHEMA_VERSION = "1.0.0"
PACKET_LINK_COLUMNS = [
    "packet_uid",
    "capture",
    "packet_index",
    "timestamp",
    "flow_id",
    "label",
    "is_attack",
    "match_status",
    "match_confidence",
    "csv_row",
    "label_source_file",
    "candidate_count",
    "mapping_accepted",
]
FLOW_ID_PATTERN = re.compile(
    r"(?P<src_ip>(?:\d{1,3}\.){3}\d{1,3})[-:](?P<src_port>\d+)[-:](?P<dst_ip>(?:\d{1,3}\.){3}\d{1,3})[-:](?P<dst_port>\d+)"
)


class MappingCacheMismatchError(ValueError):
    """A cached packet relation belongs to a different feature extraction.

    This is an expected cache-miss condition, not a label-mapping failure.  The
    caller must discard just this cache entry and rebuild the relation from the
    PCAP features and CSV labels.
    """


def _safe_timestamp(value: object) -> pd.Timestamp:
    """Return a UTC timestamp or NaT without leaking CSV-specific parsing details."""
    return pd.to_datetime(value, errors="coerce", utc=True)


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

    def parse(values: pd.Series, *, use_dayfirst: bool) -> pd.Series:
        try:
            return pd.to_datetime(
                values, format="mixed", dayfirst=use_dayfirst, errors="coerce", utc=True
            )
        except (TypeError, ValueError):
            return pd.to_datetime(values, dayfirst=use_dayfirst, errors="coerce", utc=True)

    # OT sources use unambiguous ISO timestamps (2024-07-09).  Applying the
    # regional dd/mm setting to that value can silently turn it into 7 Sep;
    # handle the two forms independently and keep the original source intact.
    text = series.astype("string").str.strip()
    iso_format = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    # Keep an object intermediary: pandas 3 may parse the all-NaT branch at a
    # seconds resolution and refuse to assign microsecond ISO values into it.
    parsed = pd.Series(pd.NaT, index=series.index, dtype="object")
    non_iso = ~iso_format
    if non_iso.any():
        parsed.loc[non_iso] = parse(text.loc[non_iso], use_dayfirst=dayfirst).astype("object")
    if iso_format.any():
        parsed.loc[iso_format] = parse(text.loc[iso_format], use_dayfirst=False).astype("object")
    return pd.to_datetime(parsed, errors="coerce", utc=True)


def normalize_label_csv(path: Path, domain: str, config: dict[str, Any]) -> pd.DataFrame:
    """Normalize IT or OT labels to a minimal common schema without changing source data."""
    raw = read_table(path)
    if raw.empty:
        raise ValueError(f"Label CSV has no rows: {path}")
    mapping = config["mapping"][domain]
    normalized = pd.DataFrame(index=raw.index)
    normalized["csv_row"] = raw.index

    # A CIC Label column is often merely "Attack" while Attack Name contains the
    # stage-two class.  Prefer the most specific column when both are present.
    preferred_labels = [
        "Attack Name",
        "attack_name",
        "Category",
        "category",
        *mapping["label_columns"],
    ]
    label_columns = list(
        dict.fromkeys(column for column in preferred_labels if column in raw.columns)
    )
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
    normalized.attrs["schema_audit"] = {
        "source_file": str(path),
        "domain": domain,
        "source_rows": int(len(raw)),
        "source_columns": [str(column) for column in raw.columns],
        "selected_columns": {
            "label": label_columns,
            "timestamp": timestamp_column,
            "end_timestamp": end_column,
            "src_ip": source_column,
            "dst_ip": destination_column,
            "src_port": source_port_column,
            "dst_port": destination_port_column,
            "flow_id": flow_id_column,
        },
        "parse_coverage": {
            "timestamp": round(float(normalized["label_timestamp"].notna().mean()), 4),
            "src_ip": round(float(normalized["src_ip"].notna().mean()), 4),
            "dst_ip": round(float(normalized["dst_ip"].notna().mean()), 4),
            "five_tuple": round(
                float(
                    normalized[["src_ip", "dst_ip", "src_port", "dst_port"]]
                    .notna()
                    .all(axis=1)
                    .mean()
                ),
                4,
            ),
        },
    }
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


def _pair_key(
    src_ip: object,
    src_port: object,
    dst_ip: object,
    dst_port: object,
    *,
    require_ports: bool,
) -> str | None:
    """Create a direction-independent endpoint or full-five-tuple key."""
    if pd.isna(src_ip) or pd.isna(dst_ip):
        return None
    if require_ports and (pd.isna(src_port) or pd.isna(dst_port)):
        return None
    if require_ports:
        endpoints = sorted((f"{src_ip}:{int(src_port)}", f"{dst_ip}:{int(dst_port)}"))
    else:
        endpoints = sorted((str(src_ip), str(dst_ip)))
    return "|".join(endpoints)


def _candidate_index(labels: pd.DataFrame) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Index full and endpoint-only labels once instead of scanning a CSV per flow."""
    full: dict[str, list[Any]] = {}
    endpoints: dict[str, list[Any]] = {}
    for index, label in labels.iterrows():
        key = _pair_key(
            label["src_ip"],
            label["src_port"],
            label["dst_ip"],
            label["dst_port"],
            require_ports=True,
        )
        if key is not None:
            full.setdefault(key, []).append(index)
            continue
        key = _pair_key(
            label["src_ip"],
            label["src_port"],
            label["dst_ip"],
            label["dst_port"],
            require_ports=False,
        )
        if key is not None:
            endpoints.setdefault(key, []).append(index)
    return full, endpoints


def _indexed_candidates(
    flow: Any,
    labels: pd.DataFrame,
    lookup: tuple[dict[str, list[Any]], dict[str, list[Any]]],
    allow_reverse: bool,
) -> pd.DataFrame:
    """Return exact field-compatible candidates using an indexed common key."""
    del allow_reverse  # Direction is intentionally normalized in the key.
    full, endpoints = lookup
    full_key = _pair_key(flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port, require_ports=True)
    endpoint_key = _pair_key(
        flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port, require_ports=False
    )
    positions: list[Any] = []
    if full_key is not None:
        positions.extend(full.get(full_key, []))
    if endpoint_key is not None:
        positions.extend(endpoints.get(endpoint_key, []))
    if positions:
        return labels.loc[list(dict.fromkeys(positions))]
    # Keep the older per-field comparison as a correctness fallback for an
    # unusual one-sided address CSV which cannot create a canonical key.
    return labels.loc[_candidate_mask(flow, labels, reverse=True)]


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


def _offset_candidates(config: dict[str, Any]) -> list[float]:
    """Return configured offset candidates in seconds, always including zero."""
    hours = config.get("mapping", {}).get("time_offset_candidates_hours", [0])
    values = {0.0}
    for hour in hours:
        try:
            values.add(float(hour) * 3600.0)
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring invalid mapping time-offset candidate: %r", hour)
    return sorted(values, key=lambda value: (abs(value), value))


def _estimate_time_offset(
    flows: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Find the timestamp offset best supported by shared endpoint evidence.

    The CSV timestamp is shifted by the selected number of seconds.  The result
    records enough evidence for a reviewer to distinguish a real clock/timezone
    offset from a weak filename match.
    """
    tolerance = float(config["mapping"]["timestamp_tolerance_seconds"])
    allow_reverse = bool(config["mapping"].get("allow_reverse_flow_match", True))
    candidates: list[dict[str, Any]] = []
    sampled_flows = flows.head(2000)
    lookup = _candidate_index(labels)
    eligible_flow_count = sum(
        not _indexed_candidates(flow, labels, lookup, allow_reverse).empty
        for flow in sampled_flows.itertuples(index=False)
    )
    for offset in _offset_candidates(config):
        supported = 0
        distances: list[float] = []
        for flow in sampled_flows.itertuples(index=False):
            matched = _indexed_candidates(flow, labels, lookup, allow_reverse).dropna(
                subset=["label_timestamp"]
            )
            if matched.empty:
                continue
            timestamps = pd.to_datetime(matched["label_timestamp"], errors="coerce", utc=True)
            delta = (
                timestamps + pd.to_timedelta(offset, unit="s") - _safe_timestamp(flow.flow_start)
            ).abs()
            nearest = delta.dt.total_seconds().min()
            if pd.notna(nearest) and float(nearest) <= tolerance:
                supported += 1
                distances.append(float(nearest))
        candidates.append(
            {
                "offset_seconds": offset,
                "supported_flows": supported,
                "median_error_seconds": float(pd.Series(distances).median()) if distances else None,
            }
        )
        # A zero-offset result already explains every endpoint-compatible flow;
        # evaluating every other timezone cannot improve this decision.
        if eligible_flow_count and supported >= eligible_flow_count:
            break
    selected = max(
        candidates,
        key=lambda item: (
            item["supported_flows"],
            -(
                item["median_error_seconds"]
                if item["median_error_seconds"] is not None
                else float("inf")
            ),
            -abs(item["offset_seconds"]),
        ),
    )
    return {"selected": selected, "candidates": candidates}


def mapping_compatibility_report(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    mapped: pd.DataFrame,
    offset_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Return a portable audit record for label/PCAP compatibility decisions."""
    matched = mapped["match_status"].ne("unmatched") if not mapped.empty else pd.Series(dtype=bool)
    rate = float(matched.mean()) if len(mapped) else 0.0
    confidence = float(mapped.loc[matched, "match_confidence"].mean()) if matched.any() else 0.0
    return {
        "feature_rows": int(len(features)),
        "flow_count": int(len(mapped)),
        "label_rows": int(len(labels)),
        "matched_flow_count": int(matched.sum()),
        "match_rate": round(rate, 4),
        "mean_match_confidence": round(confidence, 4),
        "time_offset": offset_evidence,
        "label_schema": labels.attrs.get("schema_audit", {}),
    }


def map_features_to_labels(
    features: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Map normalized labels to PCAP flows and preserve match evidence for auditability."""
    flows = _flow_summary(features)
    labels = labels.copy()
    schema_audit = labels.attrs.get("schema_audit", {})
    offset_evidence = _estimate_time_offset(flows, labels, config)
    offset_seconds = float(offset_evidence["selected"]["offset_seconds"])
    labels["label_timestamp_original"] = labels["label_timestamp"]
    labels["label_timestamp"] = pd.to_datetime(
        labels["label_timestamp"], errors="coerce", utc=True
    ) + pd.to_timedelta(offset_seconds, unit="s")
    labels["label_end_timestamp"] = pd.to_datetime(
        labels["label_end_timestamp"], errors="coerce", utc=True
    ) + pd.to_timedelta(offset_seconds, unit="s")
    labels.attrs["schema_audit"] = schema_audit
    tolerance = float(config["mapping"]["timestamp_tolerance_seconds"])
    allow_reverse = bool(config["mapping"]["allow_reverse_flow_match"])
    lookup = _candidate_index(labels)
    records: list[dict[str, Any]] = []
    for flow in flows.itertuples(index=False):
        candidates = _indexed_candidates(flow, labels, lookup, allow_reverse)
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
    mapped = pd.DataFrame(records)
    mapped.attrs["mapping_audit"] = mapping_compatibility_report(
        features, labels, mapped, offset_evidence
    )
    return mapped


def attach_flow_labels(features: pd.DataFrame, mapped_flows: pd.DataFrame) -> pd.DataFrame:
    """Attach the independently audited flow label to every extracted PCAP record."""
    columns = [
        "flow_id",
        "label",
        "match_status",
        "match_confidence",
        "csv_row",
        "label_source_file",
    ]
    available = [column for column in columns if column in mapped_flows.columns]
    labelled = features.merge(
        mapped_flows[available], on="flow_id", how="left", validate="many_to_one"
    )
    labelled["label"] = labelled.get(
        "label", pd.Series(index=labelled.index, dtype="string")
    ).fillna("unknown")
    labelled["is_attack"] = labelled["label"].astype("string").str.casefold().isin(
        {"attack", "anomaly", "malicious"}
    ) | ~labelled["label"].astype("string").str.casefold().isin(
        {"unknown", "benign", "normal", "-1", "nan", "<na>"}
    )
    return labelled


def attach_packet_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any],
    *,
    offset_seconds: float = 0.0,
) -> pd.DataFrame:
    """Attach CSV evidence at packet time, retaining flow-compatible endpoints.

    A capture can contain a long-lived TCP flow that transitions from benign
    traffic into an attack. Mapping a single flow-start label to every packet
    would erase that transition. This function uses the selected CSV, its
    audited clock offset, and the existing endpoint matching rules to label
    each PCAP-derived packet at its own timestamp.
    """
    result = features.copy()
    result["_packet_order"] = np.arange(len(result), dtype=int)
    aligned = labels.copy()
    aligned["label_timestamp"] = pd.to_datetime(
        aligned["label_timestamp"], errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]") + pd.to_timedelta(offset_seconds, unit="s")
    aligned["label_end_timestamp"] = pd.to_datetime(
        aligned["label_end_timestamp"], errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]") + pd.to_timedelta(offset_seconds, unit="s")
    lookup = _candidate_index(aligned)
    tolerance = float(config["mapping"]["timestamp_tolerance_seconds"])
    allow_reverse = bool(config["mapping"].get("allow_reverse_flow_match", True))
    label_columns = [
        "label",
        "label_timestamp",
        "label_end_timestamp",
        "csv_row",
        "source_file",
    ]
    attached: list[pd.DataFrame] = []

    for _, flow_packets in result.groupby("flow_id", sort=False, dropna=False):
        scoped = flow_packets.sort_values("timestamp", kind="stable").copy()
        flow = scoped.iloc[0]
        candidates = _indexed_candidates(flow, aligned, lookup, allow_reverse)[label_columns].copy()
        default = pd.DataFrame(
            {
                "label": config["mapping"]["default_label"],
                "match_status": "unmatched",
                "match_confidence": 0.0,
                "csv_row": pd.NA,
                "label_source_file": pd.NA,
                "candidate_count": len(candidates),
            },
            index=scoped.index,
        )
        if candidates.empty:
            attached.append(pd.concat([scoped, default], axis=1))
            continue

        candidates = candidates.dropna(subset=["label_timestamp"]).sort_values(
            "label_timestamp", kind="stable"
        )
        if candidates.empty:
            selected = _indexed_candidates(flow, aligned, lookup, allow_reverse).iloc[0]
            default["label"] = selected["label"]
            default["match_status"] = "matched_without_timestamp"
            default["match_confidence"] = 0.7
            default["csv_row"] = selected["csv_row"]
            default["label_source_file"] = selected["source_file"]
            attached.append(pd.concat([scoped, default], axis=1))
            continue

        packet_times = pd.DataFrame(
            {
                "_packet_index": scoped.index,
                "_packet_time": pd.to_datetime(
                    scoped["timestamp"], errors="coerce", utc=True
                ).astype("datetime64[ns, UTC]"),
            },
            index=scoped.index,
        ).sort_values("_packet_time", kind="stable")
        candidate_times = candidates.rename(columns={"label_timestamp": "_label_time"})
        prior = pd.merge_asof(
            packet_times,
            candidate_times,
            left_on="_packet_time",
            right_on="_label_time",
            direction="backward",
            allow_exact_matches=True,
        ).set_index("_packet_index")
        interval_match = prior["label_end_timestamp"].notna() & (
            prior["_packet_time"] <= prior["label_end_timestamp"]
        )
        nearest = pd.merge_asof(
            packet_times,
            candidate_times,
            left_on="_packet_time",
            right_on="_label_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance),
            allow_exact_matches=True,
        ).set_index("_packet_index")
        selected = prior.where(interval_match, nearest).reindex(scoped.index)
        matched = selected["label"].notna()
        interval_for_rows = interval_match.reindex(scoped.index).fillna(False)
        default.loc[matched, "label"] = selected.loc[matched, "label"].astype("string")
        default.loc[matched & interval_for_rows, "match_status"] = "matched_interval"
        nearest_match = matched & ~interval_for_rows
        default.loc[nearest_match, "match_status"] = "matched_packet_time"
        default.loc[matched & interval_for_rows, "match_confidence"] = 0.98
        distance = (selected["_packet_time"] - selected["_label_time"]).abs().dt.total_seconds()
        default.loc[nearest_match, "match_confidence"] = (
            0.95
            - distance.loc[nearest_match].clip(lower=0, upper=tolerance) / max(tolerance, 1) * 0.2
        ).clip(lower=0.6)
        default.loc[matched, "csv_row"] = selected.loc[matched, "csv_row"].to_numpy()
        default.loc[matched, "label_source_file"] = selected.loc[matched, "source_file"].to_numpy()
        attached.append(pd.concat([scoped, default], axis=1))

    labelled = pd.concat(attached, ignore_index=False, sort=False)
    labelled = labelled.sort_values("_packet_order", kind="stable").drop(columns="_packet_order")
    labelled["is_attack"] = labelled["label"].astype("string").str.casefold().isin(
        {"attack", "anomaly", "malicious"}
    ) | ~labelled["label"].astype("string").str.casefold().isin(
        {"unknown", "benign", "normal", "-1", "nan", "<na>"}
    )
    return labelled.reset_index(drop=True)


def _canonical_labelled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Stabilise CSV-derived columns across independently processed row groups."""

    result = frame.copy()
    for column in ["label", "match_status", "label_source_file", "dataset_split"]:
        if column in result:
            result[column] = result[column].astype("string")
    for column in ["match_confidence", "csv_row", "candidate_count"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    for column in ["is_attack", "mapping_accepted"]:
        if column in result:
            result[column] = result[column].fillna(False).astype(bool)
    return result


def attach_packet_labels_parquet(
    feature_path: Path,
    output_path: Path,
    labels: pd.DataFrame,
    config: dict[str, Any],
    *,
    offset_seconds: float = 0.0,
    mapping_accepted: bool,
    batch_rows: int = 50_000,
    batch_callback: Callable[[pd.DataFrame], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Label a large PCAP-derived Parquet table without loading it into RAM.

    Packet-time labels are independent for each packet once the CSV clock offset
    and endpoint lookup have been audited.  Processing row groups independently
    therefore preserves label correctness for long flows while keeping memory
    proportional to ``batch_rows``.
    """

    if batch_rows < 1:
        raise ValueError("batch_rows must be positive.")
    source = pq.ParquetFile(feature_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    attack_records = 0
    matched_records = 0
    batches = 0
    label_counts: Counter[tuple[str, bool]] = Counter()
    try:
        for batch in source.iter_batches(batch_size=batch_rows):
            features = batch.to_pandas()
            labelled = attach_packet_labels(features, labels, config, offset_seconds=offset_seconds)
            labelled["mapping_accepted"] = mapping_accepted
            labelled["dataset_split"] = "attack"
            labelled = _canonical_labelled_frame(labelled)
            table = pa.Table.from_pandas(labelled, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path, table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table, row_group_size=batch_rows)
            rows += len(labelled)
            attack_records += int(labelled["is_attack"].sum())
            matched_records += int(labelled["match_status"].ne("unmatched").sum())
            batches += 1
            label_counts.update(
                (str(label), bool(is_attack))
                for label, is_attack in labelled[["label", "is_attack"]].itertuples(
                    index=False, name=None
                )
            )
            if batch_callback is not None:
                batch_callback(labelled)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "packet_mapping_progress",
                        "rows_processed": rows,
                        "attack_records": attack_records,
                        "matched_records": matched_records,
                        "batches_completed": batches,
                        "batch_rows": batch_rows,
                    }
                )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        # A valid empty source is still a valid labelled artifact.  It has no
        # rows, so a direct empty copy is safer than inventing a schema.
        empty = source.read().to_pandas().iloc[0:0]
        empty["label"] = pd.Series(dtype="string")
        empty["is_attack"] = pd.Series(dtype=bool)
        empty["mapping_accepted"] = pd.Series(dtype=bool)
        empty["dataset_split"] = pd.Series(dtype="string")
        pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), output_path)
    return {
        "rows": rows,
        "attack_records": attack_records,
        "matched_records": matched_records,
        "batches_completed": batches,
        "label_counts": [
            {"label": label, "is_attack": is_attack, "records": records}
            for (label, is_attack), records in sorted(label_counts.items())
        ],
        "batch_rows": batch_rows,
    }


def set_mapping_acceptance_parquet(
    path: Path, accepted: bool, *, batch_rows: int = 50_000
) -> None:
    """Correct the decision bit without repeating packet-to-CSV matching."""

    source = pq.ParquetFile(path)
    temporary = path.with_suffix(".acceptance-rewrite.parquet")
    writer: pq.ParquetWriter | None = None
    try:
        for batch in source.iter_batches(batch_size=batch_rows):
            frame = batch.to_pandas()
            frame["mapping_accepted"] = bool(accepted)
            frame = _canonical_labelled_frame(frame)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table, row_group_size=batch_rows)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Cannot update mapping acceptance on an empty labelled table.")
    temporary.replace(path)


def packet_label_coverage_parquet(
    feature_path: Path,
    labels: pd.DataFrame,
    config: dict[str, Any],
    *,
    offset_seconds: float = 0.0,
    batch_rows: int = 50_000,
) -> dict[str, Any]:
    """Verify packet-time label coverage before marking a streamed mapping usable."""

    if batch_rows < 1:
        raise ValueError("batch_rows must be positive.")
    source = pq.ParquetFile(feature_path)
    rows = 0
    attack_records = 0
    label_counts: Counter[tuple[str, bool]] = Counter()
    for batch in source.iter_batches(batch_size=batch_rows):
        labelled = attach_packet_labels(
            batch.to_pandas(), labels, config, offset_seconds=offset_seconds
        )
        rows += len(labelled)
        attack_records += int(labelled["is_attack"].sum())
        label_counts.update(
            (str(label), bool(is_attack))
            for label, is_attack in labelled[["label", "is_attack"]].itertuples(
                index=False, name=None
            )
        )
    return {
        "rows": rows,
        "attack_records": attack_records,
        "label_counts": [
            {"label": label, "is_attack": is_attack, "records": records}
            for (label, is_attack), records in sorted(label_counts.items())
        ],
        "batch_rows": batch_rows,
    }


def mapping_cache_plan(
    config: dict[str, Any],
    *,
    protocol: str,
    capture_path: Path,
    label_paths: list[Path],
    max_packets: int | None,
) -> dict[str, Any]:
    """Describe a reusable packet-to-CSV mapping cache entry.

    Cache validity is deliberately based on portable operational inputs rather
    than a run directory: source names, size, modification time, all CSV
    candidates, mapping policy, protocol, and selected packet limit. A copied
    project therefore can reuse the relation when those source snapshots are
    preserved. Adding/changing a PCAP or CSV produces a different key
    automatically. ``--remap`` can still rebuild an unchanged entry explicitly.
    """

    mapping = config.get("mapping", {})

    def snapshot(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "name": path.name,
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }

    # Cache controls must not invalidate an otherwise identical mapping.
    policy = {
        key: value
        for key, value in mapping.items()
        if key
        not in {
            "cache_enabled",
            "cache_directory",
            "cache_schema_version",
        }
    }
    identity = {
        "schema_version": str(mapping.get("cache_schema_version", MAPPING_CACHE_SCHEMA_VERSION)),
        "packet_identity": "capture_name_plus_original_packet_index/v1",
        "protocol": protocol.casefold(),
        "capture": snapshot(capture_path),
        "label_candidates": [snapshot(path) for path in sorted(label_paths)],
        "max_packets_per_capture": max_packets,
        "mapping_policy": policy,
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    signature = sha256(canonical.encode("utf-8")).hexdigest()[:24]
    root = (
        Path(config["project"]["artifact_dir"])
        / str(mapping.get("cache_directory", "mapping_cache"))
        / f"v{MAPPING_CACHE_SCHEMA_VERSION.split('.', maxsplit=1)[0]}"
        / protocol.casefold()
        / signature
    )
    return {
        "enabled": bool(mapping.get("cache_enabled", True)),
        "signature": signature,
        "root": root,
        "identity": identity,
        "metadata_path": root / "mapping-cache.json",
        "flow_path": root / "flow-evidence.parquet",
        "links_path": root / "packet-csv-links.parquet",
    }


def load_mapping_cache(plan: dict[str, Any]) -> dict[str, Any] | None:
    """Load one valid cache entry, or return ``None`` without hiding a miss."""

    if not plan["enabled"]:
        return None
    metadata_path = Path(plan["metadata_path"])
    flow_path = Path(plan["flow_path"])
    links_path = Path(plan["links_path"])
    if not (metadata_path.is_file() and flow_path.is_file() and links_path.is_file()):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("MAPPING_CACHE_INVALID cache=%s reason=invalid_metadata", metadata_path)
        return None
    if (
        metadata.get("schema_version") != MAPPING_CACHE_SCHEMA_VERSION
        or metadata.get("signature") != plan["signature"]
    ):
        LOGGER.info("MAPPING_CACHE_STALE cache=%s", metadata_path)
        return None
    try:
        link_rows = int(pq.ParquetFile(links_path).metadata.num_rows)
        flow_evidence = read_table(flow_path)
    except Exception as error:  # pragma: no cover - corrupt external cache.
        LOGGER.warning("MAPPING_CACHE_INVALID cache=%s reason=%s", metadata_path, error)
        return None
    metadata["packet_link_rows"] = link_rows
    return {"metadata": metadata, "flow_evidence": flow_evidence, "links_path": links_path}


def _packet_link_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a portable, feature-independent relation from one packet to one CSV row."""

    missing = set(PACKET_LINK_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Labelled packet output is missing cache-link fields: {sorted(missing)}")
    links = _canonical_labelled_frame(frame[PACKET_LINK_COLUMNS])
    links["packet_uid"] = links["packet_uid"].astype("string")
    links["capture"] = links["capture"].astype("string")
    links["flow_id"] = links["flow_id"].astype("string")
    links["packet_index"] = pd.to_numeric(links["packet_index"], errors="coerce").astype(
        "float64"
    )
    links["timestamp"] = pd.to_datetime(links["timestamp"], errors="coerce", utc=True)
    return links


def save_mapping_cache(
    plan: dict[str, Any],
    *,
    flow_evidence: pd.DataFrame,
    labelled_path: Path,
    metadata: dict[str, Any],
    batch_rows: int = 50_000,
) -> dict[str, Any]:
    """Persist full packet↔CSV links once for later run- and model-independent reuse."""

    if not plan["enabled"]:
        return {"saved": False, "rows": 0}
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive.")
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=True)
    write_table(flow_evidence, Path(plan["flow_path"]))
    writer: pq.ParquetWriter | None = None
    rows = 0
    source = pq.ParquetFile(labelled_path)
    try:
        for batch in source.iter_batches(batch_size=batch_rows):
            links = _packet_link_frame(batch.to_pandas())
            table = pa.Table.from_pandas(links, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    Path(plan["links_path"]), table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table, row_group_size=batch_rows)
            rows += len(links)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Cannot cache an empty labelled packet table.")
    payload = {
        "schema_version": MAPPING_CACHE_SCHEMA_VERSION,
        "signature": plan["signature"],
        "created_at": datetime.now(UTC).isoformat(),
        "identity": plan["identity"],
        "artifacts": {
            "flow_evidence": Path(plan["flow_path"]).name,
            "packet_csv_links": Path(plan["links_path"]).name,
        },
        "packet_link_rows": rows,
        **metadata,
    }
    write_json(payload, Path(plan["metadata_path"]))
    LOGGER.info(
        "MAPPING_CACHE_SAVED protocol=%s signature=%s packet_links=%s path=%s",
        plan["identity"]["protocol"],
        plan["signature"],
        rows,
        root,
    )
    return {"saved": True, "rows": rows, "root": str(root)}


def apply_cached_packet_labels_parquet(
    feature_path: Path,
    links_path: Path,
    output_path: Path,
    *,
    batch_rows: int = 50_000,
    batch_callback: Callable[[pd.DataFrame], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Re-attach a validated cache relation without comparing any PCAP/CSV rows again.

    Feature extraction may complete worker batches in a different order on two
    runs.  ``packet_uid`` (capture plus original PCAP packet index) is the
    invariant relation key, so cache application deliberately joins by that
    key instead of assuming that two Parquet files have identical row order.
    """

    if batch_rows < 1:
        raise ValueError("batch_rows must be positive.")
    features = pq.ParquetFile(feature_path)
    cached_links = _packet_link_frame(read_table(links_path))
    if len(cached_links) != int(features.metadata.num_rows):
        raise MappingCacheMismatchError(
            "Cached packet links have a different row count than feature records."
        )
    if not cached_links["packet_uid"].is_unique:
        raise MappingCacheMismatchError("Cached packet links contain duplicate packet identifiers.")
    cached_links = cached_links.set_index("packet_uid", drop=False)
    remaining_packet_uids = set(cached_links.index.astype(str).tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    attack_records = 0
    matched_records = 0
    batches = 0
    label_counts: Counter[tuple[str, bool]] = Counter()
    feature_batches = features.iter_batches(batch_size=batch_rows)
    try:
        for feature_batch in feature_batches:
            feature_frame = feature_batch.to_pandas()
            packet_uids = feature_frame["packet_uid"].astype("string").astype(str).tolist()
            batch_packet_uids = set(packet_uids)
            if len(batch_packet_uids) != len(packet_uids) or not batch_packet_uids.issubset(
                remaining_packet_uids
            ):
                raise MappingCacheMismatchError(
                    "Cached links do not match this feature extraction; mapping will be rebuilt."
                )
            link_frame = cached_links.loc[packet_uids].reset_index(drop=True)
            remaining_packet_uids.difference_update(batch_packet_uids)
            mapping_fields = link_frame.drop(
                columns=["packet_uid", "capture", "packet_index", "timestamp", "flow_id"]
            )
            labelled = pd.concat(
                [feature_frame.reset_index(drop=True), mapping_fields.reset_index(drop=True)],
                axis=1,
            )
            labelled = _canonical_labelled_frame(labelled)
            table = pa.Table.from_pandas(labelled, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path, table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table, row_group_size=batch_rows)
            rows += len(labelled)
            attack_records += int(labelled["is_attack"].sum())
            matched_records += int(labelled["match_status"].ne("unmatched").sum())
            batches += 1
            label_counts.update(
                (str(label), bool(is_attack))
                for label, is_attack in labelled[["label", "is_attack"]].itertuples(
                    index=False, name=None
                )
            )
            if batch_callback is not None:
                batch_callback(labelled)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "mapping_cache_apply_progress",
                        "rows_processed": rows,
                        "attack_records": attack_records,
                        "cache_path": str(links_path),
                    }
                )
        if remaining_packet_uids:
            raise MappingCacheMismatchError(
                "Cached packet links contain identifiers absent from this feature extraction."
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Cannot apply a cache relation to an empty feature table.")
    LOGGER.info(
        "MAPPING_CACHE_APPLIED packet_links=%s rows=%s attack_records=%s",
        links_path,
        rows,
        attack_records,
    )
    return {
        "rows": rows,
        "attack_records": attack_records,
        "matched_records": matched_records,
        "batches_completed": batches,
        "label_counts": [
            {"label": label, "is_attack": is_attack, "records": records}
            for (label, is_attack), records in sorted(label_counts.items())
        ],
        "batch_rows": batch_rows,
    }


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
        "compatibility": mapped.attrs.get("mapping_audit", {}),
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
        "compatibility": mapped.attrs.get("mapping_audit", {}),
    }
    write_json(summary, output_path.with_suffix(".summary.json"))
    LOGGER.info("Mapped %s flows using existing feature evidence", len(mapped))
    return mapped, summary
