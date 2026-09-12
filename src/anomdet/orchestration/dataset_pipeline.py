"""Protocol-separated PCAP-first ingestion, evidence mapping, and analytical assets."""

from __future__ import annotations

import json
import logging
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from anomdet.analytics.reports import feature_overview
from anomdet.core.io import read_table, write_json, write_table
from anomdet.core.progress import DurableProgress
from anomdet.datasets.inventory import discover_dataset
from anomdet.features.extractor import extract_pcap_features
from anomdet.mapping.mapper import (
    MappingCacheMismatchError,
    apply_cached_packet_labels_parquet,
    attach_packet_labels,
    attach_packet_labels_parquet,
    load_mapping_cache,
    map_features_to_labels,
    mapping_cache_plan,
    normalize_label_csv,
    save_mapping_cache,
    set_mapping_acceptance_parquet,
)
from anomdet.selection.profiles import feature_quality_report
from anomdet.storage.artifacts import ArtifactStore, create_artifact_store

LOGGER = logging.getLogger("anomdet")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "capture"


def _domain(protocol: str) -> str:
    return "it" if protocol in {"dns", "http", "ssh"} else "ot"


def _attack_type_hint(label_path: Path, protocol: str) -> str:
    """Make a stable class fallback only for CSVs with no meaningful class field."""
    value = label_path.stem.casefold().replace(".pcap_flow", "")
    value = re.sub(r"(?:_|-)?\d{4}$", "", value)
    value = re.sub(r"(?:_|-)?\d+$", "", value)
    value = re.sub(r"^(schneider|siemens)(?:_|-)", "", value)
    value = re.sub(rf"(?:_|-){re.escape(protocol)}$", "", value)
    return value.strip("_-") or "unclassified_attack"


def _apply_attack_type_fallback(labels: pd.DataFrame, hint: str) -> pd.DataFrame:
    """Use a filename-derived class only when the CSV provides no class information."""
    result = labels.copy()
    invalid = {"", "-1", "needmanuallabel", "unknown", "nan", "<na>"}
    text = result["label"].astype("string").str.strip()
    meaningful = text.notna() & ~text.str.casefold().isin(invalid | {"benign", "normal"})
    if not meaningful.any():
        result["label"] = hint
        result.attrs["label_fallback"] = {"applied": True, "attack_type_hint": hint}
    else:
        result.attrs["label_fallback"] = {"applied": False, "attack_type_hint": hint}
    return result


def _mapping_candidates(pair: dict[str, Any], label_root: Path) -> list[Path]:
    primary = Path(str(pair["labels"]))
    if pair.get("pairing_method") != "needs_mapping_evidence":
        return [primary]
    candidates = sorted(label_root.glob("*.csv"), key=lambda item: item.name.casefold())
    return [primary, *[item for item in candidates if item != primary]]


def _best_mapping(
    features: pd.DataFrame,
    pair: dict[str, Any],
    label_root: Path,
    protocol: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Path, list[dict[str, Any]]]:
    """Evaluate uncertain filename pairs by endpoint/time evidence and retain the best audit."""
    candidates: list[dict[str, Any]] = []
    choices: list[tuple[tuple[float, float, float], pd.DataFrame, pd.DataFrame, Path]] = []
    for label_path in _mapping_candidates(pair, label_root):
        labels = _apply_attack_type_fallback(
            normalize_label_csv(label_path, _domain(protocol), config),
            _attack_type_hint(label_path, protocol),
        )
        mapped = map_features_to_labels(features, labels, config)
        audit = mapped.attrs.get("mapping_audit", {})
        match_rate = float(audit.get("match_rate", 0.0))
        confidence = float(audit.get("mean_match_confidence", 0.0))
        offset_support = float(
            audit.get("time_offset", {}).get("selected", {}).get("supported_flows", 0)
        )
        candidates.append(
            {
                "labels": str(label_path),
                "match_rate": match_rate,
                "mean_match_confidence": confidence,
                "offset_supported_flows": int(offset_support),
                "selected_offset_seconds": audit.get("time_offset", {})
                .get("selected", {})
                .get("offset_seconds"),
            }
        )
        choices.append(((match_rate, confidence, offset_support), mapped, labels, label_path))
    _, mapped, labels, label_path = max(choices, key=lambda item: item[0])
    mapped.attrs["mapping_audit"]["candidate_evaluations"] = candidates
    return mapped, labels, label_path, candidates


def _accept_mapping(mapped: pd.DataFrame, config: dict[str, Any]) -> bool:
    audit = mapped.attrs.get("mapping_audit", {})
    mapping = config["mapping"]
    return bool(
        audit.get("match_rate", 0.0) >= float(mapping["minimum_match_rate"])
        and audit.get("mean_match_confidence", 0.0) >= float(mapping["minimum_match_confidence"])
    )


def _register_extraction(
    store: ArtifactStore, target: Path, protocol: str, split: str, manifest: dict[str, Any]
) -> None:
    store.register_existing(
        target,
        kind="feature_records",
        protocol=protocol,
        split=split,
        description="PCAP-derived, protocol-separated feature records.",
        rows=int(manifest["rows"]),
        columns=len(manifest["columns"]),
    )
    store.register_existing(
        target.with_suffix(".manifest.json"),
        kind="feature_manifest",
        protocol=protocol,
        split=split,
        description="Feature extraction provenance and schema.",
    )


def _run_id() -> str:
    return datetime.now(UTC).strftime("dataset-%Y%m%dT%H%M%SZ")


def run_dataset_pipeline(
    config: dict[str, Any],
    output: Path | None = None,
    max_packets: int | None = None,
    *,
    allow_unbounded_streaming: bool = False,
    force_remap: bool = False,
    build_mapping: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Create a protocol-first extraction run, optionally followed by CSV mapping.

    The PCAP is always the source of features.  CSVs are only normalized and
    joined as label evidence; no CSV feature column enters the model dataset.
    """
    if allow_unbounded_streaming:
        config = {
            **config,
            "capture": {**config.get("capture", {}), "streaming_unbounded": True},
        }
    configured_cap = config.get("capture", {}).get("dataset_default_packet_cap")
    effective_max_packets = max_packets if max_packets is not None else configured_cap
    unbounded_streaming = bool(config.get("capture", {}).get("streaming_unbounded", False))
    if effective_max_packets is None and not unbounded_streaming:
        raise ValueError(
            "An unbounded dataset run is disabled because feature aggregation can exhaust memory. "
            "Set capture.dataset_default_packet_cap, pass --max-packets N, or use --all-packets "
            "with the disk-backed streaming extractor."
        )
    if effective_max_packets is not None:
        try:
            effective_max_packets = int(effective_max_packets)
        except (TypeError, ValueError) as error:
            raise ValueError("The dataset packet cap must be a positive integer.") from error
        if effective_max_packets < 1:
            raise ValueError("The dataset packet cap must be a positive integer.")
    cap_origin = (
        "all_packets_streaming"
        if effective_max_packets is None
        else "command_line"
        if max_packets is not None
        else "configuration_default"
    )
    LOGGER.info(
        "DATASET_PACKET_CAP packets_per_capture=%s source=%s",
        effective_max_packets if effective_max_packets is not None else "unlimited",
        cap_origin,
    )
    base = output or (Path(config["project"]["artifact_dir"]) / "runs" / _run_id())
    base.mkdir(parents=True, exist_ok=True)
    progress = DurableProgress(base / "dataset-progress.json", "dataset")
    store = create_artifact_store(config, base)
    inventory = discover_dataset(config)
    store.json(
        "manifests/input-inventory.json",
        inventory,
        kind="dataset_inventory",
        description="Discovered PCAP and CSV inputs.",
    )
    data = config["data"]
    label_root = Path(data["dataset_root"]) / str(data["attack_label_dir"])
    mapping_rows: list[dict[str, Any]] = []
    label_distribution_rows: list[dict[str, Any]] = []
    extracted = {"benign": 0, "attack": 0}

    benign_by_protocol: dict[str, list[Path]] = {}
    for source in inventory["benign"]:
        benign_by_protocol.setdefault(source["protocol"], []).append(Path(source["path"]))
    total_units = len(benign_by_protocol) + len(inventory["attack_pairs"])
    completed_units = 0
    cap_label = (
        f"{effective_max_packets:,}-packet cap"
        if effective_max_packets is not None
        else "unlimited disk-backed streaming"
    )
    progress.emit(
        "inventory_discovered",
        stage="inventory",
        message=(
            f"Discovered {len(benign_by_protocol)} benign protocol groups and "
            f"{len(inventory['attack_pairs'])} attack captures for processing "
            f"with {cap_label} per capture."
        ),
        protocol_count=len(benign_by_protocol),
        attack_capture_count=len(inventory["attack_pairs"]),
        packets_per_capture=effective_max_packets,
        packet_cap_source=cap_origin,
        work_units_total=total_units,
        progress_ratio=0.0,
    )

    def extraction_update(
        update: dict[str, Any], *, protocol: str, split: str, work_done: int
    ) -> None:
        """Bridge packet-level extractor events into the durable dataset monitor."""
        payload = dict(update)
        event = str(payload.pop("event", "extractor_update"))
        capture_name = str(payload.get("capture_name", ""))
        packet_count = payload.get("packets_read")
        packet_selected = payload.get("packets_selected")
        packet_limit = payload.get("packet_limit")
        source_total = payload.get("source_packets_total")
        if event == "capture_progress":
            if packet_limit:
                message = (
                    f"{protocol}/{split} · {capture_name} · processed "
                    f"{packet_selected}/{packet_limit} sampled packets from "
                    f"{source_total or packet_count} source packets."
                )
            else:
                message = f"{protocol}/{split} · {capture_name} · read {packet_count} packets."
        elif event == "capture_sampling_progress":
            message = (
                f"{protocol}/{split} · {capture_name} · scanned {packet_count} source packets; "
                f"selected {packet_selected}/{packet_limit} for the smoke test."
            )
        elif event == "capture_completed":
            message = f"{protocol}/{split} · {capture_name} completed."
        else:
            message = f"{protocol}/{split} · {event}"
        progress.emit(
            event,
            stage="feature_extraction",
            message=message,
            protocol=protocol,
            split=split,
            work_units_completed=work_done,
            work_units_total=total_units,
            overall_progress_ratio=work_done / max(total_units, 1),
            **payload,
        )

    def mapping_update(
        update: dict[str, Any], *, protocol: str, capture_name: str, work_done: int
    ) -> None:
        """Expose exhaustive packet-label mapping progress in CLI logs and dashboard."""

        payload = dict(update)
        event = str(payload.pop("event", "packet_mapping_update"))
        rows = int(payload.get("rows_processed", 0))
        attack_rows = int(payload.get("attack_records", 0))
        if event == "packet_mapping_progress":
            message = (
                f"{protocol}/attack · {capture_name} · checked {rows:,} PCAP packets "
                f"against endpoint/time-compatible CSV records; attack rows={attack_rows:,}."
            )
        elif event == "mapping_cache_apply_progress":
            message = (
                f"{protocol}/attack · {capture_name} · reapplied validated packet↔CSV links "
                f"to {rows:,} packets; no CSV comparison was repeated."
            )
        else:
            message = f"{protocol}/attack · {capture_name} · {event}"
        progress.emit(
            event,
            stage="label_mapping",
            message=message,
            protocol=protocol,
            split="attack",
            capture=capture_name,
            work_units_completed=work_done,
            work_units_total=total_units,
            overall_progress_ratio=work_done / max(total_units, 1),
            **payload,
        )

    for protocol, sources in sorted(benign_by_protocol.items()):
        # Passing the protocol folder lets the extractor compute behavioural
        # features across all benign captures while retaining a separate file per protocol.
        target = base / "features" / "benign" / protocol / "records.parquet"
        progress.emit(
            "protocol_extraction_started",
            stage="feature_extraction",
            message=f"Started benign feature extraction for {protocol}.",
            protocol=protocol,
            split="benign",
            work_units_completed=completed_units,
            work_units_total=total_units,
            progress_ratio=completed_units / max(total_units, 1),
        )
        features, manifest = extract_pcap_features(
            sources[0].parent,
            target,
            config,
            effective_max_packets,
            expected_protocol=protocol,
            progress_callback=(
                lambda update, protocol=protocol, work_done=completed_units: extraction_update(
                    update, protocol=protocol, split="benign", work_done=work_done
                )
            ),
            capture_paths=sources,
            batch_callback=lambda batch, protocol=protocol: store.mirror.write(
                f"feature_records_benign_{protocol}", batch
            ),
        )
        _register_extraction(store, target, protocol, "benign", manifest)
        if manifest.get("extraction_mode") != "streaming_causal":
            store.mirror.write(f"feature_records_benign_{protocol}", features)
        quality_path = base / "reports" / "features" / "benign" / f"{protocol}-quality.parquet"
        quality = feature_quality_report(target, quality_path)
        store.register_existing(
            quality_path,
            kind="feature_quality",
            protocol=protocol,
            split="benign",
            description="Protocol-specific feature quality.",
            rows=len(quality),
            columns=len(quality.columns),
        )
        overview = feature_overview(features, protocol)
        store.table(
            "reports/features/benign/" + f"{protocol}-overview.parquet",
            overview,
            kind="feature_overview",
            protocol=protocol,
            split="benign",
            description="Chart-ready distribution statistics.",
        )
        extracted["benign"] += 1
        completed_units += 1
        progress.emit(
            "protocol_extraction_completed",
            stage="feature_analysis",
            message=f"Saved benign features, quality report, and overview for {protocol}.",
            protocol=protocol,
            split="benign",
            rows=len(features),
            flows=manifest["flow_count"],
            work_units_completed=completed_units,
            work_units_total=total_units,
            progress_ratio=completed_units / max(total_units, 1),
        )

    for pair in inventory["attack_pairs"]:
        protocol, capture = str(pair["protocol"]), Path(str(pair["capture"]))
        capture_id = _slug(capture.stem)
        feature_target = base / "features" / "attack" / protocol / capture_id / "records.parquet"
        progress.emit(
            "attack_capture_started",
            stage="feature_extraction",
            message=f"Started attack-capture feature extraction for {protocol}: {capture.name}.",
            protocol=protocol,
            split="attack",
            capture=capture.name,
            work_units_completed=completed_units,
            work_units_total=total_units,
            progress_ratio=completed_units / max(total_units, 1),
        )
        features, manifest = extract_pcap_features(
            capture,
            feature_target,
            config,
            effective_max_packets,
            expected_protocol=protocol,
            progress_callback=(
                lambda update, protocol=protocol, work_done=completed_units: extraction_update(
                    update, protocol=protocol, split="attack", work_done=work_done
                )
            ),
            batch_callback=lambda batch, protocol=protocol: store.mirror.write(
                f"feature_records_attack_{protocol}", batch
            ),
        )
        _register_extraction(store, feature_target, protocol, "attack", manifest)
        if manifest.get("extraction_mode") != "streaming_causal":
            store.mirror.write(f"feature_records_attack_{protocol}", features)
        # The streaming extractor deliberately returns a bounded, uniform
        # reservoir.  Persist that same evidence separately so `dataset map`
        # can be run later without reopening the PCAP or loading every feature
        # record into memory merely to choose the CSV/clock-offset evidence.
        evidence_path = feature_target.with_name("mapping-evidence.parquet")
        store.table(
            evidence_path.relative_to(base),
            features,
            kind="mapping_feature_evidence",
            protocol=protocol,
            split="attack",
            description=(
                "PCAP-derived feature evidence used only to select and audit a CSV mapping; "
                "the later packet mapper still checks every stored packet record."
            ),
        )
        if not build_mapping:
            overview = feature_overview(features, protocol)
            store.table(
                "reports/features/attack/" + protocol + f"/{capture_id}-overview.parquet",
                overview,
                kind="feature_overview",
                protocol=protocol,
                split="attack",
                description="Chart-ready attack-capture feature statistics.",
            )
            extracted["attack"] += 1
            completed_units += 1
            progress.emit(
                "attack_capture_extraction_completed",
                stage="feature_analysis",
                message=(
                    f"{capture.name}: saved PCAP-derived features and mapping evidence; "
                    "CSV mapping is pending."
                ),
                protocol=protocol,
                split="attack",
                capture=capture.name,
                rows=int(manifest["rows"]),
                work_units_completed=completed_units,
                work_units_total=total_units,
                progress_ratio=completed_units / max(total_units, 1),
            )
            continue
        progress.emit(
            "mapping_started",
            stage="label_mapping",
            message=f"Started PCAP-to-CSV mapping for {capture.name}.",
            protocol=protocol,
            split="attack",
            capture=capture.name,
            rows=len(features),
            work_units_completed=completed_units,
            work_units_total=total_units,
        )
        candidate_paths = _mapping_candidates(pair, label_root / protocol)
        cache_plan = mapping_cache_plan(
            config,
            protocol=protocol,
            capture_path=capture,
            label_paths=candidate_paths,
            max_packets=effective_max_packets,
        )
        cache_entry = None if force_remap else load_mapping_cache(cache_plan)
        cache_reused = cache_entry is not None
        if cache_entry is not None:
            cache_metadata = cache_entry["metadata"]
            mapped = cache_entry["flow_evidence"]
            mapped.attrs["mapping_audit"] = dict(cache_metadata.get("mapping_audit", {}))
            selected_csv = Path(str(cache_metadata["selected_csv"]))
            candidates = list(cache_metadata.get("candidate_evaluations", []))
            mapping_evidence_accepted = bool(cache_metadata["mapping_evidence_accepted"])
            label_fallback = dict(cache_metadata.get("label_fallback", {}))
            progress.emit(
                "mapping_cache_hit",
                stage="label_mapping",
                message=(
                    f"Reusing validated packet↔CSV mapping for {capture.name}; "
                    "CSV comparison is skipped because inputs and policy are unchanged."
                ),
                protocol=protocol,
                split="attack",
                capture=capture.name,
                mapping_cache_signature=cache_plan["signature"],
                packet_link_rows=cache_metadata.get("packet_link_rows"),
            )
        else:
            progress.emit(
                "mapping_cache_miss",
                stage="label_mapping",
                message=(
                    f"Building an exhaustive packet↔CSV mapping for {capture.name}; "
                    "this is required for a new or changed source."
                ),
                protocol=protocol,
                split="attack",
                capture=capture.name,
                mapping_cache_signature=cache_plan["signature"],
                forced=force_remap,
            )
            mapped, labels, selected_csv, candidates = _best_mapping(
                features, pair, label_root / protocol, protocol, config
            )
            mapping_evidence_accepted = _accept_mapping(mapped, config)
            label_fallback = dict(labels.attrs.get("label_fallback", {}))
        mapping_path = base / "mappings" / protocol / f"{capture_id}.parquet"
        store.table(
            mapping_path.relative_to(base),
            mapped,
            kind="flow_label_mapping",
            protocol=protocol,
            split="attack",
            description="Flow-to-CSV mapping with confidence and timestamp-offset evidence.",
        )
        mapping_audit = mapped.attrs.get("mapping_audit", {})
        packet_offset = float(
            mapping_audit
            .get("time_offset", {})
            .get("selected", {})
            .get("offset_seconds", 0.0)
        )
        is_streaming = manifest.get("extraction_mode") == "streaming_causal"
        labelled_target = "labelled/attack/" + protocol + f"/{capture_id}/records.parquet"
        labelled_path = base / labelled_target
        batch_rows = int(config.get("capture", {}).get("streaming_chunk_rows", 50_000))
        if cache_entry is not None:
            mapping_result = apply_cached_packet_labels_parquet(
                feature_target,
                Path(cache_entry["links_path"]),
                labelled_path,
                batch_rows=batch_rows,
                batch_callback=lambda batch, protocol=protocol: store.mirror.write(
                    f"labelled_feature_records_{protocol}", batch
                ),
                progress_callback=lambda update, protocol=protocol, work_done=completed_units,
                capture_name=capture.name: mapping_update(
                    update,
                    protocol=protocol,
                    capture_name=capture_name,
                    work_done=work_done,
                ),
            )
            packet_attack_records = int(mapping_result["attack_records"])
            labelled_rows = int(mapping_result["rows"])
            label_count_rows = list(mapping_result["label_counts"])
            accepted = bool(cache_metadata["accepted"])
            rejection_reason = cache_metadata.get("rejection_reason")
        elif is_streaming:
            mapping_result = attach_packet_labels_parquet(
                feature_target,
                labelled_path,
                labels,
                config,
                offset_seconds=packet_offset,
                mapping_accepted=mapping_evidence_accepted,
                batch_rows=batch_rows,
                progress_callback=lambda update, protocol=protocol, work_done=completed_units,
                capture_name=capture.name: mapping_update(
                    update,
                    protocol=protocol,
                    capture_name=capture_name,
                    work_done=work_done,
                ),
            )
            packet_attack_records = int(mapping_result["attack_records"])
            labelled_rows = int(mapping_result["rows"])
            label_count_rows = list(mapping_result["label_counts"])
            accepted = mapping_evidence_accepted and packet_attack_records > 0
            rejection_reason = (
                "no_packet_time_attack_label"
                if mapping_evidence_accepted and not packet_attack_records
                else None
            )
            if accepted != mapping_evidence_accepted:
                set_mapping_acceptance_parquet(labelled_path, accepted, batch_rows=batch_rows)
            for labelled_batch in pq.ParquetFile(labelled_path).iter_batches(batch_size=batch_rows):
                store.mirror.write(
                    f"labelled_feature_records_{protocol}", labelled_batch.to_pandas()
                )
        else:
            labelled = attach_packet_labels(features, labels, config, offset_seconds=packet_offset)
            packet_attack_records = int(labelled["is_attack"].sum())
            labelled_rows = len(labelled)
            label_count_rows = [
                {"label": str(label), "is_attack": bool(is_attack), "records": int(records)}
                for (label, is_attack), records in labelled.groupby(
                    ["label", "is_attack"], dropna=False
                )
                .size()
                .items()
            ]
            accepted = mapping_evidence_accepted and packet_attack_records > 0
            rejection_reason = (
                "no_packet_time_attack_label"
                if mapping_evidence_accepted and not packet_attack_records
                else None
            )
            labelled["mapping_accepted"] = accepted
            labelled["dataset_split"] = "attack"
            write_table(labelled, labelled_path)
            store.mirror.write(f"labelled_feature_records_{protocol}", labelled)
        if rejection_reason:
            LOGGER.warning(
                "MAPPING_REJECTED_NO_ATTACK_INTERVAL protocol=%s capture=%s "
                "matching evidence is valid, "
                "but no PCAP packet overlaps a labelled attack interval.",
                protocol,
                capture.name,
            )
        progress.emit(
            "packet_mapping_completed",
            stage="label_mapping",
            message=(
                f"{protocol}/attack · {capture.name} · completed packet↔CSV relation for "
                f"{labelled_rows:,} packets; attack rows={packet_attack_records:,}; "
                f"cache={'reused' if cache_reused else 'created'}."
            ),
            protocol=protocol,
            split="attack",
            capture=capture.name,
            packet_records=labelled_rows,
            attack_records=packet_attack_records,
            mapping_accepted=accepted,
            cache_reused=cache_reused,
            work_units_completed=completed_units,
            work_units_total=total_units,
        )
        LOGGER.info(
            "MAPPING_COMPLETE protocol=%s capture=%s accepted=%s evidence_accepted=%s "
            "attack_records=%s match_rate=%.3f confidence=%.3f",
            protocol,
            capture.name,
            accepted,
            mapping_evidence_accepted,
            packet_attack_records,
            float(mapping_audit.get("match_rate", 0.0)),
            float(mapping_audit.get("mean_match_confidence", 0.0)),
        )
        audit = {
            **mapping_audit,
            "protocol": protocol,
            "capture": str(capture),
            "selected_csv": str(selected_csv),
            "accepted": accepted,
            "mapping_evidence_accepted": mapping_evidence_accepted,
            "rejection_reason": rejection_reason,
            "candidate_evaluations": candidates,
            "label_fallback": label_fallback,
            "mapping_evidence_rows": len(features),
            "pcap_feature_rows_total": int(manifest["rows"]),
            "extraction_mode": manifest.get("extraction_mode", "in_memory"),
            "mapping_cache": {
                "status": (
                    "reused"
                    if cache_reused
                    else "created"
                    if cache_plan["enabled"]
                    else "disabled"
                ),
                "signature": cache_plan["signature"],
                "path": str(cache_plan["root"]),
            },
        }
        audit["packet_label_coverage"] = {
            "records": labelled_rows,
            "attack_records": packet_attack_records,
            "non_attack_records": int(labelled_rows - packet_attack_records),
            "label_assignment": "packet_timestamp_within_endpoint_compatible_csv_interval",
            "streamed_in_batches": is_streaming,
        }
        if not cache_reused:
            save_mapping_cache(
                cache_plan,
                flow_evidence=mapped,
                labelled_path=labelled_path,
                metadata={
                    "protocol": protocol,
                    "capture": str(capture),
                    "selected_csv": str(selected_csv),
                    "mapping_audit": mapping_audit,
                    "candidate_evaluations": candidates,
                    "label_fallback": label_fallback,
                    "accepted": accepted,
                    "mapping_evidence_accepted": mapping_evidence_accepted,
                    "rejection_reason": rejection_reason,
                    "packet_label_coverage": audit["packet_label_coverage"],
                },
                batch_rows=batch_rows,
            )
        mapping_rows.append(
            {
                "protocol": protocol,
                "capture": capture.name,
                "selected_csv": selected_csv.name,
                "accepted": accepted,
                "mapping_evidence_accepted": mapping_evidence_accepted,
                "rejection_reason": rejection_reason,
                "packet_attack_records": packet_attack_records,
                "mapping_cache_status": audit["mapping_cache"]["status"],
                "mapping_cache_signature": cache_plan["signature"],
                "mapping_cache_path": str(cache_plan["root"]),
                "pairing_method": pair["pairing_method"],
                "pairing_score": pair["pairing_score"],
                "match_rate": audit.get("match_rate", 0.0),
                "mean_match_confidence": audit.get("mean_match_confidence", 0.0),
                "selected_offset_seconds": audit.get("time_offset", {})
                .get("selected", {})
                .get("offset_seconds"),
            }
        )
        store.json(
            "mappings/" + protocol + f"/{capture_id}.audit.json",
            audit,
            kind="mapping_audit",
            description="Compatibility decision and packet-time CSV label coverage.",
        )
        label_distribution_rows.extend(
            {
                "protocol": protocol,
                "capture": capture.name,
                "label": str(label),
                "is_attack": bool(is_attack),
                "records": int(records),
                "mapping_accepted": accepted,
            }
            for label, is_attack, records in (
                (row["label"], row["is_attack"], row["records"]) for row in label_count_rows
            )
        )
        labelled_columns = len(pq.ParquetFile(labelled_path).schema_arrow)
        store.register_existing(
            labelled_path,
            kind="labelled_feature_records",
            protocol=protocol,
            split="attack",
            description=(
                "PCAP feature records joined to a versioned packet-to-CSV mapping relation."
            ),
            rows=labelled_rows,
            columns=labelled_columns,
        )
        overview = feature_overview(features, protocol)
        store.table(
            "reports/features/attack/" + protocol + f"/{capture_id}-overview.parquet",
            overview,
            kind="feature_overview",
            protocol=protocol,
            split="attack",
            description="Chart-ready attack-capture feature statistics.",
        )
        extracted["attack"] += 1
        completed_units += 1
        progress.emit(
            "attack_capture_completed",
            stage="label_mapping",
            message=(
                f"{capture.name}: mapping {'accepted' if accepted else 'requires review'}; "
                "saved labelled PCAP-derived records."
            ),
            protocol=protocol,
            split="attack",
            capture=capture.name,
            rows=len(features),
            mapping_accepted=accepted,
            match_rate=float(audit.get("match_rate", 0.0)),
            mean_match_confidence=float(audit.get("mean_match_confidence", 0.0)),
            work_units_completed=completed_units,
            work_units_total=total_units,
            progress_ratio=completed_units / max(total_units, 1),
        )

    mapping_columns = [
        "protocol",
        "capture",
        "selected_csv",
        "accepted",
        "mapping_evidence_accepted",
        "rejection_reason",
        "packet_attack_records",
        "mapping_cache_status",
        "mapping_cache_signature",
        "mapping_cache_path",
        "pairing_method",
        "pairing_score",
        "match_rate",
        "mean_match_confidence",
        "selected_offset_seconds",
    ]
    mapping_frame = pd.DataFrame(mapping_rows, columns=mapping_columns)
    store.table(
        "reports/mapping-audit.parquet",
        mapping_frame,
        kind="mapping_overview",
        description="One-row mapping decision per attack capture.",
    )
    store.table(
        "reports/attack-label-distribution.parquet",
        pd.DataFrame(
            label_distribution_rows,
            columns=["protocol", "capture", "label", "is_attack", "records", "mapping_accepted"],
        ),
        kind="attack_label_distribution",
        description="Mapped PCAP-record distribution by attack type for balance review.",
    )
    catalog = store.finalize()
    summary = {
        "output_root": str(base),
        "catalog": str(catalog),
        "inventory": inventory["counts"],
        "protocol_feature_files": extracted,
        "accepted_mappings": int(mapping_frame.get("accepted", pd.Series(dtype=bool)).sum()),
        "mapping_count": int(len(mapping_frame)),
        "max_packets_per_capture": effective_max_packets,
        "packet_cap_source": cap_origin,
        "mapping_status": "completed" if build_mapping else "not_started",
    }
    write_json(summary, base / "run-summary.json")
    progress.complete(
        (
            "Protocol-first extraction, CSV mapping, and analytical reports completed."
            if build_mapping
            else "Protocol-first feature extraction completed; CSV mapping has not been started."
        ),
        accepted_mappings=summary["accepted_mappings"],
        mapping_count=summary["mapping_count"],
        output_root=str(base),
    )
    return summary, base / "run-summary.json"


def run_dataset_extraction(
    config: dict[str, Any],
    output: Path | None = None,
    max_packets: int | None = None,
    *,
    allow_unbounded_streaming: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Extract PCAP features only, leaving the CSV mapping as an explicit next step."""

    return run_dataset_pipeline(
        config,
        output,
        max_packets,
        allow_unbounded_streaming=allow_unbounded_streaming,
        build_mapping=False,
    )


def _mapping_evidence_path(feature_path: Path) -> Path:
    """Return the bounded feature evidence persisted beside one attack feature table."""

    return feature_path.with_name("mapping-evidence.parquet")


def _load_mapping_evidence(feature_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load persisted mapping evidence, with a bounded legacy-run fallback.

    New `dataset extract` runs store a uniform extractor reservoir.  A legacy
    run may predate that small file; in that case scan only the Parquet feature
    table and retain a deterministic reservoir.  This does *not* label packet
    rows: the subsequent mapper still iterates every feature row in batches.
    """

    evidence_path = _mapping_evidence_path(feature_path)
    if evidence_path.is_file():
        return read_table(evidence_path)

    source = pq.ParquetFile(feature_path)
    required = [
        "flow_id",
        "timestamp",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "protocol",
        "packet_length",
    ]
    missing = set(required).difference(source.schema_arrow.names)
    if missing:
        raise ValueError(
            f"Feature file cannot be mapped because it lacks required fields: {sorted(missing)}"
        )
    sample_size = max(
        1_000, int(config.get("capture", {}).get("streaming_return_sample_rows", 100_000))
    )
    generator = random.Random(42)
    sample: list[dict[str, Any]] = []
    seen = 0
    for batch in source.iter_batches(columns=required, batch_size=50_000):
        for record in batch.to_pandas().to_dict("records"):
            seen += 1
            if len(sample) < sample_size:
                sample.append(record)
            else:
                replacement = generator.randrange(seen)
                if replacement < sample_size:
                    sample[replacement] = record
    if not sample:
        raise ValueError(f"Feature file is empty and cannot be mapped: {feature_path}")
    LOGGER.warning(
        "MAPPING_EVIDENCE_LEGACY_FALLBACK feature_path=%s sample_rows=%s source_rows=%s; "
        "run `dataset extract` again to persist mapping-evidence.parquet alongside features.",
        feature_path,
        len(sample),
        seen,
    )
    return pd.DataFrame(sample)


def _existing_artifact_store(config: dict[str, Any], base: Path) -> ArtifactStore:
    """Reopen an extraction run without discarding its existing artifact catalog."""

    store = create_artifact_store(config, base)
    catalog_path = base / "catalog" / "artifacts.parquet"
    if catalog_path.is_file():
        existing = read_table(catalog_path)
        store.catalog = existing.to_dict("records")
    return store


def _drop_catalog_paths(store: ArtifactStore, paths: set[str]) -> None:
    """Replace catalog entries for rerunnable mapping artifacts instead of duplicating them."""

    store.catalog = [entry for entry in store.catalog if str(entry.get("path")) not in paths]


def _replace_capture_rows(
    existing_path: Path,
    updates: pd.DataFrame,
    *,
    columns: list[str],
) -> pd.DataFrame:
    """Upsert report rows for captures that were just mapped, retaining other protocols."""

    if updates.empty:
        return (
            read_table(existing_path)
            if existing_path.is_file()
            else pd.DataFrame(columns=columns)
        )
    existing = (
        read_table(existing_path)
        if existing_path.is_file()
        else pd.DataFrame(columns=columns)
    )
    if not existing.empty and {"protocol", "capture"}.issubset(existing.columns):
        captures = set(
            updates[["protocol", "capture"]].astype("string").agg("\x1f".join, axis=1)
        )
        existing_keys = existing[["protocol", "capture"]].astype("string").agg("\x1f".join, axis=1)
        existing = existing.loc[~existing_keys.isin(captures)]
    merged = pd.concat([existing, updates], ignore_index=True, sort=False)
    return merged.reindex(columns=list(dict.fromkeys([*columns, *merged.columns])))


def run_dataset_mapping(
    config: dict[str, Any],
    dataset_run: Path,
    *,
    protocol: str | None = None,
    force_remap: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Map saved attack feature tables to CSV labels without reopening any PCAP.

    The command consumes `features/attack/.../records.parquet` created by
    `dataset extract`.  It creates the reusable cache relation once and writes
    `labelled/attack/.../records.parquet` for model training.  Re-running it
    with unchanged sources only reapplies the saved relation; it does not
    compare packet rows against CSV rows again.
    """

    base = Path(dataset_run).resolve()
    inventory_path = base / "manifests" / "input-inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"{base} is not a dataset extraction run: manifests/input-inventory.json is missing."
        )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    selected_protocol = protocol.casefold() if protocol else None
    pairs = [
        pair
        for pair in inventory.get("attack_pairs", [])
        if selected_protocol is None
        or str(pair.get("protocol", "")).casefold() == selected_protocol
    ]
    if not pairs:
        scope = selected_protocol or "the saved inventory"
        raise ValueError(f"No attack PCAP/CSV pairs are available for {scope}.")

    progress = DurableProgress(base / "mapping-progress.json", "mapping")
    store = _existing_artifact_store(config, base)
    mapping_rows: list[dict[str, Any]] = []
    label_distribution_rows: list[dict[str, Any]] = []
    total = len(pairs)
    mapping_columns = [
        "protocol",
        "capture",
        "selected_csv",
        "accepted",
        "mapping_evidence_accepted",
        "rejection_reason",
        "packet_attack_records",
        "mapping_cache_status",
        "mapping_cache_signature",
        "mapping_cache_path",
        "pairing_method",
        "pairing_score",
        "match_rate",
        "mean_match_confidence",
        "selected_offset_seconds",
    ]
    distribution_columns = [
        "protocol",
        "capture",
        "label",
        "is_attack",
        "records",
        "mapping_accepted",
    ]

    def mapping_update(
        update: dict[str, Any], *, current_protocol: str, capture_name: str, completed: int
    ) -> None:
        payload = dict(update)
        event = str(payload.pop("event", "packet_mapping_update"))
        rows = int(payload.get("rows_processed", 0))
        attacks = int(payload.get("attack_records", 0))
        if event == "packet_mapping_progress":
            message = (
                f"{current_protocol}/attack · {capture_name} · checked {rows:,} saved PCAP "
                "feature records against endpoint/time-compatible CSV records; "
                f"attack rows={attacks:,}."
            )
        elif event == "mapping_cache_apply_progress":
            message = (
                f"{current_protocol}/attack · {capture_name} · reapplied saved packet↔CSV links "
                f"to {rows:,} records; no CSV comparison was repeated."
            )
        else:
            message = f"{current_protocol}/attack · {capture_name} · {event}"
        progress.emit(
            event,
            stage="label_mapping",
            message=message,
            protocol=current_protocol,
            split="attack",
            capture=capture_name,
            work_units_completed=completed,
            work_units_total=total,
            overall_progress_ratio=completed / max(total, 1),
            **payload,
        )

    for completed, pair in enumerate(pairs):
        current_protocol = str(pair["protocol"]).casefold()
        capture = Path(str(pair["capture"]))
        capture_id = _slug(capture.stem)
        feature_path = (
            base / "features" / "attack" / current_protocol / capture_id / "records.parquet"
        )
        manifest_path = feature_path.with_suffix(".manifest.json")
        if not feature_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Saved attack features are missing for "
                f"{current_protocol}/{capture.name}. Run `anomaly dataset extract` first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evidence = _load_mapping_evidence(feature_path, config)
        candidate_paths = _mapping_candidates(pair, Path(str(pair["labels"])).parent)
        cache_plan = mapping_cache_plan(
            config,
            protocol=current_protocol,
            capture_path=capture,
            label_paths=candidate_paths,
            max_packets=manifest.get("max_packets_per_capture"),
        )
        cache_entry = None if force_remap else load_mapping_cache(cache_plan)
        cache_reused = cache_entry is not None

        def build_fresh_mapping(
            raw_evidence: pd.DataFrame = evidence,
            mapping_pair: dict[str, Any] = pair,
            protocol_name: str = current_protocol,
        ) -> tuple[pd.DataFrame, pd.DataFrame, Path, list[dict[str, Any]]]:
            """Build this capture's relation without using its packet-link cache."""

            return _best_mapping(
                raw_evidence,
                mapping_pair,
                Path(str(mapping_pair["labels"])).parent,
                protocol_name,
                config,
            )

        progress.emit(
            "mapping_cache_hit" if cache_reused else "mapping_cache_miss",
            stage="label_mapping",
            message=(
                f"Reusing the saved packet↔CSV relation for {capture.name}; "
                "no CSV comparison is needed."
                if cache_reused
                else f"Building the one-time exhaustive packet↔CSV relation for {capture.name}."
            ),
            protocol=current_protocol,
            split="attack",
            capture=capture.name,
            mapping_cache_signature=cache_plan["signature"],
            forced=force_remap,
            work_units_completed=completed,
            work_units_total=total,
            progress_ratio=completed / max(total, 1),
        )
        if cache_entry is not None:
            cache_metadata = cache_entry["metadata"]
            mapped = cache_entry["flow_evidence"]
            mapped.attrs["mapping_audit"] = dict(cache_metadata.get("mapping_audit", {}))
            selected_csv = Path(str(cache_metadata["selected_csv"]))
            candidates = list(cache_metadata.get("candidate_evaluations", []))
            mapping_evidence_accepted = bool(cache_metadata["mapping_evidence_accepted"])
            label_fallback = dict(cache_metadata.get("label_fallback", {}))
        else:
            mapped, labels, selected_csv, candidates = build_fresh_mapping()
            mapping_evidence_accepted = _accept_mapping(mapped, config)
            label_fallback = dict(labels.attrs.get("label_fallback", {}))

        mapping_path = base / "mappings" / current_protocol / f"{capture_id}.parquet"
        labelled_path = (
            base / "labelled" / "attack" / current_protocol / capture_id / "records.parquet"
        )
        batch_rows = int(config.get("capture", {}).get("streaming_chunk_rows", 50_000))
        is_streaming = manifest.get("extraction_mode") == "streaming_causal"
        while True:
            mapping_audit = mapped.attrs.get("mapping_audit", {})
            packet_offset = float(
                mapping_audit.get("time_offset", {})
                .get("selected", {})
                .get("offset_seconds", 0.0)
            )
            if cache_entry is not None:
                try:
                    mapping_result = apply_cached_packet_labels_parquet(
                        feature_path,
                        Path(cache_entry["links_path"]),
                        labelled_path,
                        batch_rows=batch_rows,
                        batch_callback=lambda batch, selected=current_protocol: store.mirror.write(
                            f"labelled_feature_records_{selected}", batch
                        ),
                        progress_callback=lambda update, selected=current_protocol, done=completed,
                        capture_name=capture.name: mapping_update(
                            update,
                            current_protocol=selected,
                            capture_name=capture_name,
                            completed=done,
                        ),
                    )
                    accepted = bool(cache_metadata["accepted"])
                    rejection_reason = cache_metadata.get("rejection_reason")
                    break
                except MappingCacheMismatchError as error:
                    LOGGER.warning(
                        "MAPPING_CACHE_INVALIDATED protocol=%s capture=%s reason=%s",
                        current_protocol,
                        capture.name,
                        error,
                    )
                    progress.emit(
                        "mapping_cache_invalidated",
                        stage="label_mapping",
                        message=(
                            f"Cached packet relation for {capture.name} does not match this "
                            "extraction; rebuilding this capture's packet↔CSV relation."
                        ),
                        protocol=current_protocol,
                        split="attack",
                        capture=capture.name,
                        mapping_cache_signature=cache_plan["signature"],
                    )
                    # A mismatch can occur after one or more written batches.  Never leave a
                    # partial labelled table available to the fresh mapping path.
                    labelled_path.unlink(missing_ok=True)
                    cache_entry = None
                    cache_reused = False
                    mapped, labels, selected_csv, candidates = build_fresh_mapping()
                    mapping_evidence_accepted = _accept_mapping(mapped, config)
                    label_fallback = dict(labels.attrs.get("label_fallback", {}))
                    continue
            if is_streaming:
                mapping_result = attach_packet_labels_parquet(
                    feature_path,
                    labelled_path,
                    labels,
                    config,
                    offset_seconds=packet_offset,
                    mapping_accepted=mapping_evidence_accepted,
                    batch_rows=batch_rows,
                    progress_callback=lambda update, selected=current_protocol, done=completed,
                    capture_name=capture.name: mapping_update(
                        update,
                        current_protocol=selected,
                        capture_name=capture_name,
                        completed=done,
                    ),
                )
                accepted = mapping_evidence_accepted and int(mapping_result["attack_records"]) > 0
                rejection_reason = (
                    "no_packet_time_attack_label"
                    if mapping_evidence_accepted and not accepted
                    else None
                )
                if accepted != mapping_evidence_accepted:
                    set_mapping_acceptance_parquet(labelled_path, accepted, batch_rows=batch_rows)
                for labelled_batch in pq.ParquetFile(labelled_path).iter_batches(
                    batch_size=batch_rows
                ):
                    store.mirror.write(
                        f"labelled_feature_records_{current_protocol}", labelled_batch.to_pandas()
                    )
                break
            labelled = attach_packet_labels(
                read_table(feature_path), labels, config, offset_seconds=packet_offset
            )
            accepted = mapping_evidence_accepted and int(labelled["is_attack"].sum()) > 0
            rejection_reason = (
                "no_packet_time_attack_label"
                if mapping_evidence_accepted and not accepted
                else None
            )
            labelled["mapping_accepted"] = accepted
            labelled["dataset_split"] = "attack"
            write_table(labelled, labelled_path)
            store.mirror.write(f"labelled_feature_records_{current_protocol}", labelled)
            mapping_result = {
                "rows": len(labelled),
                "attack_records": int(labelled["is_attack"].sum()),
                "label_counts": [
                    {"label": str(label), "is_attack": bool(is_attack), "records": int(records)}
                    for (label, is_attack), records in labelled.groupby(
                        ["label", "is_attack"], dropna=False
                    )
                    .size()
                    .items()
                ],
            }
            break

        _drop_catalog_paths(
            store,
            {
                str(mapping_path.relative_to(base)).replace("\\", "/"),
                str(labelled_path.relative_to(base)).replace("\\", "/"),
                str((mapping_path.with_suffix(".audit.json")).relative_to(base)).replace("\\", "/"),
            },
        )
        store.table(
            mapping_path.relative_to(base),
            mapped,
            kind="flow_label_mapping",
            protocol=current_protocol,
            split="attack",
            description="Flow-to-CSV mapping with confidence and timestamp-offset evidence.",
        )

        labelled_rows = int(mapping_result["rows"])
        packet_attack_records = int(mapping_result["attack_records"])
        label_count_rows = list(mapping_result["label_counts"])
        audit = {
            **mapping_audit,
            "protocol": current_protocol,
            "capture": str(capture),
            "selected_csv": str(selected_csv),
            "accepted": accepted,
            "mapping_evidence_accepted": mapping_evidence_accepted,
            "rejection_reason": rejection_reason,
            "candidate_evaluations": candidates,
            "label_fallback": label_fallback,
            "mapping_evidence_rows": len(evidence),
            "pcap_feature_rows_total": int(manifest["rows"]),
            "extraction_mode": manifest.get("extraction_mode", "in_memory"),
            "mapping_cache": {
                "status": (
                    "reused" if cache_reused else "created" if cache_plan["enabled"] else "disabled"
                ),
                "signature": cache_plan["signature"],
                "path": str(cache_plan["root"]),
            },
            "packet_label_coverage": {
                "records": labelled_rows,
                "attack_records": packet_attack_records,
                "non_attack_records": labelled_rows - packet_attack_records,
                "label_assignment": "packet_timestamp_within_endpoint_compatible_csv_interval",
                "streamed_in_batches": is_streaming,
            },
        }
        if not cache_reused:
            save_mapping_cache(
                cache_plan,
                flow_evidence=mapped,
                labelled_path=labelled_path,
                metadata={
                    "protocol": current_protocol,
                    "capture": str(capture),
                    "selected_csv": str(selected_csv),
                    "mapping_audit": mapping_audit,
                    "candidate_evaluations": candidates,
                    "label_fallback": label_fallback,
                    "accepted": accepted,
                    "mapping_evidence_accepted": mapping_evidence_accepted,
                    "rejection_reason": rejection_reason,
                    "packet_label_coverage": audit["packet_label_coverage"],
                },
                batch_rows=batch_rows,
            )
        store.json(
            mapping_path.with_suffix(".audit.json").relative_to(base),
            audit,
            kind="mapping_audit",
            description="Compatibility decision and packet-time CSV label coverage.",
        )
        store.register_existing(
            labelled_path,
            kind="labelled_feature_records",
            protocol=current_protocol,
            split="attack",
            description=(
                "PCAP feature records joined to a versioned packet-to-CSV mapping relation."
            ),
            rows=labelled_rows,
            columns=len(pq.ParquetFile(labelled_path).schema_arrow),
        )
        mapping_rows.append(
            {
                "protocol": current_protocol,
                "capture": capture.name,
                "selected_csv": selected_csv.name,
                "accepted": accepted,
                "mapping_evidence_accepted": mapping_evidence_accepted,
                "rejection_reason": rejection_reason,
                "packet_attack_records": packet_attack_records,
                "mapping_cache_status": audit["mapping_cache"]["status"],
                "mapping_cache_signature": cache_plan["signature"],
                "mapping_cache_path": str(cache_plan["root"]),
                "pairing_method": pair["pairing_method"],
                "pairing_score": pair["pairing_score"],
                "match_rate": mapping_audit.get("match_rate", 0.0),
                "mean_match_confidence": mapping_audit.get("mean_match_confidence", 0.0),
                "selected_offset_seconds": mapping_audit.get("time_offset", {})
                .get("selected", {})
                .get("offset_seconds"),
            }
        )
        label_distribution_rows.extend(
            {
                "protocol": current_protocol,
                "capture": capture.name,
                "label": str(row["label"]),
                "is_attack": bool(row["is_attack"]),
                "records": int(row["records"]),
                "mapping_accepted": accepted,
            }
            for row in label_count_rows
        )
        progress.emit(
            "packet_mapping_completed",
            stage="label_mapping",
            message=(
                f"{current_protocol}/attack · {capture.name} · completed packet↔CSV relation for "
                f"{labelled_rows:,} packets; attack rows={packet_attack_records:,}; "
                f"cache={'reused' if cache_reused else 'created'}."
            ),
            protocol=current_protocol,
            split="attack",
            capture=capture.name,
            packet_records=labelled_rows,
            attack_records=packet_attack_records,
            mapping_accepted=accepted,
            cache_reused=cache_reused,
            work_units_completed=completed + 1,
            work_units_total=total,
            progress_ratio=(completed + 1) / max(total, 1),
        )

    updates = pd.DataFrame(mapping_rows, columns=mapping_columns)
    mapping_report_path = base / "reports" / "mapping-audit.parquet"
    distribution_path = base / "reports" / "attack-label-distribution.parquet"
    mapping_report = _replace_capture_rows(mapping_report_path, updates, columns=mapping_columns)
    distributions = _replace_capture_rows(
        distribution_path,
        pd.DataFrame(label_distribution_rows, columns=distribution_columns),
        columns=distribution_columns,
    )
    _drop_catalog_paths(
        store,
        {"reports/mapping-audit.parquet", "reports/attack-label-distribution.parquet"},
    )
    store.table(
        mapping_report_path.relative_to(base),
        mapping_report,
        kind="mapping_overview",
        description="One-row mapping decision per attack capture.",
    )
    store.table(
        distribution_path.relative_to(base),
        distributions,
        kind="attack_label_distribution",
        description="Mapped PCAP-record distribution by attack type for balance review.",
    )
    catalog = store.finalize()
    accepted_count = int(updates["accepted"].sum())
    summary = {
        "output_root": str(base),
        "catalog": str(catalog),
        "mapped_captures": int(len(updates)),
        "accepted_mappings": accepted_count,
        "mapping_cache_reused": int(updates["mapping_cache_status"].eq("reused").sum()),
        "mapping_cache_created": int(updates["mapping_cache_status"].eq("created").sum()),
        "protocol_scope": selected_protocol or "all",
    }
    write_json(summary, base / "mapping-summary.json")
    run_summary_path = base / "run-summary.json"
    if run_summary_path.is_file():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        run_summary.update(
            {
                "mapping_status": (
                    "completed" if selected_protocol is None else "partially_completed"
                ),
                "accepted_mappings": int(mapping_report["accepted"].sum()),
                "mapping_count": int(len(mapping_report)),
            }
        )
        write_json(run_summary, run_summary_path)
    progress.complete(
        "Saved packet-to-CSV mapping relation and labelled PCAP feature records.",
        accepted_mappings=accepted_count,
        mapped_captures=len(updates),
        output_root=str(base),
    )
    return summary, base / "mapping-summary.json"
