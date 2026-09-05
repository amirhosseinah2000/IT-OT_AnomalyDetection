"""Protocol-separated PCAP-first ingestion, evidence mapping, and analytical assets."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from anomdet.analytics.reports import feature_overview
from anomdet.core.io import write_json
from anomdet.core.progress import DurableProgress
from anomdet.datasets.inventory import discover_dataset
from anomdet.features.extractor import extract_pcap_features
from anomdet.mapping.mapper import (
    attach_flow_labels,
    map_features_to_labels,
    normalize_label_csv,
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
    config: dict[str, Any], output: Path | None = None, max_packets: int | None = None
) -> tuple[dict[str, Any], Path]:
    """Create a fully auditable, protocol-first dataset run from the declared layout.

    The PCAP is always the source of features.  CSVs are only normalized and
    joined as label evidence; no CSV feature column enters the model dataset.
    """
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
    progress.emit(
        "inventory_discovered",
        stage="inventory",
        message=(
            f"{len(benign_by_protocol)} گروه نرمال و "
            f"{len(inventory['attack_pairs'])} capture حمله برای پردازش پیدا شد."
        ),
        protocol_count=len(benign_by_protocol),
        attack_capture_count=len(inventory["attack_pairs"]),
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
        packet_limit = payload.get("packet_limit")
        if event == "capture_progress":
            message = (
                f"{protocol}/{split} · {capture_name} · "
                f"{packet_count}{'/' + str(packet_limit) if packet_limit else ''} packet خوانده شد."
            )
        elif event == "capture_completed":
            message = f"{protocol}/{split} · {capture_name} کامل شد."
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

    for protocol, sources in sorted(benign_by_protocol.items()):
        # Passing the protocol folder lets the extractor compute behavioural
        # features across all benign captures while retaining a separate file per protocol.
        target = base / "features" / "benign" / protocol / "records.parquet"
        progress.emit(
            "protocol_extraction_started",
            stage="feature_extraction",
            message=f"استخراج ترافیک نرمال {protocol} شروع شد.",
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
            max_packets,
            expected_protocol=protocol,
            progress_callback=lambda update, protocol=protocol, work_done=completed_units: extraction_update(
                update, protocol=protocol, split="benign", work_done=work_done
            ),
        )
        _register_extraction(store, target, protocol, "benign", manifest)
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
            message=f"فیچر، کیفیت و آمار نرمال {protocol} ثبت شد.",
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
            message=f"استخراج capture حمله {capture.name} برای {protocol} شروع شد.",
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
            max_packets,
            expected_protocol=protocol,
            progress_callback=lambda update, protocol=protocol, work_done=completed_units: extraction_update(
                update, protocol=protocol, split="attack", work_done=work_done
            ),
        )
        _register_extraction(store, feature_target, protocol, "attack", manifest)
        store.mirror.write(f"feature_records_attack_{protocol}", features)
        progress.emit(
            "mapping_started",
            stage="label_mapping",
            message=f"نگاشت PCAP و CSV برای {capture.name} شروع شد.",
            protocol=protocol,
            split="attack",
            capture=capture.name,
            rows=len(features),
            work_units_completed=completed_units,
            work_units_total=total_units,
        )
        mapped, labels, selected_csv, candidates = _best_mapping(
            features, pair, label_root / protocol, protocol, config
        )
        accepted = _accept_mapping(mapped, config)
        LOGGER.info(
            "MAPPING_COMPLETE protocol=%s capture=%s accepted=%s match_rate=%.3f confidence=%.3f",
            protocol,
            capture.name,
            accepted,
            float(mapped.attrs.get("mapping_audit", {}).get("match_rate", 0.0)),
            float(mapped.attrs.get("mapping_audit", {}).get("mean_match_confidence", 0.0)),
        )
        mapping_path = base / "mappings" / protocol / f"{capture_id}.parquet"
        store.table(
            mapping_path.relative_to(base),
            mapped,
            kind="flow_label_mapping",
            protocol=protocol,
            split="attack",
            description="Flow-to-CSV mapping with confidence and timestamp-offset evidence.",
        )
        audit = {
            **mapped.attrs.get("mapping_audit", {}),
            "protocol": protocol,
            "capture": str(capture),
            "selected_csv": str(selected_csv),
            "accepted": accepted,
            "candidate_evaluations": candidates,
            "label_fallback": labels.attrs.get("label_fallback", {}),
        }
        store.json(
            "mappings/" + protocol + f"/{capture_id}.audit.json",
            audit,
            kind="mapping_audit",
            description="Compatibility decision and raw matching evidence.",
        )
        mapping_rows.append(
            {
                "protocol": protocol,
                "capture": capture.name,
                "selected_csv": selected_csv.name,
                "accepted": accepted,
                "pairing_method": pair["pairing_method"],
                "pairing_score": pair["pairing_score"],
                "match_rate": audit.get("match_rate", 0.0),
                "mean_match_confidence": audit.get("mean_match_confidence", 0.0),
                "selected_offset_seconds": audit.get("time_offset", {})
                .get("selected", {})
                .get("offset_seconds"),
            }
        )
        labelled = attach_flow_labels(features, mapped)
        labelled["mapping_accepted"] = accepted
        labelled["dataset_split"] = "attack"
        label_distribution_rows.extend(
            {
                "protocol": protocol,
                "capture": capture.name,
                "label": str(label),
                "is_attack": bool(is_attack),
                "records": int(records),
                "mapping_accepted": accepted,
            }
            for (label, is_attack), records in labelled.groupby(
                ["label", "is_attack"], dropna=False
            )
            .size()
            .items()
        )
        labelled_target = "labelled/attack/" + protocol + f"/{capture_id}/records.parquet"
        store.table(
            labelled_target,
            labelled,
            kind="labelled_feature_records",
            protocol=protocol,
            split="attack",
            description="PCAP feature records with separately mapped attack labels.",
            mirror_table=f"labelled_feature_records_{protocol}",
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
                f"{capture.name}: نگاشت {'پذیرفته شد' if accepted else 'نیازمند بررسی است'} "
                f"و خروجی برچسب‌خورده ثبت شد."
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

    mapping_frame = pd.DataFrame(mapping_rows)
    store.table(
        "reports/mapping-audit.parquet",
        mapping_frame,
        kind="mapping_overview",
        description="One-row mapping decision per attack capture.",
    )
    store.table(
        "reports/attack-label-distribution.parquet",
        pd.DataFrame(label_distribution_rows),
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
        "max_packets_per_capture": max_packets,
    }
    write_json(summary, base / "run-summary.json")
    progress.complete(
        "استخراج پروتکل‌محور، نگاشت CSV و گزارش‌های تحلیلی کامل شد.",
        accepted_mappings=summary["accepted_mappings"],
        mapping_count=summary["mapping_count"],
        output_root=str(base),
    )
    return summary, base / "run-summary.json"
