"""One-command ingestion, candidate mapping, and EDA-report generation."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from anomdet.core.io import utc_now, write_json, write_table
from anomdet.core.paths import resolve_capture_paths
from anomdet.features.extractor import extract_pcap_features
from anomdet.mapping.mapper import map_feature_file_to_labels
from anomdet.modelling.training import run_feature_experiments
from anomdet.selection.profiles import feature_quality_report

LOGGER = logging.getLogger("anomdet")


def _safe_name(value: str) -> str:
    """Create a stable, filesystem-safe identifier from a configured dataset or label name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "unnamed"


def _resolve_configured_path(value: str, root: str) -> Path:
    """Resolve a configured path relative to its configured dataset root when needed."""
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


PROTOCOL_DOMAINS = {"ssh": "it", "dns": "it", "http": "it", "modbus": "ot", "s7comm": "ot"}


def _protocol_folder_inventory(
    config: dict[str, Any], folders: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Turn protocol folders into one deterministic multi-PCAP dataset each."""
    root = str(config["data"]["raw_pcap_dir"])
    supported = set(config["capture"]["supported_protocols"])
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(folders, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Protocol-folder item {index} must be a YAML mapping.")
        protocol = str(item.get("protocol", "")).strip().lower()
        if protocol not in supported:
            raise ValueError(
                f"Protocol-folder item {index} has unsupported protocol '{protocol}'. "
                f"Choose one of: {', '.join(sorted(supported))}."
            )
        dataset_id = str(item.get("id", protocol))
        if dataset_id in ids:
            raise ValueError(f"Duplicate dataset id: {dataset_id}.")
        labels = item.get("labels", [])
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"Protocol-folder '{protocol}' labels must be a list of file paths.")
        domain = str(item.get("domain", PROTOCOL_DOMAINS[protocol])).lower()
        if domain not in {"it", "ot"}:
            raise ValueError(f"Protocol-folder '{protocol}' must use domain 'it' or 'ot'.")
        folder = _resolve_configured_path(str(item.get("folder", protocol)), root)
        captures = resolve_capture_paths(folder)
        ids.add(dataset_id)
        validated.append(
            {
                "id": dataset_id,
                # Keep the configured value here.  ``run_inventory`` resolves it
                # relative to ``raw_pcap_dir`` once; storing ``folder`` (which is
                # already resolved) would otherwise create paths such as
                # ``data/raw/pcap/data/raw/pcap/modbus``.
                "pcap": str(item.get("folder", protocol)),
                "domain": domain,
                "labels": labels,
                "expected_protocol": protocol,
                "capture_count": len(captures),
            }
        )
    return validated


def _inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer protocol folders while retaining the legacy file-by-file manifest format."""
    data = config.get("data", {})
    folders = data.get("protocol_folders", [])
    if folders:
        if not isinstance(folders, list):
            raise ValueError("data.protocol_folders must be a list of YAML mappings.")
        return _protocol_folder_inventory(config, folders)
    datasets = data.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("data.datasets must be a list of YAML mappings.")
    if not datasets:
        supported = config["capture"]["supported_protocols"]
        discovered = [
            {"protocol": protocol, "folder": protocol, "domain": PROTOCOL_DOMAINS[protocol]}
            for protocol in supported
            if (Path(data["raw_pcap_dir"]) / protocol).is_dir()
        ]
        if discovered:
            return _protocol_folder_inventory(config, discovered)
        raise ValueError(
            "No protocol folders were found. Create data/raw/pcap/<protocol>/ and place PCAP files "
            "inside, or configure data.protocol_folders."
        )
    ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(datasets, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be a YAML mapping.")
        missing = {"id", "pcap", "domain"}.difference(item)
        if missing:
            raise ValueError(f"Dataset item {index} is missing: {', '.join(sorted(missing))}.")
        dataset_id = str(item["id"])
        if dataset_id in ids:
            raise ValueError(f"Duplicate dataset id: {dataset_id}.")
        if item["domain"] not in {"it", "ot"}:
            raise ValueError(f"Dataset '{dataset_id}' must use domain 'it' or 'ot'.")
        expected_protocol = item.get("protocol")
        if (
            expected_protocol is not None
            and expected_protocol not in config["capture"]["supported_protocols"]
        ):
            raise ValueError(
                f"Dataset '{dataset_id}' has unsupported protocol '{expected_protocol}'."
            )
        labels = item.get("labels", [])
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"Dataset '{dataset_id}' labels must be a list of file paths.")
        ids.add(dataset_id)
        validated.append(
            {
                "id": dataset_id,
                "pcap": str(item["pcap"]),
                "domain": item["domain"],
                "labels": labels,
                "expected_protocol": expected_protocol,
            }
        )
    return validated


def run_inventory(
    config: dict[str, Any], output_dir: Path | None = None, max_packets: int | None = None
) -> tuple[dict[str, Any], Path]:
    """Run extraction, candidate mappings, and quality reports for all configured datasets."""
    datasets = _inventory(config)
    run_id = datetime.now(UTC).strftime("batch-%Y%m%dT%H%M%SZ")
    root = output_dir or Path(config["project"]["artifact_dir"]) / "runs" / run_id
    feature_dir, mapping_dir, report_dir = root / "features", root / "mapping", root / "reports"
    for directory in (feature_dir, mapping_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []
    work_units = len(datasets) + sum(len(item["labels"]) for item in datasets)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn()
    )
    with progress:
        task = progress.add_task("Running configured datasets", total=work_units)
        for item in datasets:
            dataset_id = _safe_name(item["id"])
            capture_path = _resolve_configured_path(item["pcap"], config["data"]["raw_pcap_dir"])
            capture_paths = resolve_capture_paths(capture_path)
            dataset_summary: dict[str, Any] = {
                "dataset_id": item["id"],
                "domain": item["domain"],
                "capture_folder": str(capture_path),
                "captures": [str(path) for path in capture_paths],
                "expected_protocol": item.get("expected_protocol"),
                "status": "pending",
                "mappings": [],
            }
            feature_path = feature_dir / f"{dataset_id}.parquet"
            try:
                features, extract_manifest = extract_pcap_features(
                    capture_path,
                    feature_path,
                    config,
                    max_packets,
                    expected_protocol=item.get("expected_protocol"),
                )
                features = features.copy()
                features["dataset_id"] = item["id"]
                combined_frames.append(features)
                quality = feature_quality_report(
                    feature_path, report_dir / f"{dataset_id}-feature-quality.parquet"
                )
                dataset_summary.update(
                    {
                        "status": "extracted",
                        "features": str(feature_path),
                        "feature_rows": extract_manifest["rows"],
                        "protocol_counts": extract_manifest["protocol_counts"],
                        "capture_count": extract_manifest["capture_count"],
                        "feature_validation": quality.groupby("extraction_status").size().to_dict(),
                    }
                )
            except (
                Exception
            ) as error:  # Batch mode preserves other dataset results for investigation.
                LOGGER.exception("Dataset '%s' failed during extraction", item["id"])
                dataset_summary.update({"status": "extraction_failed", "error": str(error)})
                summaries.append(dataset_summary)
                progress.advance(task, 1 + len(item["labels"]))
                continue
            progress.advance(task)

            for label_value in item["labels"]:
                label_root = (
                    config["data"]["it_label_dir"]
                    if item["domain"] == "it"
                    else config["data"]["ot_label_dir"]
                )
                label_path = _resolve_configured_path(label_value, label_root)
                mapping_path = mapping_dir / f"{dataset_id}--{_safe_name(label_path.stem)}.parquet"
                mapping_summary: dict[str, Any] = {
                    "labels": str(label_path),
                    "output": str(mapping_path),
                    "status": "pending",
                }
                try:
                    _, result = map_feature_file_to_labels(
                        feature_path, label_path, mapping_path, item["domain"], config
                    )
                    mapping_summary.update({"status": "complete", **result})
                except (
                    Exception
                ) as error:  # A bad candidate CSV must not stop other mapping candidates.
                    LOGGER.exception("Dataset '%s' failed while mapping %s", item["id"], label_path)
                    mapping_summary.update({"status": "failed", "error": str(error)})
                dataset_summary["mappings"].append(mapping_summary)
                progress.advance(task)
            summaries.append(dataset_summary)

    combined_path: Path | None = None
    feature_evaluation: dict[str, Any] = {"status": "not_run", "runs": []}
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True, sort=False)
        combined_path = feature_dir / "all-datasets.parquet"
        write_table(combined, combined_path)
        feature_quality_report(combined_path, report_dir / "all-datasets-feature-quality.parquet")
        feature_evaluation = _run_feature_evaluation(combined_path, root, config)
    summary = {
        "created_at": utc_now(),
        "run_id": run_id,
        "output_root": str(root),
        "combined_features": str(combined_path) if combined_path else None,
        "dataset_count": len(datasets),
        "successful_extractions": sum(item["status"] == "extracted" for item in summaries),
        "feature_evaluation": feature_evaluation,
        "datasets": summaries,
    }
    summary_path = root / "run-summary.json"
    write_json(summary, summary_path)
    LOGGER.info(
        "Batch run completed: %s/%s datasets extracted",
        summary["successful_extractions"],
        len(datasets),
    )
    return summary, summary_path


def _run_feature_evaluation(
    feature_path: Path, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Compare configured detectors across all features and selected feature profiles."""
    settings = config.get("feature_evaluation", {})
    if not settings.get("enabled", False):
        return {"status": "disabled", "runs": []}
    detectors = settings.get("detectors", [])
    if not detectors:
        raise ValueError(
            "feature_evaluation.detectors must contain at least one unsupervised detector."
        )
    strategy = settings.get("strategy", "grouped")
    group = settings.get("group", "all")
    profiles: list[str] = settings.get("selected_profiles", [])
    evaluation_root = root / "feature-evaluation"
    _comparison, summary = run_feature_experiments(
        feature_path=feature_path,
        config=config,
        strategy=strategy,
        group=group,
        profiles=profiles,
        output_dir=evaluation_root,
        candidates=detectors,
    )
    return {
        "status": "complete"
        if all(item["status"] == "complete" for item in summary["runs"])
        else "partial",
        "detectors": detectors,
        "strategy": strategy,
        "group": group,
        "comparison": summary["comparison"],
        "runs": summary["runs"],
    }
