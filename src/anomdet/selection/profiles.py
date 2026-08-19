"""Reproducible, validated feature selections stored as immutable profile manifests."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from anomdet.core.io import read_table, utc_now, write_json, write_table
from anomdet.features.catalog import FEATURE_CATALOG, feature_names
from anomdet.features.extractor import NOT_YET_IMPLEMENTED_FEATURES

LOGGER = logging.getLogger("anomdet")


def _profile_directory(config: dict[str, Any]) -> Path:
    """Return and create the central location for immutable feature-profile manifests."""
    path = Path(config["project"]["artifact_dir"]) / "feature_profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_profile(
    name: str,
    features: list[str],
    config: dict[str, Any],
    description: str = "",
    protocols: list[str] | None = None,
) -> Path:
    """Validate and save an immutable, traceable feature selection manifest."""
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "Profile name may contain only letters, numbers, underscores, and hyphens."
        )
    cleaned = list(dict.fromkeys(feature.strip() for feature in features if feature.strip()))
    if not cleaned:
        raise ValueError("At least one feature must be selected.")
    valid = feature_names(tuple(protocols) if protocols else None)
    invalid = sorted(set(cleaned).difference(valid))
    if invalid:
        raise ValueError(f"Unknown or inapplicable feature names: {', '.join(invalid)}")
    version_seed = f"{name}|{','.join(cleaned)}|{utc_now()}"
    version = hashlib.sha256(version_seed.encode("utf-8")).hexdigest()[:10]
    payload = {
        "schema_version": "1.0.0",
        "name": name,
        "version": version,
        "created_at": utc_now(),
        "description": description or "User-defined feature selection.",
        "protocols": protocols or ["ssh", "dns", "http", "modbus", "s7comm"],
        "features": cleaned,
        "feature_count": len(cleaned),
    }
    path = _profile_directory(config) / f"{name}-{version}.json"
    write_json(payload, path)
    LOGGER.info("Created feature profile %s with %s features", path.name, len(cleaned))
    return path


def resolve_profile(profile: str | Path, config: dict[str, Any]) -> Path:
    """Resolve either an explicit manifest path or the newest manifest with a given name."""
    candidate = Path(profile)
    if candidate.exists():
        return candidate
    matches = sorted(_profile_directory(config).glob(f"{profile}-*.json"))
    if not matches:
        raise FileNotFoundError(f"Feature profile '{profile}' was not found.")
    return matches[-1]


def load_profile(profile: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Load a profile manifest and revalidate its schema-critical fields."""
    path = resolve_profile(profile, config)
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("features"), list) or not payload["features"]:
        raise ValueError(f"Profile has no usable features: {path}")
    payload["path"] = str(path)
    return payload


def feature_quality_report(feature_path: Path, output_path: Path) -> pd.DataFrame:
    """Validate computed catalogue features separately for every observed protocol.

    A schema column alone is not evidence that an extractor observed useful
    values. The report distinguishes absent parser evidence, constant fields,
    sparse fields, and model-usable fields without treating a legitimate rare
    security event as an extraction failure.
    """
    frame = read_table(feature_path)
    rows: list[dict[str, Any]] = []
    protocols = (
        sorted(frame["protocol"].dropna().astype("string").str.lower().unique().tolist())
        if "protocol" in frame.columns
        else ["unknown"]
    )
    for protocol in protocols:
        scoped = (
            frame[frame["protocol"].astype("string").str.lower() == protocol]
            if protocol != "unknown" and "protocol" in frame.columns
            else frame
        )
        record_count = len(scoped)
        flow_count = (
            int(scoped["flow_id"].nunique(dropna=True)) if "flow_id" in scoped.columns else 0
        )
        for definition in FEATURE_CATALOG:
            if protocol != "unknown" and protocol not in definition.protocols:
                continue
            column = definition.name
            present = column in scoped.columns
            series = scoped[column] if present else pd.Series(dtype="object")
            observed = int(series.notna().sum())
            observed_ratio = observed / max(record_count, 1)
            observed_flow_count = (
                int(scoped.loc[series.notna(), "flow_id"].nunique(dropna=True))
                if flow_count and present
                else 0
            )
            observed_flow_ratio = observed_flow_count / max(flow_count, 1)
            coverage_ratio = max(observed_ratio, observed_flow_ratio)
            unique_count = int(series.nunique(dropna=True)) if present else 0
            numeric = pd.to_numeric(series, errors="coerce") if present else pd.Series(dtype=float)
            numeric_observed = numeric.dropna()
            is_numeric = not numeric_observed.empty
            zero_ratio = float((numeric_observed == 0).mean()) if is_numeric else None
            nonzero_ratio = float((numeric_observed != 0).mean()) if is_numeric else None
            variance = float(numeric_observed.var()) if is_numeric else None
            iqr = (
                float(numeric_observed.quantile(0.75) - numeric_observed.quantile(0.25))
                if is_numeric
                else None
            )
            implementation_status = (
                "not_implemented"
                if definition.name in NOT_YET_IMPLEMENTED_FEATURES
                else "implemented"
            )
            if record_count == 0:
                status, reason = "no_records", "No records of this protocol were extracted."
            elif implementation_status == "not_implemented":
                status, reason = (
                    "not_implemented",
                    "The catalogue field needs a parser capability that is not implemented yet.",
                )
            elif not present or observed == 0:
                status, reason = (
                    "not_observed",
                    "The protocol parser did not observe this field in the available packets.",
                )
            elif coverage_ratio < 0.05:
                status, reason = (
                    "low_coverage",
                    "Observed in fewer than 5% of both protocol records and flows; inspect packet completeness.",
                )
            elif unique_count <= 1:
                status, reason = (
                    "constant",
                    "Only one observed value; it cannot separate behaviour in this capture.",
                )
            elif is_numeric and zero_ratio is not None and zero_ratio >= 0.99:
                status, reason = (
                    "near_constant",
                    "At least 99% of observed values are zero; retain only with a domain rationale.",
                )
            else:
                status, reason = (
                    "usable",
                    "Computed with sufficient coverage and variation for exploratory modelling.",
                )
            rows.append(
                {
                    "protocol": protocol,
                    "name": definition.name,
                    "display_name": definition.display_name,
                    "category": definition.category,
                    "cost": definition.cost,
                    "expected_by_catalog": True,
                    "implementation_status": implementation_status,
                    "present": present,
                    "records": record_count,
                    "flow_count": flow_count,
                    "observed_count": observed,
                    "observed_ratio": round(observed_ratio, 4),
                    "observed_flow_count": observed_flow_count,
                    "observed_flow_ratio": round(observed_flow_ratio, 4),
                    "missing_ratio": round(1 - observed_ratio, 4),
                    "unique_count": unique_count,
                    "numeric": is_numeric,
                    "zero_ratio": round(zero_ratio, 4) if zero_ratio is not None else None,
                    "nonzero_ratio": round(nonzero_ratio, 4) if nonzero_ratio is not None else None,
                    "variance": round(variance, 6) if variance is not None else None,
                    "iqr": round(iqr, 6) if iqr is not None else None,
                    "estimated_memory_mb": round(float(series.memory_usage(deep=True) / 1024**2), 4)
                    if present
                    else 0.0,
                    "extraction_status": status,
                    "model_usable": status == "usable",
                    "reason": reason,
                }
            )
    quality = pd.DataFrame(rows)
    write_table(quality, output_path)
    LOGGER.info("Wrote protocol-specific feature-quality report for %s entries", len(quality))
    return quality
