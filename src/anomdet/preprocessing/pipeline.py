"""Leakage-aware feature preparation for CPU-first unsupervised modelling."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from anomdet.core.io import read_table, utc_now, write_json, write_table
from anomdet.features.catalog import feature_names
from anomdet.selection.profiles import load_profile

LOGGER = logging.getLogger("anomdet")
IDENTIFIER_COLUMNS = {
    "capture",
    "timestamp",
    "flow_id",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "label",
}


class SafeRobustScaler(BaseEstimator, TransformerMixin):
    """Robustly scale sparse numeric columns without leaking raw magnitudes.

    ``RobustScaler`` leaves a column effectively unscaled when its IQR is zero.
    That is common for protocol-specific features represented as mostly-zero
    columns, and lets a rare large value dominate an autoencoder's loss. This
    scaler keeps the median/IQR transform where available and falls back to the
    standard deviation only for those zero-IQR columns. It then bounds the
    transformed tail so one malformed/rare raw value cannot destabilise an
    autoencoder's MSE loss.
    """

    def __init__(self, epsilon: float = 1e-12, clip_value: float | None = 12.0) -> None:
        self.epsilon = epsilon
        self.clip_value = clip_value

    def fit(self, values: np.ndarray, y: object = None) -> SafeRobustScaler:
        if self.clip_value is not None and self.clip_value <= 0:
            raise ValueError("clip_value must be positive or None.")
        matrix = np.asarray(values, dtype=float)
        self.center_ = np.nanmedian(matrix, axis=0)
        q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
        robust_scale = q75 - q25
        standard_scale = np.nanstd(matrix, axis=0)
        fallback = np.where(standard_scale > self.epsilon, standard_scale, 1.0)
        self.scale_ = np.where(robust_scale > self.epsilon, robust_scale, fallback)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (np.asarray(values, dtype=float) - self.center_) / self.scale_
        if self.clip_value is not None:
            return np.clip(transformed, -self.clip_value, self.clip_value)
        return transformed

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray:
        return np.asarray(input_features, dtype=object)


def _build_transformer(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a robust numeric/categorical transformation pipeline for the selected matrix."""
    numeric_columns = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    scaler = (
        SafeRobustScaler(clip_value=config["preprocessing"].get("numeric_clip", 12.0))
        if config["preprocessing"]["numeric_scaler"] == "robust"
        else StandardScaler()
    )
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", scaler)])
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer, numeric_columns, categorical_columns


def _limit_categories(
    frame: pd.DataFrame, maximum: int
) -> tuple[pd.DataFrame, dict[str, int], dict[str, list[Any]]]:
    """Collapse rare values in high-cardinality text features before one-hot encoding."""
    result = frame.copy()
    collapsed: dict[str, int] = {}
    vocabularies: dict[str, list[Any]] = {}
    for column in result.select_dtypes(exclude=[np.number, "bool"]).columns:
        cardinality = int(result[column].nunique(dropna=True))
        if cardinality <= maximum:
            continue
        allowed = result[column].value_counts(dropna=True).head(maximum).index
        result[column] = result[column].where(
            result[column].isin(allowed) | result[column].isna(), "__OTHER__"
        )
        collapsed[column] = cardinality
        vocabularies[column] = allowed.tolist()
    return result, collapsed, vocabularies


def read_training_source(feature_path: Path, maximum_rows: int | None = None) -> pd.DataFrame:
    """Read a deterministic, time-spread model sample without exhausting RAM on large PCAP runs."""
    if maximum_rows is not None and maximum_rows < 1:
        raise ValueError("maximum_rows must be positive or None.")
    if maximum_rows is None or feature_path.suffix.lower() not in {".parquet", ".pq"}:
        frame = read_table(feature_path)
        frame.attrs["source_rows_total"] = len(frame)
        frame.attrs["sampled_for_training"] = False
        return frame
    parquet = pq.ParquetFile(feature_path)
    source_rows = parquet.metadata.num_rows
    if source_rows <= maximum_rows:
        frame = read_table(feature_path)
        frame.attrs["source_rows_total"] = source_rows
        frame.attrs["sampled_for_training"] = False
        return frame
    group_count = parquet.num_row_groups
    selected_groups = np.unique(
        np.linspace(0, group_count - 1, num=min(group_count, 16), dtype=int)
    )
    rows_per_group = max(1, maximum_rows // len(selected_groups))
    pieces: list[pd.DataFrame] = []
    for group in selected_groups:
        piece = parquet.read_row_group(int(group)).to_pandas()
        if len(piece) > rows_per_group:
            positions = np.linspace(0, len(piece) - 1, num=rows_per_group, dtype=int)
            piece = piece.iloc[positions]
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True, sort=False)
    frame.attrs["source_rows_total"] = source_rows
    frame.attrs["sampled_for_training"] = True
    return frame


def prepare_features(
    feature_path: Path,
    output_path: Path,
    config: dict[str, Any],
    profile: str | Path | None = None,
    protocols: list[str] | None = None,
    labels_path: Path | None = None,
    fit_positions: np.ndarray | list[int] | None = None,
    fit: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    source_frame: pd.DataFrame | None = None,
    maximum_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Select, validate, transform, and persist a model-ready matrix plus its pipeline."""

    def report(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event)
        except Exception:  # pragma: no cover - an observer must not stop feature preparation.
            LOGGER.exception("Feature preparation progress observer failed")

    if source_frame is None:
        report({"event": "feature_source_loading", "source": str(feature_path)})
        source = read_training_source(feature_path, maximum_rows=maximum_rows)
    else:
        source = source_frame.copy()
    source_rows_total = int(source.attrs.get("source_rows_total", len(source)))
    sampled_for_training = bool(source.attrs.get("sampled_for_training", False))
    report(
        {
            "event": "feature_source_loaded",
            "source_rows": len(source),
            "source_rows_total": source_rows_total,
            "sampled_for_training": sampled_for_training,
        }
    )
    if labels_path is not None:
        labels = read_table(labels_path)
        required = {"flow_id", "label"}
        if missing := required.difference(labels.columns):
            raise ValueError(f"Label mapping is missing required columns: {sorted(missing)}")
        labels = labels[["flow_id", "label"]].drop_duplicates("flow_id", keep="last")
        source = source.drop(columns=["label"], errors="ignore").merge(
            labels, on="flow_id", how="left"
        )
    if source.empty:
        raise ValueError("Cannot prepare an empty feature table.")
    profile_data = load_profile(profile, config) if profile else None
    if protocols:
        source = source[source["protocol"].isin(protocols)].copy()
    source = source.reset_index(drop=True)
    observed_protocols = tuple(
        sorted(source["protocol"].dropna().astype("string").str.lower().unique().tolist())
        if "protocol" in source.columns
        else []
    )
    applicable = feature_names(observed_protocols or None)
    requested = profile_data["features"] if profile_data else sorted(applicable)
    ignored_inapplicable = sorted(set(requested).difference(applicable))
    requested = [feature for feature in requested if feature in applicable]
    selected = [column for column in requested if column in source.columns]
    if "protocol" in source.columns and "protocol" not in selected:
        selected.append("protocol")
    if not selected:
        raise ValueError("None of the requested features exists in the source table.")
    matrix = source[selected].copy()
    min_non_null = float(config["preprocessing"]["min_non_null_ratio"])
    retained = [
        column for column in matrix.columns if matrix[column].notna().mean() >= min_non_null
    ]
    dropped = sorted(set(matrix.columns).difference(retained))
    matrix = matrix[retained]
    constant_features = [
        column for column in matrix.columns if matrix[column].nunique(dropna=True) <= 1
    ]
    if constant_features:
        matrix = matrix.drop(columns=constant_features)
        retained = [column for column in retained if column not in constant_features]
    if matrix.empty:
        raise ValueError(
            "No variable model features remain after removing missing and constant fields. "
            "Inspect the protocol-specific feature-quality report."
        )
    matrix, collapsed_categories, category_vocabularies = _limit_categories(
        matrix, int(config["features"]["high_cardinality_max_categories"])
    )
    report(
        {
            "event": "feature_transform_started",
            "source_rows": len(source),
            "selected_features": len(retained),
        }
    )
    transformer, numeric_columns, categorical_columns = _build_transformer(matrix, config)
    if fit:
        fit_matrix = matrix.iloc[fit_positions] if fit_positions is not None else matrix
        transformer.fit(fit_matrix)
    transformed = transformer.transform(matrix)
    transformed_feature_names = transformer.get_feature_names_out().tolist()
    report(
        {
            "event": "feature_transform_completed",
            "prepared_rows": len(transformed),
            "transformed_features": len(transformed_feature_names),
        }
    )
    prepared = pd.DataFrame(transformed, columns=transformed_feature_names, index=source.index)
    prepared.insert(0, "row_id", source.index)
    metadata_columns = ["label", "protocol", "flow_id", "timestamp", "capture"]
    insert_position = 1
    for column in metadata_columns:
        if column not in source.columns:
            continue
        prepared.insert(insert_position, column, source[column])
        insert_position += 1
    write_table(prepared, output_path)
    pipeline_path = output_path.with_suffix(".pipeline.joblib")
    joblib.dump(transformer, pipeline_path)
    manifest = {
        "created_at": utc_now(),
        "source": str(feature_path),
        "output": str(output_path),
        "pipeline": str(pipeline_path),
        "profile": profile_data,
        "protocols": protocols,
        "labels": str(labels_path) if labels_path else None,
        "source_rows": len(source),
        "source_rows_total": source_rows_total,
        "sampled_for_training": sampled_for_training,
        "prepared_rows": len(prepared),
        "selected_input_features": retained,
        "inapplicable_profile_features": ignored_inapplicable,
        "dropped_missing_features": dropped,
        "dropped_constant_features": constant_features,
        "numeric_input_features": numeric_columns,
        "numeric_clip": config["preprocessing"].get("numeric_clip", 12.0),
        "categorical_input_features": categorical_columns,
        "collapsed_high_cardinality_features": collapsed_categories,
        # This is part of the model contract: the same values must collapse to
        # __OTHER__ at inference, even when it runs on another host.
        "categorical_value_vocabulary": category_vocabularies,
        "transformer_fit_rows": len(fit_positions) if fit_positions is not None else len(matrix),
        "transformed_feature_count": len(transformed_feature_names),
    }
    write_json(manifest, output_path.with_suffix(".manifest.json"))
    LOGGER.info(
        "Prepared %s rows and %s model columns", len(prepared), len(transformed_feature_names)
    )
    return prepared, manifest, pipeline_path


def transform_with_manifest(
    source: pd.DataFrame, pipeline_path: Path, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply a frozen training feature contract to raw PCAP features at inference.

    Missing fields remain missing for the fitted imputers and unseen categories
    either collapse exactly as during training or are safely ignored by the
    frozen one-hot encoder.  The returned column order is the trained order.
    """
    transformer = joblib.load(pipeline_path)
    selected = list(manifest["selected_input_features"])
    matrix = source.copy()
    for column in selected:
        if column not in matrix.columns:
            matrix[column] = np.nan
    matrix = matrix[selected]
    for column, vocabulary in manifest.get("categorical_value_vocabulary", {}).items():
        if column not in matrix:
            continue
        allowed = set(vocabulary)
        matrix[column] = matrix[column].where(
            matrix[column].isin(allowed) | matrix[column].isna(), "__OTHER__"
        )
    transformed = transformer.transform(matrix)
    names = transformer.get_feature_names_out().tolist()
    return pd.DataFrame(transformed, columns=names, index=source.index), names
