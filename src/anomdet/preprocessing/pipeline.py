"""Leakage-aware feature preparation for CPU-first unsupervised modelling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
    standard deviation only for those zero-IQR columns.
    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon

    def fit(self, values: np.ndarray, y: object = None) -> SafeRobustScaler:
        matrix = np.asarray(values, dtype=float)
        self.center_ = np.nanmedian(matrix, axis=0)
        q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
        robust_scale = q75 - q25
        standard_scale = np.nanstd(matrix, axis=0)
        fallback = np.where(standard_scale > self.epsilon, standard_scale, 1.0)
        self.scale_ = np.where(robust_scale > self.epsilon, robust_scale, fallback)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.center_) / self.scale_

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
        SafeRobustScaler()
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


def _limit_categories(frame: pd.DataFrame, maximum: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse rare values in high-cardinality text features before one-hot encoding."""
    result = frame.copy()
    collapsed: dict[str, int] = {}
    for column in result.select_dtypes(exclude=[np.number, "bool"]).columns:
        cardinality = int(result[column].nunique(dropna=True))
        if cardinality <= maximum:
            continue
        allowed = result[column].value_counts(dropna=True).head(maximum).index
        result[column] = result[column].where(
            result[column].isin(allowed) | result[column].isna(), "__OTHER__"
        )
        collapsed[column] = cardinality
    return result, collapsed


def prepare_features(
    feature_path: Path,
    output_path: Path,
    config: dict[str, Any],
    profile: str | Path | None = None,
    protocols: list[str] | None = None,
    labels_path: Path | None = None,
    fit_positions: np.ndarray | list[int] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Select, validate, transform, and persist a model-ready matrix plus its pipeline."""
    source = read_table(feature_path)
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
    matrix, collapsed_categories = _limit_categories(
        matrix, int(config["features"]["high_cardinality_max_categories"])
    )
    transformer, numeric_columns, categorical_columns = _build_transformer(matrix, config)
    if fit:
        fit_matrix = matrix.iloc[fit_positions] if fit_positions is not None else matrix
        transformer.fit(fit_matrix)
    transformed = transformer.transform(matrix)
    transformed_feature_names = transformer.get_feature_names_out().tolist()
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
        "prepared_rows": len(prepared),
        "selected_input_features": retained,
        "inapplicable_profile_features": ignored_inapplicable,
        "dropped_missing_features": dropped,
        "dropped_constant_features": constant_features,
        "numeric_input_features": numeric_columns,
        "categorical_input_features": categorical_columns,
        "collapsed_high_cardinality_features": collapsed_categories,
        "transformer_fit_rows": len(fit_positions) if fit_positions is not None else len(matrix),
        "transformed_feature_count": len(transformed_feature_names),
    }
    write_json(manifest, output_path.with_suffix(".manifest.json"))
    LOGGER.info(
        "Prepared %s rows and %s model columns", len(prepared), len(transformed_feature_names)
    )
    return prepared, manifest, pipeline_path
