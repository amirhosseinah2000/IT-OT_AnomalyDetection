"""Train and compare resource-conscious unsupervised detectors across deployment strategies."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from anomdet.core.io import read_table, utc_now, write_json, write_table
from anomdet.core.resources import effective_workers
from anomdet.modelling.detectors import PCAAutoencoder
from anomdet.modelling.lstm_autoencoder import LSTMAutoencoder
from anomdet.preprocessing.pipeline import prepare_features

LOGGER = logging.getLogger("anomdet")
METADATA_COLUMNS = {"row_id", "label", "protocol", "flow_id", "timestamp", "capture"}
IT_PROTOCOLS = ["ssh", "dns", "http"]
OT_PROTOCOLS = ["modbus", "s7comm"]
NORMAL_LABELS = {"benign", "normal", "0", "false", "no", "non-anomaly", "non_anomaly"}


def _detectors(
    config: dict[str, Any], model_overrides: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Construct configurable detector candidates with a common score convention."""
    models = config["models"]
    overrides = model_overrides or {}
    workers = effective_workers(int(config["runtime"]["cpu_workers"]))
    lstm_settings = {**models["lstm_autoencoder"], **overrides.get("lstm_autoencoder", {})}
    return {
        "isolation_forest": IsolationForest(
            n_estimators=int(models["isolation_forest"]["n_estimators"]),
            max_samples=models["isolation_forest"]["max_samples"],
            contamination=float(models["contamination"]),
            n_jobs=workers,
            random_state=int(config["project"]["random_seed"]),
        ),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=int(models["local_outlier_factor"]["n_neighbors"]),
            novelty=True,
            n_jobs=workers,
        ),
        "one_class_svm": OneClassSVM(
            nu=float(models["one_class_svm"]["nu"]), gamma=models["one_class_svm"]["gamma"]
        ),
        "pca_autoencoder": PCAAutoencoder(float(models["pca_autoencoder"]["explained_variance"])),
        "lstm_autoencoder": LSTMAutoencoder(
            **{
                **lstm_settings,
                "random_seed": int(
                    lstm_settings.get("random_seed", config["project"]["random_seed"])
                ),
            }
        ),
    }


def _anomaly_scores(model_name: str, model: Any, values: np.ndarray) -> np.ndarray:
    """Normalize every detector so larger scores always mean more anomalous."""
    raw = model.score_samples(values)
    return raw if model_name in {"pca_autoencoder", "lstm_autoencoder"} else -raw


def _binary_labels(labels: pd.Series | None) -> np.ndarray | None:
    """Convert known labels to normal/anomaly targets only when both classes are present."""
    if labels is None:
        return None
    clean = labels.astype("string").str.strip().str.lower()
    known = ~clean.isin(["", "unknown", "nan", "none", "<na>"])
    if known.sum() < 2:
        return None
    binary = (~clean.isin(NORMAL_LABELS)).astype(int)
    return binary.to_numpy() if binary.nunique() == 2 else None


def _normal_training_mask(labels: pd.Series | None) -> np.ndarray | None:
    """Use labelled normal observations for fitting when the dataset provides them."""
    if labels is None:
        return None
    clean = labels.astype("string").str.strip().str.lower()
    mask = clean.isin(NORMAL_LABELS).to_numpy()
    return mask if mask.sum() >= 10 else None


def _scope_protocols(strategy: str, group: str, present: list[str]) -> dict[str, list[str]]:
    """Return deployment scopes for per-protocol or grouped model training."""
    if strategy == "per_protocol":
        return {protocol: [protocol] for protocol in present}
    candidates = (
        IT_PROTOCOLS
        if group == "it"
        else OT_PROTOCOLS
        if group == "ot"
        else IT_PROTOCOLS + OT_PROTOCOLS
    )
    selected = [protocol for protocol in candidates if protocol in present]
    return {f"grouped_{group}": selected} if selected else {}


def _metrics(
    scores: np.ndarray, labels: np.ndarray | None, threshold: float
) -> dict[str, float | None]:
    """Compute unsupervised operational metrics and supervised evidence when labels permit it."""
    result: dict[str, float | None] = {
        "score_mean": round(float(np.mean(scores)), 6),
        "score_std": round(float(np.std(scores)), 6),
        "score_p50": round(float(np.quantile(scores, 0.50)), 6),
        "score_p95": round(float(np.quantile(scores, 0.95)), 6),
        "score_p99": round(float(np.quantile(scores, 0.99)), 6),
        "threshold": round(float(threshold), 6),
        "predicted_anomaly_rate": round(float((scores >= threshold).mean()), 6),
        "roc_auc": None,
        "average_precision": None,
    }
    if labels is not None and len(np.unique(labels)) == 2:
        result["roc_auc"] = round(float(roc_auc_score(labels, scores)), 6)
        result["average_precision"] = round(float(average_precision_score(labels, scores)), 6)
    return result


def _raw_feature_name(transformed: str, selected_features: list[str]) -> str:
    """Map one-hot transformer output back to its source catalogue feature when possible."""
    for feature in sorted(selected_features, key=len, reverse=True):
        if transformed == feature or transformed.startswith(f"{feature}_"):
            return feature
    return transformed


def _importance(
    model_name: str,
    model: Any,
    values: np.ndarray,
    input_columns: list[str],
    selected_features: list[str],
    random_seed: int,
    maximum_rows: int = 4000,
) -> pd.DataFrame:
    """Estimate model-specific feature contribution on an evaluation sample.

    Tree and PCA models expose learned structure directly. For other detectors,
    including LSTM AE, score-shift permutation estimates how much each feature
    affects the detector's anomaly assessment without needing labels.
    """
    rows: list[dict[str, float | int | str]] = []
    if model_name == "pca_autoencoder":
        weighted = (
            np.square(model.model.components_) * model.model.explained_variance_ratio_[:, None]
        )
        values_by_feature = np.sqrt(weighted.sum(axis=0))
        method = "pca_weighted_loading"
    elif model_name == "isolation_forest":
        trees = [tree.feature_importances_ for tree in model.estimators_]
        values_by_feature = np.mean(trees, axis=0)
        method = "isolation_tree_split_gain"
    else:
        sample = values
        if len(sample) > maximum_rows:
            positions = np.linspace(0, len(sample) - 1, maximum_rows, dtype=int)
            sample = sample[positions]
        baseline = _anomaly_scores(model_name, model, sample)
        rng = np.random.default_rng(random_seed)
        values_by_feature = np.zeros(len(input_columns), dtype=float)
        for position in range(len(input_columns)):
            permuted = sample.copy()
            permuted[:, position] = rng.permutation(permuted[:, position])
            shifted = _anomaly_scores(model_name, model, permuted)
            values_by_feature[position] = float(np.mean(np.abs(shifted - baseline)))
        method = "permutation_score_shift"
    for transformed, importance in zip(input_columns, values_by_feature, strict=True):
        rows.append(
            {
                "feature": _raw_feature_name(transformed, selected_features),
                "transformed_feature": transformed,
                "importance": float(importance),
                "method": method,
            }
        )
    result = pd.DataFrame(rows)
    result = (
        result.groupby(["feature", "method"], as_index=False)
        .agg(
            raw_importance=("importance", "sum"),
            transformed_columns=("transformed_feature", "count"),
        )
        .sort_values("raw_importance", ascending=False, kind="stable")
    )
    total = max(float(result["raw_importance"].sum()), np.finfo(float).eps)
    result["importance"] = result["raw_importance"] / total
    result["importance_rank"] = np.arange(1, len(result) + 1)
    result["raw_importance"] = result["raw_importance"].round(6)
    result["importance"] = result["importance"].round(8)
    return result


def _split_positions(length: int, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible hold-out split, with temporal order available for sequence models."""
    split = float(config["models"]["train_split"])
    if config["models"].get("split_strategy", "temporal") == "temporal":
        boundary = max(1, min(length - 1, int(length * split)))
        return np.arange(boundary), np.arange(boundary, length)
    positions = np.arange(length)
    train, test = train_test_split(
        positions,
        train_size=split,
        random_state=int(config["project"]["random_seed"]),
    )
    return np.sort(train), np.sort(test)


def train_models(
    feature_path: Path,
    config: dict[str, Any],
    strategy: str,
    group: str,
    profile: str | Path | None,
    output_dir: Path,
    labels_path: Path | None = None,
    candidates: list[str] | None = None,
    model_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare selected data and compare detectors for each requested deployment scope."""
    if strategy not in {"per_protocol", "grouped"}:
        raise ValueError("Strategy must be 'per_protocol' or 'grouped'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = read_table(feature_path)
    if "protocol" not in source.columns:
        raise ValueError("Feature input must include a protocol column.")
    present = sorted(source["protocol"].dropna().unique().tolist())
    scopes = _scope_protocols(strategy, group, present)
    if not scopes:
        raise ValueError(f"No protocols available for group '{group}'.")

    candidate_names = candidates or config["models"]["candidates"]
    available_models = _detectors(config, model_overrides)
    unknown = set(candidate_names).difference(available_models)
    if unknown:
        raise ValueError(f"Unknown configured detector names: {', '.join(sorted(unknown))}")
    report_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn()
    )
    with progress:
        task = progress.add_task(
            "Training anomaly detector candidates", total=len(scopes) * len(candidate_names)
        )
        for scope_name, protocols in scopes.items():
            scope_dir = output_dir / scope_name
            prepared_path = scope_dir / "prepared.parquet"
            scope_source = source[source["protocol"].isin(protocols)].reset_index(drop=True)
            if len(scope_source) < 12:
                LOGGER.warning("Skipping %s: fewer than 12 rows are available", scope_name)
                progress.advance(task, len(candidate_names))
                continue
            train_idx, test_idx = _split_positions(len(scope_source), config)
            prepared, _prepare_manifest, pipeline_path = prepare_features(
                feature_path,
                prepared_path,
                config,
                profile=profile,
                protocols=protocols,
                labels_path=labels_path,
                fit_positions=train_idx,
            )
            input_columns = [
                column for column in prepared.columns if column not in METADATA_COLUMNS
            ]
            if len(prepared) < 12 or not input_columns:
                LOGGER.warning(
                    "Skipping %s: insufficient data (%s rows, %s features)",
                    scope_name,
                    len(prepared),
                    len(input_columns),
                )
                progress.advance(task, len(candidate_names))
                continue
            values = prepared[input_columns].to_numpy(dtype=float)
            labels = prepared["label"] if "label" in prepared.columns else None
            normal_mask = _normal_training_mask(
                labels.iloc[train_idx] if labels is not None else None
            )
            fit_values = (
                values[train_idx][normal_mask] if normal_mask is not None else values[train_idx]
            )
            if len(fit_values) < 10:
                fit_values = values[train_idx]
            test_labels = _binary_labels(labels.iloc[test_idx] if labels is not None else None)
            for model_name in candidate_names:
                model = _detectors(config, model_overrides)[model_name]
                if model_name == "local_outlier_factor":
                    model.n_neighbors = min(model.n_neighbors, max(2, len(fit_values) - 1))
                memory_before_mb = psutil.Process().memory_info().rss / 1024**2
                started_at = time.perf_counter()
                model.fit(fit_values)
                fit_seconds = time.perf_counter() - started_at
                memory_after_mb = psutil.Process().memory_info().rss / 1024**2
                train_scores = _anomaly_scores(model_name, model, fit_values)
                threshold = float(
                    np.quantile(train_scores, 1 - float(config["models"]["contamination"]))
                )
                scores = _anomaly_scores(model_name, model, values[test_idx])
                score_frame = prepared.iloc[test_idx][
                    [column for column in METADATA_COLUMNS if column in prepared.columns]
                ].copy()
                score_frame["scope"] = scope_name
                score_frame["model"] = model_name
                score_frame["anomaly_score"] = scores
                score_frame["score_percentile"] = pd.Series(scores).rank(pct=True).to_numpy()
                score_frame["is_anomaly"] = scores >= threshold
                model_dir = scope_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                selected_features = _prepare_manifest["selected_input_features"]
                model_metadata = {
                    "model": model_name,
                    "pipeline_path": str(pipeline_path),
                    "input_columns": input_columns,
                    "selected_input_features": selected_features,
                    "scope": scope_name,
                    "protocols": protocols,
                }
                if model_name == "lstm_autoencoder":
                    model.save(model_dir / "model.pt")
                    model_metadata["model_artifact"] = str(model_dir / "model.pt")
                    model_metadata["parameters"] = model._parameters()
                else:
                    joblib.dump(
                        {
                            "model": model,
                            "pipeline_path": str(pipeline_path),
                            "input_columns": input_columns,
                        },
                        model_dir / "model.joblib",
                    )
                    model_metadata["model_artifact"] = str(model_dir / "model.joblib")
                write_table(score_frame, model_dir / "scores.parquet")
                metrics = _metrics(scores, test_labels, threshold)
                if model_name in {"pca_autoencoder", "lstm_autoencoder"}:
                    reconstruction_mse = float(np.mean(scores))
                    metrics["reconstruction_mse_mean"] = round(reconstruction_mse, 8)
                    metrics["reconstruction_rmse"] = round(float(np.sqrt(reconstruction_mse)), 8)
                if model_name == "lstm_autoencoder":
                    history = pd.DataFrame(model.training_history_)
                    if not history.empty:
                        write_table(history, model_dir / "training-history.parquet")
                        metrics["epochs_completed"] = int(len(history))
                        metrics["best_training_loss"] = round(float(history["train_loss"].min()), 8)
                        if history["validation_loss"].notna().any():
                            metrics["best_validation_loss"] = round(
                                float(history["validation_loss"].min()), 8
                            )
                importance = _importance(
                    model_name,
                    model,
                    values[test_idx],
                    input_columns,
                    selected_features,
                    int(config["project"]["random_seed"]),
                )
                importance.insert(0, "model", model_name)
                importance.insert(0, "scope", scope_name)
                write_table(importance, model_dir / "feature-importance.parquet")
                importance_rows.append(importance)
                write_json(metrics, model_dir / "metrics.json")
                write_json(model_metadata, model_dir / "model.json")
                report_rows.append(
                    {
                        "scope": scope_name,
                        "protocols": ",".join(protocols),
                        "model": model_name,
                        "training_rows": len(fit_values),
                        "test_rows": len(test_idx),
                        "input_features": len(input_columns),
                        "fit_seconds": round(fit_seconds, 4),
                        "process_memory_delta_mb": round(memory_after_mb - memory_before_mb, 3),
                        "importance_method": str(importance["method"].iloc[0])
                        if not importance.empty
                        else None,
                        **metrics,
                    }
                )
                progress.advance(task)
    comparison = pd.DataFrame(report_rows)
    if not comparison.empty:
        ordering = ["average_precision", "roc_auc", "score_p95"]
        comparison = comparison.sort_values(
            ordering, ascending=[False, False, True], na_position="last", kind="stable"
        )
    write_table(comparison, output_dir / "comparison.parquet")
    if importance_rows:
        write_table(
            pd.concat(importance_rows, ignore_index=True), output_dir / "feature-importance.parquet"
        )
    summary = {
        "created_at": utc_now(),
        "source_features": str(feature_path),
        "labels": str(labels_path) if labels_path else None,
        "strategy": strategy,
        "group": group,
        "profile": str(profile) if profile else "all_catalogue_features",
        "detectors": candidate_names,
        "scopes": scopes,
        "comparison": str(output_dir / "comparison.parquet"),
        "feature_importance": str(output_dir / "feature-importance.parquet")
        if importance_rows
        else None,
        "model_count": len(comparison),
    }
    write_json(summary, output_dir / "experiment.json")
    LOGGER.info("Completed %s detector runs in %s", len(comparison), output_dir)
    return comparison, summary


def _safe_profile_id(profile: str | Path | None) -> str:
    """Create a stable directory name while keeping the all-feature baseline prominent."""
    if profile is None:
        return "all-features"
    raw = Path(profile).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-") or "selected-features"


def run_feature_experiments(
    feature_path: Path,
    config: dict[str, Any],
    strategy: str,
    group: str,
    profiles: list[str | Path | None],
    output_dir: Path,
    labels_path: Path | None = None,
    candidates: list[str] | None = None,
    model_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run an all-feature baseline and any number of selected feature profiles.

    This is the common contract used by the batch pipeline, CLI integrations,
    and the dashboard's experiment studio. It makes profile-vs-baseline
    comparisons explicit in one aggregate table rather than requiring users to
    discover separate per-profile directories.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    requested: list[str | Path | None] = [None]
    for profile in profiles:
        if profile is not None and str(profile) not in {
            str(item) for item in requested if item is not None
        }:
            requested.append(profile)
    all_rows: list[pd.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []
    for profile in requested:
        profile_id = _safe_profile_id(profile)
        target = output_dir / profile_id
        try:
            comparison, summary = train_models(
                feature_path=feature_path,
                config=config,
                strategy=strategy,
                group=group,
                profile=profile,
                output_dir=target,
                labels_path=labels_path,
                candidates=candidates,
                model_overrides=model_overrides,
            )
            comparison = comparison.copy()
            comparison.insert(
                0, "feature_profile", "all_features" if profile is None else str(profile)
            )
            all_rows.append(comparison)
            run_summaries.append(
                {
                    "feature_profile": "all_features" if profile is None else str(profile),
                    "status": "complete",
                    "output": str(target),
                    "model_runs": len(comparison),
                    "experiment": summary,
                }
            )
        except Exception as error:  # Keep other selected profiles inspectable after one failure.
            LOGGER.exception("Feature experiment failed for profile '%s'", profile_id)
            run_summaries.append(
                {
                    "feature_profile": "all_features" if profile is None else str(profile),
                    "status": "failed",
                    "error": str(error),
                }
            )
    aggregate = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    comparison_path = output_dir / "comparison.parquet"
    write_table(aggregate, comparison_path)
    summary = {
        "created_at": utc_now(),
        "source_features": str(feature_path),
        "strategy": strategy,
        "group": group,
        "detectors": candidates or config["models"]["candidates"],
        "comparison": str(comparison_path),
        "runs": run_summaries,
        "successful_profiles": sum(item["status"] == "complete" for item in run_summaries),
    }
    write_json(summary, output_dir / "experiment-batch.json")
    return aggregate, summary


def run_lstm_sweep(
    feature_path: Path,
    config: dict[str, Any],
    strategy: str,
    group: str,
    profiles: list[str | Path | None],
    parameter_sets: list[dict[str, Any]],
    output_dir: Path,
    labels_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run labelled LSTM parameter variants across the baseline and selected profiles."""
    if not parameter_sets:
        raise ValueError("At least one LSTM parameter set is required for a sweep.")
    requested: list[str | Path | None] = [None]
    for profile in profiles:
        if profile is not None and str(profile) not in {
            str(item) for item in requested if item is not None
        }:
            requested.append(profile)
    rows: list[pd.DataFrame] = []
    runs: list[dict[str, Any]] = []
    for variant_index, parameters in enumerate(parameter_sets, start=1):
        variant_id = f"variant-{variant_index:02d}"
        for profile in requested:
            profile_id = _safe_profile_id(profile)
            target = output_dir / variant_id / profile_id
            try:
                comparison, summary = train_models(
                    feature_path=feature_path,
                    config=config,
                    strategy=strategy,
                    group=group,
                    profile=profile,
                    output_dir=target,
                    labels_path=labels_path,
                    candidates=["lstm_autoencoder"],
                    model_overrides={"lstm_autoencoder": parameters},
                )
                comparison = comparison.copy()
                comparison.insert(
                    0, "feature_profile", "all_features" if profile is None else str(profile)
                )
                comparison.insert(1, "sweep_variant", variant_id)
                for key, value in parameters.items():
                    comparison[key] = value
                rows.append(comparison)
                runs.append(
                    {
                        "variant": variant_id,
                        "parameters": parameters,
                        "feature_profile": "all_features" if profile is None else str(profile),
                        "status": "complete",
                        "output": str(target),
                        "experiment": summary,
                    }
                )
            except (
                Exception
            ) as error:  # Preserve successful variants for comparison after one failure.
                LOGGER.exception("LSTM sweep failed for %s / %s", variant_id, profile_id)
                runs.append(
                    {
                        "variant": variant_id,
                        "parameters": parameters,
                        "feature_profile": "all_features" if profile is None else str(profile),
                        "status": "failed",
                        "error": str(error),
                    }
                )
    aggregate = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    comparison_path = output_dir / "sweep-comparison.parquet"
    write_table(aggregate, comparison_path)
    summary = {
        "created_at": utc_now(),
        "source_features": str(feature_path),
        "strategy": strategy,
        "group": group,
        "comparison": str(comparison_path),
        "runs": runs,
    }
    write_json(summary, output_dir / "lstm-sweep.json")
    return aggregate, summary
