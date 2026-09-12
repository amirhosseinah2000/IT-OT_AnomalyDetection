"""CPU-first two-stage detector: LSTM-AE + Isolation Forest, then Random Forest."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from anomdet.analytics.reports import preprocessing_comparison
from anomdet.core.io import utc_now, write_json, write_table
from anomdet.core.progress import DurableProgress
from anomdet.core.resources import effective_workers, snapshot
from anomdet.modelling.lstm_autoencoder import LSTMAutoencoder
from anomdet.modelling.random_forest import (
    make_stage_two_matrix,
    predict_attack_type,
    stage_two_readiness,
    train_attack_random_forest,
)
from anomdet.preprocessing.pipeline import (
    prepare_features,
    read_training_source,
    transform_with_manifest,
)

METADATA_COLUMNS = {"row_id", "label", "protocol", "flow_id", "timestamp", "capture"}
NORMAL_LABELS = {"benign", "normal", "0", "false", "no", "non-anomaly", "non_anomaly"}
UNKNOWN_LABELS = {"", "unknown", "nan", "none", "<na>", "-1", "needmanuallabel"}


def _load_sources(
    paths: list[Path], config: dict[str, Any], protocol: str | None = None
) -> pd.DataFrame:
    sources = [path for path in paths if path.exists()]
    maximum = config["models"].get("max_source_rows")
    source_sizes = [
        int(pq.ParquetFile(path).metadata.num_rows)
        if path.suffix.lower() in {".parquet", ".pq"}
        else 0
        for path in sources
    ]
    total_source_rows = sum(source_sizes)
    per_source_limits: list[int | None] = [None] * len(sources)
    if maximum is not None and total_source_rows > int(maximum) and sources:
        # A protocol can include dozens of captures.  Allocate one global,
        # time-spread budget across them instead of silently reading the full
        # cap from every capture; this keeps CPU/RAM bounded without dropping a
        # rare attack capture from the training/evaluation population.
        budget = int(maximum)
        minimum = min(500, max(1, budget // len(sources)))
        base = [min(size, minimum) for size in source_sizes]
        remaining = max(0, budget - sum(base))
        capacities = [max(0, size - allocated) for size, allocated in zip(source_sizes, base)]
        capacity_total = max(sum(capacities), 1)
        additions = [
            min(capacity, int(remaining * capacity / capacity_total)) for capacity in capacities
        ]
        allocated = [base_value + addition for base_value, addition in zip(base, additions)]
        leftover = max(0, budget - sum(allocated))
        for index in np.argsort(-np.asarray(capacities)):
            if leftover == 0:
                break
            if allocated[int(index)] < source_sizes[int(index)]:
                allocated[int(index)] += 1
                leftover -= 1
        per_source_limits = allocated
    frames = [
        read_training_source(path, per_source_limits[index])
        for index, path in enumerate(sources)
    ]
    if not frames:
        raise FileNotFoundError("No source feature artifacts were found for this model run.")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if protocol is not None and "protocol" in frame:
        frame = frame[
            frame["protocol"].astype("string").str.casefold() == protocol.casefold()
        ].copy()
    frame = frame.reset_index(drop=True)
    frame.attrs["source_rows_total"] = total_source_rows
    frame.attrs["loaded_rows"] = len(frame)
    frame.attrs["global_model_row_cap"] = maximum
    frame.attrs["source_count"] = len(sources)
    return frame


def discover_dataset_run_sources(
    dataset_run: Path, protocol: str | None = None
) -> tuple[list[Path], list[Path]]:
    """Return protocol-separated benign and already mapped attack records from a dataset run."""
    benign = sorted((dataset_run / "features" / "benign").glob("*/records.parquet"))
    attacks = sorted((dataset_run / "labelled" / "attack").glob("*/*/records.parquet"))
    if protocol is not None:
        scope = protocol.casefold()
        benign = [path for path in benign if path.parent.name.casefold() == scope]
        attacks = [path for path in attacks if path.parents[1].name.casefold() == scope]
    return benign, attacks


def _binary_label(frame: pd.DataFrame) -> np.ndarray:
    if "is_attack" in frame:
        return frame["is_attack"].fillna(False).astype(bool).to_numpy(dtype=int)
    labels = (
        frame.get("label", pd.Series("unknown", index=frame.index)).astype("string").str.casefold()
    )
    return (~labels.isin(NORMAL_LABELS | {"unknown", "nan", "<na>", "-1"})).to_numpy(dtype=int)


def _known_binary_label_mask(frame: pd.DataFrame) -> np.ndarray:
    """Identify packet records with a trustworthy normal-vs-attack target.

    A capture-level mapping may be accepted while individual packets remain
    `unknown`; treating those as benign silently poisons calibration and makes
    recall/FPR claims invalid.  An explicit boolean target is accepted only
    when no textual label is present.
    """

    if "label" in frame:
        labels = frame["label"].astype("string").str.strip().str.casefold()
        return (~labels.isin(UNKNOWN_LABELS)).to_numpy(dtype=bool)
    if "is_attack" in frame:
        return frame["is_attack"].notna().to_numpy(dtype=bool)
    return np.zeros(len(frame), dtype=bool)


def _split_normal(length: int, fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ordered train, calibration, and untouched normal-evaluation slices.

    Stage 1 is unsupervised, but threshold selection still needs normal traffic.
    Keeping a third slice that is never used for fitting *or* threshold tuning
    makes the reported false-positive rate meaningful instead of optimistic.
    """
    if length < 12:
        raise ValueError("Stage one needs at least 12 normal PCAP feature records.")
    requested = max(2, int(round(length * fraction)))
    # Retain at least eight records for model fitting, including in compact
    # smoke fixtures.  Large production runs use the configured fraction.
    split_size = min(requested, max(2, (length - 8) // 2))
    train_end = length - (2 * split_size)
    calibration_end = train_end + split_size
    return (
        np.arange(0, train_end),
        np.arange(train_end, calibration_end),
        np.arange(calibration_end, length),
    )


def _sequence_groups(frame: pd.DataFrame) -> np.ndarray:
    """Return a stable capture key for boundary-safe LSTM sequences."""

    return (
        frame.get("capture", pd.Series("unknown-capture", index=frame.index))
        .astype("string")
        .fillna("unknown-capture")
        .astype(str)
        .to_numpy(dtype=object)
    )


def _normal_capture_split(
    frame: pd.DataFrame, fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Split clean normal traffic by capture before falling back to time slices.

    A point-in-time split of one concatenated capture is optimistic for network
    traffic: packets in the same flow can land in both fit and test.  When at
    least three benign captures exist, the fit, calibration and test partitions
    are strictly capture-disjoint.  Two-capture protocols still reserve one
    entire capture for the normal test and use a tail of the training capture
    only for calibration.
    """

    capture_values = _sequence_groups(frame)
    captures = list(dict.fromkeys(capture_values.tolist()))
    if len(captures) >= 3:
        requested = max(1, int(round(len(captures) * fraction)))
        reserved = min(requested, max(1, (len(captures) - 1) // 2))
        test_captures = captures[-reserved:]
        calibration_captures = captures[-2 * reserved : -reserved]
        train_captures = captures[: -2 * reserved]
        train = np.flatnonzero(np.isin(capture_values, train_captures))
        calibration = np.flatnonzero(np.isin(capture_values, calibration_captures))
        evaluation = np.flatnonzero(np.isin(capture_values, test_captures))
        if len(train) >= 8 and len(calibration) and len(evaluation):
            return train, calibration, evaluation, {
                "strategy": "capture_disjoint",
                "training_captures": train_captures,
                "calibration_captures": calibration_captures,
                "evaluation_captures": test_captures,
            }
    if len(captures) == 2:
        train_capture, evaluation_capture = captures
        train_candidates = np.flatnonzero(capture_values == train_capture)
        requested = max(2, int(round(len(train_candidates) * fraction)))
        calibration_size = min(requested, max(2, len(train_candidates) - 8))
        train = train_candidates[:-calibration_size]
        calibration = train_candidates[-calibration_size:]
        evaluation = np.flatnonzero(capture_values == evaluation_capture)
        if len(train) >= 8 and len(calibration) and len(evaluation):
            return train, calibration, evaluation, {
                "strategy": "capture_holdout_with_within_capture_calibration",
                "training_captures": [train_capture],
                "calibration_captures": [train_capture],
                "evaluation_captures": [evaluation_capture],
            }
    train, calibration, evaluation = _split_normal(len(frame), fraction)
    return train, calibration, evaluation, {
        "strategy": "ordered_row_fallback",
        "training_captures": captures,
        "calibration_captures": captures,
        "evaluation_captures": captures,
    }


def _attack_capture_roles(
    frame: pd.DataFrame, attack_truth: np.ndarray, fraction: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign known attack captures to train/calibration/test roles.

    Only captures with at least one packet-level attack label may make a
    supervised decision or metric.  Records that lack a trusted attack label
    are deliberately excluded from accuracy claims.
    """

    capture_values = _sequence_groups(frame)
    positive_captures = list(
        dict.fromkeys(capture_values[np.asarray(attack_truth, dtype=bool)].tolist())
    )
    roles = np.full(len(frame), "unlabelled", dtype=object)
    if len(positive_captures) < 3:
        return roles, {
            "available": False,
            "reason": "requires_at_least_three_distinct_labelled_attack_captures",
            "positive_captures": positive_captures,
            "training_captures": [],
            "calibration_captures": [],
            "evaluation_captures": [],
        }
    requested = max(1, int(round(len(positive_captures) * fraction)))
    reserved = min(requested, max(1, (len(positive_captures) - 1) // 2))
    evaluation_captures = positive_captures[-reserved:]
    calibration_captures = positive_captures[-2 * reserved : -reserved]
    training_captures = positive_captures[: -2 * reserved]
    roles[np.isin(capture_values, training_captures)] = "train"
    roles[np.isin(capture_values, calibration_captures)] = "calibration"
    roles[np.isin(capture_values, evaluation_captures)] = "evaluation"
    return roles, {
        "available": True,
        "positive_captures": positive_captures,
        "training_captures": training_captures,
        "calibration_captures": calibration_captures,
        "evaluation_captures": evaluation_captures,
    }


def _fit_stage_one_binary_gate(
    normal_values: np.ndarray,
    normal_train_idx: np.ndarray,
    attack_values: np.ndarray,
    attack_truth: np.ndarray,
    attack_roles: np.ndarray,
    config: dict[str, Any],
) -> tuple[HistGradientBoostingClassifier | None, dict[str, Any]]:
    """Fit a light binary gate only when packet labels cover multiple captures."""

    settings = dict(config["stage_one"].get("supervised_gate", {}))
    if not bool(settings.get("enabled", True)):
        return None, {"status": "skipped", "reason": "disabled"}
    attack_train = np.flatnonzero(attack_roles == "train")
    if not len(attack_train):
        return None, {
            "status": "skipped",
            "reason": "no_capture_disjoint_labelled_attack_train_rows",
        }
    values = np.vstack([normal_values[normal_train_idx], attack_values[attack_train]])
    labels = np.concatenate(
        [np.zeros(len(normal_train_idx), dtype=int), attack_truth[attack_train]]
    )
    minimum = int(settings.get("min_labelled_attack_records", 50))
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    if positives < minimum or negatives < 2 or len(np.unique(labels)) < 2:
        return None, {
            "status": "skipped",
            "reason": "insufficient_capture_disjoint_binary_labels",
            "positive_records": positives,
            "negative_records": negatives,
            "minimum_labelled_attack_records": minimum,
        }
    classifier = HistGradientBoostingClassifier(
        learning_rate=float(settings.get("learning_rate", 0.08)),
        max_iter=int(settings.get("max_iter", 180)),
        max_leaf_nodes=int(settings.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(settings.get("min_samples_leaf", 20)),
        l2_regularization=float(settings.get("l2_regularization", 1.0)),
        early_stopping=True,
        validation_fraction=float(settings.get("validation_fraction", 0.15)),
        random_state=int(config["project"]["random_seed"]),
    )
    classifier.fit(values, labels, sample_weight=compute_sample_weight("balanced", labels))
    return classifier, {
        "status": "trained",
        "input_features": int(values.shape[1]),
        "positive_records": positives,
        "negative_records": negatives,
        "training_records": int(len(labels)),
        "parameters": {
            "learning_rate": classifier.learning_rate,
            "max_iter": classifier.max_iter,
            "max_leaf_nodes": classifier.max_leaf_nodes,
            "min_samples_leaf": classifier.min_samples_leaf,
            "l2_regularization": classifier.l2_regularization,
        },
    }


def _stage_one_hybrid_decision(
    lstm_scores: np.ndarray,
    forest_scores: np.ndarray,
    thresholds: dict[str, Any],
    mode: str,
    weights: dict[str, Any] | None = None,
    supervised_probability: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fuse unsupervised novelty and a capture-disjoint binary attack gate."""

    unsupervised, unsupervised_normalized, raw_score, lstm_hit, forest_hit = _stage1_decision(
        lstm_scores, forest_scores, thresholds, mode, weights
    )
    if supervised_probability is None or "supervised" not in thresholds:
        return {
            "is_anomaly": unsupervised,
            "normalized_score": unsupervised_normalized,
            "ensemble_raw_score": raw_score,
            "lstm_hit": lstm_hit,
            "forest_hit": forest_hit,
            "unsupervised_normalized_score": unsupervised_normalized,
            "supervised_probability": np.full(len(unsupervised), np.nan),
        }
    probability = np.asarray(supervised_probability, dtype=float)
    supervised_threshold = max(float(thresholds["supervised"]["threshold"]), 1e-12)
    combined = np.maximum(unsupervised_normalized, probability / supervised_threshold)
    final_threshold = max(float(thresholds["final"]["threshold"]), 1e-12)
    normalized = combined / final_threshold
    return {
        "is_anomaly": normalized >= 1.0,
        "normalized_score": normalized,
        "ensemble_raw_score": raw_score,
        "lstm_hit": lstm_hit,
        "forest_hit": forest_hit,
        "unsupervised_normalized_score": unsupervised_normalized,
        "supervised_probability": probability,
    }


def _threshold(scores: np.ndarray, target_fpr: float) -> dict[str, float]:
    """Use a validation quantile and robust MAD guardrail for low-FPR calibration."""
    target = min(max(float(target_fpr), 0.0001), 0.25)
    quantile = float(np.quantile(scores, 1 - target))
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust = median + 3.5 * 1.4826 * mad
    raw_threshold = max(quantile, robust)
    # A quantile can coincide with one or more validation values.  The model
    # decision uses ``>=`` so retain the next representable float; otherwise
    # all tied boundary points become false alarms and violate the FPR budget.
    decision_threshold = float(np.nextafter(raw_threshold, np.inf))
    return {
        "threshold": decision_threshold,
        "raw_threshold": raw_threshold,
        "quantile_threshold": quantile,
        "mad_threshold": robust,
        "target_false_positive_rate": target,
        "decision_operator": ">=",
    }


def _ensemble_score(
    lstm_scores: np.ndarray,
    forest_scores: np.ndarray,
    lstm_threshold: float,
    forest_threshold: float,
    mode: str,
    weights: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return one stable, dimensionless Stage-1 risk score.

    Scores from reconstruction and isolation models have different units.  We
    first express each against its normal-only calibration threshold.  The
    default weighted form rewards agreement, but does not require it: a very
    strong single-model signal can still cross the final calibrated threshold.
    """
    lstm_normalized = np.asarray(lstm_scores, dtype=float) / max(float(lstm_threshold), 1e-12)
    forest_normalized = np.asarray(forest_scores, dtype=float) / max(
        float(forest_threshold), 1e-12
    )
    if mode == "calibrated_weighted":
        source = weights or {}
        lstm_weight = max(float(source.get("lstm", 0.55)), 0.0)
        forest_weight = max(float(source.get("isolation_forest", 0.45)), 0.0)
        agreement_bonus = max(float(source.get("agreement_bonus", 0.20)), 0.0)
        denominator = max(lstm_weight + forest_weight + agreement_bonus, 1e-12)
        return (
            lstm_weight * lstm_normalized
            + forest_weight * forest_normalized
            + agreement_bonus * np.minimum(lstm_normalized, forest_normalized)
        ) / denominator
    if mode == "both_detectors":
        return np.minimum(lstm_normalized, forest_normalized)
    # ``calibrated_max`` and the legacy ``either_detector`` are intentionally
    # equivalent as a score; only calibration/decision policy differs.
    return np.maximum(lstm_normalized, forest_normalized)


def _stage1_decision(
    lstm_scores: np.ndarray,
    forest_scores: np.ndarray,
    thresholds: dict[str, Any],
    mode: str,
    weights: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen Stage-1 contract and return decision plus evidence.

    The returned normalized ensemble score has a fixed decision boundary of
    one.  Dashboard charts and external consumers can therefore compare runs
    safely even when their raw score ranges differ.
    """
    lstm_threshold = float(thresholds["lstm"]["threshold"])
    forest_threshold = float(thresholds["isolation_forest"]["threshold"])
    lstm_hit = np.asarray(lstm_scores, dtype=float) >= lstm_threshold
    forest_hit = np.asarray(forest_scores, dtype=float) >= forest_threshold
    raw_score = _ensemble_score(
        lstm_scores,
        forest_scores,
        lstm_threshold,
        forest_threshold,
        mode,
        weights,
    )
    if mode in {"calibrated_max", "calibrated_weighted"}:
        ensemble_threshold = float(thresholds["ensemble"]["threshold"])
        normalized_score = raw_score / max(ensemble_threshold, 1e-12)
        is_anomaly = normalized_score >= 1.0
    else:
        normalized_score = raw_score
        is_anomaly = (
            (lstm_hit | forest_hit)
            if mode == "either_detector"
            else (lstm_hit & forest_hit)
        )
    return is_anomaly, normalized_score, raw_score, lstm_hit, forest_hit


def _metrics(y_true: np.ndarray, predicted: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    output: dict[str, Any] = {
        "records": int(len(y_true)),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "accuracy": round(float(accuracy_score(y_true, predicted)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "false_positive_rate": round(fp / max(fp + tn, 1), 6),
        "false_negative_rate": round(fn / max(fn + tp, 1), 6),
        "specificity": round(tn / max(tn + fp, 1), 6),
    }
    if len(np.unique(y_true)) > 1:
        output["roc_auc"] = round(float(roc_auc_score(y_true, score)), 6)
        output["average_precision"] = round(float(average_precision_score(y_true, score)), 6)
    return output


def _contributions(
    values: np.ndarray,
    feature_names: list[str],
    normal_values: np.ndarray,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Give per-record, scaled feature evidence alongside the model decision.

    LSTM reconstruction and isolation scores are global decisions.  The evidence
    table is deliberately transparent: deviation from the normal median in the
    exact transformed feature space, weighted by isolation-forest importance.
    """
    medians = np.nanmedian(normal_values, axis=0)
    iqr = np.nanpercentile(normal_values, 75, axis=0) - np.nanpercentile(normal_values, 25, axis=0)
    safe_iqr = np.where(iqr > 1e-9, iqr, 1.0)
    deviations = np.abs(values - medians) / safe_iqr
    rows: list[dict[str, Any]] = []
    for record_index, ranking in enumerate(np.argsort(-deviations, axis=1)[:, :3]):
        for rank, feature_index in enumerate(ranking, start=1):
            rows.append(
                {
                    "record_id": int(scores.iloc[record_index]["record_id"]),
                    "rank": rank,
                    "feature": feature_names[int(feature_index)],
                    "robust_deviation": round(float(deviations[record_index, feature_index]), 6),
                    "feature_value": round(float(values[record_index, feature_index]), 6),
                    "stage1_anomaly": bool(scores.iloc[record_index]["stage1_anomaly"]),
                    "evidence_method": "scaled_distance_from_normal_median",
                }
            )
    return pd.DataFrame(rows)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a stable, dashboard-safe ratio for diagnostic artifacts."""
    return float(numerator) / max(float(denominator), 1.0)


def _end_to_end_metrics(
    true_attack: np.ndarray,
    true_type: np.ndarray,
    predicted_type: np.ndarray,
) -> dict[str, float]:
    """Metrics for the complete normal-vs-typed-attack deployment decision."""
    truth = np.where(true_attack.astype(bool), true_type.astype(str), "normal")
    predicted = predicted_type.astype(str)
    false_alarm = (~true_attack.astype(bool)) & (predicted != "normal")
    attack_predicted = predicted != "normal"
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "attack_recall": _safe_ratio(
            int((true_attack.astype(bool) & attack_predicted).sum()), int(true_attack.sum())
        ),
        "false_positive_rate": _safe_ratio(int(false_alarm.sum()), int((~true_attack).sum())),
    }


def _write_pipeline_diagnostics(
    *,
    output_dir: Path,
    config: dict[str, Any],
    model_columns: list[str],
    normal_values: np.ndarray,
    train_idx: np.ndarray,
    evaluation_idx: np.ndarray,
    type_values: np.ndarray,
    type_rows: pd.DataFrame,
    random_forest: Any,
    attack_score_positions: np.ndarray,
    evaluation_values: np.ndarray,
    scores: pd.DataFrame,
    lstm: LSTMAutoencoder,
    forest: IsolationForest,
    lstm_scores: np.ndarray,
    forest_scores: np.ndarray,
    classifier: RandomForestClassifier,
    stage2_columns: list[str],
    include_stage_one_scores: bool,
) -> dict[str, Any]:
    """Create held-out, end-to-end diagnostic artifacts for the dashboard.

    The random-forest split is reused exactly: its test attacks and Stage 1's
    untouched normal-evaluation slice form the evaluation set. None of the
    diagnostics refit Stage 1 or score training attack labels as test data.
    """
    pipeline_dir = output_dir / "pipeline"
    evaluation_config = config.get("evaluation", {})
    validation_count = len(evaluation_idx)
    attack_test_positions = np.asarray(random_forest.test_positions, dtype=int)
    held_out_positions = np.concatenate(
        [np.arange(validation_count, dtype=int), attack_score_positions[attack_test_positions]]
    )
    benchmark = scores.iloc[held_out_positions].reset_index(drop=True).copy()
    benchmark_values = evaluation_values[held_out_positions]
    benchmark_lstm = lstm_scores[held_out_positions]
    benchmark_forest = forest_scores[held_out_positions]
    benchmark_stage2, resolved_columns = make_stage_two_matrix(
        benchmark_values,
        model_columns,
        benchmark_lstm,
        benchmark_forest,
        include_stage_one_scores,
    )
    if resolved_columns != stage2_columns:
        raise RuntimeError("Stage 2 diagnostic matrix does not match the frozen model contract.")

    candidate_predictions = classifier.predict(benchmark_stage2).astype(str)
    probabilities = classifier.predict_proba(benchmark_stage2)
    max_probability = probabilities.max(axis=1)
    gate = benchmark["stage1_anomaly"].to_numpy(dtype=bool)
    final_predictions = np.where(gate, candidate_predictions, "normal")
    true_attack = benchmark["true_is_attack"].to_numpy(dtype=bool)
    true_type = benchmark.get("label", pd.Series("unknown", index=benchmark.index)).astype(str).to_numpy()
    final_success = true_attack & gate & (candidate_predictions == true_type)

    probability_frame = benchmark.copy()
    probability_frame["candidate_predicted_attack_type"] = candidate_predictions
    probability_frame["predicted_attack_type"] = final_predictions
    probability_frame["stage2_max_probability"] = max_probability
    # This is a ranking score rather than a calibrated probability. Naming it
    # explicitly avoids presenting product-of-model scores as a probability.
    probability_frame["end_to_end_confidence"] = (
        probability_frame["stage1_normalized_score"].clip(lower=0).to_numpy() * max_probability
    )
    probability_frame["end_to_end_success"] = final_success
    probability_frame["is_held_out_evaluation"] = True
    for column_index, label in enumerate(classifier.classes_.astype(str)):
        probability_frame[f"probability__{label}"] = probabilities[:, column_index]
    write_table(probability_frame, pipeline_dir / "end-to-end-probabilities.parquet")
    write_table(probability_frame, pipeline_dir / "end-to-end-evaluation.parquet")

    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(
                    probability_frame["stage1_normalized_score"].to_numpy(dtype=float),
                    np.linspace(
                        0.01,
                        0.99,
                        max(3, int(evaluation_config.get("threshold_points", 41))),
                    ),
                ),
                np.asarray([1.0]),
            ]
        )
    )
    threshold_rows: list[dict[str, Any]] = []
    normalized = probability_frame["stage1_normalized_score"].to_numpy(dtype=float)
    for threshold in thresholds:
        threshold_gate = normalized >= threshold
        threshold_predictions = np.where(threshold_gate, candidate_predictions, "normal")
        success = true_attack & threshold_gate & (candidate_predictions == true_type)
        false_alarm = (~true_attack) & threshold_gate
        missed = true_attack & ~threshold_gate
        wrong_type = true_attack & threshold_gate & (candidate_predictions != true_type)
        precision = _safe_ratio(int(success.sum()), int(threshold_gate.sum()))
        recall = _safe_ratio(int(success.sum()), int(true_attack.sum()))
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "stage1_flagged": int(threshold_gate.sum()),
                "final_precision": precision,
                "final_recall": recall,
                "final_f1": _safe_ratio(2 * precision * recall, precision + recall),
                "false_positive_rate": _safe_ratio(int(false_alarm.sum()), int((~true_attack).sum())),
                "successful_type_predictions": int(success.sum()),
                "false_alarms": int(false_alarm.sum()),
                "missed_attacks": int(missed.sum()),
                "wrong_attack_type": int(wrong_type.sum()),
                "end_to_end_accuracy": _end_to_end_metrics(
                    true_attack, true_type, threshold_predictions
                )["accuracy"],
            }
        )
    write_table(pd.DataFrame(threshold_rows), pipeline_dir / "threshold-sensitivity.parquet")

    attack_train_positions = np.asarray(random_forest.train_positions, dtype=int)
    baseline_train_values = np.vstack([normal_values[train_idx], type_values[attack_train_positions]])
    baseline_train_labels = np.concatenate(
        [
            np.full(len(train_idx), "normal", dtype=object),
            type_rows.get("label", pd.Series("unknown", index=type_rows.index))
            .iloc[attack_train_positions]
            .astype(str)
            .to_numpy(),
        ]
    )
    diagnostic_estimators = min(
        int(config["stage_two"]["n_estimators"]),
        max(10, int(evaluation_config.get("stage2_diagnostic_estimators", 40))),
    )
    diagnostic_workers = min(
        effective_workers(int(config["runtime"]["cpu_workers"])),
        max(1, int(evaluation_config.get("diagnostic_cpu_workers", 4))),
    )
    baseline = RandomForestClassifier(
        n_estimators=diagnostic_estimators,
        min_samples_leaf=int(config["stage_two"]["min_samples_leaf"]),
        max_features=config["stage_two"].get("max_features", "sqrt"),
        max_samples=config["stage_two"].get("max_samples"),
        max_depth=config["stage_two"].get("max_depth"),
        class_weight="balanced_subsample",
        n_jobs=diagnostic_workers,
        random_state=int(config["project"]["random_seed"]),
    ).fit(baseline_train_values, baseline_train_labels)
    baseline_predictions = baseline.predict(benchmark_values).astype(str)
    two_stage_metrics = _end_to_end_metrics(true_attack, true_type, final_predictions)
    baseline_metrics = _end_to_end_metrics(true_attack, true_type, baseline_predictions)
    ablation = pd.DataFrame(
        [
            {
                "model": "Two-stage LSTM-AE + Isolation Forest → Random Forest",
                "evaluation_scope": "held_out_normal_evaluation_plus_rf_test_attacks",
                "records": len(probability_frame),
                **two_stage_metrics,
            },
            {
                "model": "Single-stage Random Forest baseline",
                "evaluation_scope": "same_held_out_records",
                "records": len(probability_frame),
                "diagnostic_tree_cap": diagnostic_estimators,
                **baseline_metrics,
            },
        ]
    )
    write_table(ablation, pipeline_dir / "ablation-comparison.parquet")

    requested_sample = max(1, int(evaluation_config.get("latency_sample_records", 2000)))
    sample_count = min(len(evaluation_values), requested_sample)
    sample_positions = np.linspace(0, len(evaluation_values) - 1, sample_count, dtype=int)
    latency_values = evaluation_values[sample_positions]
    latency_groups = _sequence_groups(scores.iloc[sample_positions])
    elapsed_start = time.perf_counter()
    latency_lstm_scores = lstm.score_samples(
        latency_values, sequence_groups=latency_groups
    )
    lstm_seconds = time.perf_counter() - elapsed_start
    elapsed_start = time.perf_counter()
    latency_forest_scores = -forest.score_samples(latency_values)
    forest_seconds = time.perf_counter() - elapsed_start
    latency_stage2, _ = make_stage_two_matrix(
        latency_values,
        model_columns,
        latency_lstm_scores,
        latency_forest_scores,
        include_stage_one_scores,
    )
    elapsed_start = time.perf_counter()
    classifier.predict_proba(latency_stage2)
    stage2_seconds = time.perf_counter() - elapsed_start
    latency_rows: list[dict[str, Any]] = []
    for component, elapsed in [
        ("LSTM-AE", lstm_seconds),
        ("Isolation Forest", forest_seconds),
        ("Stage 2 Random Forest", stage2_seconds),
        ("Stage 1 total", lstm_seconds + forest_seconds),
        ("Full pipeline", lstm_seconds + forest_seconds + stage2_seconds),
    ]:
        latency_rows.append(
            {
                "component": component,
                "records": sample_count,
                "seconds": elapsed,
                "milliseconds_per_record": elapsed * 1000 / max(sample_count, 1),
                "records_per_second": sample_count / max(elapsed, 1e-12),
            }
        )
    write_table(pd.DataFrame(latency_rows), pipeline_dir / "inference-latency.parquet")

    train_type_values = type_values[attack_train_positions]
    train_type_lstm = lstm.score_samples(
        train_type_values,
        sequence_groups=_sequence_groups(type_rows.iloc[attack_train_positions]),
    )
    train_type_forest = -forest.score_samples(train_type_values)
    train_stage2, _ = make_stage_two_matrix(
        train_type_values,
        model_columns,
        train_type_lstm,
        train_type_forest,
        include_stage_one_scores,
    )
    train_type_labels = (
        type_rows.get("label", pd.Series("unknown", index=type_rows.index))
        .iloc[attack_train_positions]
        .astype(str)
    )
    stability_rows: list[dict[str, Any]] = []
    seeds = evaluation_config.get("seed_stability_seeds", [17, 37, 73, 101, 211])
    stability_estimators = min(
        int(config["stage_two"]["n_estimators"]),
        max(10, int(evaluation_config.get("seed_stability_estimators", 75))),
    )
    diagnostic_workers = min(
        effective_workers(int(config["runtime"]["cpu_workers"])),
        max(1, int(evaluation_config.get("diagnostic_cpu_workers", 4))),
    )
    for seed in [int(value) for value in seeds]:
        seeded = RandomForestClassifier(
            n_estimators=stability_estimators,
            min_samples_leaf=int(config["stage_two"]["min_samples_leaf"]),
            max_features=config["stage_two"].get("max_features", "sqrt"),
            max_samples=config["stage_two"].get("max_samples"),
            max_depth=config["stage_two"].get("max_depth"),
            class_weight="balanced_subsample",
            n_jobs=diagnostic_workers,
            random_state=seed,
        ).fit(train_stage2, train_type_labels)
        seed_candidates = seeded.predict(benchmark_stage2).astype(str)
        seed_predictions = np.where(gate, seed_candidates, "normal")
        stability_rows.append(
            {
                "seed": seed,
                "scope": "Stage 1 frozen; bounded Stage 2 Random Forest diagnostic refit",
                "records": len(probability_frame),
                "diagnostic_tree_cap": stability_estimators,
                **_end_to_end_metrics(true_attack, true_type, seed_predictions),
            }
        )
    write_table(pd.DataFrame(stability_rows), pipeline_dir / "seed-stability.parquet")
    return {
        "held_out_records": len(probability_frame),
        "threshold_points": len(threshold_rows),
        "latency_records": sample_count,
        "seed_runs": len(stability_rows),
    }


def _write_stage1_learning_curve(
    *,
    normal_values: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    sequence_groups: np.ndarray,
    lstm_config: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Measure Stage 1 reconstruction quality after controlled data-size refits.

    This is intentionally a bounded diagnostic, distinct from the production
    LSTM. It records its epoch cap in every row so its comparison is useful
    without pretending those auxiliary models are the deployed detector.
    """
    evaluation = config.get("evaluation", {})
    diagnostic_epochs = min(
        int(lstm_config["epochs"]), max(1, int(evaluation.get("stage1_learning_curve_epochs", 20)))
    )
    rows: list[dict[str, Any]] = []
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        training_records = max(12, int(round(len(train_idx) * fraction)))
        training_records = min(training_records, len(train_idx))
        candidate_config = dict(lstm_config)
        candidate_config["epochs"] = diagnostic_epochs
        candidate_config["patience"] = min(int(candidate_config["patience"]), diagnostic_epochs)
        candidate_positions = train_idx[:training_records]
        candidate = LSTMAutoencoder(**candidate_config).fit(
            normal_values[candidate_positions],
            sequence_groups=sequence_groups[candidate_positions],
        )
        validation_score = candidate.score_samples(
            normal_values[validation_idx],
            sequence_groups=sequence_groups[validation_idx],
        )
        history = candidate.training_history_
        final_history = history[-1] if history else {}
        rows.append(
            {
                "training_fraction": training_records / max(len(train_idx), 1),
                "training_records": training_records,
                "mean_validation_reconstruction_score": float(np.mean(validation_score)),
                "median_validation_reconstruction_score": float(np.median(validation_score)),
                "final_train_loss": float(
                    final_history["train_loss"]
                ) if final_history.get("train_loss") is not None else np.nan,
                "final_validation_loss": float(
                    final_history["validation_loss"]
                ) if final_history.get("validation_loss") is not None else np.nan,
                "diagnostic_epochs": diagnostic_epochs,
                "scope": "controlled_stage1_refit_not_deployed_model",
            }
        )
    write_table(pd.DataFrame(rows), output_path)


def _export_onnx(
    lstm: LSTMAutoencoder,
    forest: IsolationForest,
    classifier: Any | None,
    binary_gate: Any | None,
    input_features: int,
    output_dir: Path,
) -> dict[str, str]:
    """Export standard ONNX assets without making local model use depend on ONNX."""
    outcomes: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        if lstm.model_ is None or lstm.effective_sequence_length_ is None:
            raise RuntimeError("LSTM model is not fitted.")
        path = output_dir / "lstm-autoencoder.onnx"
        example = torch.zeros(
            (1, lstm.effective_sequence_length_, input_features), dtype=torch.float32
        )
        torch.onnx.export(
            lstm.model_,
            example,
            path,
            input_names=["sequence"],
            output_names=["reconstruction"],
            dynamic_axes={
                "sequence": {0: "batch", 1: "sequence"},
                "reconstruction": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            # The Torch dynamo exporter can spend minutes compiling a tiny
            # CPU LSTM and has no benefit for this fixed deployment graph.
            # The legacy path is deterministic, fast, and broadly supported
            # by ONNX Runtime on Linux servers.
            dynamo=False,
        )
        outcomes["lstm_autoencoder"] = str(path)
    except Exception as error:  # ONNX is optional at import time; record the exact cause.
        outcomes["lstm_autoencoder_error"] = str(error)
    try:
        from skl2onnx import to_onnx

        path = output_dir / "isolation-forest.onnx"
        sklearn_opset = {"": 17, "ai.onnx.ml": 3}
        path.write_bytes(
            to_onnx(
                forest,
                np.zeros((1, input_features), dtype=np.float32),
                target_opset=sklearn_opset,
            ).SerializeToString()
        )
        outcomes["isolation_forest"] = str(path)
        if classifier is not None:
            class_path = output_dir / "attack-classifier.onnx"
            class_path.write_bytes(
                to_onnx(
                    classifier,
                    np.zeros((1, classifier.n_features_in_), dtype=np.float32),
                    target_opset=sklearn_opset,
                ).SerializeToString()
            )
            outcomes["attack_classifier"] = str(class_path)
        if binary_gate is not None:
            gate_path = output_dir / "binary-attack-gate.onnx"
            gate_path.write_bytes(
                to_onnx(
                    binary_gate,
                    np.zeros((1, input_features), dtype=np.float32),
                    target_opset=sklearn_opset,
                ).SerializeToString()
            )
            outcomes["binary_attack_gate"] = str(gate_path)
    except Exception as error:  # pragma: no cover - depends on optional ONNX conversion support.
        outcomes["sklearn_export_error"] = str(error)
    write_json(outcomes, output_dir / "onnx-export.json")
    return outcomes


def train_two_stage(
    benign_paths: list[Path],
    attack_paths: list[Path],
    config: dict[str, Any],
    output_dir: Path,
    profile: str | Path | None = None,
    protocol: str | None = None,
) -> dict[str, Any]:
    """Train, validate, and package the two-stage pipeline under one frozen feature contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = DurableProgress(output_dir / "training-progress.json", "training")

    def report(
        event: str,
        stage: str,
        message: str,
        progress_ratio: float | None = None,
        **details: Any,
    ) -> None:
        progress.emit(
            event,
            stage=stage,
            message=message,
            progress_ratio=progress_ratio,
            protocol=protocol or "all",
            **details,
        )

    report(
        "training_started",
        "initialization",
        "Loading benign and attack sources and building the immutable model contract.",
        0.0,
        benign_sources=len(benign_paths),
        attack_sources=len(attack_paths),
        output_dir=str(output_dir),
    )
    process = psutil.Process()
    started = time.perf_counter()
    resources_before = snapshot().as_dict() | {
        "process_rss_mb": round(process.memory_info().rss / 1024**2, 2)
    }
    normal = _load_sources(benign_paths, config, protocol)
    attacks = _load_sources(attack_paths, config, protocol) if attack_paths else pd.DataFrame()
    # DataFrame filtering below intentionally drops pandas attrs, so retain
    # source-coverage evidence before selecting accepted mapping rows.
    normal_source_rows = int(normal.attrs.get("source_rows_total", len(normal)))
    attack_source_rows = int(attacks.attrs.get("source_rows_total", len(attacks)))
    normal_loaded_rows = int(normal.attrs.get("loaded_rows", len(normal)))
    attack_loaded_rows = int(attacks.attrs.get("loaded_rows", len(attacks)))
    model_row_cap = config["models"].get("max_source_rows")
    unlabelled_attack_rows = pd.DataFrame()
    if "mapping_accepted" in attacks:
        # Attack PCAPs are intentionally treated as mixed traffic.  A row that
        # cannot be mapped to a trusted CSV label is useful for operational
        # scoring, but it is *not* a negative or positive example for a metric.
        unlabelled_attack_rows = attacks[~attacks["mapping_accepted"].fillna(False)].copy()
        attacks = attacks[attacks["mapping_accepted"].fillna(False)].copy()
    if not attacks.empty:
        known_target_mask = _known_binary_label_mask(attacks)
        unlabelled_attack_rows = pd.concat(
            [unlabelled_attack_rows, attacks[~known_target_mask]],
            ignore_index=True,
            sort=False,
        )
        attacks = attacks[known_target_mask].copy()
    if normal.empty:
        reason = "Stage 1 requires benign PCAP-derived feature records."
        progress.fail(
            reason,
            normal_rows=len(normal),
            accepted_attack_rows=len(attacks),
            normal_source_rows=normal_source_rows,
            attack_source_rows=attack_source_rows,
        )
        raise ValueError(reason)
    report(
        "sources_loaded",
        "source_loading",
        (
            f"Loaded {len(normal):,} benign records, {len(attacks):,} trusted mapped-label "
            f"records, and {len(unlabelled_attack_rows):,} mixed-capture observations. "
            "Stage 1 remains runnable when trusted attack labels are unavailable."
        ),
        0.05,
        normal_rows=len(normal),
        accepted_attack_rows=len(attacks),
        normal_source_rows=normal_source_rows,
        attack_source_rows=attack_source_rows,
        normal_loaded_rows=normal_loaded_rows,
        attack_loaded_rows=attack_loaded_rows,
        unlabelled_attack_capture_observations=len(unlabelled_attack_rows),
        model_row_cap=model_row_cap,
    )
    train_idx, calibration_idx, evaluation_idx, normal_split = _normal_capture_split(
        normal, float(config["stage_one"]["validation_fraction"])
    )
    stage1_dir = output_dir / "stage1"
    prepared_path = stage1_dir / "prepared-normal.parquet"
    report(
        "preprocessing_started",
        "preprocessing",
        "Starting feature-profile selection, imputation, encoding, and robust scaling.",
        0.10,
    )
    prepared_normal, manifest, pipeline_path = prepare_features(
        feature_path=benign_paths[0],
        output_path=prepared_path,
        config=config,
        profile=profile,
        protocols=[protocol] if protocol else None,
        fit_positions=train_idx,
        source_frame=normal,
    )
    model_columns = [column for column in prepared_normal.columns if column not in METADATA_COLUMNS]
    normal_values = prepared_normal[model_columns].to_numpy(dtype=float)
    report(
        "preprocessing_completed",
        "preprocessing",
        (
            f"Frozen contract ready with {len(model_columns):,} transformed columns and "
            f"{len(prepared_normal):,} records."
        ),
        0.18,
        selected_features=len(manifest["selected_input_features"]),
        transformed_features=len(model_columns),
        training_rows=len(train_idx),
        calibration_rows=len(calibration_idx),
        untouched_evaluation_rows=len(evaluation_idx),
        normal_split=normal_split,
    )
    lstm_config = {
        **config["models"]["lstm_autoencoder"],
        "random_seed": config["project"]["random_seed"],
    }
    def lstm_update(update: dict[str, Any]) -> None:
        payload = dict(update)
        event = str(payload.pop("event", "lstm_update"))
        epoch = payload.get("epoch")
        epochs = payload.get("epochs")
        if event == "epoch":
            ratio = 0.18 + 0.40 * (float(epoch) / max(float(epochs), 1.0))
            message = (
                f"LSTM epoch {epoch}/{epochs} · "
                f"train loss={float(payload.get('train_loss', 0)):.6f}"
            )
            if payload.get("validation_loss") is not None:
                message += f" · validation loss={float(payload['validation_loss']):.6f}"
        elif event == "lstm_started":
            ratio = 0.18
            message = (
                f"Started LSTM with {payload.get('training_windows', 0):,} windows and "
                f"{payload.get('input_features', 0):,} features."
            )
        else:
            ratio = 0.58
            message = "LSTM-AE training completed; best weights are ready for the next stage."
        report(event, "stage1_lstm", message, ratio, **payload)

    report(
        "lstm_preparing",
        "stage1_lstm",
        "Building the dynamic LSTM Autoencoder and training it on benign traffic.",
        0.18,
    )
    lstm = LSTMAutoencoder(**lstm_config).fit(
        normal_values[train_idx],
        progress_callback=lstm_update,
        sequence_groups=_sequence_groups(normal.iloc[train_idx]),
    )
    report(
        "stage1_learning_curve_started",
        "stage1_lstm",
        "Running bounded refits for the Stage 1 training-size learning curve.",
        0.585,
    )
    _write_stage1_learning_curve(
        normal_values=normal_values,
        train_idx=train_idx,
        validation_idx=evaluation_idx,
        sequence_groups=_sequence_groups(normal),
        lstm_config=lstm_config,
        config=config,
        output_path=stage1_dir / "learning-curve.parquet",
    )
    report(
        "stage1_learning_curve_completed",
        "stage1_lstm",
        "Saved the Stage 1 training-size learning curve.",
        0.595,
    )
    report(
        "isolation_forest_started",
        "stage1_isolation_forest",
        "Fitting Isolation Forest in the same benign feature space.",
        0.60,
        estimators=int(config["models"]["isolation_forest"]["n_estimators"]),
        workers=effective_workers(int(config["runtime"]["cpu_workers"])),
    )
    forest = IsolationForest(
        n_estimators=int(config["models"]["isolation_forest"]["n_estimators"]),
        max_samples=config["models"]["isolation_forest"]["max_samples"],
        contamination="auto",
        n_jobs=effective_workers(int(config["runtime"]["cpu_workers"])),
        random_state=int(config["project"]["random_seed"]),
    ).fit(normal_values[train_idx])
    report(
        "attack_transform_started",
        "stage1_evaluation",
        (
            "Transforming accepted attack data with the frozen pipeline and evaluating Stage 1."
            if not attacks.empty
            else "No accepted attack labels are available; evaluating Stage 1 on untouched benign data."
        ),
        0.68,
    )
    if attacks.empty:
        attack_values = np.empty((0, len(model_columns)), dtype=float)
        attack_truth = np.empty(0, dtype=int)
    else:
        prepared_attack, transformed_names = transform_with_manifest(attacks, pipeline_path, manifest)
        if transformed_names != model_columns:
            raise RuntimeError(
                "Frozen feature contract did not reproduce the stage-one model column order."
            )
        attack_values = prepared_attack.to_numpy(dtype=float)
        attack_truth = _binary_label(attacks)
    attack_roles, attack_split = _attack_capture_roles(
        attacks, attack_truth, float(config["stage_one"]["validation_fraction"])
    )
    binary_gate, binary_gate_metrics = _fit_stage_one_binary_gate(
        normal_values,
        train_idx,
        attack_values,
        attack_truth,
        attack_roles,
        config,
    )
    mapped_benign_positions = np.flatnonzero(attack_truth == 0)
    mapped_benign_calibration_mask = np.zeros(len(attacks), dtype=bool)
    if len(mapped_benign_positions) and bool(
        config["stage_one"].get("calibration_include_mapped_benign", True)
    ):
        # Only a temporal tail of mapped benign records is used to tune the
        # threshold.  The remaining mapped benign packets stay in the reported
        # evaluation so calibration cannot conceal background false alarms.
        calibration_count = max(
            1,
            int(
                round(
                    len(mapped_benign_positions)
                    * float(config["stage_one"]["validation_fraction"])
                )
            ),
        )
        calibration_positions = mapped_benign_positions[-calibration_count:]
        mapped_benign_calibration_mask[calibration_positions] = True
    mapped_benign_values = attack_values[mapped_benign_calibration_mask]
    calibration_values = normal_values[calibration_idx]
    calibration_groups = _sequence_groups(normal.iloc[calibration_idx])
    if bool(config["stage_one"].get("calibration_include_mapped_benign", True)):
        calibration_values = np.vstack([calibration_values, mapped_benign_values])
        calibration_groups = np.concatenate(
            [calibration_groups, _sequence_groups(attacks.iloc[mapped_benign_calibration_mask])]
        )
    target_fpr = float(config["stage_one"]["target_false_positive_rate"])
    ensemble_mode = str(config["stage_one"].get("ensemble_mode", "calibrated_weighted"))
    ensemble_weights = dict(config["stage_one"].get("ensemble_weights", {}))
    validation_lstm_scores = lstm.score_samples(
        calibration_values, sequence_groups=calibration_groups
    )
    validation_forest_scores = -forest.score_samples(calibration_values)
    # Stage 1 is still fitted only with normal-PCAP records.  Explicitly mapped
    # benign packets from attack captures are calibration-only: they make the
    # threshold robust to the deployment capture distribution without training
    # either detector on attack labels.
    lstm_threshold = _threshold(validation_lstm_scores, target_fpr)
    forest_threshold = _threshold(validation_forest_scores, target_fpr)
    calibration_thresholds: dict[str, Any] = {
        "lstm": lstm_threshold,
        "isolation_forest": forest_threshold,
    }
    calibration_score = _ensemble_score(
        validation_lstm_scores,
        validation_forest_scores,
        lstm_threshold["threshold"],
        forest_threshold["threshold"],
        ensemble_mode,
        ensemble_weights,
    )
    if ensemble_mode in {"calibrated_max", "calibrated_weighted"}:
        calibration_thresholds["ensemble"] = _threshold(calibration_score, target_fpr)
    else:
        calibration_thresholds["ensemble"] = {
            "threshold": 1.0,
            "target_false_positive_rate": target_fpr,
            "decision_policy": "legacy_detector_vote",
        }
    calibration_unsupervised = _stage1_decision(
        validation_lstm_scores,
        validation_forest_scores,
        calibration_thresholds,
        ensemble_mode,
        ensemble_weights,
    )[1]
    if binary_gate is not None:
        calibration_probability = binary_gate.predict_proba(calibration_values)[:, 1]
        calibration_thresholds["supervised"] = _threshold(calibration_probability, target_fpr)
        combined_calibration = np.maximum(
            calibration_unsupervised,
            calibration_probability
            / max(float(calibration_thresholds["supervised"]["threshold"]), 1e-12),
        )
        calibration_thresholds["final"] = _threshold(combined_calibration, target_fpr)
    report(
        "stage1_thresholds_calibrated",
        "stage1_calibration",
        "Calibrated dynamic thresholds; the evaluation slice remains untouched.",
        0.66,
        lstm_threshold=lstm_threshold["threshold"],
        isolation_forest_threshold=forest_threshold["threshold"],
        ensemble_threshold=calibration_thresholds["ensemble"]["threshold"],
        ensemble_mode=ensemble_mode,
        target_false_positive_rate=target_fpr,
        mapped_benign_calibration_records=len(mapped_benign_values),
        binary_gate_status=binary_gate_metrics["status"],
    )
    evaluation_normal_raw = normal.iloc[evaluation_idx].copy()
    evaluation_normal_raw["is_attack"] = False
    evaluation_raw = pd.concat([evaluation_normal_raw, attacks], ignore_index=True, sort=False)
    evaluation_values = np.vstack([normal_values[evaluation_idx], attack_values])
    y_true = np.concatenate([np.zeros(len(evaluation_idx), dtype=int), attack_truth])
    # Scores for all accepted attack-capture records are retained for the
    # dashboard.  Only the small mapped-benign calibration tail is excluded
    # from metrics, while every positive attack record remains evaluated.
    metric_evaluation_mask = np.concatenate(
        [np.ones(len(evaluation_idx), dtype=bool), ~mapped_benign_calibration_mask]
    )
    evaluation_groups = np.concatenate(
        [_sequence_groups(normal.iloc[evaluation_idx]), _sequence_groups(attacks)]
    )
    lstm_scores = lstm.score_samples(evaluation_values, sequence_groups=evaluation_groups)
    forest_scores = -forest.score_samples(evaluation_values)
    supervised_probability = (
        binary_gate.predict_proba(evaluation_values)[:, 1] if binary_gate is not None else None
    )
    stage1_decision = _stage_one_hybrid_decision(
        lstm_scores,
        forest_scores,
        calibration_thresholds,
        ensemble_mode,
        ensemble_weights,
        supervised_probability,
    )
    predicted = stage1_decision["is_anomaly"]
    normalized_score = stage1_decision["normalized_score"]
    ensemble_raw_score = stage1_decision["ensemble_raw_score"]
    lstm_hit = stage1_decision["lstm_hit"]
    forest_hit = stage1_decision["forest_hit"]
    scores = evaluation_raw[
        [
            column
            for column in [
                "capture",
                "timestamp",
                "protocol",
                "flow_id",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "label",
            ]
            if column in evaluation_raw
        ]
    ].copy()
    scores.insert(0, "record_id", np.arange(len(scores), dtype=int))
    scores["true_is_attack"] = y_true.astype(bool)
    scores["stage1_split"] = np.concatenate(
        [
            np.full(len(evaluation_idx), "normal_evaluation", dtype=object),
            np.where(
                mapped_benign_calibration_mask,
                "mapped_benign_calibration",
                "attack_capture_evaluation",
            ),
        ]
    )
    scores["stage1_capture_role"] = np.concatenate(
        [
            np.full(len(evaluation_idx), "normal_evaluation", dtype=object),
            attack_roles,
        ]
    )
    scores["stage1_metric_evaluation"] = metric_evaluation_mask
    scores["lstm_reconstruction_score"] = lstm_scores
    scores["isolation_forest_score"] = forest_scores
    scores["lstm_threshold"] = lstm_threshold["threshold"]
    scores["isolation_forest_threshold"] = forest_threshold["threshold"]
    scores["stage1_ensemble_raw_score"] = ensemble_raw_score
    scores["stage1_ensemble_threshold"] = calibration_thresholds["ensemble"]["threshold"]
    scores["stage1_unsupervised_normalized_score"] = stage1_decision[
        "unsupervised_normalized_score"
    ]
    scores["stage1_supervised_probability"] = stage1_decision["supervised_probability"]
    scores["stage1_normalized_score"] = normalized_score
    scores["lstm_detector_vote"] = lstm_hit
    scores["isolation_forest_detector_vote"] = forest_hit
    scores["stage1_anomaly"] = predicted
    contributions = _contributions(
        evaluation_values, model_columns, normal_values[train_idx], scores
    )
    # Persist the exact transformed inputs used for evaluation.  Besides making
    # the run reproducible outside the dashboard, this permits post-training
    # latent/cluster/interaction diagnostics without re-reading or changing a
    # PCAP source file.
    transformed_frame = pd.DataFrame(evaluation_values, columns=model_columns)
    raw_feature_columns = [name for name in model_columns if name in evaluation_raw]
    raw_frame = evaluation_raw[raw_feature_columns].apply(pd.to_numeric, errors="coerce")
    raw_frame = raw_frame.rename(columns={name: f"raw__{name}" for name in raw_feature_columns})
    evaluation_prepared = pd.concat(
        [scores.reset_index(drop=True), transformed_frame, raw_frame.reset_index(drop=True)], axis=1
    )
    development_metrics = _metrics(
        y_true[metric_evaluation_mask],
        predicted.astype(int)[metric_evaluation_mask],
        normalized_score[metric_evaluation_mask],
    )
    normal_evaluation_predictions = predicted[: len(evaluation_idx)]
    normal_only_metrics = _metrics(
        np.zeros(len(evaluation_idx), dtype=int),
        normal_evaluation_predictions.astype(int),
        normalized_score[: len(evaluation_idx)],
    )
    mapped_benign_evaluation_mask = (attack_truth == 0) & ~mapped_benign_calibration_mask
    mapped_benign_evaluation_predictions = predicted[len(evaluation_idx) :][
        mapped_benign_evaluation_mask
    ]
    strict_attack_positions = np.flatnonzero(
        (attack_roles == "evaluation") & ~mapped_benign_calibration_mask
    )
    strict_metrics: dict[str, Any] | None = None
    if len(strict_attack_positions):
        strict_positions = np.concatenate(
            [
                np.arange(len(evaluation_idx), dtype=int),
                len(evaluation_idx) + strict_attack_positions,
            ]
        )
        strict_metrics = _metrics(
            y_true[strict_positions],
            predicted.astype(int)[strict_positions],
            normalized_score[strict_positions],
        )
        strict_metrics["scope"] = "capture_disjoint_normal_and_known_labelled_attack_test"
        strict_metrics["attack_test_captures"] = attack_split["evaluation_captures"]
        strict_metrics["normal_test_captures"] = normal_split["evaluation_captures"]

    # The top-level values are always the most deployment-faithful available
    # test: a capture-disjoint normal + known-labelled-attack test when label
    # coverage permits it, otherwise the explicitly named development scope.
    stage1_metrics = dict(strict_metrics or development_metrics)
    primary_metric_mask = metric_evaluation_mask.copy()
    if strict_metrics is not None:
        primary_metric_mask = np.zeros(len(scores), dtype=bool)
        primary_metric_mask[strict_positions] = True
    # Dashboard charts consume this column directly.  It must therefore match
    # the headline metric scope rather than quietly mixing train/calibration
    # attack captures into a capture-held-out result.
    scores["stage1_metric_evaluation"] = primary_metric_mask
    scores["stage1_development_evaluation"] = metric_evaluation_mask
    evaluation_prepared["stage1_metric_evaluation"] = primary_metric_mask
    evaluation_prepared["stage1_development_evaluation"] = metric_evaluation_mask
    stage1_metrics["evaluation_scope"] = (
        "capture_disjoint_normal_and_known_labelled_attack_test"
        if strict_metrics is not None
        else "normal_capture_holdout_plus_all_accepted_labelled_attack_rows_development_only"
    )
    stage1_metrics.update(
        {
            "thresholds": calibration_thresholds,
            "ensemble_mode": ensemble_mode,
            "ensemble_weights": ensemble_weights,
            "binary_gate": binary_gate_metrics,
            "normal_capture_split": normal_split,
            "attack_capture_split": attack_split,
            "normal_only_capture_holdout": normal_only_metrics,
            "all_accepted_labelled_development_evaluation": development_metrics,
            "input_features": len(model_columns),
            "training_normal_records": int(len(train_idx)),
            "calibration_normal_records": int(len(calibration_idx)),
            "evaluation_normal_records": int(len(evaluation_idx)),
            "mapped_benign_calibration_records": int(len(mapped_benign_values)),
            "mapped_benign_evaluation_records": int(mapped_benign_evaluation_mask.sum()),
            "normal_evaluation_false_positive_rate": round(
                float(normal_evaluation_predictions.mean()), 6
            ),
            "mapped_benign_evaluation_false_positive_rate": round(
                float(mapped_benign_evaluation_predictions.mean())
                if len(mapped_benign_evaluation_predictions)
                else 0.0,
                6,
            ),
            "attack_evaluation_records": int(len(attacks)),
            "normal_source_rows": normal_source_rows,
            "attack_source_rows": attack_source_rows,
            "normal_loaded_rows": normal_loaded_rows,
            "attack_loaded_rows": attack_loaded_rows,
            "model_row_cap": model_row_cap,
        }
    )
    if strict_metrics is not None:
        stage1_metrics["capture_holdout_evaluation"] = strict_metrics
    else:
        stage1_metrics["capture_holdout_evaluation"] = {
            "status": "unavailable",
            "reason": attack_split.get("reason", "no_capture_disjoint_attack_test_rows"),
        }

    observation_summary: dict[str, Any] = {
        "status": "no_unmapped_attack_capture_records_loaded",
        "records": 0,
        "captures": 0,
        "metric_policy": "not_used_for_accuracy_metrics",
    }
    if not unlabelled_attack_rows.empty:
        observation_values_frame, observation_names = transform_with_manifest(
            unlabelled_attack_rows, pipeline_path, manifest
        )
        if observation_names != model_columns:
            raise RuntimeError(
                "Frozen feature contract did not reproduce the stage-one model columns "
                "for unlabelled attack-capture observations."
            )
        observation_values = observation_values_frame.to_numpy(dtype=float)
        observation_lstm = lstm.score_samples(
            observation_values,
            sequence_groups=_sequence_groups(unlabelled_attack_rows),
        )
        observation_forest = -forest.score_samples(observation_values)
        observation_probability = (
            binary_gate.predict_proba(observation_values)[:, 1]
            if binary_gate is not None
            else None
        )
        observation_decision = _stage_one_hybrid_decision(
            observation_lstm,
            observation_forest,
            calibration_thresholds,
            ensemble_mode,
            ensemble_weights,
            observation_probability,
        )
        observation_columns = [
            column
            for column in [
                "capture",
                "timestamp",
                "protocol",
                "flow_id",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "label",
                "mapping_accepted",
            ]
            if column in unlabelled_attack_rows
        ]
        observations = unlabelled_attack_rows[observation_columns].copy().reset_index(drop=True)
        observations.insert(0, "record_id", np.arange(len(observations), dtype=int))
        observations["lstm_reconstruction_score"] = observation_lstm
        observations["isolation_forest_score"] = observation_forest
        observations["stage1_unsupervised_normalized_score"] = observation_decision[
            "unsupervised_normalized_score"
        ]
        observations["stage1_supervised_probability"] = observation_decision[
            "supervised_probability"
        ]
        observations["stage1_normalized_score"] = observation_decision["normalized_score"]
        observations["stage1_anomaly"] = observation_decision["is_anomaly"]
        observations["metric_policy"] = "unlabelled_observation_not_used_for_accuracy_metrics"
        write_table(observations, stage1_dir / "unlabelled-mixed-observations.parquet")
        capture_summary = (
            observations.groupby("capture", dropna=False)
            .agg(
                records=("record_id", "size"),
                stage1_anomalies=("stage1_anomaly", "sum"),
                anomaly_rate=("stage1_anomaly", "mean"),
                median_normalized_score=("stage1_normalized_score", "median"),
                p95_normalized_score=(
                    "stage1_normalized_score",
                    lambda values: float(np.quantile(values, 0.95)),
                ),
                median_supervised_probability=("stage1_supervised_probability", "median"),
            )
            .reset_index()
        )
        write_table(capture_summary, stage1_dir / "unlabelled-mixed-capture-summary.parquet")
        observation_summary = {
            "status": "scored_without_ground_truth",
            "records": int(len(observations)),
            "captures": int(observations.get("capture", pd.Series(dtype="string")).nunique()),
            "anomalies": int(observation_decision["is_anomaly"].sum()),
            "metric_policy": "not_used_for_accuracy_metrics",
            "record_artifact": "stage1/unlabelled-mixed-observations.parquet",
            "capture_summary_artifact": "stage1/unlabelled-mixed-capture-summary.parquet",
        }
    stage1_metrics["unlabelled_mixed_capture_observations"] = observation_summary
    write_json(
        {
            "normal_only_capture_holdout": normal_only_metrics,
            "capture_disjoint_known_labelled_attack_test": stage1_metrics[
                "capture_holdout_evaluation"
            ],
            "all_accepted_labelled_development_evaluation": development_metrics,
            "unlabelled_mixed_capture_observations": observation_summary,
            "interpretation": (
                "Only normal-only and known CSV-mapped records contribute to accuracy metrics. "
                "Unmapped attack-PCAP records are scored for operational review only."
            ),
        },
        stage1_dir / "validation-summary.json",
    )
    report(
        "stage1_evaluated",
        "stage1_evaluation",
        (
            "Stage 1 evaluated on the untouched evaluation slice: "
            f"FPR={stage1_metrics['false_positive_rate']:.2%} · "
            f"Recall={stage1_metrics['recall']:.2%}."
        ),
        0.75,
        stage1_metrics=stage1_metrics,
    )
    lstm.save(stage1_dir / "lstm-autoencoder.pt")
    joblib.dump(forest, stage1_dir / "isolation-forest.joblib")
    if binary_gate is not None:
        joblib.dump(binary_gate, stage1_dir / "binary-attack-gate.joblib")
    write_table(pd.DataFrame(lstm.training_history_), stage1_dir / "lstm-training-history.parquet")
    write_table(scores, stage1_dir / "scores.parquet")
    write_table(contributions, stage1_dir / "feature-evidence.parquet")
    write_table(evaluation_prepared, stage1_dir / "evaluation-prepared.parquet")
    write_json(stage1_metrics, stage1_dir / "metrics.json")
    pre_comparison = preprocessing_comparison(
        normal, prepared_normal, manifest["selected_input_features"], protocol or "all"
    )
    write_table(pre_comparison, output_dir / "reports" / "preprocessing-comparison.parquet")

    # Stage two is implemented in its own module. It receives only accepted,
    # PCAP-derived attack rows and the exact transformed Stage 1 contract.
    stage2_dir = output_dir / "stage2"
    type_rows = attacks[_binary_label(attacks).astype(bool)].copy()
    attack_score_positions = len(evaluation_idx) + np.flatnonzero(attack_truth.astype(bool))
    stage2_readiness = stage_two_readiness(type_rows, config)
    write_json(stage2_readiness, stage2_dir / "readiness.json")
    report(
        "stage2_matrix_preparing",
        "stage2_random_forest",
        (
            "Building the Stage 2 matrix from frozen features and Stage 1 scores."
            if stage2_readiness["ready"]
            else "Stage 2 eligibility was checked; Stage 1 remains available if labels are insufficient."
        ),
        0.78,
        attack_type_rows=len(type_rows),
        stage2_readiness=stage2_readiness,
    )
    random_forest = None
    classifier: RandomForestClassifier | None = None
    type_values = np.empty((0, len(model_columns)), dtype=float)
    stage2_values = np.empty((0, len(model_columns)), dtype=float)
    stage2_columns: list[str] = []
    if stage2_readiness["ready"]:
        type_values_frame, _ = transform_with_manifest(type_rows, pipeline_path, manifest)
        type_values = type_values_frame.to_numpy(dtype=float)
        stage1_attack_lstm = lstm.score_samples(
            type_values, sequence_groups=_sequence_groups(type_rows)
        )
        stage1_attack_forest = -forest.score_samples(type_values)
        stage2_values, stage2_columns = make_stage_two_matrix(
            type_values,
            model_columns,
            stage1_attack_lstm,
            stage1_attack_forest,
            bool(config["stage_two"].get("include_stage_one_scores", True)),
        )
        report(
            "random_forest_started",
            "stage2_random_forest",
            (
                f"Training Random Forest with {len(stage2_values):,} records and "
                f"{len(stage2_columns):,} columns."
            ),
            0.82,
            input_features=len(stage2_columns),
            classes=int(type_rows.get("label", pd.Series(dtype="string")).nunique(dropna=True)),
        )
        random_forest = train_attack_random_forest(
            type_rows, stage2_values, stage2_columns, config, stage2_dir
        )
        classifier = random_forest.classifier
        stage2_metrics = random_forest.metrics
        report(
            "random_forest_completed",
            "stage2_random_forest",
            (
                f"Stage 2 completed: weighted F1={stage2_metrics['weighted_f1']:.2%} "
                f"on {stage2_metrics['test_records']:,} held-out test records."
            ),
            0.89,
            stage2_metrics=stage2_metrics,
        )
    else:
        stage2_metrics = {
            "status": "skipped",
            "reason": stage2_readiness["reason"],
            "records": stage2_readiness["records"],
            "classes": stage2_readiness["classes"],
            "training_records": 0,
            "test_records": 0,
            "weighted_f1": None,
            "test_accuracy": None,
            "readiness": stage2_readiness,
        }
        write_json(stage2_metrics, stage2_dir / "metrics.json")
        report(
            "random_forest_skipped",
            "stage2_random_forest",
            f"Stage 2 skipped safely: {stage2_readiness['reason']}.",
            0.89,
            stage2_metrics=stage2_metrics,
        )

    # End-to-end evaluation: only packets raised by stage one are eligible for a
    # type prediction.  This is the same gate a deployment uses in real time.
    pipeline_rows = scores.copy()
    pipeline_rows["predicted_attack_type"] = "normal"
    attack_gate = pipeline_rows["stage1_anomaly"].to_numpy()
    gated_values = evaluation_values[attack_gate]
    pipeline_rows["stage2_status"] = stage2_metrics["status"]
    scores["stage2_evaluation_role"] = "not_stage2_attack"
    if classifier is not None and len(gated_values):
        stage2_input = gated_values
        if config["stage_two"].get("include_stage_one_scores", True):
            stage2_input, _ = make_stage_two_matrix(
                stage2_input,
                model_columns,
                lstm_scores[attack_gate],
                forest_scores[attack_gate],
                True,
            )
        pipeline_rows.loc[attack_gate, "predicted_attack_type"] = predict_attack_type(
            classifier, stage2_input
        )
    elif attack_gate.any():
        pipeline_rows.loc[attack_gate, "predicted_attack_type"] = "stage1_anomaly_untyped"
    write_table(pipeline_rows, output_dir / "pipeline" / "end-to-end-predictions.parquet")
    report(
        "pipeline_evaluated",
        "end_to_end",
        f"End-to-end pipeline evaluated; {int(attack_gate.sum()):,} records passed the Stage 1 gate.",
        0.93,
        stage1_anomalies=int(attack_gate.sum()),
        evaluation_records=len(pipeline_rows),
    )
    if random_forest is not None and classifier is not None:
        test_score_positions = attack_score_positions[random_forest.test_positions]
        train_score_positions = attack_score_positions[random_forest.train_positions]
        scores.loc[test_score_positions, "stage2_evaluation_role"] = "held_out_test"
        scores.loc[train_score_positions, "stage2_evaluation_role"] = "stage2_train"
        held_out_positions = np.concatenate(
            [np.arange(len(evaluation_idx), dtype=int), test_score_positions]
        )
        held_out_stage1 = _metrics(
            y_true[held_out_positions],
            predicted.astype(int)[held_out_positions],
            normalized_score[held_out_positions],
        )
        held_out_stage1["scope"] = "untouched_normal_evaluation_plus_stage2_held_out_attacks"
        held_out_stage1["attack_test_records"] = int(len(test_score_positions))
        stage1_metrics["held_out_attack_test"] = held_out_stage1
        report(
            "pipeline_diagnostics_started",
            "end_to_end_diagnostics",
            "Building baseline, threshold-sensitivity, latency, and seed-stability diagnostics.",
            0.94,
        )
        diagnostics = _write_pipeline_diagnostics(
            output_dir=output_dir,
            config=config,
            model_columns=model_columns,
            normal_values=normal_values,
            train_idx=train_idx,
            evaluation_idx=evaluation_idx,
            type_values=type_values,
            type_rows=type_rows,
            random_forest=random_forest,
            attack_score_positions=attack_score_positions,
            evaluation_values=evaluation_values,
            scores=scores,
            lstm=lstm,
            forest=forest,
            lstm_scores=lstm_scores,
            forest_scores=forest_scores,
            classifier=classifier,
            stage2_columns=stage2_columns,
            include_stage_one_scores=bool(config["stage_two"].get("include_stage_one_scores", True)),
        )
        report(
            "pipeline_diagnostics_completed",
            "end_to_end_diagnostics",
            "Saved held-out end-to-end analytical artifacts for the dashboard and external use.",
            0.96,
            **diagnostics,
        )
    else:
        stage1_metrics["held_out_attack_test"] = None
        diagnostics = {
            "status": "skipped",
            "reason": "Stage 2 is unavailable, so typed end-to-end diagnostics cannot be computed.",
        }
        write_json(diagnostics, output_dir / "pipeline" / "diagnostics-status.json")
        report(
            "pipeline_diagnostics_skipped",
            "end_to_end_diagnostics",
            "Stage 1 predictions were saved; typed pipeline diagnostics are unavailable without Stage 2.",
            0.96,
            **diagnostics,
        )
    pipeline_rows["stage2_evaluation_role"] = scores["stage2_evaluation_role"].to_numpy()
    write_table(pipeline_rows, output_dir / "pipeline" / "end-to-end-predictions.parquet")
    write_table(scores, stage1_dir / "scores.parquet")
    write_json(stage1_metrics, stage1_dir / "metrics.json")
    contract = {
        "schema_version": "1.1.0",
        "created_at": utc_now(),
        "protocol": protocol,
        "feature_profile": manifest.get("profile"),
        "selected_input_features": manifest["selected_input_features"],
        "transformed_model_columns": model_columns,
        "stage2_model_columns": stage2_columns,
        "preprocessor": str(pipeline_path),
        "preprocessor_manifest": str(prepared_path.with_suffix(".manifest.json")),
        "stage1": {
            "lstm": str(stage1_dir / "lstm-autoencoder.pt"),
            "isolation_forest": str(stage1_dir / "isolation-forest.joblib"),
            "binary_attack_gate": (
                str(stage1_dir / "binary-attack-gate.joblib") if binary_gate is not None else None
            ),
            "binary_gate_status": binary_gate_metrics["status"],
            "thresholds": stage1_metrics["thresholds"],
            "ensemble_mode": ensemble_mode,
            "ensemble_weights": ensemble_weights,
        },
        "stage2": {
            "status": stage2_metrics["status"],
            "random_forest": (
                str(stage2_dir / "attack-random-forest.joblib") if classifier is not None else None
            ),
            "reason": stage2_metrics.get("reason"),
        },
    }
    write_json(contract, output_dir / "model-contract.json")
    exports = _export_onnx(
        lstm,
        forest,
        classifier,
        binary_gate,
        len(model_columns),
        output_dir / "onnx",
    )
    report(
        "onnx_export_completed",
        "packaging",
        "Saved primary model artifacts and recorded ONNX export results.",
        0.97,
        onnx_exports=exports,
    )
    resources_after = snapshot().as_dict() | {
        "process_rss_mb": round(process.memory_info().rss / 1024**2, 2)
    }
    resource_rows = pd.DataFrame(
        [
            {"point": "before_training", **resources_before},
            {"point": "after_training", **resources_after},
        ]
    )
    write_table(resource_rows, output_dir / "reports" / "resource-usage.parquet")
    summary = {
        "created_at": utc_now(),
        "output_dir": str(output_dir),
        "protocol": protocol or "all",
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "resource_before": resources_before,
        "resource_after": resources_after,
        "training_seconds": round(time.perf_counter() - started, 4),
        "onnx_exports": exports,
        "pipeline_diagnostics": diagnostics,
        "contract": str(output_dir / "model-contract.json"),
    }
    write_json(summary, output_dir / "training-summary.json")
    progress.complete(
        "Saved the two-stage model, inference contract, and evaluation reports.",
        training_seconds=summary["training_seconds"],
        stage1_fpr=stage1_metrics["false_positive_rate"],
        stage1_recall=stage1_metrics["recall"],
        stage2_weighted_f1=stage2_metrics["weighted_f1"],
    )
    return summary


def score_two_stage(source: pd.DataFrame, contract_path: Path) -> pd.DataFrame:
    """Run portable inference with the exact frozen profile and model artifacts."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    root = contract_path.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        if candidate.exists() or candidate.is_absolute():
            return candidate
        return root / candidate

    pipeline_path = resolve(contract["preprocessor"])
    manifest_path = resolve(contract["preprocessor_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values_frame, names = transform_with_manifest(source, pipeline_path, manifest)
    if names != contract["transformed_model_columns"]:
        raise ValueError("Incoming data does not satisfy the frozen model feature contract.")
    values = values_frame.to_numpy(dtype=float)
    lstm = LSTMAutoencoder.load(resolve(contract["stage1"]["lstm"]), device="cpu")
    forest = joblib.load(resolve(contract["stage1"]["isolation_forest"]))
    lstm_scores = lstm.score_samples(values, sequence_groups=_sequence_groups(source))
    forest_scores = -forest.score_samples(values)
    mode = str(contract["stage1"].get("ensemble_mode", "either_detector"))
    binary_gate_path = contract["stage1"].get("binary_attack_gate")
    binary_gate = joblib.load(resolve(str(binary_gate_path))) if binary_gate_path else None
    supervised_probability = (
        binary_gate.predict_proba(values)[:, 1] if binary_gate is not None else None
    )
    stage1_decision = _stage_one_hybrid_decision(
        lstm_scores,
        forest_scores,
        contract["stage1"]["thresholds"],
        mode,
        dict(contract["stage1"].get("ensemble_weights", {})),
        supervised_probability,
    )
    is_anomaly = stage1_decision["is_anomaly"]
    normalized_score = stage1_decision["normalized_score"]
    ensemble_raw_score = stage1_decision["ensemble_raw_score"]
    lstm_hit = stage1_decision["lstm_hit"]
    forest_hit = stage1_decision["forest_hit"]
    output = source[
        [
            column
            for column in [
                "capture",
                "timestamp",
                "protocol",
                "flow_id",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
            ]
            if column in source
        ]
    ].copy()
    output["lstm_reconstruction_score"] = lstm_scores
    output["isolation_forest_score"] = forest_scores
    output["stage1_ensemble_raw_score"] = ensemble_raw_score
    output["stage1_unsupervised_normalized_score"] = stage1_decision[
        "unsupervised_normalized_score"
    ]
    output["stage1_supervised_probability"] = stage1_decision["supervised_probability"]
    output["stage1_normalized_score"] = normalized_score
    output["lstm_detector_vote"] = lstm_hit
    output["isolation_forest_detector_vote"] = forest_hit
    output["stage1_anomaly"] = is_anomaly
    output["predicted_attack_type"] = "normal"
    stage2_contract = dict(contract.get("stage2", {}))
    random_forest_path = stage2_contract.get("random_forest")
    stage2_status = str(
        stage2_contract.get("status", "trained" if random_forest_path else "skipped")
    )
    output["stage2_status"] = stage2_status
    if is_anomaly.any() and random_forest_path:
        classifier = joblib.load(resolve(str(random_forest_path)))
        stage2_values = values[is_anomaly]
        if len(contract["stage2_model_columns"]) > len(names):
            stage2_values, _ = make_stage_two_matrix(
                stage2_values,
                names,
                lstm_scores[is_anomaly],
                forest_scores[is_anomaly],
                True,
            )
        output.loc[is_anomaly, "predicted_attack_type"] = predict_attack_type(
            classifier, stage2_values
        )
    elif is_anomaly.any():
        output.loc[is_anomaly, "predicted_attack_type"] = "stage1_anomaly_untyped"
    return output
