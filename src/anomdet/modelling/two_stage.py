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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from anomdet.analytics.reports import preprocessing_comparison
from anomdet.core.io import utc_now, write_json, write_table
from anomdet.core.progress import DurableProgress
from anomdet.core.resources import effective_workers, snapshot
from anomdet.modelling.lstm_autoencoder import LSTMAutoencoder
from anomdet.modelling.random_forest import (
    make_stage_two_matrix,
    predict_attack_type,
    train_attack_random_forest,
)
from anomdet.preprocessing.pipeline import (
    prepare_features,
    read_training_source,
    transform_with_manifest,
)

METADATA_COLUMNS = {"row_id", "label", "protocol", "flow_id", "timestamp", "capture"}
NORMAL_LABELS = {"benign", "normal", "0", "false", "no", "non-anomaly", "non_anomaly"}


def _load_sources(
    paths: list[Path], config: dict[str, Any], protocol: str | None = None
) -> pd.DataFrame:
    maximum = config["models"].get("max_source_rows")
    frames = [read_training_source(path, maximum) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No source feature artifacts were found for this model run.")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if protocol is not None and "protocol" in frame:
        frame = frame[
            frame["protocol"].astype("string").str.casefold() == protocol.casefold()
        ].copy()
    return frame.reset_index(drop=True)


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


def _split_normal(length: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if length < 12:
        raise ValueError("Stage one needs at least 12 normal PCAP feature records.")
    validation = max(2, int(round(length * fraction)))
    validation = min(validation, max(2, length // 3))
    return np.arange(0, length - validation), np.arange(length - validation, length)


def _threshold(scores: np.ndarray, target_fpr: float) -> dict[str, float]:
    """Use a validation quantile and robust MAD guardrail for low-FPR calibration."""
    target = min(max(float(target_fpr), 0.0001), 0.25)
    quantile = float(np.quantile(scores, 1 - target))
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust = median + 3.5 * 1.4826 * mad
    return {
        "threshold": max(quantile, robust),
        "quantile_threshold": quantile,
        "mad_threshold": robust,
        "target_false_positive_rate": target,
    }


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


def _export_onnx(
    lstm: LSTMAutoencoder,
    forest: IsolationForest,
    classifier: Any | None,
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
        "بارگذاری منابع نرمال و حمله و ساخت قرارداد immutable مدل شروع شد.",
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
    attacks = _load_sources(attack_paths, config, protocol)
    if "mapping_accepted" in attacks:
        attacks = attacks[attacks["mapping_accepted"].fillna(False)].copy()
    if normal.empty or attacks.empty:
        raise ValueError("Both benign records and accepted mapped attack records are required.")
    report(
        "sources_loaded",
        "source_loading",
        f"{len(normal):,} ردیف نرمال و {len(attacks):,} ردیف حملهٔ دارای نگاشت پذیرفته‌شده آماده است.",
        0.05,
        normal_rows=len(normal),
        accepted_attack_rows=len(attacks),
    )
    train_idx, validation_idx = _split_normal(
        len(normal), float(config["stage_one"]["validation_fraction"])
    )
    stage1_dir = output_dir / "stage1"
    prepared_path = stage1_dir / "prepared-normal.parquet"
    report(
        "preprocessing_started",
        "preprocessing",
        "انتخاب feature profile، imputation، encoding و robust scaling شروع شد.",
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
        f"قرارداد ثابت با {len(model_columns):,} ستون transformed و {len(prepared_normal):,} ردیف آماده شد.",
        0.18,
        selected_features=len(manifest["selected_input_features"]),
        transformed_features=len(model_columns),
        training_rows=len(train_idx),
        validation_rows=len(validation_idx),
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
                f"LSTM epoch {epoch}/{epochs} · train loss={float(payload.get('train_loss', 0)):.6f}"
            )
            if payload.get("validation_loss") is not None:
                message += f" · validation loss={float(payload['validation_loss']):.6f}"
        elif event == "lstm_started":
            ratio = 0.18
            message = (
                f"LSTM با {payload.get('training_windows', 0):,} window و "
                f"{payload.get('input_features', 0):,} feature آغاز شد."
            )
        else:
            ratio = 0.58
            message = "آموزش LSTM-AE کامل شد؛ بهترین وزن‌ها برای مرحلهٔ بعد آماده است."
        report(event, "stage1_lstm", message, ratio, **payload)

    report(
        "lstm_preparing",
        "stage1_lstm",
        "ساخت معماری پویا و آموزش LSTM Autoencoder روی ترافیک نرمال شروع شد.",
        0.18,
    )
    lstm = LSTMAutoencoder(**lstm_config).fit(normal_values[train_idx], progress_callback=lstm_update)
    report(
        "isolation_forest_started",
        "stage1_isolation_forest",
        "Isolation Forest روی همان فضای feature نرمال fit می‌شود.",
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
    target_fpr = float(config["stage_one"]["target_false_positive_rate"])
    lstm_threshold = _threshold(lstm.score_samples(normal_values[validation_idx]), target_fpr / 2)
    forest_threshold = _threshold(
        -forest.score_samples(normal_values[validation_idx]), target_fpr / 2
    )
    report(
        "stage1_thresholds_calibrated",
        "stage1_calibration",
        "آستانهٔ پویا با quantile اعتبارسنجی و guardrail مبتنی بر MAD تنظیم شد.",
        0.66,
        lstm_threshold=lstm_threshold["threshold"],
        isolation_forest_threshold=forest_threshold["threshold"],
        target_false_positive_rate=target_fpr,
    )

    report(
        "attack_transform_started",
        "stage1_evaluation",
        "دادهٔ حمله با pipeline frozen تبدیل و Stage 1 ارزیابی می‌شود.",
        0.68,
    )
    prepared_attack, transformed_names = transform_with_manifest(attacks, pipeline_path, manifest)
    if transformed_names != model_columns:
        raise RuntimeError(
            "Frozen feature contract did not reproduce the stage-one model column order."
        )
    attack_values = prepared_attack.to_numpy(dtype=float)
    validation_raw = normal.iloc[validation_idx].copy()
    validation_raw["is_attack"] = False
    evaluation_raw = pd.concat([validation_raw, attacks], ignore_index=True, sort=False)
    evaluation_values = np.vstack([normal_values[validation_idx], attack_values])
    y_true = np.concatenate([np.zeros(len(validation_idx), dtype=int), _binary_label(attacks)])
    lstm_scores = lstm.score_samples(evaluation_values)
    forest_scores = -forest.score_samples(evaluation_values)
    lstm_hit = lstm_scores >= lstm_threshold["threshold"]
    forest_hit = forest_scores >= forest_threshold["threshold"]
    ensemble_mode = str(config["stage_one"].get("ensemble_mode", "either_detector"))
    predicted = (
        (lstm_hit | forest_hit) if ensemble_mode == "either_detector" else (lstm_hit & forest_hit)
    )
    normalized_score = np.maximum(
        lstm_scores / max(lstm_threshold["threshold"], 1e-12),
        forest_scores / max(forest_threshold["threshold"], 1e-12),
    )
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
    scores["lstm_reconstruction_score"] = lstm_scores
    scores["isolation_forest_score"] = forest_scores
    scores["lstm_threshold"] = lstm_threshold["threshold"]
    scores["isolation_forest_threshold"] = forest_threshold["threshold"]
    scores["stage1_normalized_score"] = normalized_score
    scores["stage1_anomaly"] = predicted
    contributions = _contributions(
        evaluation_values, model_columns, normal_values[train_idx], scores
    )
    # Persist the exact transformed inputs used for evaluation.  Besides making
    # the run reproducible outside the dashboard, this permits post-training
    # latent/cluster/interaction diagnostics without re-reading or changing a
    # PCAP source file.
    evaluation_prepared = scores.copy()
    for feature_index, feature_name in enumerate(model_columns):
        evaluation_prepared[feature_name] = evaluation_values[:, feature_index]
        if feature_name in evaluation_raw:
            evaluation_prepared[f"raw__{feature_name}"] = pd.to_numeric(
                evaluation_raw[feature_name], errors="coerce"
            ).to_numpy()
    stage1_metrics = _metrics(y_true, predicted.astype(int), normalized_score)
    stage1_metrics.update(
        {
            "thresholds": {"lstm": lstm_threshold, "isolation_forest": forest_threshold},
            "ensemble_mode": ensemble_mode,
            "input_features": len(model_columns),
            "training_normal_records": int(len(train_idx)),
            "validation_normal_records": int(len(validation_idx)),
            "attack_evaluation_records": int(len(attacks)),
        }
    )
    report(
        "stage1_evaluated",
        "stage1_evaluation",
        (
            f"Stage 1 ارزیابی شد: FPR={stage1_metrics['false_positive_rate']:.2%} · "
            f"Recall={stage1_metrics['recall']:.2%}."
        ),
        0.75,
        stage1_metrics=stage1_metrics,
    )
    lstm.save(stage1_dir / "lstm-autoencoder.pt")
    joblib.dump(forest, stage1_dir / "isolation-forest.joblib")
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
    report(
        "stage2_matrix_preparing",
        "stage2_random_forest",
        "ماتریس Stage 2 از featureهای frozen و scoreهای Stage 1 ساخته می‌شود.",
        0.78,
        attack_type_rows=len(type_rows),
    )
    type_values_frame, _ = transform_with_manifest(type_rows, pipeline_path, manifest)
    type_values = type_values_frame.to_numpy(dtype=float)
    stage1_attack_lstm = lstm.score_samples(type_values)
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
        f"Random Forest با {len(stage2_values):,} ردیف و {len(stage2_columns):,} ستون آموزش می‌بیند.",
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
        f"Stage 2 کامل شد: weighted F1={stage2_metrics['weighted_f1']:.2%} روی {stage2_metrics['test_records']:,} ردیف آزمون.",
        0.89,
        stage2_metrics=stage2_metrics,
    )

    # End-to-end evaluation: only packets raised by stage one are eligible for a
    # type prediction.  This is the same gate a deployment uses in real time.
    pipeline_rows = scores.copy()
    pipeline_rows["predicted_attack_type"] = "normal"
    attack_gate = pipeline_rows["stage1_anomaly"].to_numpy()
    gated_values = evaluation_values[attack_gate]
    if len(gated_values):
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
    write_table(pipeline_rows, output_dir / "pipeline" / "end-to-end-predictions.parquet")
    report(
        "pipeline_evaluated",
        "end_to_end",
        f"پایپ‌لاین end-to-end ارزیابی شد؛ {int(attack_gate.sum()):,} ردیف از گیت Stage 1 عبور کرد.",
        0.93,
        stage1_anomalies=int(attack_gate.sum()),
        evaluation_records=len(pipeline_rows),
    )
    contract = {
        "schema_version": "1.0.0",
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
            "thresholds": stage1_metrics["thresholds"],
            "ensemble_mode": ensemble_mode,
        },
        "stage2": {"random_forest": str(stage2_dir / "attack-random-forest.joblib")},
    }
    write_json(contract, output_dir / "model-contract.json")
    exports = _export_onnx(lstm, forest, classifier, len(model_columns), output_dir / "onnx")
    report(
        "onnx_export_completed",
        "packaging",
        "artifactهای اصلی و تلاش برای export ONNX ثبت شد.",
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
        "contract": str(output_dir / "model-contract.json"),
    }
    write_json(summary, output_dir / "training-summary.json")
    progress.complete(
        "مدل دو مرحله‌ای، قرارداد inference و گزارش‌های ارزیابی با موفقیت ذخیره شدند.",
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
    lstm_scores = lstm.score_samples(values)
    forest_scores = -forest.score_samples(values)
    lstm_threshold = float(contract["stage1"]["thresholds"]["lstm"]["threshold"])
    forest_threshold = float(contract["stage1"]["thresholds"]["isolation_forest"]["threshold"])
    lstm_hit = lstm_scores >= lstm_threshold
    forest_hit = forest_scores >= forest_threshold
    is_anomaly = (
        (lstm_hit | forest_hit)
        if contract["stage1"]["ensemble_mode"] == "either_detector"
        else (lstm_hit & forest_hit)
    )
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
    output["stage1_anomaly"] = is_anomaly
    output["predicted_attack_type"] = "normal"
    if is_anomaly.any():
        classifier = joblib.load(resolve(contract["stage2"]["random_forest"]))
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
    return output
