"""Standalone Stage 2 Random Forest training and inference for attack classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from anomdet.core.io import write_json, write_table
from anomdet.core.resources import effective_workers

PACKET_COLUMNS = [
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


@dataclass
class RandomForestTrainingResult:
    """The fitted classifier and its frozen input order."""

    classifier: RandomForestClassifier
    input_columns: list[str]
    metrics: dict[str, Any]
    train_positions: np.ndarray
    test_positions: np.ndarray


def stage_two_readiness(attack_rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Decide whether accepted packet labels can support a trustworthy Stage 2.

    Stage 1 is useful with benign traffic alone.  Stage 2 is intentionally
    stricter: an attack-type classifier needs enough accepted labels in more
    than one class to produce an honest held-out performance result.
    """

    labels = attack_rows.get("label", pd.Series("unknown", index=attack_rows.index)).astype(
        "string"
    )
    counts = labels.value_counts(dropna=False)
    minimum_records = max(4, int(config["stage_two"].get("min_training_records", 20)))
    minimum_per_class = max(2, int(config["stage_two"].get("min_records_per_class", 4)))
    require_multiple = bool(config["stage_two"].get("require_multiple_attack_types", True))
    if not bool(config["stage_two"].get("enabled", True)):
        return {
            "ready": False,
            "reason": "disabled_for_stage_one_only_run",
            "records": int(len(attack_rows)),
            "classes": {str(label): int(records) for label, records in counts.items()},
            "class_count": int(len(counts)),
            "minimum_records": minimum_records,
            "minimum_records_per_class": minimum_per_class,
            "require_multiple_attack_types": require_multiple,
        }
    class_count = int(len(counts))
    reasons: list[str] = []
    if len(attack_rows) < minimum_records:
        reasons.append(f"requires at least {minimum_records} accepted attack records")
    if require_multiple and class_count < 2:
        reasons.append("requires at least two labelled attack types")
    if not counts.empty and int(counts.min()) < minimum_per_class:
        reasons.append(f"requires at least {minimum_per_class} records in every attack type")
    if counts.empty:
        reasons.append("contains no accepted attack labels")
    return {
        "ready": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "records": int(len(attack_rows)),
        "classes": {str(label): int(records) for label, records in counts.items()},
        "class_count": class_count,
        "minimum_records": minimum_records,
        "minimum_records_per_class": minimum_per_class,
        "require_multiple_attack_types": require_multiple,
    }


def make_stage_two_matrix(
    transformed_values: np.ndarray,
    transformed_columns: list[str],
    lstm_scores: np.ndarray,
    isolation_scores: np.ndarray,
    include_stage_one_scores: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build the optional Stage 1 enriched matrix in one explicit reusable place."""
    matrix = np.asarray(transformed_values, dtype=float)
    columns = list(transformed_columns)
    if include_stage_one_scores:
        matrix = np.column_stack([matrix, lstm_scores, isolation_scores])
        columns += ["stage1_lstm_score", "stage1_isolation_forest_score"]
    return matrix, columns


def _max_samples_for_records(config: dict[str, Any], record_count: int) -> float | int | None:
    """Avoid undersized bootstrap samples in smoke tests and rare attack classes."""
    configured = config["stage_two"].get("max_samples")
    if configured is None:
        return None
    if isinstance(configured, float) and configured * record_count < 16:
        return None
    return configured


def _diagnostic_workers(config: dict[str, Any]) -> int:
    """Keep auxiliary chart refits from competing with the deployed training job."""
    available = effective_workers(int(config["runtime"]["cpu_workers"]))
    cap = max(1, int(config.get("evaluation", {}).get("diagnostic_cpu_workers", 4)))
    return min(available, cap)


def _held_out_positions(
    attack_rows: pd.DataFrame, labels: pd.Series, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    """Prefer a capture-held-out test set, falling back to stratified packets.

    Holding out a whole capture is the strongest guard against packet/flow
    leakage.  Small datasets often expose one capture per attack class, where
    that split would remove a class from training; the explicit fallback keeps
    the evaluation possible and records that limitation in the artifact.
    """

    positions = np.arange(len(attack_rows))
    target_fraction = float(config["stage_two"]["test_fraction"])
    captures = attack_rows.get("capture")
    if captures is not None:
        groups = captures.astype("string").fillna("__missing_capture__")
        per_class_captures = (
            pd.DataFrame({"label": labels.to_numpy(), "capture": groups.to_numpy()})
            .groupby("label", dropna=False)["capture"]
            .nunique()
        )
        max_splits = int(per_class_captures.min()) if not per_class_captures.empty else 0
        if groups.nunique(dropna=False) >= 2 and max_splits >= 2:
            splits = min(5, max_splits)
            splitter = StratifiedGroupKFold(
                n_splits=splits,
                shuffle=True,
                random_state=int(config["project"]["random_seed"]),
            )
            candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
            for train, test in splitter.split(np.zeros(len(labels)), labels, groups):
                if labels.iloc[train].nunique() != labels.nunique():
                    continue
                candidates.append((abs(len(test) / max(len(labels), 1) - target_fraction), train, test))
            if candidates:
                _, train, test = min(candidates, key=lambda item: item[0])
                return train, test, "stratified_group_by_capture"

    counts = labels.value_counts()
    stratify = labels if len(counts) > 1 and int(counts.min()) >= 2 else None
    train, test = train_test_split(
        positions,
        test_size=target_fraction,
        random_state=int(config["project"]["random_seed"]),
        stratify=stratify,
    )
    strategy = "stratified_packet_fallback" if stratify is not None else "random_packet_fallback"
    return train, test, strategy


def _diagnostic_forest(
    config: dict[str, Any], *, estimators: int, seed: int, training_records: int
) -> RandomForestClassifier:
    """Build bounded CPU diagnostics without changing the deployed RF model."""
    return RandomForestClassifier(
        n_estimators=estimators,
        min_samples_leaf=int(config["stage_two"]["min_samples_leaf"]),
        max_features=config["stage_two"].get("max_features", "sqrt"),
        max_samples=_max_samples_for_records(config, training_records),
        max_depth=config["stage_two"].get("max_depth"),
        class_weight="balanced_subsample",
        n_jobs=_diagnostic_workers(config),
        random_state=seed,
    )


def _write_stage_two_diagnostics(
    *,
    classifier: RandomForestClassifier,
    matrix: np.ndarray,
    labels: pd.Series,
    input_columns: list[str],
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist chart-ready RF diagnostics from the same held-out split.

    Diagnostic sweeps use a bounded tree count configured independently from
    the deployed classifier. Their role is comparative, while the main model
    remains trained with ``stage_two.n_estimators``.
    """
    evaluation = config.get("evaluation", {})
    seed = int(config["project"]["random_seed"])
    diagnostic_trees = min(
        int(config["stage_two"]["n_estimators"]),
        max(10, int(evaluation.get("stage2_diagnostic_estimators", 100))),
    )
    repeats = max(1, int(evaluation.get("permutation_repeats", 5)))
    test_labels = labels.iloc[test_positions]

    permutation = permutation_importance(
        classifier,
        matrix[test_positions],
        test_labels,
        scoring="f1_weighted",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=effective_workers(int(config["runtime"]["cpu_workers"])),
    )
    write_table(
        pd.DataFrame(
            {
                "feature": input_columns,
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
                "scoring": "f1_weighted",
                "repeats": repeats,
            }
        ).sort_values("importance_mean", ascending=False, kind="stable"),
        output_dir / "permutation-importance.parquet",
    )

    sweep_sizes = sorted(
        {
            min(diagnostic_trees, count)
            for count in [
                max(10, diagnostic_trees // 4),
                max(20, diagnostic_trees // 2),
                max(30, int(diagnostic_trees * 0.75)),
                diagnostic_trees,
            ]
        }
    )
    estimator_rows: list[dict[str, Any]] = []
    for estimators in sweep_sizes:
        model = _diagnostic_forest(
            config,
            estimators=estimators,
            seed=seed,
            training_records=len(train_positions),
        ).fit(
            matrix[train_positions], labels.iloc[train_positions]
        )
        prediction = model.predict(matrix[test_positions])
        estimator_rows.append(
            {
                "n_estimators": estimators,
                "weighted_f1": f1_score(test_labels, prediction, average="weighted", zero_division=0),
                "accuracy": accuracy_score(test_labels, prediction),
                "diagnostic_tree_cap": diagnostic_trees,
            }
        )
    write_table(pd.DataFrame(estimator_rows), output_dir / "n-estimators-sweep.parquet")

    classes = labels.iloc[train_positions].value_counts()
    minimum_fraction = min(1.0, max(0.2, (len(classes) * 2) / max(len(train_positions), 1)))
    train_label_counts = labels.iloc[train_positions].value_counts()
    sampling_stratify = (
        labels.iloc[train_positions]
        if len(train_label_counts) > 1 and int(train_label_counts.min()) >= 2
        else None
    )
    learning_rows: list[dict[str, Any]] = []
    for fraction in sorted({minimum_fraction, 0.4, 0.6, 0.8, 1.0}):
        if fraction >= 1.0:
            sampled_positions = train_positions
        else:
            sampled_positions, _ = train_test_split(
                train_positions,
                train_size=fraction,
                random_state=seed,
                stratify=sampling_stratify,
            )
        model = _diagnostic_forest(
            config,
            estimators=diagnostic_trees,
            seed=seed,
            training_records=len(sampled_positions),
        ).fit(
            matrix[sampled_positions], labels.iloc[sampled_positions]
        )
        prediction = model.predict(matrix[test_positions])
        learning_rows.append(
            {
                "training_fraction": fraction,
                "training_records": len(sampled_positions),
                "weighted_f1": f1_score(test_labels, prediction, average="weighted", zero_division=0),
                "accuracy": accuracy_score(test_labels, prediction),
                "diagnostic_tree_cap": diagnostic_trees,
            }
        )
    write_table(pd.DataFrame(learning_rows), output_dir / "learning-curve.parquet")

    minimum_class = int(classes.min()) if not classes.empty else 0
    split_count = min(5, minimum_class)
    cross_validation_rows: list[dict[str, Any]] = []
    if split_count >= 2:
        splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=seed)
        training_values = matrix[train_positions]
        training_labels = labels.iloc[train_positions].reset_index(drop=True)
        for fold, (fit_index, validation_index) in enumerate(
            splitter.split(training_values, training_labels), start=1
        ):
            model = _diagnostic_forest(
                config,
                estimators=diagnostic_trees,
                seed=seed + fold,
                training_records=len(fit_index),
            ).fit(
                training_values[fit_index], training_labels.iloc[fit_index]
            )
            prediction = model.predict(training_values[validation_index])
            cross_validation_rows.append(
                {
                    "fold": fold,
                    "weighted_f1": f1_score(
                        training_labels.iloc[validation_index], prediction, average="weighted", zero_division=0
                    ),
                    "accuracy": accuracy_score(training_labels.iloc[validation_index], prediction),
                    "validation_records": len(validation_index),
                    "diagnostic_tree_cap": diagnostic_trees,
                }
            )
    write_table(pd.DataFrame(cross_validation_rows), output_dir / "cross-validation.parquet")

    leaf_values = sorted(
        {
            max(1, int(config["stage_two"]["min_samples_leaf"]) // 2),
            int(config["stage_two"]["min_samples_leaf"]),
            max(2, int(config["stage_two"]["min_samples_leaf"]) * 2),
            max(3, int(config["stage_two"]["min_samples_leaf"]) * 4),
        }
    )
    leaf_rows: list[dict[str, Any]] = []
    for leaf in leaf_values:
        model = _diagnostic_forest(
            config,
            estimators=diagnostic_trees,
            seed=seed,
            training_records=len(train_positions),
        )
        model.set_params(min_samples_leaf=leaf)
        model.fit(matrix[train_positions], labels.iloc[train_positions])
        prediction = model.predict(matrix[test_positions])
        leaf_rows.append(
            {
                "min_samples_leaf": leaf,
                "weighted_f1": f1_score(test_labels, prediction, average="weighted", zero_division=0),
                "accuracy": accuracy_score(test_labels, prediction),
                "diagnostic_tree_cap": diagnostic_trees,
            }
        )
    write_table(pd.DataFrame(leaf_rows), output_dir / "min-samples-leaf-sweep.parquet")


def train_attack_random_forest(
    attack_rows: pd.DataFrame,
    matrix: np.ndarray,
    input_columns: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> RandomForestTrainingResult:
    """Train only the attack-type classifier from accepted PCAP-derived attack rows.

    ``attack_rows`` must already have passed PCAP/CSV mapping acceptance.  The
    function deliberately accepts a preprocessed matrix so that it cannot fit
    a second, incompatible preprocessing pipeline behind Stage 1's back.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = attack_rows.get("label", pd.Series("unknown", index=attack_rows.index)).astype(
        "string"
    )
    readiness = stage_two_readiness(attack_rows, config)
    if not readiness["ready"]:
        raise ValueError(f"Random Forest Stage 2 is not ready: {readiness['reason']}.")
    if len(matrix) != len(attack_rows):
        raise ValueError("Random Forest rows and transformed feature matrix do not align.")
    counts = labels.value_counts()
    train_positions, test_positions, split_strategy = _held_out_positions(attack_rows, labels, config)
    resolved_max_samples = _max_samples_for_records(config, len(train_positions))
    classifier = RandomForestClassifier(
        n_estimators=int(config["stage_two"]["n_estimators"]),
        min_samples_leaf=int(config["stage_two"]["min_samples_leaf"]),
        max_features=config["stage_two"].get("max_features", "sqrt"),
        max_samples=resolved_max_samples,
        max_depth=config["stage_two"].get("max_depth"),
        class_weight="balanced_subsample",
        n_jobs=effective_workers(int(config["runtime"]["cpu_workers"])),
        random_state=int(config["project"]["random_seed"]),
    ).fit(matrix[train_positions], labels.iloc[train_positions])
    predicted = classifier.predict(matrix[test_positions])
    probabilities = classifier.predict_proba(matrix[test_positions])
    metrics: dict[str, Any] = {
        "records": int(len(attack_rows)),
        "classes": {str(key): int(value) for key, value in counts.items()},
        "test_accuracy": round(float(accuracy_score(labels.iloc[test_positions], predicted)), 6),
        "weighted_f1": round(
            float(
                f1_score(
                    labels.iloc[test_positions], predicted, average="weighted", zero_division=0
                )
            ),
            6,
        ),
        "class_weight": "balanced_subsample",
        "max_features": config["stage_two"].get("max_features", "sqrt"),
        "max_samples": resolved_max_samples,
        "max_depth": config["stage_two"].get("max_depth"),
        "input_features": len(input_columns),
        "training_records": int(len(train_positions)),
        "test_records": int(len(test_positions)),
        "status": "trained",
        "split_strategy": split_strategy,
        "held_out_capture_count": int(
            attack_rows.iloc[test_positions]
            .get("capture", pd.Series("__missing_capture__", index=test_positions))
            .nunique(dropna=False)
        ),
        "readiness": readiness,
    }
    class_order = classifier.classes_.astype(str).tolist()
    confusion = (
        pd.DataFrame(
            confusion_matrix(labels.iloc[test_positions], predicted, labels=classifier.classes_),
            index=class_order,
            columns=class_order,
        )
        .rename_axis("true_attack_type")
        .reset_index()
    )
    scores = attack_rows.iloc[test_positions][
        [column for column in PACKET_COLUMNS if column in attack_rows]
    ].copy()
    scores["predicted_attack_type"] = predicted
    scores["correct"] = labels.iloc[test_positions].to_numpy() == predicted
    scores["prediction_confidence"] = probabilities.max(axis=1)
    probability_columns = pd.DataFrame(
        probabilities,
        columns=[f"probability__{label}" for label in classifier.classes_.astype(str)],
        index=scores.index,
    )
    probabilities_frame = pd.concat([scores, probability_columns], axis=1)
    importance = pd.DataFrame(
        {"feature": input_columns, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False, kind="stable")
    joblib.dump(classifier, output_dir / "attack-random-forest.joblib")
    write_table(scores, output_dir / "scores.parquet")
    write_table(probabilities_frame, output_dir / "probabilities.parquet")
    write_table(confusion, output_dir / "confusion-matrix.parquet")
    write_table(importance, output_dir / "feature-importance.parquet")
    split_manifest = attack_rows.iloc[
        np.concatenate([train_positions, test_positions])
    ][[column for column in PACKET_COLUMNS if column in attack_rows]].copy()
    split_manifest["stage2_split"] = [
        *("train" for _ in train_positions),
        *("test" for _ in test_positions),
    ]
    split_manifest["split_strategy"] = split_strategy
    write_table(split_manifest, output_dir / "held-out-split.parquet")
    _write_stage_two_diagnostics(
        classifier=classifier,
        matrix=matrix,
        labels=labels,
        input_columns=input_columns,
        train_positions=train_positions,
        test_positions=test_positions,
        config=config,
        output_dir=output_dir,
    )
    write_json(metrics, output_dir / "metrics.json")
    return RandomForestTrainingResult(
        classifier=classifier,
        input_columns=input_columns,
        metrics=metrics,
        train_positions=train_positions,
        test_positions=test_positions,
    )


def predict_attack_type(classifier: RandomForestClassifier, matrix: np.ndarray) -> np.ndarray:
    """Predict attack types with the separately persisted Stage 2 model."""
    return classifier.predict(np.asarray(matrix, dtype=float))
