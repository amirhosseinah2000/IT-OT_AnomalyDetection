"""Standalone Stage 2 Random Forest training and inference for attack classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

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
    if len(attack_rows) < 4:
        raise ValueError("Random Forest Stage 2 needs at least four mapped attack records.")
    if len(matrix) != len(attack_rows):
        raise ValueError("Random Forest rows and transformed feature matrix do not align.")
    counts = labels.value_counts()
    stratify = labels if len(counts) > 1 and int(counts.min()) >= 2 else None
    positions = np.arange(len(attack_rows))
    train_positions, test_positions = train_test_split(
        positions,
        test_size=float(config["stage_two"]["test_fraction"]),
        random_state=int(config["project"]["random_seed"]),
        stratify=stratify,
    )
    classifier = RandomForestClassifier(
        n_estimators=int(config["stage_two"]["n_estimators"]),
        min_samples_leaf=int(config["stage_two"]["min_samples_leaf"]),
        class_weight="balanced_subsample",
        n_jobs=effective_workers(int(config["runtime"]["cpu_workers"])),
        random_state=int(config["project"]["random_seed"]),
    ).fit(matrix[train_positions], labels.iloc[train_positions])
    predicted = classifier.predict(matrix[test_positions])
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
        "input_features": len(input_columns),
        "training_records": int(len(train_positions)),
        "test_records": int(len(test_positions)),
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
    importance = pd.DataFrame(
        {"feature": input_columns, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False, kind="stable")
    joblib.dump(classifier, output_dir / "attack-random-forest.joblib")
    write_table(scores, output_dir / "scores.parquet")
    write_table(confusion, output_dir / "confusion-matrix.parquet")
    write_table(importance, output_dir / "feature-importance.parquet")
    write_json(metrics, output_dir / "metrics.json")
    return RandomForestTrainingResult(classifier, input_columns, metrics)


def predict_attack_type(classifier: RandomForestClassifier, matrix: np.ndarray) -> np.ndarray:
    """Predict attack types with the separately persisted Stage 2 model."""
    return classifier.predict(np.asarray(matrix, dtype=float))
