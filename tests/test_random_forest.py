"""Contract tests for the standalone Stage 2 Random Forest script."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from anomdet.core.config import load_config
from anomdet.modelling.random_forest import make_stage_two_matrix, train_attack_random_forest


def test_trains_random_forest_in_its_own_module(tmp_path) -> None:
    """The classifier owns its outputs and preserves the supplied Stage 1 feature order."""
    config = copy.deepcopy(load_config())
    config["stage_two"].update({"n_estimators": 8, "min_samples_leaf": 1, "test_fraction": 0.25})
    rows = pd.DataFrame(
        {
            "flow_id": [f"flow-{index}" for index in range(12)],
            "label": ["scan"] * 6 + ["flood"] * 6,
            "protocol": ["dns"] * 12,
        }
    )
    base = np.arange(36, dtype=float).reshape(12, 3)
    matrix, columns = make_stage_two_matrix(
        base,
        ["packet_length", "payload_size", "dns_rcode"],
        np.linspace(0.1, 1.2, 12),
        np.linspace(1.2, 0.1, 12),
        True,
    )

    result = train_attack_random_forest(rows, matrix, columns, config, tmp_path)

    assert result.input_columns[-2:] == ["stage1_lstm_score", "stage1_isolation_forest_score"]
    assert (tmp_path / "attack-random-forest.joblib").exists()
    assert (tmp_path / "feature-importance.parquet").exists()
