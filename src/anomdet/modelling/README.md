# Unsupervised model comparison

The Phase 1 training command compares five unsupervised anomaly detectors:

- Isolation Forest
- Local Outlier Factor in novelty mode
- One-Class SVM
- Sequence LSTM autoencoder (LSTM AE)

```powershell
uv run anomaly train artifacts/features/capture.parquet --profile compact-it --strategy grouped --group it
uv run anomaly train artifacts/features/capture.parquet --strategy per_protocol --labels artifacts/mapping/it.csv
uv run anomaly train artifacts/features/capture.parquet --strategy per_protocol --models isolation_forest,lstm_autoencoder
```

Every model gets the same prepared input, a score where higher means more anomalous, a contamination-derived threshold, persisted model artefact, scores, metrics, feature contribution, and a comparison row. The preprocessing transformer is fit on the training partition only before transforming evaluation data. The default temporal split keeps later traffic outside training; set `models.split_strategy: random` only for a deliberate i.i.d. study. When trustworthy mapped labels contain both normal and anomalous data, ROC-AUC and average precision are reported; otherwise the command reports operational score distributions only.

The LSTM AE turns adjacent prepared records into configurable windows (`sequence_length` and `sequence_stride`), reconstructs each window, and aggregates reconstruction MSE back to a record-level anomaly score. It writes `model.pt`, `training-history.parquet`, and model metadata for every run. `max_train_windows` caps training cost while score generation still covers the full held-out partition.

`run_feature_experiments()` is the common API for all-features baseline versus multiple selected profiles. `run_lstm_sweep()` adds a named LSTM parameter variant to every comparison row, which the dashboard plots as training-loss and trade-off curves. See the [integration guide](../../../docs/INTEGRATION_GUIDE.md) for the complete module contract.

`per_protocol` creates independent training scopes. `grouped` trains one scope for the IT, OT, or all-protocols group; protocol is retained as an input category in grouped mode so that the model can learn protocol-specific distributions without activating unrelated protocol fields.
