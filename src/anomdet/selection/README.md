# Feature selection

Feature selection is manifest-based, so a model never relies on an untracked list passed in a shell command. Every profile receives a content-derived version and records selected features, protocol scope, description, and creation time.

```powershell
uv run anomaly select create compact-it --features packet_length,payload_entropy,flow_duration,packet_rate --protocols ssh,dns,http
uv run anomaly select list
uv run anomaly select analyze artifacts/features/capture.parquet --output artifacts/reports/feature-quality.parquet
```

The dashboard reads both profile manifests and `feature-quality` reports. It exposes missingness, cardinality, variance, approximate storage cost, feature description, protocol applicability, and cost tier before a user selects features.
