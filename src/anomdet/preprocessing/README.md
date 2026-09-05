# Preprocessing

`prepare` applies a selected profile to extracted features, removes features below the configured non-null threshold, imputes missing values, robust-scales numeric values, and one-hot encodes categorical values. The exact fitted transformer is stored with the matrix.

```powershell
uv run anomaly prepare artifacts/features/capture.parquet --profile compact-it --output artifacts/prepared/compact-it.parquet
```

The output has traceable `row_id`, `protocol`, `flow_id`, and optional `label` columns. These are metadata only and are never used as model inputs. Model columns follow them in the output table. High-cardinality text values are collapsed to the configured most-frequent categories before one-hot encoding, keeping CPU and memory use bounded.
