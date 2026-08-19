# Integration guide

This guide describes the stable input, output, and invocation contract for integrating the platform into another service or module. The Streamlit workbench is optional; every capability is available through the Python package and CLI.

## System boundary

```text
PCAP / PCAPNG + optional CSV labels
        |
        v
Extraction -> feature table -> feature profile -> preprocessing -> detector experiment
        |              |                 |                  |              |
        v              v                 v                  v              v
run summary       quality report    JSON manifest     pipeline/model     scores, metrics,
mapping evidence                                                       importance, curves
```

The canonical configuration is `config/default.yaml`. Create an environment-specific YAML override for machine paths, dataset inventory, runtime limits, and model settings; do not edit source code to configure a deployment.

## Module contracts

| Module | Input | Output | Primary entry point |
|---|---|---|---|
| Extraction | `.pcap`/`.pcapng` capture or protocol folder | Protocol-aware Parquet/CSV/JSONL feature table and per-capture manifest | `anomaly extract` / `extract_pcap_features()` |
| Mapping | Capture + candidate label CSV + domain (`it` or `ot`) | Mapped flow table, status/confidence evidence, summary | `anomaly map` / `map_pcap_to_labels()` |
| Feature quality | Extracted feature table | Protocol-specific extraction validation: record/flow coverage, variation, zero ratio, parser capability, model usability | `anomaly select analyze` / `feature_quality_report()` |
| Feature selection | Feature names from the catalogue, scope, rationale | Immutable JSON profile under `artifacts/feature_profiles/` | `anomaly select create` / `create_profile()` |
| Preprocessing | Feature table + optional profile, protocol scope, mapped labels | Numeric prepared table, fitted transformer, manifest | `anomaly prepare` / `prepare_features()` |
| Training | Feature table + strategy + selected profile(s) + detector config | Prepared inputs, persisted models, scores, metrics, importance, comparison | `anomaly train` / `train_models()` |
| Feature experiment | Feature table + baseline/profile list + detector list | Aggregate `comparison.parquet` across all profiles | `run_feature_experiments()` |
| LSTM parameter sweep | Feature table + profiles + LSTM parameter sets | `sweep-comparison.parquet`, per-variant curves and models | `run_lstm_sweep()` |
| Configured inventory | Protocol folders and optional `data.protocol_folders` metadata | Self-contained run directory with all preceding artefacts | `anomaly pipeline run` / `run_inventory()` |

## Required input conventions

### Capture input

- Accepts one `.pcap`/`.pcapng` file or every direct capture in a protocol-named folder.
- Phase 1 extracts SSH, DNS, HTTP, Modbus TCP, and S7comm. A capture with unsupported packets still produces a clear zero-row or partial-protocol manifest rather than silently inventing a protocol.
- Put files directly under a supported protocol folder. The inventory discovers those folders automatically; configure `protocol_folders` only for labels or overrides:

```yaml
data:
  raw_pcap_dir: D:/network-data/pcap
  it_label_dir: D:/network-data/labels/it
  ot_label_dir: D:/network-data/labels/ot
  protocol_folders:
    - id: dns-observation
      protocol: dns
      domain: it
      labels: [DNS_Spoofing.csv]
```

### Label input

Label files are candidate evidence, not automatic training truth. IT mapping expects one of the configured flow/timestamp/IP/port column aliases; OT mapping expects the configured start/end and source/destination aliases. Inspect `match_status` and `match_confidence` before passing a mapping output to `anomaly train --labels`.

### Feature profile input

A profile is an immutable JSON manifest created from catalogue names. The profile contains its own version, scope, rationale, and feature list. Pass either its path or its stable name to CLI commands. The all-catalogue-feature baseline is always included by `run_feature_experiments()` and by the dashboard's multi-profile experiment form.

## Outputs and layout

One configured run is self-contained:

```text
artifacts/runs/<run-id>/
  run-summary.json
  features/
    all-datasets.parquet             # dashboard and module source of truth
    <dataset-id>.parquet
  reports/
    all-datasets-feature-quality.parquet  # one row per protocol-feature validation result
  mapping/
    <candidate>.parquet
    <candidate>.summary.json
  feature-evaluation/
    comparison.parquet
    all-features/
      <scope>/<model>/{scores.parquet,metrics.json,model.*,feature-importance.parquet}
  experiments/
    <experiment-id>/
      comparison.parquet
      experiment-batch.json
      lstm-sweep/sweep-comparison.parquet
```

`comparison.parquet` is the integration-friendly model summary. Important fields include `feature_profile`, `scope`, `protocols`, `model`, `input_features`, `training_rows`, `test_rows`, `fit_seconds`, `process_memory_delta_mb`, score mean and percentiles, anomaly rate, and, when validated labels are supplied, ROC-AUC and average precision. LSTM AE persists `reconstruction_mse_mean` and `reconstruction_rmse`; its per-record `anomaly_score` is reconstruction MSE. PCA is retained only for reading legacy artefacts and is not a new-training candidate.

Every LSTM AE run additionally persists:

- `model.pt`: explicit PyTorch state dictionary and architecture parameters.
- `training-history.parquet`: epoch, training loss, validation loss, and learning rate.
- `feature-importance.parquet`: score-shift permutation contribution per original feature.

Other detectors persist `model.joblib`, and all detectors persist scores, metrics, model metadata, and feature importance. Never load model artefacts from an untrusted source.

## CLI integration

```powershell
# One protocol folder and its model comparison.
uv run anomaly extract D:/network-data/pcap/modbus --output artifacts/features/modbus.parquet
uv run anomaly select analyze artifacts/features/modbus.parquet --output artifacts/reports/modbus-quality.parquet
uv run anomaly select create modbus-operations --features packet_length,flow_duration,modbus_function_code,modbus_quantity
uv run anomaly train artifacts/features/modbus.parquet --profile modbus-operations --strategy per_protocol --models isolation_forest,lstm_autoencoder

# Configured multi-capture run.
uv run anomaly --config config/lab.yaml pipeline run --output artifacts/runs/lab-01
```

The CLI returns non-zero on unrecoverable input or model errors. A configured inventory continues mapping other candidate CSVs when one candidate fails, and records that failure in `run-summary.json`.

## Python integration

```python
from pathlib import Path

from anomdet.core.config import load_config
from anomdet.modelling.training import run_feature_experiments

config = load_config(Path("config/lab.yaml"))
comparison, manifest = run_feature_experiments(
    feature_path=Path("artifacts/runs/lab-01/features/all-datasets.parquet"),
    config=config,
    strategy="per_protocol",
    group="all",
    profiles=["modbus-operations"],
    output_dir=Path("artifacts/runs/lab-01/experiments/integration-01"),
    candidates=["isolation_forest", "lstm_autoencoder"],
    model_overrides={
        "lstm_autoencoder": {
            "sequence_length": 10,
            "sequence_stride": 10,
            "hidden_size": 128,
            "latent_size": 64,
            "epochs": 100,
        }
    },
)
print(manifest["comparison"])
```

Use `run_lstm_sweep()` when comparing several LSTM configurations. Keep sweeps bounded and start with a small packet cap or small `max_train_windows` during structural validation.

## Deployment notes

- Keep PCAPs, CSV labels, generated artefacts, and model files outside source control.
- Set `runtime.cpu_workers` and `runtime.memory_limit_gb` in the environment override. LSTM defaults to CPU and caps its training windows; it scores all held-out records.
- `models.split_strategy: temporal` is the default so later traffic is held out from training. Switch to `random` only for an explicitly i.i.d. study.
- The dashboard is a local operator interface. A production module should call the package functions or the CLI, read the persisted comparison/manifests, and own authentication, scheduling, retention, and alert delivery at its boundary.
