# Project Structure and Code Map

## Purpose and boundaries

This repository is a portable, CPU-first network-anomaly platform. Its source
of truth is local Parquet/JSON/model files; the dashboard is only a reader and
operator interface. A normal deployment path is:

```text
PCAP/PCAPNG
  -> protocol-aware packet/flow features
  -> saved packet-to-CSV mapping (when labels exist)
  -> immutable feature profile and fitted preprocessing contract
  -> Stage 1: LSTM-AE + Isolation Forest (+ optional known-attack gate)
  -> Stage 2: Random Forest attack-type classifier
  -> native model artifacts, ONNX, Parquet reports, dashboard/API/CLI use
```

The two important rules are:

1. CSV data supplies labels and mapping evidence; it is never silently turned
   into model features.
2. Every protocol, capture, feature profile, model contract, and output run is
   stored independently so a consumer does not have to depend on Streamlit.

## Repository root

| Path | Contents and responsibility | Change it? |
|---|---|---|
| `README.md` | Project overview, primary commands, data layout, and quick start. | Update when the user-facing workflow changes. |
| `pyproject.toml` | Package metadata, Python range, dependencies, CLI entry point (`anomaly`), CPU PyTorch index, and test/lint settings. | Change for dependency or packaging changes. |
| `uv.lock` | Exact resolved dependency graph used by `uv sync`. | Regenerate through `uv lock`/`uv sync`; do not hand-edit. |
| `requirements.txt` / `requirements-offline.txt` | Dependency manifests for conventional or offline installation. | Update with dependency changes. |
| `run_full.txt` | Saved operational command example. | Optional operational note; not runtime code. |
| `dataset-progress.json` | Latest local progress state from an execution. | Generated; do not treat as source code. |
| `build-deps.tar.gz`, `wheelhouse.tar.gz`, `wheelhouse/` | Offline installation bundles. | Generated/distribution assets; keep only when needed for an offline server. |
| `.venv/`, `.uv-cache/`, `.download-venv/` | Local virtual environment and package/download caches. | Generated; never commit or copy as project source. |
| `.streamlit/` | Streamlit-specific local settings, if configured. | Only change for dashboard runtime settings. |
| `.gitignore` | Excludes data, artifacts, caches, and local state from Git. | Change only when repository tracking policy changes. |

## Configuration: `config/`

Configuration files are YAML overrides read by `anomdet.core.config.load_config`.
They define data roots, protocol discovery, extraction limits, mapping policy,
preprocessing, CPU limits, model architecture, threshold policy, and optional
ClickHouse mirroring.

| File | What it is for |
|---|---|
| `default.yaml` | Baseline portable configuration. All commands use it unless `--config` is supplied. |
| `production-streaming.yaml` | Large-data/server settings. It enables disk-backed/streaming extraction and bounded model sampling. |
| `smoke-4protocol-200.yaml` | Fast four-protocol smoke-test setup with a small packet cap. It validates output production, not final accuracy. |
| `README.md` | Explains configuration sections and safe override practice. |

Do not modify `default.yaml` for a one-off server if an override YAML can be
used instead. The override is deeply merged with the default configuration.

## Input data: `data/`

The preferred raw layout is deliberately protocol-first:

```text
data/raw/
  benign/pcap/<protocol>/*.pcap|*.pcapng
  attack/pcap/<protocol>/*.pcap|*.pcapng
  attack/labels/<protocol>/*.csv
```

`<protocol>` is currently `ssh`, `dns`, `http`, `modbus`, or `s7comm`.
Benign PCAPs train Stage 1. Attack PCAPs are feature sources; their CSV files
are independently matched to the extracted packet records to establish trusted
labels. The folders under `data/interim` and `data/processed`, if present, are
working-data locations defined by configuration, not a replacement for the
artifact contracts below.

## Generated outputs: `artifacts/`

`artifacts/` is the portable output root. It can be copied to another system,
served by an API, read by the dashboard, or mirrored to ClickHouse. It is not
source code and should normally be excluded from Git.

### Feature profiles

```text
artifacts/feature_profiles/<profile-name>-<content-hash>.json
```

Each JSON file records the selected raw feature names, protocol scope,
description, creation timestamp, and content-derived version. The model
contract stores the exact profile used at training time; inference reuses its
fitted preprocessing pipeline instead of accepting an arbitrary column list.

### One dataset run

```text
artifacts/runs/<run-id>/
  dataset-progress.json
  run-summary.json
  mapping-summary.json                 # after `dataset map`
  manifests/input-inventory.json
  catalog/artifacts.parquet
  catalog/summary.json
  features/benign/<protocol>/records.parquet
  features/attack/<protocol>/<capture>/records.parquet
  labelled/attack/<protocol>/<capture>/records.parquet
  mappings/<protocol>/<capture>.parquet
  mappings/<protocol>/<capture>.audit.json
  reports/
  models/<model-run>/<protocol>/
```

| Path | Exact role |
|---|---|
| `manifests/input-inventory.json` | Physical PCAP/CSV discovery result before processing. Use it to confirm a capture was found. |
| `catalog/artifacts.parquet` | Machine-readable index of produced artifacts, their paths, type, protocol, split, description, and schema information. External systems should discover files through this catalog where possible. |
| `features/benign/<protocol>/records.parquet` | PCAP-derived, one-record-per-supported-packet feature table for normal traffic. It is the Stage 1 normal-training source. |
| `features/attack/<protocol>/<capture>/records.parquet` | Unlabelled feature extraction from an attack-side PCAP. It is never automatically considered all-malicious. |
| `mappings/...parquet` | Audited PCAP/CSV relation: endpoint/time evidence, candidate count, confidence, selected CSV row, and mapping status. |
| `mappings/...audit.json` | Mapping quality, time offset, match rate/confidence, and acceptance evidence for one capture/CSV relation. |
| `labelled/attack/.../records.parquet` | The attack-side PCAP feature rows after packet-time CSV label attachment. `mapping_accepted`, `label`, and `is_attack` determine whether a row may be used for supervised evaluation. |
| `reports/` | Feature quality, preprocessing comparison, resource usage, mapping distribution, and other tabular reports. |
| `models/` | Frozen model runs. Never mix artifacts from two model directories. |

The mapping cache is configured under `mapping.cache_directory`. It preserves a
packet-to-CSV relation keyed by PCAP/CSV identity and mapping policy. A later
`dataset map` reuses it unless `--remap` is requested or source evidence
changes. It is a cache of mapping evidence, not a training-data replacement.

### One two-stage model run

```text
models/<model-run>/<protocol>/
  training-progress.json
  training-summary.json
  model-contract.json
  onnx/
  reports/
  stage1/
  stage2/
  pipeline/
```

| Directory/file | What a consumer should use it for |
|---|---|
| `training-progress.json` | Durable event log for CLI/dashboard live monitoring. It contains stage, message, progress ratio, and resource-relevant details. |
| `training-summary.json` | High-level model, metrics, resources, elapsed time, ONNX export status, and contract path. |
| `model-contract.json` | Primary deployment entry point. It freezes profile, transformed feature order, preprocessing path, Stage 1 thresholds/models, Stage 2 status/model, and feature contract. |
| `onnx/` | Portable ONNX exports for LSTM-AE, Isolation Forest, Stage 2 RF, and the optional binary gate where conversion is supported. `onnx-export.json` records success/failure precisely. |
| `reports/resource-usage.parquet` | Before/after memory and host-resource snapshot. |
| `reports/preprocessing-comparison.parquet` | Raw versus transformed feature-health comparison. |
| `stage1/prepared-normal.parquet` | Prepared normal feature matrix; metadata is retained separately from model columns. |
| `stage1/prepared-normal.pipeline.joblib` and `.manifest.json` | Fitted imputation/encoding/scaling pipeline and immutable feature manifest. Required by portable inference. |
| `stage1/lstm-autoencoder.pt` | Native PyTorch LSTM-AE state and architecture metadata. |
| `stage1/isolation-forest.joblib` | Native Isolation Forest. |
| `stage1/binary-attack-gate.joblib` | Optional lightweight known-attack gate. It exists only when trusted labels span enough distinct captures. |
| `stage1/scores.parquet` | Packet-level Stage 1 decisions, scores, thresholds, votes, true labels where trustworthy, and evaluation-role flags. |
| `stage1/feature-evidence.parquet` | Top robust deviations from normal per record; use it to explain why an alert was raised. |
| `stage1/validation-summary.json` | Separates normal-only holdout, capture-held-out known-label testing, development-only metrics, and unlabelled mixed-capture observations. |
| `stage1/unlabelled-mixed-*.parquet` | Scores for PCAP packets that lack trusted labels. They are operational observations only and are excluded from accuracy metrics. |
| `stage2/attack-random-forest.joblib` | Native attack-type classifier when label readiness passes. |
| `stage2/readiness.json` / `metrics.json` | Explains whether Stage 2 trained or safely skipped, class coverage, split strategy, and metrics. |
| `pipeline/end-to-end-predictions.parquet` | Stage 1 gate plus Stage 2 type prediction for each scored evaluation packet. |
| `pipeline/*.parquet` | Held-out diagnostics: probabilities, ablation comparison, threshold sensitivity, inference latency, and seed stability when Stage 2 is available. |

## Installable code: `src/anomdet/`

`src/anomdet` is the product. `pyproject.toml` exposes `anomdet.cli:app` as the
`anomaly` command. `__init__.py` files only mark Python packages unless they
explicitly export a small API.

### Entry points and public integration

| File | What the code does |
|---|---|
| `cli.py` | Typer command-line adapter. It parses commands, loads configuration, calls the domain modules, writes concise Rich summaries, and turns exceptions into non-zero exits. It does not implement feature extraction or model mathematics itself. Commands include `dataset inspect/run/extract/map`, `select`, `prepare`, `train`, `two-stage train/score`, `dashboard`, `resources`, and `replay`. `--stage-one-only` deliberately skips the attack-type RF. |
| `service.py` | Programmatic façade for embedding the platform in another Python system. `AnomalyService` extracts features, creates profiles, trains/scores models, and returns explicit artifact paths through `ModelArtifacts`/`TrainResult`. `quick_anomaly_detection` is a convenience wrapper. |
| `analytics/reports.py` | Produces reusable DataFrames for feature overview and raw-vs-preprocessed comparison; it contains no UI logic. |

### `core/`: common, side-effect-safe utilities

| File | What the code does |
|---|---|
| `config.py` | Reads `config/default.yaml`, deep-merges an optional override, and resolves the artifact root. |
| `io.py` | Reads/writes Parquet, CSV, JSONL, and JSON with parent-directory creation. It standardizes the project's table persistence. |
| `logging.py` | Configures structured console/file logging and records exceptions with context. |
| `paths.py` | Resolves a requested PCAP file or directory safely into one or more valid capture paths. |
| `progress.py` | `DurableProgress` appends/replaces durable JSON events so a dashboard or external monitor can follow long extraction/training work. |
| `resources.py` | Captures CPU/memory snapshot and derives an effective worker count while reserving one logical CPU by default. |

### `datasets/`: input discovery

| File | What the code does |
|---|---|
| `inventory.py` | Discovers the protocol-first benign/attack PCAP and CSV layout. It applies optional allowlists/overrides, scores file-name similarity for candidate pairings, and emits `CaptureSource`/`AttackPair` records and an inventory report. |

### `features/`: PCAP parsing and causal feature creation

| File | What the code does |
|---|---|
| `extractor.py` | Main PCAP engine. It reads PCAP/PCAPNG with Scapy, identifies protocol, creates base packet fields, maintains bounded causal flow/host state, adds timing/behaviour fields, writes fixed-size Parquet batches for large captures, samples deterministically when capped, and emits extraction progress/manifests. No future packet is used to calculate a packet's causal feature. |
| `catalog.py` | Static feature catalogue: names, applicable protocols, category, description, and estimated CPU cost. Used by selection/report/dashboard layers. |
| `protocols/base.py` | Protocol-extractor interface. Each protocol implementation accepts a packet/raw payload and returns observed fields only. |
| `protocols/common.py` | Safe payload text decoding and entropy helpers shared by protocol parsers. |
| `protocols/dns.py` | Extracts DNS name, labels, query type, response code, TTL, answer count, and DNS-specific lexical/behaviour fields when present. |
| `protocols/http.py` | Parses request/response line and headers, method, status, host/path/user-agent and content-related fields; context is propagated only within the same capture/flow/direction. |
| `protocols/modbus.py` | Parses Modbus TCP header/function code, unit ID, address/quantity/value fields and function category. |
| `protocols/s7comm.py` | Parses observable S7comm header/function/parameter/data-size fields. |
| `protocols/ssh.py` | Parses observable unencrypted SSH handshake/banner/algorithm fields. It does not invent encrypted authentication information. |

### `mapping/`: CSV normalization and evidence-based labelling

| File | What the code does |
|---|---|
| `mapper.py` | Normalizes IT/OT CSV schemas; summarizes PCAP flows; generates endpoint-compatible candidates; estimates CSV/PCAP clock offset; chooses the strongest candidate; records confidence/status; attaches labels at packet time; supports batch Parquet labeling; calculates coverage; and saves/reuses an exhaustive packet-link cache. `unknown` remains unknown rather than becoming a fabricated label. |

### `selection/` and `preprocessing/`: frozen feature contract

| File | What the code does |
|---|---|
| `selection/profiles.py` | Creates content-hashed profile JSONs, resolves/loads profiles, and generates feature quality reports (coverage, cardinality, variance, usability, cost). |
| `preprocessing/pipeline.py` | Fits preprocessing only on training rows. It validates selected fields, removes unusable/constant fields, normalizes missing values, limits high-cardinality categories, imputes, robust-scales numeric fields with `SafeRobustScaler`, one-hot encodes categorical fields, writes a prepared table and manifest, then reuses the same fitted transform at inference. |

### `modelling/`: training, validation, scoring, and packaging

| File | What the code does |
|---|---|
| `detectors.py` | Lightweight PCA/MLP autoencoder implementations used by generic comparison experiments. New two-stage production flow does not depend on PCA. |
| `lstm_autoencoder.py` | Defines the dynamic PyTorch sequence autoencoder. It derives dimensions from selected feature count, trains on normal windows, preserves capture boundaries when sequence groups are supplied, aggregates reconstruction MSE back to packets, and saves/loads a state dictionary. |
| `training.py` | Generic unsupervised experiment engine. It prepares inputs, fits selected detector candidates, makes score/metric/importance tables, compares all-feature versus selected profiles, and performs bounded LSTM sweeps. |
| `random_forest.py` | Stage 2 module. It checks label readiness, builds the Stage-2 matrix (optionally including Stage-1 scores), chooses a held-out capture split when possible, trains a balanced Random Forest, writes per-class/probability/importance/sweep diagnostics, and predicts attack type. |
| `two_stage.py` | Deployable two-stage workflow. It loads separate normal and mapped attack feature sources; creates capture-aware normal/attack roles; fits normal-only LSTM-AE and Isolation Forest; calibrates dynamic thresholds; optionally fits a capture-disjoint known-attack gate; records per-packet evidence; invokes Stage 2 only when labels are adequate; writes contract/native/ONNX artifacts; and implements portable `score_two_stage` inference. |

### `orchestration/`: composition of independent stages

| File | What the code does |
|---|---|
| `dataset_pipeline.py` | Current protocol-first dataset workflow. `run_dataset_extraction` writes PCAP features first; `run_dataset_mapping` later maps only those saved records; `run_dataset_pipeline` is the convenience composition. It updates the catalog, mapping cache, audits, label distribution, and durable progress. |
| `batch.py` | Earlier/general inventory workflow. It composes configured discovery, extraction, quality analysis, optional mapping candidates, and optional generic feature evaluation. |

### `storage/`: portable local storage and optional ClickHouse mirror

| File | What the code does |
|---|---|
| `artifacts.py` | `ArtifactStore` registers artifacts in a local catalog and writes local Parquet/JSON as the authoritative output. `NullMirror` keeps default operation database-free; `ClickHouseMirror` lazily creates compatible tables and mirrors frames only when configured. |

### `dashboard/`: local Streamlit presentation layer

| File | What the code does |
|---|---|
| `app.py` | Streamlit application entry point and legacy/original explorer. It loads local artifacts with cache-aware readers, lets the operator choose a run/protocol, renders EDA/mapping/model/resource pages, creates profiles, and launches bounded jobs. |
| `workbench.py` | Main enhanced Persian RTL workbench. It implements theme/RTL styling, system-wide and per-protocol dashboards, model pipeline diagram, Stage 1/2/end-to-end analytics, feature evidence, mapping/EDA tables, file browser/downloads, resource monitor, and live job monitor. It reads artifacts; it does not redefine model results. |
| `experiment_worker.py` | Subprocess worker launched by the dashboard for long-running training. It invokes training safely and writes structured progress events for the UI. |

## Automated tests: `tests/`

Tests use small synthetic records and verify contracts rather than requiring a
production PCAP archive.

| File | Coverage |
|---|---|
| `test_protocol_extractors.py` | Packet/protocol parser fields. |
| `test_mapping.py`, `test_dataset_mapping_cache.py` | Endpoint/time mapping rules, packet labels, cache reuse/mismatch handling. |
| `test_dataset_inventory_allowlist.py`, `test_protocol_folders_and_quality.py` | Folder discovery, allowlists, inventory, and feature quality. |
| `test_selection_and_preprocessing.py` | Immutable profiles and preprocessing contract. |
| `test_model_experiments.py` | Generic experiments and LSTM capture-boundary behaviour. |
| `test_two_stage.py` | Portable two-stage training/scoring contract and capture-disjoint gate logic. |
| `test_random_forest.py` | Independent Stage 2 RF outputs/readiness. |
| `test_dashboard_protocol_sources.py`, `test_experiment_worker.py` | Dashboard source selection and worker behavior. |

Run all checks with:

```bash
uv run ruff check .
uv run pytest
```

## Documentation: `guides/` and `docs/`

`guides/` contains Persian operator/developer documentation: module usage,
adding a protocol, dynamic feature selection, feature processing, dashboards,
metrics, output analytics, external-system integration, and this project map.
`docs/` contains the integration contract and full dashboard guide. Generated
Word reports under `guides/` are delivery documents; their Python builder is
`guides/build_technical_report.py`.

## Safe change map

- Add a protocol: parser in `features/protocols/`, catalogue entries in
  `features/catalog.py`, supported-protocol configuration, tests, then update
  the protocol guide.
- Add/change a raw feature: extractor/parser + catalogue + quality tests. Do
  not insert labels or metadata as model columns.
- Change a model: `modelling/`; preserve `model-contract.json` compatibility
  and add a focused test.
- Change output locations/schema: `core/io.py`, `storage/artifacts.py`,
  orchestration/model writers, dashboard readers, and external integration docs
  together.
- Change visual presentation only: `dashboard/workbench.py` (and, if needed,
  `dashboard/app.py`), never the artifact values themselves.
