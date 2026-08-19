# Network Anomaly Detection Platform

A Phase 1, PCAP-first platform for unsupervised anomaly detection in IT and OT traffic. It supports SSH, DNS, HTTP, Modbus TCP, and S7comm, with two deployment strategies:

- `per_protocol`: one detector per protocol.
- `grouped`: one detector for the selected IT protocols or OT protocols.

The current phase provides mapping, feature extraction, selection, preprocessing, model comparison, resource monitoring, and a Streamlit EDA dashboard. Supervised anomaly-type classification is intentionally deferred to Phase 2, after mapped labels are validated.

## Requirements

- Windows, Linux, or macOS
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) 0.4 or later
- For PCAP parsing: no external capture tool is required.
- For packet replay: install `tcpreplay` separately and configure its binary and interface in `config/default.yaml`.

## Quick start

```powershell
uv sync
uv run anomaly --help
uv run anomaly resources
uv run anomaly extract capture.pcap --output artifacts/features/capture.parquet
uv run anomaly select create baseline --features packet_length,flow_duration,packet_rate
uv run anomaly train artifacts/features/capture.parquet --strategy per_protocol --models isolation_forest,lstm_autoencoder
uv run anomaly dashboard
```

All commands accept `--config path/to/config.yaml`. Generated artefacts are stored beneath `artifacts/` unless an explicit output path is supplied.

## Execute the workflow end to end

1. Set the standard dataset roots in `config/default.yaml` or an environment-specific override. Put any number of PCAP/PCAPNG files directly inside one folder per supported protocol, for example `data/raw/pcap/modbus/` or `data/raw/pcap/http/`. IT labels remain under `data/raw/labels/it` and OT labels under `data/raw/labels/ot`. Keep raw data outside version control.
2. Map a CSV to a PCAP when labels are available:

   ```powershell
   uv run anomaly map capture.pcap labels.csv --domain it --output artifacts/mapping/mapped.csv
   ```

3. Extract protocol-aware packet and flow features from one file or a whole protocol folder:

   ```powershell
   uv run anomaly extract data/raw/pcap/modbus --output artifacts/features/modbus.parquet
   uv run anomaly select analyze artifacts/features/modbus.parquet --output artifacts/reports/modbus-quality.parquet
   ```

4. Create one or more feature profiles, prepare the selected feature matrix, and compare detectors:

   ```powershell
   uv run anomaly select create compact-it --features packet_length,flow_duration,packet_rate,payload_entropy
   uv run anomaly prepare artifacts/features/capture.parquet --profile compact-it
   uv run anomaly train artifacts/features/capture.parquet --profile compact-it --strategy grouped --group it
   ```

5. Start the dashboard and select the artefact directory in the sidebar:

   ```powershell
   uv run streamlit run src/anomdet/dashboard/app.py
   ```

### One-command ingestion and mapping

For a multi-PCAP test, create the protocol folders. `pipeline run` discovers them automatically; use `data.protocol_folders` only when labels or non-default folder names must be specified.

```powershell
uv run anomaly --config config/lab-64gb.yaml pipeline run
uv run anomaly dashboard
```

An inventory for the test files already placed in this repository is available at `config/test-datasets.yaml`:

```powershell
uv run anomaly --config config/test-datasets.yaml pipeline run
```

This batch path reads every direct PCAP in each protocol folder, preserves the source capture name, and evaluates every listed CSV candidate. It writes a protocol-specific feature-quality report that labels every catalogue field as `usable`, `constant`, `near_constant`, `low_coverage`, `not_observed`, or `not_implemented`. Only variable fields are retained for model preparation. CSV mapping remains separate evidence-validation for later supervised work.

## Individual components

| Component | Purpose | Documentation |
|---|---|---|
| Mapping | Associates PCAP flows with CSV labels via 5-tuple and time | [mapping README](src/anomdet/mapping/README.md) |
| Extraction | Produces protocol-aware packet, flow, timing, and behaviour features | [features README](src/anomdet/features/README.md) |
| Selection | Creates reproducible feature-profile manifests | [selection README](src/anomdet/selection/README.md) |
| Preprocessing | Validates and transforms selected model features | [preprocessing README](src/anomdet/preprocessing/README.md) |
| Modelling | Compares CPU-first anomaly detectors, including a sequence LSTM autoencoder | [modelling README](src/anomdet/modelling/README.md) |
| Dashboard | Provides controlled execution plus protocol-centred results analysis | [dashboard README](src/anomdet/dashboard/README.md) |
| Integration | Defines stable module inputs, outputs, artefacts, and embedding patterns | [integration guide](docs/INTEGRATION_GUIDE.md) |

## Run individual stages

Use these commands when diagnosing a single stage rather than executing the full inventory.

| Stage | Command |
|---|---|
| Inspect capacity | `uv run anomaly resources` |
| Extract one PCAP | `uv run anomaly extract data/raw/pcap/modbus.pcap --output artifacts/features/modbus.parquet` |
| Map one PCAP/CSV candidate | `uv run anomaly map data/raw/pcap/modbus.pcap data/raw/labels/ot/Schneider_fraggle_faster_0710_01_.csv --domain ot --output artifacts/mapping/modbus-candidate.parquet` |
| Analyze extracted-feature quality | `uv run anomaly select analyze artifacts/features/modbus.parquet` |
| Create a selected-feature profile | `uv run anomaly select create compact-it --features packet_length,payload_entropy,flow_duration,packet_rate` |
| Prepare a selected matrix | `uv run anomaly prepare artifacts/features/modbus.parquet --profile compact-it` |
| Train an individual unsupervised comparison | `uv run anomaly train artifacts/features/modbus.parquet --profile compact-it --strategy per_protocol --models isolation_forest,lstm_autoencoder` |
| Run the configured inventory | `uv run anomaly --config config/test-datasets.yaml pipeline run --output artifacts/runs/full-test-01` |
| Start the dashboard | `uv run anomaly dashboard` |

## View results and dashboard

Each `pipeline run` creates a self-contained directory below `artifacts/runs/`. For the example full run, use `artifacts/runs/full-test-01/`:

| Result | Location | What to inspect |
|---|---|---|
| Overall status | `run-summary.json` | Extracted datasets, mapping-candidate status, and feature-evaluation status. |
| Combined feature data | `features/all-datasets.parquet` | All supported PCAP records in one feature table. |
| Per-protocol features | `features/<dataset-id>.parquet` | DNS, HTTP, Modbus, or S7comm data separately. |
| Mapping evidence | `mapping/*.parquet` and `mapping/*.summary.json` | `match_status`, `match_confidence`, and candidate label distribution. |
| Feature EDA reports | `reports/*feature-quality.parquet` | Missingness, cardinality, variance, cost, and catalogue descriptions. |
| Feature/profile comparison | `feature-evaluation/comparison.parquet` | All-features baseline versus each selected profile, with feature count, fit time, memory, scores, and anomaly rate. |
| Individual model outputs | `feature-evaluation/all-features/grouped_all/<model>/` | Persisted model, anomaly scores, model metrics, and feature importance. LSTM AE also writes an epoch training history and `model.pt`. |

Start the dashboard:

```powershell
uv run anomaly dashboard
```

Open `http://127.0.0.1:8501` in the browser. In the sidebar, set **Artifact directory** to:

```text
artifacts/runs/full-test-01
```

Choose the pipeline run and then the protocol in the sidebar. The dashboard reads the matching protocol feature table first rather than loading the combined multi-GB table, displays an audit of every included PCAP, and separates all views by **protocol**. **Experiment studio** runs configured ingestion, creates multiple immutable feature profiles, launches profile comparisons, and can execute LSTM parameter sweeps. **Results explorer** provides protocol feature guides, EDA, model/profile comparisons, feature contribution, LSTM training curves, mapping audit, and runtime evidence. See the [Persian dashboard guide](docs/DASHBOARD_GUIDE_FA.md) for every control and table column.

## Operational notes

- `runtime.memory_limit_gb` is a soft validation limit. Set `runtime.cpu_workers` to a positive value to cap parallel CPU work.
- For a 16-core / 64 GB machine, the supplied defaults use at most 48 GB and automatically reserve one logical CPU.
- PCAP-to-CSV labels are evidence-based matches, not inferred labels. Inspect `match_status` and `match_confidence` before using labels for evaluation.
- IP geolocation and ASN enrichment are deliberately optional; no external data is sent by default.

## Project layout

```text
config/                 Shared configuration
src/anomdet/            Installable application package
  mapping/              PCAP/CSV flow label mapping
  features/             Protocol parsing and feature catalogue
  selection/            Versioned feature profiles
  preprocessing/        Feature matrix preparation
  modelling/            Unsupervised model comparison
  dashboard/            Streamlit EDA application
tests/                  Focused automated tests
artifacts/              Runtime outputs, ignored by Git
```

## Development checks

```powershell
uv run ruff check .
uv run pytest
```
