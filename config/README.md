# Configuration

`default.yaml` is the canonical configuration contract for all commands. Do not edit it for an environment-specific deployment. Create an override instead:

```yaml
# config/lab-64gb.yaml
runtime:
  execution_device: cpu
  cpu_workers: 12
  memory_limit_gb: 48

data:
  raw_pcap_dir: D:/network-data/pcap
  it_label_dir: D:/network-data/labels/it
  ot_label_dir: D:/network-data/labels/ot

capture:
  behavior_window_seconds: 60
```

Run a command with `uv run anomaly --config config/lab-64gb.yaml extract capture.pcap`.

For a deliberately small reproducible run, `data.capture_allowlist` can name
the exact PCAP/PCAPNG filenames to use for each protocol and split.  It is
intended for smoke tests and targeted re-processing; omit it for a complete
folder-first run.  `config/smoke-4protocol-200.yaml` is the repository's
ready-made four-protocol example.

`runtime.memory_limit_gb` is a soft safety threshold for scheduling and operator review. The platform records resource snapshots at command startup; it does not attempt to reserve host memory or alter system-wide CPU affinity.

`capture.dataset_default_packet_cap` is the safe default for `dataset extract` and `dataset run`.
For a trial it retains a deterministic, time-spread sample per capture.  Large
values passed through `--max-packets N` automatically use the disk-backed
extractor: it writes fixed-size Parquet row groups and never accumulates raw
packets or the whole feature table in RAM.

For a full archive, use the explicit production override and `--all-packets`:

```bash
RUN="artifacts/runs/final-20260907"
uv run anomaly --config config/production-streaming.yaml dataset extract \
  --output "$RUN" \
  --all-packets
uv run anomaly --config config/production-streaming.yaml dataset map "$RUN"
```

`dataset extract` reads every packet once and writes one feature Parquet per
protocol/capture. `dataset map` subsequently maps CSV labels in batches from
those saved files, without reopening the PCAP. It needs free disk space for the
output, but has a bounded working-memory footprint. Keep `streaming_unbounded:
false` in configuration; the explicit CLI flag is a guard against starting an
unlimited run by accident.

`mapping.timestamp_dayfirst` controls the parsing order for ambiguous label timestamps. It defaults to `true`, which suits common CIC-style `day/month/year` CSV exports. Change it only if a dataset documents month-first timestamps.

## Reusable packet-to-CSV mapping

`mapping.cache_enabled` is on by default. The first `dataset map` creates a compact,
versioned `packet-csv-links.parquet` relation under
`<artifact_dir>/mapping_cache/`. It records `packet_uid`, original packet
index, timestamp, flow, CSV row, label, confidence, and acceptance decision.
Later mapping runs reuse that relation when the PCAP/CSV file snapshots, mapping
policy, protocol, and packet cap are unchanged. New or changed inputs automatically
create a new entry. Use `anomaly dataset map <dataset-run> --remap` only to force a
rebuild of an unchanged entry.

The mapping still visits every extracted PCAP packet. It uses endpoint/time
indexes to test every compatible CSV candidate rather than performing an
unbounded Cartesian comparison with unrelated rows. `mapping-progress.json`
and console logs expose `mapping_cache_hit`, `mapping_cache_miss`, and
`packet_mapping_progress` events.

## Stage 2 readiness and held-out evaluation

`stage_two.min_training_records` (default `20`),
`stage_two.min_records_per_class` (default `4`), and
`stage_two.require_multiple_attack_types` decide whether the Random Forest can
be trained honestly. If they are not met, `two-stage train` still trains and
packages LSTM-AE + Isolation Forest, writes `stage2/readiness.json`, and marks
Stage 2 as `skipped`; it never fabricates an RF metric.

When possible, Stage 2 holds out complete captures for test using a stratified
group split. Small datasets that cannot retain every class in the training
captures use a recorded `stratified_packet_fallback`. Inspect
`stage2/held-out-split.parquet` and `stage2/metrics.json` before comparing
models.

The `data` section defines the standard dataset roots for the deployment. `anomaly extract` accepts either one capture or a protocol folder. `pipeline run` discovers direct PCAP/PCAPNG files in protocol-named folders below `raw_pcap_dir`, keeping each source capture in the output manifest.

## Dataset inventory for one-command runs

Create one direct folder per supported protocol, such as `D:/network-data/pcap/modbus/`; each can hold any number of PCAP/PCAPNG files. `pipeline run` discovers these folders automatically. Use `data.protocol_folders` only to attach label files, override a directory name, or choose a stable dataset id. Each label filename is resolved under the matching label directory unless it is absolute.

```yaml
data:
  raw_pcap_dir: D:/network-data/pcap
  it_label_dir: D:/network-data/labels/it
  ot_label_dir: D:/network-data/labels/ot
  protocol_folders:
    - id: ssh-test
      protocol: ssh
      domain: it
      labels: [it-flows-a.csv, it-flows-b.csv]
    - id: dns-test
      protocol: dns
      domain: it
      labels: [it-flows-a.csv]
    - id: modbus-test
      protocol: modbus
      domain: ot
      labels: [schneider-flows.csv]
```

Run `uv run anomaly --config config/lab-64gb.yaml pipeline run`. The batch run extracts every PCAP in each selected protocol folder, attempts every configured PCAP/CSV candidate pairing, produces protocol-specific feature-validation reports, and writes a single run summary. Mapping candidates are not used to control unsupervised feature evaluation; they remain evidence for later label validation.

`config/test-datasets.yaml` is an inventory created for the captures and CSVs currently in this repository. It includes DNS, HTTP, Modbus, and S7comm. `tls.pcap.pcapng` is excluded because TLS is outside the Phase 1 protocol scope, and no SSH PCAP is currently present.

## Unsupervised feature evaluation

Feature evaluation is independent of mapping and is disabled by default so ingestion does not begin model training. Enable it only for a controlled run. New candidates exclude PCA reconstruction because this project uses a sequence-oriented Modbus workflow; use the LSTM autoencoder for reconstruction experiments. Add as many subset profile names as needed to `feature_evaluation.selected_profiles`:

```yaml
feature_evaluation:
  enabled: true
  detectors: [isolation_forest, lstm_autoencoder]
  strategy: grouped
  group: all
  selected_profiles: [compact-it, operations-only]
```

The following `pipeline run` compares `all_features` against every selected profile using the same detector set. It reports transformed feature count, detector fit time, process-memory delta, anomaly-score distribution, anomaly rate, and model-specific feature contribution. CSV labels are not passed to this evaluation.

### LSTM autoencoder settings

`models.lstm_autoencoder` controls the sequence model. `sequence_length` and `sequence_stride` define the contiguous prepared-record windows; `hidden_size`, `latent_size`, and `num_layers` define the network; and `epochs`, `patience`, `batch_size`, and `max_train_windows` govern resource usage. The default is CPU-first and temporal-safe:

```yaml
models:
  # Deterministic, time-spread fit population from the disk-backed corpus.
  # It is a global cap across a protocol's capture files, not a cap per file.
  max_source_rows: 250000
  split_strategy: temporal
  lstm_autoencoder:
    sequence_length: 8
    sequence_stride: 8
    hidden_size: null # resolved from the selected feature count
    latent_size: null # resolved from the selected feature count
    epochs: 60
    max_train_windows: 5000
    device: cpu
```

Set `models.split_strategy: random` only when rows are independently distributed and a temporal hold-out is not appropriate. `max_source_rows` is a deterministic time-spread model population, designed to keep CPU/RAM bounded while preserving coverage of each capture. Increase it gradually (for example, 250,000 to 500,000) only after observing RAM and training time. Setting it to `null` deliberately loads the entire table and is not the safe production path.
