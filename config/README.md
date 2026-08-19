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

`runtime.memory_limit_gb` is a soft safety threshold for scheduling and operator review. The platform records resource snapshots at command startup; it does not attempt to reserve host memory or alter system-wide CPU affinity.

`mapping.timestamp_dayfirst` controls the parsing order for ambiguous label timestamps. It defaults to `true`, which suits common CIC-style `day/month/year` CSV exports. Change it only if a dataset documents month-first timestamps.

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
  split_strategy: temporal
  lstm_autoencoder:
    sequence_length: 10
    sequence_stride: 10
    hidden_size: 128
    latent_size: 64
    epochs: 100
    max_train_windows: 5000
    device: cpu
```

Set `models.split_strategy: random` only when rows are independently distributed and a temporal hold-out is not appropriate. For trials, reduce `max_train_windows`, epochs, or the pipeline packet cap first; do not reduce the all-record scoring stage if anomaly coverage is required.
