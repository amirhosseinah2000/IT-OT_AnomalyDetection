# Batch orchestration

Batch orchestration provides the integrated validation path for multiple test captures. It does not replace individual commands; it composes them and preserves the artefacts for every capture and label candidate.

```powershell
uv run anomaly --config config/lab-64gb.yaml pipeline run
```

For every discovered protocol folder (or configured `data.protocol_folders` item), it:

1. extracts every direct PCAP/PCAPNG file while preserving its capture name and flow identity;
2. creates a protocol-specific feature-validation report with parser coverage and variation evidence;
3. maps each listed CSV candidate using the existing extracted feature evidence;
4. combines all extracted feature rows; and
5. writes `run-summary.json` with every output and failure.

Automatic feature evaluation is disabled by default so a PCAP ingestion run remains an extraction, mapping, and evidence-generation operation. Enable it deliberately only for a controlled experiment, or use the dashboard/CLI after reviewing feature quality. New training candidates exclude PCA; use the LSTM autoencoder for sequence-oriented Modbus experiments.

Outputs are placed in `artifacts/runs/batch-<timestamp>/`. Mapping candidates remain separate deliberately. Review their `match_status` and `match_confidence` before deciding which label source is fit for training.
