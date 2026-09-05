# Dashboard

Start the local workbench after generating at least one configured pipeline run:

```powershell
uv run anomaly dashboard
```

The dashboard reads only local artefacts. Select an artefact root, a **pipeline run**, and then an **analysis protocol** in the sidebar. It loads that protocol's own feature table (for example `features/modbus-observation.parquet`) before falling back to `features/all-datasets.parquet` for legacy runs. Large Parquet files are read as a deterministic, time-spread dashboard sample; the source row count remains visible and no raw artefact is changed.

The **PCAP inclusion audit** expander lists every physical capture recorded in the extraction manifest. It is the first place to verify that newly copied Modbus PCAP files were actually included, together with packet counts and packets filtered because they belonged to another supported protocol.

It has two deliberately separate workspaces:

- **Experiment studio**: runs the configured inventory, creates any number of immutable client feature profiles, compares the all-features baseline against selected profiles, optionally uses a reviewed mapping output to calculate ROC-AUC and average precision, selects detectors, and can run a bounded LSTM parameter sweep.
- **Results explorer**: separates every chart, table, and score inspection by protocol name. Its on-demand protocol areas cover traffic and timing, endpoints and flows, feature health, feature value and catalogue guidance, and model diagnostics. Model diagnostics lets the operator select one persisted detector run at a time—including LSTM AE—and review score distributions, anomaly decisions, MSE, training curves, and detector-specific feature contribution.

All compute-heavy actions are initiated only by a submitted form. Automatic feature evaluation is disabled by default so a pipeline run can finish extraction and evidence generation without model training. The dashboard is an operator convenience layer; the same operations remain available through the CLI and package APIs described in the [integration guide](../../../docs/INTEGRATION_GUIDE.md).

### Reading feature value and model evidence

The **Feature value & guide** area ranks features for the selected protocol from availability, robust spread (IQR), low redundancy, and rare-event response. It is an operational-selection score, not a claim of causal model importance; the table states the reason for every rank. Use **Model diagnostics** to see persisted detector-specific importance next to that protocol value.

For the LSTM autoencoder, each held-out `anomaly_score` is reconstruction MSE. New model runs persist `reconstruction_mse_mean`, `reconstruction_rmse`, score percentiles, and score rank; older artefacts remain viewable and the dashboard derives mean MSE from their scores. PCA is excluded from all new training choices because it is not suitable for the sequence-oriented Modbus workflow; legacy PCA output remains readable. To obtain one LSTM result per protocol, select that protocol in the sidebar, then run **Experiment studio → Train and compare** with `per_protocol` and `lstm_autoencoder`.

For a complete Persian guide to every control, chart, card, and table column, see [DASHBOARD_GUIDE_FA.md](../../../docs/DASHBOARD_GUIDE_FA.md).
