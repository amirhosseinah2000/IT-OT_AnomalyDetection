# Dashboard

Start the local workbench after generating at least one configured pipeline run:

```powershell
uv run anomaly dashboard
```

The dashboard uses a dark navy, blue, and neutral theme and reads only local artefacts. Select an artefact root and then a **pipeline run** in the sidebar. It automatically reads `features/all-datasets.parquet`; users never need to choose individual feature files.

It has two deliberately separate workspaces:

- **Experiment studio**: runs the configured inventory, creates any number of immutable client feature profiles, compares the all-features baseline against selected profiles, optionally uses a reviewed mapping output to calculate ROC-AUC and average precision, selects detectors, and can run a bounded LSTM parameter sweep.
- **Results explorer**: separates every chart, table, and score inspection by protocol name. Its on-demand protocol areas cover traffic and timing, endpoints and flows, feature health, feature value and catalogue guidance, and model diagnostics. Model diagnostics lets the operator select one persisted detector run at a time—including LSTM AE—and review score distributions, anomaly decisions, MSE, training curves, and detector-specific feature contribution.

All compute-heavy actions are initiated only by a submitted form. The dashboard is an operator convenience layer; the same operations remain available through the CLI and package APIs described in the [integration guide](../../../docs/INTEGRATION_GUIDE.md).

### Reading feature value and model evidence

The **Feature value & guide** area ranks features for the selected protocol from availability, robust spread (IQR), low redundancy, and rare-event response. It is an operational-selection score, not a claim of causal model importance; the table states the reason for every rank. Use **Model diagnostics** to see persisted detector-specific importance next to that protocol value.

For an autoencoder, each held-out `anomaly_score` is reconstruction MSE. New model runs persist `reconstruction_mse_mean`, `reconstruction_rmse`, score percentiles, and score rank; older artefacts remain viewable and the dashboard derives mean MSE from their scores. A detector is shown only for protocols with persisted held-out scores. To obtain one LSTM result per protocol, run **Experiment studio → Train and compare** with `per_protocol` strategy and select `lstm_autoencoder`.
