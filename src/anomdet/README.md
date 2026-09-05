# Application package

`anomdet` is the installable package behind the `anomaly` command. Its subpackages map directly to the Phase 1 pipeline boundaries: shared core services, PCAP/CSV mapping, feature extraction, feature selection, preprocessing, modelling, and the dashboard.

Each boundary writes self-describing local artefacts and includes a focused README. This keeps the pipeline suitable for later embedding inside a larger module without coupling its inputs or outputs to the dashboard.
