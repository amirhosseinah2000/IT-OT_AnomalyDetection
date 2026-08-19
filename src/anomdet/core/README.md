# Core services

`core` owns shared YAML configuration, structured logging, safe table IO, artefact metadata, and host-resource snapshots. Every command loads the same configuration and writes logs beneath `artifacts/logs/platform.log` by default.

Use `uv run anomaly resources` before large work. Set `runtime.cpu_workers` and `runtime.memory_limit_gb` in a configuration override to match the target deployment hardware.
