# Feature extraction

`extract` streams one PCAP/PCAPNG file or every direct capture inside a protocol folder with Scapy and emits one row per supported packet. It then adds capture-isolated bidirectional flow, timing, request-response, and host-behaviour features.

```powershell
uv run anomaly extract data/raw/pcap/modbus --output artifacts/features/modbus.parquet
uv run anomaly select analyze artifacts/features/modbus.parquet --output artifacts/reports/modbus-quality.parquet
```

Use one direct subfolder for each supported protocol, such as `pcap/dns/`, `pcap/http/`, `pcap/modbus/`, and `pcap/s7comm/`. The output manifest records every source capture, parser protocol counts, records filtered because they did not match the folder protocol, and schema version.

`select analyze` verifies each catalogue feature separately per protocol. It records record-level and flow-level coverage, variation, zero ratio, and one of `usable`, `constant`, `near_constant`, `low_coverage`, `not_observed`, or `not_implemented`. A field is only considered model-usable when it has adequate coverage and variation. HTTP header fields are propagated within the same capture, TCP flow, and direction after being parsed, so their packet sparsity does not erase valid flow context.

Protocol parsers only claim values observable in PCAP payloads. For example, SSH authentication events are generally encrypted after key exchange and remain null unless an upstream parser exposes them. This protects the pipeline from manufacturing evidence.
