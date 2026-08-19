# PCAP/CSV mapping

The mapper maps labelled CSV rows to PCAP-derived bidirectional flows. It uses the smallest reliable common contract across data sources: source/destination IP, ports when present, and timestamps when present.

```powershell
uv run anomaly map capture.pcap labels.csv --domain it --output artifacts/mapping/it.csv
```

The command writes three outputs:

- The requested mapping table, with `match_status`, `match_confidence`, candidate count, and source CSV row.
- A packet-feature evidence file named `<output>.pcap-features.parquet`.
- A JSON summary with label and matching distributions.

The mapper never invents a label. Unmatched traffic is labelled `unknown`. Treat `matched_from_candidates` as a review signal; it means more than one CSV row was compatible before temporal ranking.
