# Tests

The tests cover deterministic contracts that do not require a real production capture: feature-profile versioning, mapper evidence rules, and preprocessing output shape. Run them with:

```powershell
uv run pytest
```

Use a representative but non-sensitive PCAP fixture in the target environment before approving a new protocol parser or deploying changed extraction logic.
