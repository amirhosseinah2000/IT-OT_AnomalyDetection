"""Safe dataframe IO and metadata persistence used by all pipeline stages."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    """Return a sortable ISO-8601 timestamp in UTC."""
    return datetime.now(UTC).isoformat()


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, Parquet, or JSON Lines input with a predictable error message."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format: {path.suffix}. Use CSV, Parquet, or JSONL.")


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write a dataframe based on the output suffix and return the resolved path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True, date_format="iso")
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}. Use CSV, Parquet, or JSONL.")
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Persist JSON metadata with stable formatting for reviews and diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path
