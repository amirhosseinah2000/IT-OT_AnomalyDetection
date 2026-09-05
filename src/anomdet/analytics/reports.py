"""Generate chart-ready, portable analytical tables rather than UI-only calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_overview(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Return one explicit row per numeric feature for charts, APIs, and CSV export."""
    excluded = {"row_id", "src_port", "dst_port"}
    columns = [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
    ]
    rows: list[dict[str, float | str | int]] = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        observed = series.dropna()
        q25, q75 = observed.quantile([0.25, 0.75]) if not observed.empty else (np.nan, np.nan)
        rows.append(
            {
                "protocol": protocol,
                "feature": column,
                "records": int(len(series)),
                "availability": round(float(series.notna().mean()), 6),
                "unique_values": int(series.nunique(dropna=True)),
                "mean": float(observed.mean()) if not observed.empty else np.nan,
                "median": float(observed.median()) if not observed.empty else np.nan,
                "std": float(observed.std()) if not observed.empty else np.nan,
                "iqr": float(q75 - q25) if not observed.empty else np.nan,
                "zero_ratio": float((observed == 0).mean()) if not observed.empty else np.nan,
                "p01": float(observed.quantile(0.01)) if not observed.empty else np.nan,
                "p99": float(observed.quantile(0.99)) if not observed.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["availability", "iqr"], ascending=[False, False])


def preprocessing_comparison(
    raw: pd.DataFrame, prepared: pd.DataFrame, selected_features: list[str], protocol: str
) -> pd.DataFrame:
    """Expose before/after distribution changes in a shareable comparison table."""
    rows: list[dict[str, float | str | int]] = []
    for feature in selected_features:
        if feature not in raw.columns:
            continue
        before = pd.to_numeric(raw[feature], errors="coerce")
        transformed = [
            column
            for column in prepared.columns
            if column == feature or column.startswith(f"{feature}_")
        ]
        after = (
            pd.to_numeric(prepared[transformed].stack(), errors="coerce")
            if transformed
            else pd.Series(dtype=float)
        )
        for stage, values in [("raw", before), ("prepared", after)]:
            valid = values.dropna()
            rows.append(
                {
                    "protocol": protocol,
                    "feature": feature,
                    "stage": stage,
                    "records": int(len(values)),
                    "mean": float(valid.mean()) if not valid.empty else np.nan,
                    "std": float(valid.std()) if not valid.empty else np.nan,
                    "median": float(valid.median()) if not valid.empty else np.nan,
                    "iqr": float(valid.quantile(0.75) - valid.quantile(0.25))
                    if not valid.empty
                    else np.nan,
                    "min": float(valid.min()) if not valid.empty else np.nan,
                    "max": float(valid.max()) if not valid.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)
