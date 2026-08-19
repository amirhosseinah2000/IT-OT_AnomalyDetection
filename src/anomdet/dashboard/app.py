"""Protocol-centred experiment studio and results explorer for local artefacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from anomdet.core.config import load_config
from anomdet.core.io import read_table
from anomdet.core.resources import snapshot
from anomdet.features.catalog import available_features
from anomdet.modelling.training import run_feature_experiments, run_lstm_sweep
from anomdet.orchestration.batch import run_inventory
from anomdet.selection.profiles import create_profile

st.set_page_config(
    page_title="Network anomaly workbench",
    page_icon=":material/network_intelligence:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALL_PROTOCOLS = ["ssh", "dns", "http", "modbus", "s7comm"]
MODEL_OPTIONS = [
    "isolation_forest",
    "pca_autoencoder",
    "lstm_autoencoder",
    "local_outlier_factor",
    "one_class_svm",
]
METADATA_COLUMNS = {"row_id", "label", "protocol", "flow_id", "capture", "timestamp"}
CHART_SAMPLE_SIZE = 12_000


@st.cache_data(max_entries=24, show_spinner="Loading local artefact...")
def _load_table(path_text: str, modified_ns: int) -> pd.DataFrame:
    """Read an artefact once per file version rather than on every interaction."""
    del modified_ns
    return read_table(Path(path_text))


@st.cache_data(max_entries=48, show_spinner=False)
def _load_json(path_text: str, modified_ns: int) -> dict[str, Any]:
    """Read small JSON manifests with the same cache invalidation policy."""
    del modified_ns
    with Path(path_text).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read(path: Path) -> pd.DataFrame:
    """Load a table while including its modification time in the cache key."""
    return _load_table(str(path), path.stat().st_mtime_ns)


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON manifest while including its modification time in the cache key."""
    return _load_json(str(path), path.stat().st_mtime_ns)


def _find_files(root: Path, pattern: str) -> list[Path]:
    """Find local artefacts newest first without making feature files the dashboard unit."""
    if not root.exists():
        return []
    return sorted(root.rglob(pattern), key=lambda item: item.stat().st_mtime_ns, reverse=True)


def _discover_runs(artifact_root: Path) -> list[Path]:
    """Discover self-contained pipeline runs from their combined feature output."""
    if not artifact_root.exists():
        return []
    runs: set[Path] = set()
    direct = artifact_root / "features" / "all-datasets.parquet"
    if direct.exists():
        runs.add(artifact_root)
    for combined in artifact_root.rglob("all-datasets.parquet"):
        if combined.parent.name == "features":
            runs.add(combined.parent.parent)
    return sorted(
        runs,
        key=lambda item: (
            (item / "run-summary.json").stat().st_mtime_ns
            if (item / "run-summary.json").exists()
            else item.stat().st_mtime_ns
        ),
        reverse=True,
    )


def _run_label(run: Path, artifact_root: Path) -> str:
    """Show a compact stable label in the run picker."""
    try:
        return str(run.relative_to(artifact_root)) or run.name
    except ValueError:
        return str(run)


def _feature_path(run: Path | None) -> Path | None:
    """Return the single combined feature source for a selected pipeline run."""
    if run is None:
        return None
    candidate = run / "features" / "all-datasets.parquet"
    return candidate if candidate.exists() else None


def _safe_numeric(frame: pd.DataFrame) -> list[str]:
    """Return analysable numerical columns without technical record identifiers."""
    excluded = METADATA_COLUMNS | {"src_port", "dst_port"}
    return [
        column for column in frame.select_dtypes(include="number").columns if column not in excluded
    ]


def _sample(frame: pd.DataFrame, limit: int = CHART_SAMPLE_SIZE) -> pd.DataFrame:
    """Keep browser-side charts responsive for million-record PCAP runs."""
    return frame if len(frame) <= limit else frame.sample(limit, random_state=42)


def _chart_sample(frame: pd.DataFrame, limit: int = 4_000) -> pd.DataFrame:
    """Downsample in source order when a chart needs temporal or rank continuity."""
    if len(frame) <= limit:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=limit, dtype=int)
    return frame.iloc[positions]


def _profile_records(artifact_root: Path) -> list[dict[str, Any]]:
    """List immutable feature profiles and retain the path needed for experiments."""
    directory = artifact_root / "feature_profiles"
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), reverse=True) if directory.exists() else []:
        try:
            payload = _read_json(path)
            records.append(
                {
                    "label": (
                        f"{payload.get('name', path.stem)} / {payload.get('version', path.stem)} "
                        f"({payload.get('feature_count', 0)} features)"
                    ),
                    "path": str(path),
                    **payload,
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _quality_report(run: Path) -> pd.DataFrame | None:
    """Load the combined quality report when the pipeline produced one."""
    candidates = _find_files(run / "reports", "*all-datasets-feature-quality.parquet")
    return _read(candidates[0]) if candidates else None


def _protocols(frame: pd.DataFrame) -> list[str]:
    """Return present protocols in a stable, expected order."""
    present = set(frame.get("protocol", pd.Series(dtype="string")).dropna().astype(str).str.lower())
    return [protocol for protocol in ALL_PROTOCOLS if protocol in present]


def _metric_row(frame: pd.DataFrame) -> None:
    """Render a compact operational snapshot for the selected run or protocol."""
    timestamps = (
        pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if "timestamp" in frame.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    flows = int(frame["flow_id"].nunique()) if "flow_id" in frame.columns else 0
    with st.container(horizontal=True):
        st.metric("Records", f"{len(frame):,}", border=True)
        st.metric("Flows", f"{flows:,}", border=True)
        st.metric(
            "Protocols",
            f"{frame['protocol'].nunique() if 'protocol' in frame.columns else 0:,}",
            border=True,
        )
        st.metric("Feature columns", f"{len(_safe_numeric(frame)):,}", border=True)
        st.metric(
            "Observed period",
            "Unavailable"
            if timestamps.isna().all()
            else f"{(timestamps.max() - timestamps.min()).total_seconds() / 60:.1f} min",
            border=True,
        )


def _protocol_volume_chart(frame: pd.DataFrame) -> None:
    """Plot record volume by protocol with readable totals."""
    if "protocol" not in frame.columns:
        st.info("This artefact has no protocol column.", icon=":material/info:")
        return
    counts = frame["protocol"].value_counts().rename_axis("protocol").reset_index(name="records")
    chart = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("protocol:N", title="Protocol", sort="-y"),
            y=alt.Y("records:Q", title="Records"),
            color=alt.Color("protocol:N", legend=None),
            tooltip=[
                alt.Tooltip("protocol:N", title="Protocol"),
                alt.Tooltip("records:Q", title="Records", format=","),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart)


def _timeline_chart(frame: pd.DataFrame, color_by_protocol: bool = True) -> None:
    """Plot packet/feature record volume over time at a practical minute resolution."""
    if "timestamp" not in frame.columns:
        st.info("Timestamp is unavailable for this artefact.", icon=":material/schedule:")
        return
    timed = frame[
        [column for column in ["timestamp", "protocol"] if column in frame.columns]
    ].copy()
    timed["timestamp"] = pd.to_datetime(timed["timestamp"], errors="coerce", utc=True)
    timed = timed.dropna(subset=["timestamp"])
    if timed.empty:
        st.info("No valid timestamps are available.", icon=":material/schedule:")
        return
    timed["minute"] = timed["timestamp"].dt.floor("min")
    group_columns = ["minute"] + (
        ["protocol"] if color_by_protocol and "protocol" in timed.columns else []
    )
    timeline = timed.groupby(group_columns).size().rename("records").reset_index()
    encoding: dict[str, Any] = {
        "x": alt.X("minute:T", title="Time"),
        "y": alt.Y("records:Q", title="Records"),
        "tooltip": [
            alt.Tooltip("minute:T", title="Minute"),
            alt.Tooltip("records:Q", title="Records", format=","),
        ],
    }
    if "protocol" in timeline.columns:
        encoding["color"] = alt.Color("protocol:N", title="Protocol")
        encoding["tooltip"].insert(1, alt.Tooltip("protocol:N", title="Protocol"))
    chart = alt.Chart(timeline).mark_line().encode(**encoding).properties(height=320)
    st.altair_chart(chart)


def _endpoint_chart(frame: pd.DataFrame) -> None:
    """Show the most active source endpoints for protocol investigation."""
    if "src_ip" not in frame.columns:
        st.info("Source endpoint data is unavailable.", icon=":material/device_hub:")
        return
    endpoints = (
        frame["src_ip"]
        .astype("string")
        .value_counts()
        .head(15)
        .rename_axis("source")
        .reset_index(name="records")
    )
    chart = (
        alt.Chart(endpoints)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("source:N", sort="-x", title="Source"),
            x=alt.X("records:Q", title="Records"),
            color=alt.Color("records:Q", legend=None),
            tooltip=[
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("records:Q", title="Records", format=","),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart)


def _distribution_chart(frame: pd.DataFrame, feature: str) -> None:
    """Plot a sampled numeric distribution without sending the full capture to the browser."""
    sample = _sample(
        frame[[feature, "protocol"]] if "protocol" in frame.columns else frame[[feature]]
    ).dropna(subset=[feature])
    if sample.empty:
        st.info(
            "The selected feature has no numeric values in this protocol.", icon=":material/info:"
        )
        return
    encoding: dict[str, Any] = {
        "x": alt.X(
            f"{feature}:Q", bin=alt.Bin(maxbins=50), title=feature.replace("_", " ").title()
        ),
        "y": alt.Y("count():Q", title="Records"),
        "tooltip": [alt.Tooltip("count():Q", title="Records", format=",")],
    }
    if "protocol" in sample.columns:
        encoding["color"] = alt.Color("protocol:N", title="Protocol")
    chart = alt.Chart(sample).mark_bar(opacity=0.65).encode(**encoding).properties(height=320)
    st.altair_chart(chart)


def _missingness_chart(frame: pd.DataFrame) -> None:
    """Visualise the most incomplete numerical features in the current protocol scope."""
    numerical = _safe_numeric(frame)
    if not numerical:
        st.info("No numeric features are present.", icon=":material/info:")
        return
    missing = (
        frame[numerical]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .head(24)
        .rename("missing_ratio")
        .reset_index()
    )
    missing.columns = ["feature", "missing_ratio"]
    chart = (
        alt.Chart(missing)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            x=alt.X("missing_ratio:Q", title="Missing ratio", axis=alt.Axis(format="%")),
            color=alt.Color("missing_ratio:Q", legend=None),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("missing_ratio:Q", title="Missing", format=".1%"),
            ],
        )
        .properties(height=520)
    )
    st.altair_chart(chart)


def _feature_guide(protocol: str, available_columns: set[str]) -> None:
    """Explain every catalogue feature relevant to the selected protocol in one dedicated view."""
    guide = pd.DataFrame(available_features((protocol,)))
    if guide.empty:
        st.info("No catalogue definitions are available for this protocol.", icon=":material/info:")
        return
    guide["present_in_run"] = guide["name"].isin(available_columns)
    with st.container(horizontal=True):
        st.metric("Catalogue features", f"{len(guide):,}", border=True)
        st.metric("Observed in this run", f"{int(guide['present_in_run'].sum()):,}", border=True)
        st.metric(
            "Medium-cost features", f"{int((guide['cost'] == 'medium').sum()):,}", border=True
        )
    category_counts = (
        guide["category"].value_counts().rename_axis("category").reset_index(name="features")
    )
    chart = (
        alt.Chart(category_counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("category:N", title="Feature category", sort="-y"),
            y=alt.Y("features:Q", title="Features"),
            color=alt.Color("category:N", legend=None),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("features:Q", title="Features"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart)
    st.dataframe(
        guide[["name", "display_name", "category", "description", "cost", "present_in_run"]],
        hide_index=True,
        height=560,
        column_config={
            "present_in_run": st.column_config.CheckboxColumn("Present in run"),
            "cost": st.column_config.TextColumn("Compute cost"),
        },
    )


def _chart_card(title: str, chart: alt.Chart | alt.LayerChart, caption: str | None = None) -> None:
    """Place a chart in a consistently styled analytical card."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        st.altair_chart(chart)


def _horizontal_bar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    *,
    title: str,
    limit: int = 20,
    value_format: str = ".3f",
) -> alt.Chart:
    """Build a consistent ranked horizontal bar chart for analytical summaries."""
    shown = frame.nlargest(limit, value)
    return (
        alt.Chart(shown)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y(f"{category}:N", sort="-x", title=None),
            x=alt.X(f"{value}:Q", title=title),
            color=alt.Color(f"{value}:Q", legend=None),
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category.replace("_", " ").title()),
                alt.Tooltip(f"{value}:Q", title=title, format=value_format),
            ],
        )
        .properties(height=max(260, min(600, 28 * len(shown))))
    )


def _feature_reason(row: pd.Series) -> str:
    """Explain an operational signal-value score without overstating causal importance."""
    reasons: list[str] = []
    if float(row["availability"]) >= 0.95:
        reasons.append("coverage is high")
    elif float(row["availability"]) < 0.60:
        reasons.append("coverage is limited")
    if float(row["spread_rank"]) >= 0.75:
        reasons.append("captures a wide operating range")
    if float(row["max_abs_correlation"]) <= 0.65:
        reasons.append("is not strongly redundant")
    if float(row["outlier_rate"]) >= 0.05:
        reasons.append("reacts to rare behaviour")
    return "; ".join(reasons) or "has a stable but lower-priority signal"


def _feature_evidence(frame: pd.DataFrame, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score feature usefulness from coverage, spread, uniqueness, and rare-event response.

    This is a protocol-local observability score, not a claim of model causality.
    Model-specific contribution is presented separately from persisted importance artefacts.
    """
    numerical = _safe_numeric(frame)
    if not numerical:
        return pd.DataFrame(), pd.DataFrame()
    sample = _sample(frame[numerical], 40_000)
    availability = 1 - sample.isna().mean()
    q25 = sample.quantile(0.25, numeric_only=True)
    q75 = sample.quantile(0.75, numeric_only=True)
    iqr = (q75 - q25).clip(lower=0)
    variance = sample.var(numeric_only=True).clip(lower=0)
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outlier_rate = (
        (sample.lt(lower) | sample.gt(upper)).sum() / sample.notna().sum().clip(lower=1)
    ).fillna(0)
    zero_rate = (sample.fillna(0) == 0).mean()
    cardinality_ratio = (sample.nunique(dropna=True) / max(len(sample), 1)).fillna(0)
    skewness = sample.skew(numeric_only=True).abs().fillna(0)
    correlation = sample.corr(numeric_only=True).abs().fillna(0)
    if not correlation.empty:
        correlation = correlation.where(~np.eye(len(correlation), dtype=bool), 0.0)
        max_correlation = correlation.max(axis=1).reindex(numerical, fill_value=0)
    else:
        max_correlation = pd.Series(0.0, index=numerical)
    evidence = pd.DataFrame(
        {
            "feature": numerical,
            "availability": availability.reindex(numerical).fillna(0).to_numpy(),
            "missing_ratio": (1 - availability).reindex(numerical).fillna(1).to_numpy(),
            "variance": variance.reindex(numerical).fillna(0).to_numpy(),
            "iqr": iqr.reindex(numerical).fillna(0).to_numpy(),
            "outlier_rate": outlier_rate.reindex(numerical).fillna(0).to_numpy(),
            "zero_rate": zero_rate.reindex(numerical).fillna(0).to_numpy(),
            "cardinality_ratio": cardinality_ratio.reindex(numerical).fillna(0).to_numpy(),
            "skewness": skewness.reindex(numerical).fillna(0).to_numpy(),
            "max_abs_correlation": max_correlation.reindex(numerical).fillna(0).to_numpy(),
        }
    )
    evidence["spread_rank"] = np.log1p(evidence["iqr"]).rank(pct=True).fillna(0)
    evidence["coverage_rank"] = evidence["availability"].rank(pct=True).fillna(0)
    evidence["novelty_rank"] = (1 - evidence["max_abs_correlation"]).rank(pct=True).fillna(0)
    evidence["rare_event_rank"] = evidence["outlier_rate"].rank(pct=True).fillna(0)
    evidence["signal_value"] = (
        100
        * (
            0.35 * evidence["coverage_rank"]
            + 0.30 * evidence["spread_rank"]
            + 0.20 * evidence["novelty_rank"]
            + 0.15 * evidence["rare_event_rank"]
        )
    ).round(1)
    catalogue = pd.DataFrame(available_features((protocol,)))
    if not catalogue.empty:
        evidence = evidence.merge(
            catalogue[["name", "display_name", "category", "description", "cost"]],
            left_on="feature",
            right_on="name",
            how="left",
        ).drop(columns="name")
    else:
        evidence["display_name"] = evidence["feature"]
        evidence["category"] = "observed"
        evidence["description"] = "Observed numerical feature"
        evidence["cost"] = "unknown"
    evidence["reason"] = evidence.apply(_feature_reason, axis=1)
    return evidence.sort_values("signal_value", ascending=False, kind="stable"), correlation


def _protocol_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a practical time series for one protocol without excessive chart points."""
    if "timestamp" not in frame.columns:
        return pd.DataFrame()
    columns = [
        column
        for column in [
            "timestamp",
            "packet_length",
            "payload_size",
            "payload_entropy",
            "inter_arrival_time",
            "jitter",
            "packet_rate",
            "flow_id",
            "src_ip",
        ]
        if column in frame.columns
    ]
    timed = frame[columns].copy()
    timed["timestamp"] = pd.to_datetime(timed["timestamp"], errors="coerce", utc=True)
    timed = timed.dropna(subset=["timestamp"]).sort_values("timestamp")
    if timed.empty:
        return pd.DataFrame()
    duration = timed["timestamp"].max() - timed["timestamp"].min()
    frequency = "1D" if duration.days > 120 else "1H" if duration.days > 14 else "15min"
    indexed = timed.set_index("timestamp")
    timeline = indexed.resample(frequency).size().rename("records").to_frame()
    for column in [
        "packet_length",
        "payload_size",
        "payload_entropy",
        "inter_arrival_time",
        "jitter",
        "packet_rate",
    ]:
        if column in indexed.columns:
            timeline[f"{column}_mean"] = indexed[column].resample(frequency).mean()
    if "packet_length" in indexed.columns:
        timeline["packet_length_p95"] = indexed["packet_length"].resample(frequency).quantile(0.95)
    if "flow_id" in indexed.columns:
        timeline["active_flows"] = indexed["flow_id"].resample(frequency).nunique()
    if "src_ip" in indexed.columns:
        timeline["active_sources"] = indexed["src_ip"].resample(frequency).nunique()
    return timeline.reset_index()


def _line_chart(frame: pd.DataFrame, x: str, y: str, title: str) -> alt.Chart:
    """Create a readable line chart with a consistent tooltip contract."""
    return (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X(f"{x}:T", title="Time"),
            y=alt.Y(f"{y}:Q", title=title),
            tooltip=[
                alt.Tooltip(f"{x}:T", title="Time"),
                alt.Tooltip(f"{y}:Q", title=title, format=".4g"),
            ],
        )
        .properties(height=280)
    )


def _render_protocol_traffic(frame: pd.DataFrame) -> None:
    """Render a multi-angle traffic and timing examination for one protocol."""
    _metric_row(frame)
    timeline = _protocol_timeline(frame)
    if timeline.empty:
        st.info("No timestamp is available for traffic-time analysis.", icon=":material/schedule:")
        return
    timeline["rolling_records"] = timeline["records"].rolling(5, min_periods=1).mean()
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Traffic volume over time", _line_chart(timeline, "timestamp", "records", "Records")
        )
    with right:
        _chart_card(
            "Smoothed traffic rate",
            _line_chart(timeline, "timestamp", "rolling_records", "Rolling records"),
            "Five-bucket rolling mean exposes sustained load rather than isolated spikes.",
        )
    timing_columns = [
        ("packet_length_mean", "Mean packet length"),
        ("packet_length_p95", "95th percentile packet length"),
        ("payload_size_mean", "Mean payload size"),
        ("payload_entropy_mean", "Mean payload entropy"),
        ("inter_arrival_time_mean", "Mean inter-arrival time"),
        ("jitter_mean", "Mean jitter"),
        ("packet_rate_mean", "Mean packet rate"),
        ("active_flows", "Active flows"),
        ("active_sources", "Active sources"),
    ]
    for offset in range(0, len(timing_columns), 2):
        left, right = st.columns(2)
        for column, title in timing_columns[offset : offset + 2]:
            target = left if column == timing_columns[offset][0] else right
            if column in timeline.columns:
                with target:
                    _chart_card(title, _line_chart(timeline, "timestamp", column, title))

    timed = frame[
        [column for column in ["timestamp", "packet_length"] if column in frame.columns]
    ].copy()
    if "timestamp" not in timed.columns:
        return
    timed["timestamp"] = pd.to_datetime(timed["timestamp"], errors="coerce", utc=True)
    timed = timed.dropna(subset=["timestamp"])
    if timed.empty:
        return
    hourly = (
        timed["timestamp"]
        .dt.hour.value_counts()
        .sort_index()
        .rename_axis("hour")
        .reset_index(name="records")
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = (
        timed["timestamp"]
        .dt.day_name()
        .value_counts()
        .reindex(weekday_order, fill_value=0)
        .rename_axis("weekday")
        .reset_index(name="records")
    )
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Traffic by hour of day",
            alt.Chart(hourly)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("hour:O", title="Hour (UTC)"),
                y=alt.Y("records:Q", title="Records"),
                color=alt.Color("records:Q", legend=None),
                tooltip=[alt.Tooltip("hour:O", title="Hour"), alt.Tooltip("records:Q", format=",")],
            )
            .properties(height=280),
        )
    with right:
        _chart_card(
            "Traffic by weekday",
            alt.Chart(weekday)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("weekday:N", title=None, sort=weekday_order),
                y=alt.Y("records:Q", title="Records"),
                color=alt.Color("records:Q", legend=None),
                tooltip=[
                    alt.Tooltip("weekday:N", title="Weekday"),
                    alt.Tooltip("records:Q", format=","),
                ],
            )
            .properties(height=280),
        )
    if "packet_length" in frame.columns:
        _chart_card(
            "Packet length distribution",
            alt.Chart(_sample(frame[["packet_length"]]).dropna())
            .mark_bar(opacity=0.72)
            .encode(
                x=alt.X("packet_length:Q", bin=alt.Bin(maxbins=70), title="Packet length"),
                y=alt.Y("count():Q", title="Records"),
                tooltip=[alt.Tooltip("count():Q", title="Records", format=",")],
            )
            .properties(height=300),
        )


def _render_protocol_endpoints(frame: pd.DataFrame) -> None:
    """Expose endpoint, port, direction, and flow relationships for one protocol."""
    source = (
        frame["src_ip"]
        .astype("string")
        .value_counts()
        .head(20)
        .rename_axis("source")
        .reset_index(name="records")
        if "src_ip" in frame.columns
        else pd.DataFrame()
    )
    destination = (
        frame["dst_ip"]
        .astype("string")
        .value_counts()
        .head(20)
        .rename_axis("destination")
        .reset_index(name="records")
        if "dst_ip" in frame.columns
        else pd.DataFrame()
    )
    left, right = st.columns(2)
    if not source.empty:
        with left:
            _chart_card(
                "Most active sources", _horizontal_bar(source, "source", "records", title="Records")
            )
    if not destination.empty:
        with right:
            _chart_card(
                "Most active destinations",
                _horizontal_bar(destination, "destination", "records", title="Records"),
            )
    if {"src_ip", "dst_ip"}.issubset(frame.columns):
        pairs = (
            (frame["src_ip"].astype("string") + " → " + frame["dst_ip"].astype("string"))
            .value_counts()
            .head(25)
            .rename_axis("endpoint_pair")
            .reset_index(name="records")
        )
        _chart_card(
            "Most frequent endpoint pairs",
            _horizontal_bar(pairs, "endpoint_pair", "records", title="Records", limit=25),
        )
    port_cards: list[tuple[str, pd.DataFrame]] = []
    for column, label in [("src_port", "Source port"), ("dst_port", "Destination port")]:
        if column in frame.columns:
            counts = (
                frame[column]
                .value_counts()
                .head(20)
                .rename_axis("port")
                .reset_index(name="records")
            )
            port_cards.append((label, counts))
    if port_cards:
        left, right = st.columns(2)
        for position, (label, counts) in enumerate(port_cards):
            target = left if position == 0 else right
            with target:
                _chart_card(
                    f"{label} concentration",
                    _horizontal_bar(counts, "port", "records", title="Records"),
                )
    categoricals = [column for column in ["transport", "direction"] if column in frame.columns]
    if categoricals:
        left, right = st.columns(2)
        for position, column in enumerate(categoricals):
            target = left if position == 0 else right
            counts = (
                frame[column]
                .astype("string")
                .value_counts()
                .rename_axis("value")
                .reset_index(name="records")
            )
            with target:
                _chart_card(
                    f"Traffic by {column.replace('_', ' ')}",
                    alt.Chart(counts)
                    .mark_arc(innerRadius=55)
                    .encode(
                        theta=alt.Theta("records:Q"),
                        color=alt.Color("value:N", title=column.replace("_", " ").title()),
                        tooltip=[
                            alt.Tooltip("value:N", title="Value"),
                            alt.Tooltip("records:Q", format=","),
                        ],
                    )
                    .properties(height=300),
                )
    if "flow_id" not in frame.columns:
        return
    aggregations: dict[str, tuple[str, str]] = {"records": ("flow_id", "size")}
    if "packet_length" in frame.columns:
        aggregations["total_bytes"] = ("packet_length", "sum")
        aggregations["mean_packet_length"] = ("packet_length", "mean")
    if "timestamp" in frame.columns:
        aggregations["first_seen"] = ("timestamp", "min")
        aggregations["last_seen"] = ("timestamp", "max")
    flows = frame.groupby("flow_id", dropna=False).agg(**aggregations).reset_index()
    if {"first_seen", "last_seen"}.issubset(flows.columns):
        flows["flow_duration_seconds"] = (
            pd.to_datetime(flows["last_seen"], utc=True)
            - pd.to_datetime(flows["first_seen"], utc=True)
        ).dt.total_seconds()
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Flow record-count distribution",
            alt.Chart(_sample(flows[["records"]]))
            .mark_bar(opacity=0.72)
            .encode(
                x=alt.X("records:Q", bin=alt.Bin(maxbins=50), title="Records per flow"),
                y=alt.Y("count():Q", title="Flows"),
                tooltip=[alt.Tooltip("count():Q", title="Flows", format=",")],
            )
            .properties(height=300),
        )
    if "total_bytes" in flows.columns:
        with right:
            _chart_card(
                "Flow volume versus record count",
                alt.Chart(_sample(flows[["records", "total_bytes"]]))
                .mark_circle(size=55, opacity=0.65)
                .encode(
                    x=alt.X("records:Q", title="Records per flow"),
                    y=alt.Y("total_bytes:Q", title="Bytes per flow"),
                    color=alt.Color("total_bytes:Q", legend=None),
                    tooltip=[
                        alt.Tooltip("records:Q", title="Records"),
                        alt.Tooltip("total_bytes:Q", title="Bytes", format=","),
                    ],
                )
                .properties(height=300),
            )
        _chart_card(
            "Largest flows by bytes",
            _horizontal_bar(
                flows.nlargest(25, "total_bytes"),
                "flow_id",
                "total_bytes",
                title="Bytes",
                limit=25,
                value_format=",",
            ),
        )
    if "flow_duration_seconds" in flows.columns:
        _chart_card(
            "Flow duration distribution",
            alt.Chart(_sample(flows[["flow_duration_seconds"]]).dropna())
            .mark_bar(opacity=0.72)
            .encode(
                x=alt.X(
                    "flow_duration_seconds:Q", bin=alt.Bin(maxbins=50), title="Flow duration (s)"
                ),
                y=alt.Y("count():Q", title="Flows"),
                tooltip=[alt.Tooltip("count():Q", title="Flows", format=",")],
            )
            .properties(height=300),
        )


def _render_feature_health(frame: pd.DataFrame, protocol: str) -> None:
    """Show non-redundant, protocol-local health evidence for numeric features."""
    evidence, correlation = _feature_evidence(frame, protocol)
    if evidence.empty:
        st.info("No numeric features are available for health analysis.", icon=":material/info:")
        return
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Missing-data burden",
            _horizontal_bar(
                evidence, "feature", "missing_ratio", title="Missing ratio", value_format=".1%"
            ),
            "High missingness means the feature may be less reliable for this protocol.",
        )
    with right:
        _chart_card(
            "Feature coverage",
            _horizontal_bar(
                evidence, "feature", "availability", title="Availability", value_format=".1%"
            ),
        )
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Robust feature spread (IQR)", _horizontal_bar(evidence, "feature", "iqr", title="IQR")
        )
    with right:
        _chart_card(
            "Feature variance", _horizontal_bar(evidence, "feature", "variance", title="Variance")
        )
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Zero-value concentration",
            _horizontal_bar(
                evidence, "feature", "zero_rate", title="Zero ratio", value_format=".1%"
            ),
        )
    with right:
        _chart_card(
            "Rare-event response",
            _horizontal_bar(
                evidence, "feature", "outlier_rate", title="Outlier ratio", value_format=".1%"
            ),
        )
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Observed uniqueness",
            _horizontal_bar(
                evidence,
                "feature",
                "cardinality_ratio",
                title="Unique value ratio",
                value_format=".1%",
            ),
        )
    with right:
        scatter = evidence.assign(log_iqr=np.log1p(evidence["iqr"]))
        _chart_card(
            "Coverage versus usable spread",
            alt.Chart(scatter)
            .mark_circle(size=90, opacity=0.75)
            .encode(
                x=alt.X("availability:Q", title="Availability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("log_iqr:Q", title="log(1 + IQR)"),
                color=alt.Color("signal_value:Q", title="Signal value"),
                size=alt.Size("outlier_rate:Q", title="Rare-event ratio"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("availability:Q", title="Availability", format=".1%"),
                    alt.Tooltip("iqr:Q", title="IQR", format=".5g"),
                    alt.Tooltip("outlier_rate:Q", title="Outlier ratio", format=".1%"),
                ],
            )
            .properties(height=360),
        )
    correlated_features = evidence.nlargest(20, "signal_value")["feature"].tolist()
    correlated_features = [
        feature for feature in correlated_features if feature in correlation.columns
    ]
    if len(correlated_features) >= 2:
        matrix = (
            correlation.loc[correlated_features, correlated_features]
            .reset_index()
            .melt(id_vars="index", var_name="feature_y", value_name="absolute_correlation")
            .rename(columns={"index": "feature_x"})
        )
        _chart_card(
            "Correlation map of high-value features",
            alt.Chart(matrix)
            .mark_rect()
            .encode(
                x=alt.X("feature_x:N", title=None, sort=None),
                y=alt.Y("feature_y:N", title=None, sort=None),
                color=alt.Color(
                    "absolute_correlation:Q",
                    title="|Correlation|",
                    scale=alt.Scale(domain=[0, 1], scheme="blues"),
                ),
                tooltip=[
                    alt.Tooltip("feature_x:N", title="Feature"),
                    alt.Tooltip("feature_y:N", title="Feature"),
                    alt.Tooltip("absolute_correlation:Q", title="|Correlation|", format=".3f"),
                ],
            )
            .properties(height=560),
            "High correlation suggests a duplicate signal; a compact profile can keep one representative.",
        )
    selected_feature = st.selectbox(
        "Inspect a feature distribution",
        evidence["feature"].tolist(),
        key=f"health_distribution_{protocol}",
    )
    _distribution_chart(frame, selected_feature)


def _render_feature_value(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Explain which protocol-local features are operationally valuable and why."""
    evidence, _correlation = _feature_evidence(frame, protocol)
    if evidence.empty:
        st.info("No numeric features are available for value analysis.", icon=":material/info:")
        return evidence
    st.caption(
        "Signal value combines coverage, robust spread, low redundancy, and rare-event response. "
        "It is an operational-selection score; detector-specific impact is shown in Model diagnostics."
    )
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Highest-value features",
            _horizontal_bar(
                evidence,
                "feature",
                "signal_value",
                title="Signal value (0–100)",
                limit=25,
                value_format=".1f",
            ),
        )
    with right:
        category = (
            evidence.fillna({"category": "uncatalogued"})
            .groupby("category", as_index=False)
            .agg(mean_signal_value=("signal_value", "mean"), features=("feature", "count"))
        )
        _chart_card(
            "Value by feature category",
            alt.Chart(category)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("category:N", title="Category", sort="-y"),
                y=alt.Y("mean_signal_value:Q", title="Mean signal value"),
                color=alt.Color("features:Q", title="Feature count"),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("mean_signal_value:Q", title="Mean value", format=".1f"),
                    alt.Tooltip("features:Q", title="Features"),
                ],
            )
            .properties(height=320),
        )
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Value versus redundancy",
            alt.Chart(evidence)
            .mark_circle(size=90, opacity=0.75)
            .encode(
                x=alt.X(
                    "max_abs_correlation:Q",
                    title="Maximum |correlation|",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y("signal_value:Q", title="Signal value"),
                color=alt.Color("availability:Q", title="Availability"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("signal_value:Q", title="Signal value", format=".1f"),
                    alt.Tooltip("max_abs_correlation:Q", title="Max |correlation|", format=".3f"),
                ],
            )
            .properties(height=360),
        )
    with right:
        _chart_card(
            "Value versus rare-event sensitivity",
            alt.Chart(evidence)
            .mark_circle(size=90, opacity=0.75)
            .encode(
                x=alt.X("outlier_rate:Q", title="Outlier ratio"),
                y=alt.Y("signal_value:Q", title="Signal value"),
                color=alt.Color("cost:N", title="Compute cost"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("signal_value:Q", title="Signal value", format=".1f"),
                    alt.Tooltip("outlier_rate:Q", title="Outlier ratio", format=".1%"),
                    alt.Tooltip("cost:N", title="Compute cost"),
                ],
            )
            .properties(height=360),
        )
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Value versus feature uniqueness",
            alt.Chart(evidence)
            .mark_circle(size=90, opacity=0.75)
            .encode(
                x=alt.X("cardinality_ratio:Q", title="Unique value ratio"),
                y=alt.Y("signal_value:Q", title="Signal value"),
                color=alt.Color("iqr:Q", title="IQR"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("cardinality_ratio:Q", title="Unique ratio", format=".1%"),
                    alt.Tooltip("signal_value:Q", title="Signal value", format=".1f"),
                ],
            )
            .properties(height=360),
        )
    with right:
        _chart_card(
            "Value versus data completeness",
            alt.Chart(evidence)
            .mark_circle(size=90, opacity=0.75)
            .encode(
                x=alt.X("availability:Q", title="Availability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("signal_value:Q", title="Signal value"),
                color=alt.Color("max_abs_correlation:Q", title="Redundancy"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("availability:Q", title="Availability", format=".1%"),
                    alt.Tooltip("signal_value:Q", title="Signal value", format=".1f"),
                ],
            )
            .properties(height=360),
        )
    st.subheader("Feature-value evidence table")
    shown = evidence.head(40)[
        [
            "feature",
            "category",
            "signal_value",
            "availability",
            "iqr",
            "outlier_rate",
            "max_abs_correlation",
            "cost",
            "reason",
            "description",
        ]
    ]
    st.dataframe(
        shown,
        hide_index=True,
        height=620,
        column_config={
            "signal_value": st.column_config.ProgressColumn(
                "Signal value", min_value=0, max_value=100, format="%.1f"
            ),
            "availability": st.column_config.NumberColumn("Availability", format="percent"),
            "outlier_rate": st.column_config.NumberColumn("Outlier ratio", format="percent"),
            "max_abs_correlation": st.column_config.NumberColumn(
                "Max |correlation|", format="%.3f"
            ),
        },
    )
    return evidence


def _comparison_files(run: Path) -> list[Path]:
    """Prefer aggregate comparison outputs and exclude a profile's internal duplicate table."""
    return [
        path
        for path in _find_files(run, "comparison.parquet")
        if path.parent.name not in {"all-features"}
    ]


def _importance_files(run: Path) -> list[Path]:
    """Return aggregate and individual model importance artefacts newest first."""
    return _find_files(run, "feature-importance.parquet")


def _history_files(run: Path) -> list[Path]:
    """Return LSTM training history files for loss-curve inspection."""
    return _find_files(run, "training-history.parquet")


def _score_files(run: Path) -> list[Path]:
    """Return persisted held-out scores that can be inspected without rerunning a model."""
    return _find_files(run, "scores.parquet")


def _label_mapping_files(run: Path) -> list[Path]:
    """Return mapping outputs that satisfy the training label contract."""
    compatible: list[Path] = []
    for path in _find_files(run / "mapping", "*.parquet"):
        try:
            if {"flow_id", "label"}.issubset(_read(path).columns):
                compatible.append(path)
        except (OSError, ValueError):
            continue
    return compatible


def _comparison_selector(files: list[Path], run: Path, key: str) -> Path | None:
    """Select one aggregate model report while exposing its relative destination."""
    if not files:
        return None
    labels = {_run_label(path, run): path for path in files}
    return labels[st.selectbox("Experiment comparison", list(labels), key=key)]


def _model_manifest(score_path: Path) -> dict[str, Any]:
    """Load model metadata next to a score artefact without making it a hard dependency."""
    path = score_path.parent / "model.json"
    try:
        return _read_json(path) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _model_metrics(score_path: Path) -> dict[str, Any]:
    """Load persisted detector metrics when the artefact supplies them."""
    path = score_path.parent / "metrics.json"
    try:
        return _read_json(path) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _model_label(score_path: Path, run: Path) -> str:
    """Give every persisted detector run a unique human-readable selector label."""
    metadata = _model_manifest(score_path)
    model = str(metadata.get("model", score_path.parent.name))
    return f"{model} · {_run_label(score_path.parent, run)}"


def _protocol_score_sources(run: Path, protocol: str) -> dict[str, tuple[Path, pd.DataFrame]]:
    """Find every detector that actually produced scores for one selected protocol."""
    sources: dict[str, tuple[Path, pd.DataFrame]] = {}
    for path in _score_files(run):
        try:
            scores = _read(path)
        except (OSError, ValueError):
            continue
        if "anomaly_score" not in scores.columns or "protocol" not in scores.columns:
            continue
        scoped = scores[scores["protocol"].astype("string").str.lower() == protocol].copy()
        if not scoped.empty:
            sources[_model_label(path, run)] = (path, scoped)
    return sources


def _score_context(scores: pd.DataFrame, frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Attach raw protocol evidence to scores when row IDs allow a reliable local join."""
    if "row_id" not in scores.columns:
        return scores.copy()
    scope = str(scores.get("scope", pd.Series([""])).iloc[0]) if not scores.empty else ""
    reference = (
        frame
        if scope.startswith("grouped_")
        else frame[frame["protocol"].astype("string").str.lower() == protocol]
    )
    raw_columns = [
        column
        for column in [
            "timestamp",
            "packet_length",
            "payload_size",
            "payload_entropy",
            "inter_arrival_time",
            "jitter",
            "flow_duration",
            "packet_rate",
        ]
        if column in reference.columns and column not in scores.columns
    ]
    if not raw_columns:
        return scores.copy()
    context = reference.reset_index(drop=True).reset_index(names="row_id")
    return scores.merge(context[["row_id", *raw_columns]], on="row_id", how="left")


def _render_model_comparison(comparison: pd.DataFrame, protocol: str) -> None:
    """Compare detector runs in the selected protocol context and expose trade-offs."""
    if comparison.empty:
        st.info("This experiment has no completed model run.", icon=":material/info:")
        return
    scoped = comparison.copy()
    if "protocols" in scoped.columns:
        mask = scoped["protocols"].astype("string").str.lower().str.contains(protocol, regex=False)
        if mask.any():
            scoped = scoped[mask].copy()
    profile_column = "feature_profile" if "feature_profile" in scoped.columns else "scope"
    metric_options = [
        column
        for column in [
            "roc_auc",
            "average_precision",
            "reconstruction_mse_mean",
            "reconstruction_rmse",
            "score_mean",
            "score_p95",
            "score_p99",
            "predicted_anomaly_rate",
            "fit_seconds",
            "process_memory_delta_mb",
        ]
        if column in scoped.columns and scoped[column].notna().any()
    ]
    if not metric_options:
        st.info(
            "No comparable numerical metric is available in this artefact.", icon=":material/info:"
        )
        return
    metric = st.selectbox("Comparison measure", metric_options, key=f"comparison_metric_{protocol}")
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Detector comparison",
            alt.Chart(scoped)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("model:N", title="Detector"),
                xOffset=alt.XOffset(f"{profile_column}:N"),
                y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                color=alt.Color(f"{profile_column}:N", title="Feature profile"),
                tooltip=[
                    alt.Tooltip("model:N", title="Detector"),
                    alt.Tooltip(f"{profile_column}:N", title="Profile"),
                    alt.Tooltip(
                        f"{metric}:Q", title=metric.replace("_", " ").title(), format=".6g"
                    ),
                ],
            )
            .properties(height=350),
        )
    if {"input_features", "fit_seconds"}.issubset(scoped.columns):
        with right:
            _chart_card(
                "Feature-cost trade-off",
                alt.Chart(scoped)
                .mark_circle(size=150, opacity=0.8)
                .encode(
                    x=alt.X("input_features:Q", title="Prepared columns"),
                    y=alt.Y("fit_seconds:Q", title="Fit time (seconds)"),
                    color=alt.Color("model:N", title="Detector"),
                    shape=alt.Shape(f"{profile_column}:N", title="Feature profile"),
                    size=alt.Size("process_memory_delta_mb:Q", title="Memory delta (MB)"),
                    tooltip=[
                        alt.Tooltip("model:N", title="Detector"),
                        alt.Tooltip("input_features:Q", title="Columns"),
                        alt.Tooltip("fit_seconds:Q", title="Fit time", format=".3f"),
                        alt.Tooltip(
                            "process_memory_delta_mb:Q", title="Memory delta", format=".3f"
                        ),
                    ],
                )
                .properties(height=350),
            )
    if {"score_p50", "score_p95", "score_p99", "model"}.issubset(scoped.columns):
        percentiles = scoped.melt(
            id_vars=["model", profile_column],
            value_vars=["score_p50", "score_p95", "score_p99"],
            var_name="percentile",
            value_name="score",
        )
        _chart_card(
            "Anomaly-score tail by detector",
            alt.Chart(percentiles)
            .mark_line(point=True)
            .encode(
                x=alt.X("percentile:N", title="Score percentile"),
                y=alt.Y("score:Q", title="Anomaly score"),
                color=alt.Color("model:N", title="Detector"),
                strokeDash=alt.StrokeDash(f"{profile_column}:N", title="Feature profile"),
                tooltip=[
                    alt.Tooltip("model:N", title="Detector"),
                    alt.Tooltip("percentile:N", title="Percentile"),
                    alt.Tooltip("score:Q", title="Score", format=".6g"),
                ],
            )
            .properties(height=330),
            "A large P99-to-P50 gap means the detector separates a small high-score tail.",
        )
    summary_columns = [
        column
        for column in [
            "input_features",
            "fit_seconds",
            "process_memory_delta_mb",
            "roc_auc",
            "average_precision",
            "reconstruction_mse_mean",
            "score_p95",
            "predicted_anomaly_rate",
        ]
        if column in scoped.columns
    ]
    if summary_columns:
        summary = scoped.groupby(profile_column, as_index=False)[summary_columns].mean(
            numeric_only=True
        )
        if "input_features" in summary.columns:
            baseline = summary.loc[summary[profile_column] == "all_features", "input_features"]
            if not baseline.empty:
                summary["feature_reduction_vs_all"] = (
                    1 - summary["input_features"] / baseline.iloc[0]
                )
        st.dataframe(
            summary,
            hide_index=True,
            column_config={
                "feature_reduction_vs_all": st.column_config.NumberColumn(
                    "Feature reduction vs all", format="percent"
                ),
                "predicted_anomaly_rate": st.column_config.NumberColumn(
                    "Flag rate", format="percent"
                ),
            },
        )


def _render_model_inspection(
    frame: pd.DataFrame, run: Path, protocol: str, evidence: pd.DataFrame
) -> None:
    """Render deep score, MSE, learning, and contribution evidence for one detector artefact."""
    sources = _protocol_score_sources(run, protocol)
    if not sources:
        st.info(
            "No persisted detector has held-out scores for this protocol. Run a per-protocol "
            "experiment in Experiment studio to create them.",
            icon=":material/info:",
        )
        return
    selected = st.selectbox("Detector run", list(sources), key=f"detector_run_{protocol}")
    score_path, scores = sources[selected]
    metadata = _model_manifest(score_path)
    metrics = _model_metrics(score_path)
    model_name = str(metadata.get("model", score_path.parent.name))
    diagnostic = _score_context(scores, frame, protocol)
    protocol_frame = frame[frame["protocol"].astype("string").str.lower() == protocol]
    flow_count = (
        int(protocol_frame["flow_id"].nunique(dropna=True))
        if "flow_id" in protocol_frame.columns
        else 0
    )
    if 0 < flow_count < 10:
        st.warning(
            f"This protocol has only {flow_count} distinct flows in the current run. "
            "Treat detector importance as capture-specific exploratory evidence, not a "
            "generalizable protocol ranking.",
            icon=":material/science:",
        )
    flags = diagnostic.get("is_anomaly", pd.Series(False, index=diagnostic.index)).astype(bool)
    reconstruction_mse = metrics.get("reconstruction_mse_mean")
    if reconstruction_mse is None and model_name in {"lstm_autoencoder", "pca_autoencoder"}:
        reconstruction_mse = float(diagnostic["anomaly_score"].mean())
    with st.container(horizontal=True):
        st.metric("Detector", model_name.replace("_", " ").title(), border=True)
        st.metric(
            "Scope", str(diagnostic.get("scope", pd.Series(["unknown"])).iloc[0]), border=True
        )
        st.metric("Held-out records", f"{len(diagnostic):,}", border=True)
        st.metric("Flagged rate", f"{flags.mean():.2%}", border=True)
        st.metric(
            "Reconstruction MSE" if reconstruction_mse is not None else "Mean anomaly score",
            f"{float(reconstruction_mse if reconstruction_mse is not None else diagnostic['anomaly_score'].mean()):.6g}",
            border=True,
        )
        st.metric("Threshold", f"{float(metrics.get('threshold', np.nan)):.6g}", border=True)
    threshold = metrics.get("threshold")
    histogram = (
        alt.Chart(_sample(diagnostic[["anomaly_score"]], 8_000))
        .mark_bar(opacity=0.72)
        .encode(
            x=alt.X(
                "anomaly_score:Q",
                bin=alt.Bin(maxbins=70),
                title="Anomaly score / reconstruction MSE",
            ),
            y=alt.Y("count():Q", title="Held-out records"),
            tooltip=[alt.Tooltip("count():Q", title="Records", format=",")],
        )
        .properties(height=320)
    )
    if isinstance(threshold, int | float):
        histogram = histogram + alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
            color="#FB7185", strokeDash=[6, 4], size=2
        ).encode(x="threshold:Q")
    left, right = st.columns(2)
    with left:
        _chart_card("Score distribution and decision threshold", histogram)
    with right:
        ecdf = diagnostic[["anomaly_score"]].sort_values("anomaly_score").reset_index(drop=True)
        ecdf["cumulative_share"] = (ecdf.index + 1) / max(len(ecdf), 1)
        _chart_card(
            "Cumulative score distribution",
            alt.Chart(_chart_sample(ecdf))
            .mark_line()
            .encode(
                x=alt.X("anomaly_score:Q", title="Anomaly score"),
                y=alt.Y("cumulative_share:Q", title="Cumulative share", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("anomaly_score:Q", title="Score", format=".6g"),
                    alt.Tooltip("cumulative_share:Q", title="Share", format=".1%"),
                ],
            )
            .properties(height=320),
        )
    diagnostic = diagnostic.reset_index(drop=True)
    diagnostic["evaluation_order"] = np.arange(1, len(diagnostic) + 1)
    x_column = (
        "timestamp"
        if "timestamp" in diagnostic.columns and diagnostic["timestamp"].notna().any()
        else "evaluation_order"
    )
    x_encoding = alt.X(
        f"{x_column}:{'T' if x_column == 'timestamp' else 'Q'}",
        title="Time" if x_column == "timestamp" else "Evaluation order",
    )
    diagnostic["rolling_score"] = diagnostic["anomaly_score"].rolling(25, min_periods=1).mean()
    left, right = st.columns(2)
    with left:
        _chart_card(
            "Score sequence",
            alt.Chart(_chart_sample(diagnostic))
            .mark_line()
            .encode(
                x=x_encoding,
                y=alt.Y("anomaly_score:Q", title="Anomaly score"),
                color=alt.Color("is_anomaly:N", title="Flagged"),
                tooltip=[
                    alt.Tooltip("anomaly_score:Q", title="Score", format=".6g"),
                    alt.Tooltip("is_anomaly:N", title="Flagged"),
                ],
            )
            .properties(height=320),
        )
    with right:
        _chart_card(
            "Rolling anomaly-score baseline",
            alt.Chart(_chart_sample(diagnostic))
            .mark_line(color="#38BDF8")
            .encode(
                x=x_encoding,
                y=alt.Y("rolling_score:Q", title="Rolling mean score"),
                tooltip=[alt.Tooltip("rolling_score:Q", title="Rolling score", format=".6g")],
            )
            .properties(height=320),
        )
    grouping = (
        "label"
        if "label" in diagnostic.columns and diagnostic["label"].notna().any()
        else "is_anomaly"
    )
    _chart_card(
        "Score spread by label or detector decision",
        alt.Chart(_chart_sample(diagnostic[["anomaly_score", grouping]].dropna()))
        .mark_boxplot(size=38)
        .encode(
            x=alt.X(f"{grouping}:N", title=grouping.replace("_", " ").title()),
            y=alt.Y("anomaly_score:Q", title="Anomaly score"),
            color=alt.Color(f"{grouping}:N", legend=None),
            tooltip=[alt.Tooltip(f"{grouping}:N", title="Group")],
        )
        .properties(height=320),
    )
    raw_signals = [
        column
        for column in [
            "packet_length",
            "payload_size",
            "payload_entropy",
            "inter_arrival_time",
            "jitter",
            "packet_rate",
        ]
        if column in diagnostic.columns and diagnostic[column].notna().any()
    ][:2]
    if raw_signals:
        columns = st.columns(len(raw_signals))
        for target, signal in zip(columns, raw_signals, strict=True):
            with target:
                _chart_card(
                    f"Score versus {signal.replace('_', ' ')}",
                    alt.Chart(
                        _chart_sample(diagnostic[[signal, "anomaly_score", "is_anomaly"]].dropna())
                    )
                    .mark_circle(size=45, opacity=0.6)
                    .encode(
                        x=alt.X(f"{signal}:Q", title=signal.replace("_", " ").title()),
                        y=alt.Y("anomaly_score:Q", title="Anomaly score"),
                        color=alt.Color("is_anomaly:N", title="Flagged"),
                        tooltip=[
                            alt.Tooltip(
                                f"{signal}:Q", title=signal.replace("_", " ").title(), format=".5g"
                            ),
                            alt.Tooltip("anomaly_score:Q", title="Score", format=".6g"),
                        ],
                    )
                    .properties(height=320),
                )
    importance_path = score_path.parent / "feature-importance.parquet"
    if importance_path.exists():
        importance = _read(importance_path)
        if not importance.empty and "importance" in importance.columns:
            legacy_importance = "raw_importance" not in importance.columns
            raw_column = "raw_importance" if not legacy_importance else "importance"
            importance = importance.copy()
            importance["raw_importance"] = pd.to_numeric(
                importance[raw_column], errors="coerce"
            ).fillna(0)
            importance = importance.sort_values("raw_importance", ascending=False, kind="stable")
            total_importance = max(float(importance["raw_importance"].sum()), np.finfo(float).eps)
            importance["importance_share"] = importance["raw_importance"] / total_importance
            importance["cumulative_importance"] = importance["importance_share"].cumsum()
            importance = importance.merge(
                evidence[["feature", "signal_value", "reason"]], on="feature", how="left"
            )
            if legacy_importance:
                st.warning(
                    "This detector was trained before the sparse-feature scaling safeguard. "
                    "The chart is normalized for readability, but rerun this detector before "
                    "treating its ranking as validated.",
                    icon=":material/warning:",
                )
            left, right = st.columns(2)
            with left:
                _chart_card(
                    "Detector-specific feature contribution",
                    _horizontal_bar(
                        importance,
                        "feature",
                        "importance_share",
                        title="Relative contribution",
                        limit=25,
                        value_format=".1%",
                    ),
                    "For LSTM/LOF/SVM, this is normalized score change after permuting the feature.",
                )
            with right:
                _chart_card(
                    "Cumulative contribution coverage",
                    alt.Chart(
                        importance.reset_index(drop=True).assign(rank=lambda data: data.index + 1)
                    )
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("rank:Q", title="Feature rank"),
                        y=alt.Y(
                            "cumulative_importance:Q",
                            title="Cumulative contribution",
                            axis=alt.Axis(format="%"),
                        ),
                        tooltip=[
                            alt.Tooltip("rank:Q", title="Rank"),
                            alt.Tooltip("feature:N", title="Feature"),
                            alt.Tooltip(
                                "cumulative_importance:Q", title="Cumulative", format=".1%"
                            ),
                        ],
                    )
                    .properties(height=320),
                )
            _chart_card(
                "Operational value versus detector contribution",
                alt.Chart(importance.dropna(subset=["signal_value"]))
                .mark_circle(size=90, opacity=0.78)
                .encode(
                    x=alt.X("signal_value:Q", title="Protocol signal value"),
                    y=alt.Y("importance_share:Q", title="Relative detector contribution"),
                    color=alt.Color("method:N", title="Method"),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Feature"),
                        alt.Tooltip("signal_value:Q", title="Signal value", format=".1f"),
                        alt.Tooltip("importance_share:Q", title="Contribution", format=".1%"),
                        alt.Tooltip("reason:N", title="Operational reason"),
                    ],
                )
                .properties(height=340),
            )
            st.dataframe(
                importance.head(40)[
                    [
                        "feature",
                        "importance_share",
                        "raw_importance",
                        "method",
                        "signal_value",
                        "reason",
                    ]
                ],
                hide_index=True,
                height=500,
                column_config={
                    "importance_share": st.column_config.ProgressColumn(
                        "Relative contribution", min_value=0, max_value=1, format="percent"
                    ),
                    "raw_importance": st.column_config.NumberColumn(
                        "Raw model signal", format="%.6g"
                    ),
                    "signal_value": st.column_config.ProgressColumn(
                        "Signal value", min_value=0, max_value=100
                    ),
                },
            )
    history_path = score_path.parent / "training-history.parquet"
    if history_path.exists():
        history = _read(history_path)
        if not history.empty and "train_loss" in history.columns:
            melted = history.melt(
                id_vars="epoch",
                value_vars=[
                    column
                    for column in ["train_loss", "validation_loss"]
                    if column in history.columns
                ],
                var_name="series",
                value_name="mse_loss",
            ).dropna()
            history["loss_change"] = history["train_loss"].diff()
            left, right = st.columns(2)
            with left:
                _chart_card(
                    "LSTM reconstruction-MSE learning curve",
                    alt.Chart(melted)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("epoch:Q", title="Epoch"),
                        y=alt.Y(
                            "mse_loss:Q", title="Reconstruction MSE", scale=alt.Scale(zero=False)
                        ),
                        color=alt.Color("series:N", title="Series"),
                        tooltip=[
                            alt.Tooltip("epoch:Q", title="Epoch"),
                            alt.Tooltip("series:N", title="Series"),
                            alt.Tooltip("mse_loss:Q", title="MSE", format=".8g"),
                        ],
                    )
                    .properties(height=330),
                )
            with right:
                _chart_card(
                    "LSTM per-epoch training improvement",
                    alt.Chart(history.dropna(subset=["loss_change"]))
                    .mark_bar()
                    .encode(
                        x=alt.X("epoch:O", title="Epoch"),
                        y=alt.Y("loss_change:Q", title="Change in train MSE"),
                        color=alt.condition(
                            alt.datum.loss_change < 0, alt.value("#34D399"), alt.value("#FB7185")
                        ),
                        tooltip=[
                            alt.Tooltip("epoch:Q", title="Epoch"),
                            alt.Tooltip("loss_change:Q", title="MSE change", format=".8g"),
                        ],
                    )
                    .properties(height=330),
                )
    st.subheader("Highest-scoring held-out records")
    columns = [
        column
        for column in [
            "timestamp",
            "anomaly_score",
            "score_percentile",
            "is_anomaly",
            "label",
            "flow_id",
            "row_id",
        ]
        if column in diagnostic.columns
    ]
    st.dataframe(
        diagnostic.nlargest(100, "anomaly_score")[columns],
        hide_index=True,
        height=420,
        column_config={
            "score_percentile": st.column_config.NumberColumn("Score percentile", format="percent"),
            "is_anomaly": st.column_config.CheckboxColumn("Flagged"),
        },
    )


def _render_model_results(frame: pd.DataFrame, run: Path) -> None:
    """Render protocol-specific model comparison and deep per-detector diagnostics."""
    protocols = _protocols(frame)
    if not protocols:
        st.info("The current run has no supported protocol to analyse.", icon=":material/info:")
        return
    protocol = st.selectbox("Protocol for model results", protocols, key="model_results_protocol")
    area = st.segmented_control(
        "Model results area",
        ["Compare models", "Inspect one detector"],
        default="Compare models",
        required=True,
        key="model_results_area",
        width="stretch",
    )
    if area == "Compare models":
        files = _comparison_files(run)
        if not files:
            st.info(
                "Run a model experiment to populate model comparison.",
                icon=":material/play_circle:",
            )
            return
        labels = {_run_label(path, run): path for path in files}
        path = labels[
            st.selectbox("Experiment comparison", list(labels), key="comparison_selector_v2")
        ]
        st.caption(f"Comparison source: {path.relative_to(run)}")
        _render_model_comparison(_read(path), protocol)
        return
    evidence, _correlation = _feature_evidence(
        frame[frame["protocol"].astype("string").str.lower() == protocol], protocol
    )
    _render_model_inspection(frame, run, protocol, evidence)
    return

    comparison_path = _comparison_selector(_comparison_files(run), run, "comparison_selector")
    if comparison_path is None:
        st.info(
            "Run a model experiment in the experiment studio to populate this section.",
            icon=":material/play_circle:",
        )
        return
    comparison = _read(comparison_path)
    if comparison.empty:
        st.warning(
            "The selected comparison contains no completed model runs.", icon=":material/warning:"
        )
        return
    st.caption(f"Reading {comparison_path.relative_to(run)}")
    profile_column = "feature_profile" if "feature_profile" in comparison.columns else "scope"
    with st.container(horizontal=True):
        st.metric("Completed model runs", f"{len(comparison):,}", border=True)
        st.metric("Feature profiles", f"{comparison[profile_column].nunique():,}", border=True)
        st.metric(
            "Detectors",
            f"{comparison['model'].nunique() if 'model' in comparison.columns else 0:,}",
            border=True,
        )
        st.metric(
            "Median fit time",
            f"{comparison['fit_seconds'].median():.2f} s"
            if "fit_seconds" in comparison.columns
            else "Unavailable",
            border=True,
        )

    metric_options = [
        column
        for column in [
            "average_precision",
            "roc_auc",
            "predicted_anomaly_rate",
            "score_p95",
            "best_validation_loss",
            "best_training_loss",
            "fit_seconds",
            "process_memory_delta_mb",
        ]
        if column in comparison.columns and comparison[column].notna().any()
    ]
    if metric_options:
        metric = st.selectbox("Comparison metric", metric_options, key="comparison_metric")
        chart = (
            alt.Chart(comparison)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title="Detector"),
                xOffset=alt.XOffset(f"{profile_column}:N"),
                y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                color=alt.Color(f"{profile_column}:N", title="Feature profile"),
                tooltip=[
                    alt.Tooltip("model:N", title="Detector"),
                    alt.Tooltip(f"{profile_column}:N", title="Feature profile"),
                    alt.Tooltip(
                        f"{metric}:Q", title=metric.replace("_", " ").title(), format=".5f"
                    ),
                    alt.Tooltip("input_features:Q", title="Prepared columns"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart)

    if {"fit_seconds", "input_features", "model"}.issubset(comparison.columns):
        tradeoff = (
            alt.Chart(comparison)
            .mark_circle(size=130, opacity=0.75)
            .encode(
                x=alt.X("input_features:Q", title="Prepared columns"),
                y=alt.Y("fit_seconds:Q", title="Fit time (seconds)"),
                color=alt.Color("model:N", title="Detector"),
                shape=alt.Shape(f"{profile_column}:N", title="Feature profile"),
                tooltip=[
                    alt.Tooltip("model:N", title="Detector"),
                    alt.Tooltip(f"{profile_column}:N", title="Feature profile"),
                    alt.Tooltip("input_features:Q", title="Prepared columns"),
                    alt.Tooltip("fit_seconds:Q", title="Fit time", format=".3f"),
                    alt.Tooltip("process_memory_delta_mb:Q", title="Memory delta", format=".3f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(tradeoff)

    st.subheader("All features versus selected profiles")
    summary_columns = [
        column
        for column in [
            "input_features",
            "fit_seconds",
            "process_memory_delta_mb",
            "predicted_anomaly_rate",
            "score_p95",
            "roc_auc",
            "average_precision",
            "best_validation_loss",
            "best_training_loss",
        ]
        if column in comparison.columns
    ]
    if summary_columns:
        profile_summary = comparison.groupby(profile_column, as_index=False)[summary_columns].mean(
            numeric_only=True
        )
        if "input_features" in profile_summary.columns:
            baseline = profile_summary.loc[
                profile_summary[profile_column] == "all_features", "input_features"
            ]
            if not baseline.empty:
                profile_summary["feature_reduction_vs_all"] = 1 - (
                    profile_summary["input_features"] / baseline.iloc[0]
                )
        st.dataframe(
            profile_summary,
            hide_index=True,
            column_config={
                "feature_reduction_vs_all": st.column_config.NumberColumn(
                    "Feature reduction vs all", format="percent"
                )
            },
        )

    importance_paths = _importance_files(run)
    if importance_paths:
        st.subheader("Feature contribution")
        options = {_run_label(path, run): path for path in importance_paths}
        importance_path = options[
            st.selectbox("Feature importance source", list(options), key="importance_selector")
        ]
        importance = _read(importance_path)
        if not importance.empty:
            model_options = (
                sorted(importance["model"].dropna().unique().tolist())
                if "model" in importance.columns
                else []
            )
            selected_model = (
                st.selectbox("Detector importance", model_options, key="importance_model")
                if model_options
                else None
            )
            shown = (
                importance
                if selected_model is None
                else importance[importance["model"] == selected_model]
            )
            shown = shown.nlargest(25, "importance")
            chart = (
                alt.Chart(shown)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                    x=alt.X("importance:Q", title="Contribution"),
                    color=alt.Color("method:N", title="Method"),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Feature"),
                        alt.Tooltip("importance:Q", title="Contribution", format=".6f"),
                        alt.Tooltip("method:N", title="Method"),
                        alt.Tooltip("transformed_columns:Q", title="Transformed columns"),
                    ],
                )
                .properties(height=600)
            )
            st.altair_chart(chart)

    score_paths = _score_files(run)
    if score_paths:
        st.subheader("Held-out anomaly triage")
        score_options = {_run_label(path, run): path for path in score_paths}
        score_path = score_options[
            st.selectbox("Detector score source", list(score_options), key="score_selector")
        ]
        scores = _read(score_path)
        if "anomaly_score" in scores.columns and not scores.empty:
            flags = scores.get("is_anomaly", pd.Series(False, index=scores.index)).astype(bool)
            with st.container(horizontal=True):
                st.metric("Held-out records", f"{len(scores):,}", border=True)
                st.metric("Flagged records", f"{int(flags.sum()):,}", border=True)
                st.metric("Flagged rate", f"{flags.mean():.2%}", border=True)
                st.metric("Maximum score", f"{scores['anomaly_score'].max():.5g}", border=True)
            distribution = _sample(
                pd.DataFrame(
                    {
                        "anomaly_score": scores["anomaly_score"],
                        "decision": flags.map({True: "flagged", False: "not flagged"}),
                    }
                )
            )
            chart = (
                alt.Chart(distribution)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X("anomaly_score:Q", bin=alt.Bin(maxbins=60), title="Anomaly score"),
                    y=alt.Y("count():Q", title="Held-out records"),
                    color=alt.Color("decision:N", title="Detector decision"),
                    tooltip=[
                        alt.Tooltip("decision:N", title="Decision"),
                        alt.Tooltip("count():Q", title="Records", format=","),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart)
            triage_columns = [
                column
                for column in [
                    "anomaly_score",
                    "is_anomaly",
                    "label",
                    "protocol",
                    "flow_id",
                    "scope",
                    "model",
                ]
                if column in scores.columns
            ]
            st.dataframe(
                scores.nlargest(100, "anomaly_score")[triage_columns],
                hide_index=True,
                height=330,
                column_config={
                    "is_anomaly": st.column_config.CheckboxColumn("Flagged by detector")
                },
            )

    histories = _history_files(run)
    if histories:
        st.subheader("LSTM autoencoder training curves")
        labels = {_run_label(path, run): path for path in histories}
        history_path = labels[
            st.selectbox("Training history", list(labels), key="history_selector")
        ]
        history = _read(history_path)
        if not history.empty:
            melted = history.melt(
                id_vars=["epoch"],
                value_vars=[
                    column
                    for column in ["train_loss", "validation_loss"]
                    if column in history.columns
                ],
                var_name="series",
                value_name="loss",
            ).dropna()
            curve = (
                alt.Chart(melted)
                .mark_line(point=True)
                .encode(
                    x=alt.X("epoch:Q", title="Epoch"),
                    y=alt.Y("loss:Q", title="Reconstruction loss", scale=alt.Scale(zero=False)),
                    color=alt.Color("series:N", title="Series"),
                    tooltip=[
                        alt.Tooltip("epoch:Q", title="Epoch"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("loss:Q", title="Loss", format=".8f"),
                    ],
                )
                .properties(height=340)
            )
            st.altair_chart(curve)

    sweep_files = _find_files(run, "sweep-comparison.parquet")
    if sweep_files:
        st.subheader("LSTM parameter sweep")
        sweep = _read(sweep_files[0])
        sweep_metrics = [
            column
            for column in [
                "best_validation_loss",
                "best_training_loss",
                "roc_auc",
                "average_precision",
                "predicted_anomaly_rate",
                "score_p95",
                "fit_seconds",
            ]
            if column in sweep.columns and sweep[column].notna().any()
        ]
        sweep_metric = (
            st.selectbox(
                "Sweep measure",
                sweep_metrics,
                key="sweep_metric",
            )
            if sweep_metrics
            else None
        )
        if sweep_metric and {"hidden_size", "sequence_length"}.issubset(sweep.columns):
            chart = (
                alt.Chart(sweep)
                .mark_line(point=True)
                .encode(
                    x=alt.X("hidden_size:O", title="Hidden size"),
                    y=alt.Y(
                        f"{sweep_metric}:Q",
                        title=sweep_metric.replace("_", " ").title(),
                        scale=alt.Scale(zero=False),
                    ),
                    color=alt.Color("sequence_length:N", title="Sequence length"),
                    strokeDash=alt.StrokeDash("feature_profile:N", title="Feature profile"),
                    tooltip=[
                        alt.Tooltip("sweep_variant:N", title="Variant"),
                        alt.Tooltip("feature_profile:N", title="Feature profile"),
                        alt.Tooltip("hidden_size:Q", title="Hidden size"),
                        alt.Tooltip("sequence_length:Q", title="Sequence length"),
                        alt.Tooltip(
                            f"{sweep_metric}:Q",
                            title=sweep_metric.replace("_", " ").title(),
                            format=".8f",
                        ),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(chart)
        st.dataframe(sweep, hide_index=True, height=360)

    st.subheader("Complete comparison table")
    st.dataframe(comparison, hide_index=True, height=420)


def _render_mapping_audit(run: Path) -> None:
    """Make CSV-to-PCAP mapping evidence visible without treating it as model truth."""
    summaries = _find_files(run / "mapping", "*.summary.json")
    if not summaries:
        st.info("No mapping summaries are available in this run.", icon=":material/info:")
        return
    rows: list[dict[str, Any]] = []
    for path in summaries:
        try:
            payload = _read_json(path)
            rows.append(
                {
                    "mapping": path.name,
                    "feature_rows": payload.get("feature_rows"),
                    "label_rows": payload.get("label_rows"),
                    "matched": payload.get("matched_rows"),
                    "unmatched": payload.get("unmatched_rows"),
                    "status": payload.get("status", "complete"),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    audit = pd.DataFrame(rows)
    if audit.empty:
        st.info("The mapping manifests could not be read.", icon=":material/info:")
        return
    st.warning(
        "Mapping is evidence for later label validation. It is not automatically used as "
        "the truth source for unsupervised training.",
        icon=":material/warning:",
    )
    st.dataframe(audit, hide_index=True)


def _render_runtime(run: Path) -> None:
    """Show live capacity and persisted training resource evidence side by side."""
    current = snapshot()
    with st.container(horizontal=True):
        st.metric("Logical CPUs", current.logical_cpus, border=True)
        st.metric("CPU utilisation", f"{current.cpu_percent:.1f}%", border=True)
        st.metric("Available memory", f"{current.available_memory_gb:.1f} GB", border=True)
        st.metric("Memory utilisation", f"{current.memory_percent:.1f}%", border=True)
    comparison_files = _comparison_files(run)
    if comparison_files:
        comparison = _read(comparison_files[0])
        if {"model", "fit_seconds", "process_memory_delta_mb"}.issubset(comparison.columns):
            resource = comparison.groupby("model", as_index=False)[
                ["fit_seconds", "process_memory_delta_mb"]
            ].mean(numeric_only=True)
            chart = (
                alt.Chart(resource)
                .mark_circle(size=180)
                .encode(
                    x=alt.X("fit_seconds:Q", title="Mean fit time (seconds)"),
                    y=alt.Y("process_memory_delta_mb:Q", title="Mean process memory delta (MB)"),
                    color=alt.Color("model:N", title="Detector"),
                    tooltip=[
                        alt.Tooltip("model:N", title="Detector"),
                        alt.Tooltip("fit_seconds:Q", title="Fit time", format=".3f"),
                        alt.Tooltip(
                            "process_memory_delta_mb:Q", title="Memory delta", format=".3f"
                        ),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(chart)


def _render_protocol_explorer(frame: pd.DataFrame, run: Path) -> None:
    """Provide 25+ useful, dynamically rendered charts for every protocol independently."""
    protocols = _protocols(frame)
    if not protocols:
        st.warning(
            "The combined feature table contains no supported protocols.", icon=":material/warning:"
        )
        return
    protocol = st.selectbox("Protocol section", protocols, key="results_protocol")
    scoped = frame[frame["protocol"].astype("string").str.lower() == protocol].copy()
    section = st.segmented_control(
        "Protocol analysis area",
        [
            "Traffic & timing",
            "Endpoints & flows",
            "Feature health",
            "Feature value & guide",
            "Model diagnostics",
        ],
        default="Traffic & timing",
        required=True,
        key="protocol_view_v2",
        width="stretch",
    )
    st.caption(
        f"All evidence below is calculated only from the {protocol.upper()} records in this run. "
        "Views render on demand so the dashboard stays responsive on large PCAPs."
    )
    if section == "Traffic & timing":
        _render_protocol_traffic(scoped)
    elif section == "Endpoints & flows":
        _render_protocol_endpoints(scoped)
    elif section == "Feature health":
        _render_feature_health(scoped, protocol)
    elif section == "Feature value & guide":
        _render_feature_value(scoped, protocol)
        st.subheader("Protocol feature catalogue")
        _feature_guide(protocol, set(scoped.columns))
    else:
        evidence, _correlation = _feature_evidence(scoped, protocol)
        _render_model_inspection(frame, run, protocol, evidence)
    return

    protocols = _protocols(frame)
    if not protocols:
        st.warning(
            "The combined feature table contains no supported protocols.", icon=":material/warning:"
        )
        return
    protocol = st.selectbox("Protocol section", protocols, key="results_protocol")
    scoped = frame[frame["protocol"].astype(str).str.lower() == protocol].copy()
    section = st.segmented_control(
        "Protocol view",
        ["Traffic profile", "Feature health", "Feature guide", "Records"],
        default="Traffic profile",
        required=True,
        key="protocol_view",
        width="stretch",
    )
    if section == "Traffic profile":
        _metric_row(scoped)
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("**Traffic over time**")
                _timeline_chart(scoped, color_by_protocol=False)
        with right:
            with st.container(border=True):
                st.markdown("**Most active sources**")
                _endpoint_chart(scoped)
        numerical = _safe_numeric(scoped)
        if numerical:
            with st.container(border=True):
                feature = st.selectbox(
                    "Distribution feature", numerical, key="protocol_distribution_feature"
                )
                _distribution_chart(scoped, feature)
    elif section == "Feature health":
        _missingness_chart(scoped)
        numerical = _safe_numeric(scoped)
        if len(numerical) >= 2:
            selected = st.multiselect(
                "Correlation features",
                numerical,
                default=numerical[: min(12, len(numerical))],
                key="protocol_correlation_features",
            )
            if len(selected) >= 2:
                correlation = (
                    _sample(scoped[selected], 5000)
                    .corr(numeric_only=True)
                    .reset_index()
                    .melt(id_vars="index", var_name="feature_y", value_name="correlation")
                    .rename(columns={"index": "feature_x"})
                )
                heatmap = (
                    alt.Chart(correlation)
                    .mark_rect()
                    .encode(
                        x=alt.X("feature_x:N", title=None, sort=None),
                        y=alt.Y("feature_y:N", title=None, sort=None),
                        color=alt.Color(
                            "correlation:Q",
                            scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                            title="Correlation",
                        ),
                        tooltip=[
                            alt.Tooltip("feature_x:N", title="Feature"),
                            alt.Tooltip("feature_y:N", title="Feature"),
                            alt.Tooltip("correlation:Q", title="Correlation", format=".3f"),
                        ],
                    )
                    .properties(height=560)
                )
                st.altair_chart(heatmap)
    elif section == "Feature guide":
        _feature_guide(protocol, set(frame.columns))
    else:
        st.caption(
            "A deterministic sample is shown; the dashboard never changes your raw artefact."
        )
        st.dataframe(_sample(scoped, 1000), hide_index=True, height=560)


def _render_results(frame: pd.DataFrame, run: Path) -> None:
    """Render one of the protocol-first result areas, avoiding feature-file navigation."""
    area = st.segmented_control(
        "Results area",
        ["Run overview", "Protocol explorer", "Model results", "Mapping audit", "Runtime"],
        default="Run overview",
        required=True,
        key="results_area",
        width="stretch",
    )
    if area == "Run overview":
        _metric_row(frame)
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("**Protocol volume**")
                _protocol_volume_chart(frame)
        with right:
            with st.container(border=True):
                st.markdown("**Traffic timeline**")
                _timeline_chart(frame)
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("**Most active sources**")
                _endpoint_chart(frame)
        with right:
            numerical = _safe_numeric(frame)
            if numerical:
                with st.container(border=True):
                    st.markdown("**Cross-protocol distribution**")
                    feature = st.selectbox(
                        "Overview distribution feature",
                        numerical,
                        key="overview_distribution_feature",
                    )
                    _distribution_chart(frame, feature)
    elif area == "Protocol explorer":
        _render_protocol_explorer(frame, run)
    elif area == "Model results":
        _render_model_results(frame, run)
    elif area == "Mapping audit":
        _render_mapping_audit(run)
    else:
        _render_runtime(run)


def _config_for_dashboard(config_path: str, artifact_root: Path) -> dict[str, Any]:
    """Load a requested config while keeping dashboard profiles under the chosen artefact root."""
    path = Path(config_path)
    settings = load_config(path if config_path.strip() and path.exists() else None)
    settings["project"]["artifact_dir"] = str(artifact_root)
    return settings


def _parse_numbers(raw: str) -> list[int]:
    """Parse comma-separated positive integer variants from the sweep form."""
    values: list[int] = []
    for item in raw.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        value = int(cleaned)
        if value < 2:
            raise ValueError("Sweep values must be integers of at least 2.")
        values.append(value)
    return list(dict.fromkeys(values))


def _render_pipeline_launcher(artifact_root: Path) -> None:
    """Expose the existing config-driven full pipeline as an intentional dashboard action."""
    st.subheader("Run configured ingestion")
    st.caption(
        "This runs the same extraction, mapping-candidate, quality-report, and "
        "baseline-evaluation pipeline as `anomaly pipeline run`."
    )
    with st.form("pipeline_launcher"):
        config_path = st.text_input("Configuration override", value="config/test-datasets.yaml")
        run_name = st.text_input(
            "New run name", value=f"dashboard-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        packet_cap = st.number_input(
            "Optional packet cap per capture",
            min_value=0,
            value=0,
            help="Use a cap for a quick structural test; zero processes complete captures.",
        )
        submitted = st.form_submit_button(
            "Run configured pipeline", type="primary", icon=":material/play_arrow:"
        )
    if not submitted:
        return
    try:
        settings = _config_for_dashboard(config_path, artifact_root)
        target = artifact_root / "runs" / run_name.strip()
        if target.exists():
            st.error(
                "Choose a new run name; this directory already exists.", icon=":material/error:"
            )
            return
        with st.status("Running configured pipeline", expanded=True) as status:
            st.write(f"Output: `{target}`")
            summary, summary_path = run_inventory(
                settings, output_dir=target, max_packets=int(packet_cap) or None
            )
            st.write(
                "Extracted "
                f"{summary['successful_extractions']}/"
                f"{summary['dataset_count']} configured captures."
            )
            st.write(f"Summary: `{summary_path}`")
            status.update(label="Configured pipeline complete", state="complete", expanded=False)
        _load_table.clear()
        _load_json.clear()
        st.success(
            "The run is ready. Select it from the sidebar and open Results explorer.",
            icon=":material/check_circle:",
        )
    except Exception as error:
        st.error(f"Pipeline did not complete: {error}", icon=":material/error:")


def _render_profile_builder(
    artifact_root: Path, run: Path | None, frame: pd.DataFrame | None
) -> None:
    """Create as many immutable client feature sets as needed before training."""
    st.subheader("Build feature profiles")
    if run is None or frame is None:
        st.info(
            "Select a completed pipeline run in the sidebar to build a profile from its "
            "observed protocols.",
            icon=":material/info:",
        )
        return
    present_protocols = _protocols(frame)
    quality = _quality_report(run)
    protocol_scope = st.multiselect(
        "Profile protocol scope",
        present_protocols,
        default=present_protocols,
        key="profile_protocol_scope",
    )
    definitions = pd.DataFrame(available_features(tuple(protocol_scope)))
    definitions = definitions[definitions["name"].isin(frame.columns)].copy()
    if quality is not None:
        quality_columns = [
            column
            for column in ["name", "missing_ratio", "variance", "estimated_memory_mb"]
            if column in quality.columns
        ]
        definitions = definitions.merge(quality[quality_columns], on="name", how="left")
    with st.container(border=True):
        st.markdown("**Available client-selectable features**")
        st.dataframe(definitions, hide_index=True, height=300)
    with st.form("profile_builder"):
        choices = definitions["name"].tolist()
        selected = st.multiselect(
            "Features in this profile", choices, default=choices[: min(12, len(choices))]
        )
        name = st.text_input("Profile name", value="client-selection")
        description = st.text_area(
            "Selection rationale",
            value="Selected in the experiment studio after protocol and feature review.",
        )
        submitted = st.form_submit_button(
            "Create immutable feature profile", type="primary", icon=":material/save:"
        )
    if submitted:
        try:
            settings = _config_for_dashboard("", artifact_root)
            path = create_profile(name, selected, settings, description, protocol_scope)
            _load_json.clear()
            st.success(
                f"Created {path.name}. It is now available for multi-profile model comparison.",
                icon=":material/check_circle:",
            )
        except (ValueError, OSError) as error:
            st.error(str(error), icon=":material/error:")


def _render_experiment_runner(
    artifact_root: Path, run: Path | None, frame: pd.DataFrame | None
) -> None:
    """Run baseline/profile comparisons and optional LSTM parameter sweeps from one form."""
    st.subheader("Train and compare")
    if run is None or frame is None:
        st.info(
            "Select a completed pipeline run in the sidebar before starting model training.",
            icon=":material/info:",
        )
        return
    feature_path = _feature_path(run)
    if feature_path is None:
        st.error("The selected run has no combined feature table.", icon=":material/error:")
        return
    profiles = _profile_records(artifact_root)
    profile_labels = {item["label"]: item["path"] for item in profiles}
    label_sources: dict[str, Path | None] = {"No mapped labels (unsupervised)": None}
    label_sources.update({_run_label(path, run): path for path in _label_mapping_files(run)})
    with st.form("experiment_runner"):
        strategy = st.selectbox("Deployment strategy", ["per_protocol", "grouped"], index=0)
        group = st.selectbox(
            "Grouped deployment scope",
            ["all", "it", "ot"],
            index=0,
            help="Used only when the strategy is grouped.",
        )
        models = st.multiselect(
            "Detectors",
            MODEL_OPTIONS,
            default=["isolation_forest", "pca_autoencoder", "lstm_autoencoder"],
        )
        selected_profiles = st.multiselect(
            "Selected feature profiles",
            list(profile_labels),
            help="The all-features baseline is always included automatically.",
        )
        label_source = st.selectbox(
            "Evaluation labels (optional)",
            list(label_sources),
            help=(
                "Choose a reviewed mapping output to calculate ROC-AUC and average precision. "
                "Leave this unset for a fully unsupervised run."
            ),
        )
        st.markdown("**LSTM autoencoder settings**")
        left, middle, right = st.columns(3)
        with left:
            hidden_size = st.number_input(
                "Hidden size", min_value=4, max_value=512, value=32, step=4
            )
            latent_size = st.number_input(
                "Latent size", min_value=2, max_value=256, value=16, step=2
            )
        with middle:
            sequence_length = st.number_input(
                "Sequence length", min_value=2, max_value=256, value=16, step=2
            )
            epochs = st.number_input("Epochs", min_value=1, max_value=200, value=20, step=1)
        with right:
            batch_size = st.number_input(
                "Batch size", min_value=8, max_value=4096, value=256, step=8
            )
            max_windows = st.number_input(
                "Maximum training windows", min_value=100, max_value=100_000, value=5000, step=100
            )
        run_sweep = st.checkbox("Also run an LSTM parameter sweep", value=False)
        sweep_hidden = st.text_input("Sweep hidden sizes", value="16,32,64", disabled=not run_sweep)
        sweep_sequences = st.text_input(
            "Sweep sequence lengths", value="8,16", disabled=not run_sweep
        )
        submitted = st.form_submit_button(
            "Run experiment", type="primary", icon=":material/play_arrow:"
        )
    if not submitted:
        return
    if not models:
        st.error("Select at least one detector.", icon=":material/error:")
        return
    selected_paths = [profile_labels[label] for label in selected_profiles]
    selected_labels = label_sources[label_source]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = run / "experiments" / f"studio-{timestamp}"
    parameters = {
        "hidden_size": int(hidden_size),
        "latent_size": int(latent_size),
        "sequence_length": int(sequence_length),
        "sequence_stride": int(sequence_length),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "max_train_windows": int(max_windows),
    }
    try:
        with st.status("Training selected detectors", expanded=True) as status:
            st.write("Baseline: all catalogue features")
            st.write(f"Selected profiles: {len(selected_paths)}")
            st.write(
                f"Evaluation labels: `{selected_labels}`"
                if selected_labels
                else "Evaluation labels: none"
            )
            comparison, summary = run_feature_experiments(
                feature_path=feature_path,
                config=_config_for_dashboard("", artifact_root),
                strategy=strategy,
                group=group,
                profiles=selected_paths,
                output_dir=output_root,
                labels_path=selected_labels,
                candidates=models,
                model_overrides={"lstm_autoencoder": parameters},
            )
            st.write(f"Completed {len(comparison)} detector runs.")
            st.write(f"Comparison: `{summary['comparison']}`")
            if run_sweep:
                hidden_values = _parse_numbers(sweep_hidden)
                sequence_values = _parse_numbers(sweep_sequences)
                variants = [
                    {
                        **parameters,
                        "hidden_size": hidden,
                        "sequence_length": length,
                        "sequence_stride": length,
                    }
                    for hidden, length in product(hidden_values, sequence_values)
                ]
                if len(variants) > 8:
                    raise ValueError(
                        "Limit the LSTM sweep to eight parameter combinations per run."
                    )
                st.write(
                    f"Running {len(variants)} LSTM variants across baseline and selected profiles."
                )
                sweep, sweep_summary = run_lstm_sweep(
                    feature_path=feature_path,
                    config=_config_for_dashboard("", artifact_root),
                    strategy=strategy,
                    group=group,
                    profiles=selected_paths,
                    parameter_sets=variants,
                    output_dir=output_root / "lstm-sweep",
                    labels_path=selected_labels,
                )
                st.write(f"Completed {len(sweep)} LSTM sweep runs: `{sweep_summary['comparison']}`")
            status.update(label="Experiment complete", state="complete", expanded=False)
        _load_table.clear()
        _load_json.clear()
        st.success(
            "Open Results explorer > Model results to inspect the new comparison.",
            icon=":material/check_circle:",
        )
    except Exception as error:
        st.error(f"Experiment did not complete: {error}", icon=":material/error:")


def _render_execution_history(run: Path | None) -> None:
    """Expose saved pipeline, profile, and experiment manifests as audit-ready records."""
    st.subheader("Execution history")
    if run is None:
        st.info("Select a run to inspect its execution manifests.", icon=":material/info:")
        return
    manifests = _find_files(run, "experiment-batch.json") + _find_files(run, "lstm-sweep.json")
    if (run / "run-summary.json").exists():
        manifests.append(run / "run-summary.json")
    if not manifests:
        st.info("No execution manifest exists yet.", icon=":material/info:")
        return
    rows: list[dict[str, Any]] = []
    for path in sorted(set(manifests), key=lambda item: item.stat().st_mtime_ns, reverse=True):
        payload = _read_json(path)
        rows.append(
            {
                "manifest": _run_label(path, run),
                "created_at": payload.get("created_at"),
                "detectors": ", ".join(payload.get("detectors", []))
                if isinstance(payload.get("detectors"), list)
                else None,
                "runs": len(payload.get("runs", []))
                if isinstance(payload.get("runs"), list)
                else payload.get("dataset_count"),
                "comparison": payload.get("comparison"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _render_experiment_studio(
    artifact_root: Path, run: Path | None, frame: pd.DataFrame | None
) -> None:
    """Group all write actions separately from the read-only analytics explorer."""
    area = st.segmented_control(
        "Experiment studio area",
        ["Configured pipeline", "Feature profiles", "Train and compare", "Execution history"],
        default="Configured pipeline",
        required=True,
        key="studio_area",
        width="stretch",
    )
    if area == "Configured pipeline":
        _render_pipeline_launcher(artifact_root)
    elif area == "Feature profiles":
        _render_profile_builder(artifact_root, run, frame)
    elif area == "Train and compare":
        _render_experiment_runner(artifact_root, run, frame)
    else:
        _render_execution_history(run)


def _init_state() -> None:
    """Initialize only true per-user state; persisted artefacts remain on disk."""
    st.session_state.setdefault("workspace", "Results explorer")


def main() -> None:
    """Render a two-part workbench: controlled execution first, evidence review second."""
    _init_state()
    st.title("Network anomaly workbench", anchor=False)
    st.caption(
        "Protocol-centred traffic analysis, feature governance, and reproducible "
        "unsupervised experiments."
    )
    with st.sidebar:
        st.header("Run context")
        artifact_root_text = st.text_input(
            "Artifact directory", value="artifacts", key="artifact_root"
        )
        artifact_root = Path(artifact_root_text)
        runs = _discover_runs(artifact_root)
        run: Path | None = None
        if runs:
            labels = {_run_label(item, artifact_root): item for item in runs}
            run = labels[st.selectbox("Pipeline run", list(labels), key="active_run")]
            st.caption(
                "Results are scoped by protocol from the run's combined feature table, "
                "never by individual feature-file names."
            )
        else:
            st.caption("No completed pipeline run found. Create one in Experiment studio.")
        if st.button("Refresh artefacts", icon=":material/refresh:"):
            _load_table.clear()
            _load_json.clear()
            st.rerun()
    feature_path = _feature_path(run)
    frame: pd.DataFrame | None = None
    if feature_path is not None:
        try:
            frame = _read(feature_path)
        except Exception as error:
            st.error(f"Could not load the combined feature table: {error}", icon=":material/error:")
    workspace = st.segmented_control(
        "Workspace",
        ["Experiment studio", "Results explorer"],
        required=True,
        key="workspace",
        width="stretch",
    )
    if workspace == "Experiment studio":
        _render_experiment_studio(artifact_root, run, frame)
        return
    if run is None or frame is None:
        st.info(
            "Select a completed run in the sidebar, or use Experiment studio to run the "
            "configured inventory.",
            icon=":material/info:",
        )
        return
    _render_results(frame, run)


if __name__ == "__main__":
    main()
