"""A small, artifact-first Streamlit dashboard for the portable two-stage platform."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.neighbors import NearestNeighbors

from anomdet.core.io import read_table
from anomdet.core.resources import snapshot
from anomdet.features.catalog import available_features
from anomdet.preprocessing.pipeline import transform_with_manifest
from anomdet.selection.profiles import create_profile, load_profile


@st.cache_data(max_entries=64, show_spinner="در حال خواندن خروجی ذخیره‌شده…")
def _table(path_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return read_table(Path(path_text))


@st.cache_data(max_entries=64, show_spinner=False)
def _json(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    with Path(path_text).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read(path: Path) -> pd.DataFrame:
    return _table(str(path), path.stat().st_mtime_ns)


def _read_json(path: Path) -> dict[str, Any]:
    return _json(str(path), path.stat().st_mtime_ns)


def _runs(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    if (root / "catalog" / "artifacts.parquet").exists():
        candidates.add(root)
    for base in (root, root / "runs"):
        if base.is_dir():
            candidates.update(
                item for item in base.iterdir() if (item / "catalog" / "artifacts.parquet").exists()
            )
    return sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True)


def _catalog(run: Path) -> pd.DataFrame:
    return _read(run / "catalog" / "artifacts.parquet")


def _paths(catalog: pd.DataFrame, run: Path, kind: str, protocol: str | None = None) -> list[Path]:
    scoped = catalog[catalog["kind"].eq(kind)]
    if protocol and "protocol" in scoped:
        scoped = scoped[scoped["protocol"].fillna("").eq(protocol)]
    return [run / item for item in scoped["path"].astype(str) if (run / item).exists()]


@st.fragment(run_every="5s")
def _resource_monitor() -> None:
    """A lightweight live host status panel; it never starts a training job."""
    current = snapshot()
    cpu, memory, available = st.columns(3)
    cpu.metric("CPU", f"{current.cpu_percent:.1f}%", f"{current.logical_cpus} logical cores")
    memory.metric(
        "RAM", f"{current.memory_percent:.1f}%", f"{current.total_memory_gb:.1f} GB total"
    )
    available.metric("Available RAM", f"{current.available_memory_gb:.1f} GB")


def _notice_missing(title: str, expected: str) -> None:
    st.info(f"{title} هنوز برای این اجرا تولید نشده است. مسیر مورد انتظار: `{expected}`")


def _overview(run: Path, catalog: pd.DataFrame, protocol: str | None) -> None:
    scoped = catalog if protocol is None else catalog[catalog["protocol"].fillna("").eq(protocol)]
    mapping = run / "reports" / "mapping-audit.parquet"
    mapping_rows = _read(mapping) if mapping.exists() else pd.DataFrame()
    if protocol and not mapping_rows.empty:
        mapping_rows = mapping_rows[mapping_rows["protocol"].eq(protocol)]
    artifacts, feature_rows, mappings, accepted = st.columns(4)
    artifacts.metric("خروجی‌های ثبت‌شده", f"{len(scoped):,}")
    feature_rows.metric(
        "رکوردهای فیچر",
        f"{int(scoped.loc[scoped.kind.eq('feature_records'), 'rows'].fillna(0).sum()):,}",
    )
    mappings.metric("نگاشت‌های بررسی‌شده", f"{len(mapping_rows):,}")
    accepted.metric(
        "نگاشت‌های پذیرفته‌شده", f"{int(mapping_rows.get('accepted', pd.Series(dtype=bool)).sum()):,}"
    )
    st.subheader("مانیتور منابع", anchor=False)
    _resource_monitor()
    st.subheader("آنچه این اجرا تولید کرده", anchor=False)
    by_kind = (
        scoped.groupby("kind", dropna=False)
        .agg(files=("path", "count"), rows=("rows", "sum"))
        .reset_index()
    )
    st.dataframe(by_kind, hide_index=True, width="stretch")


def _mapping_view(run: Path, protocol: str | None) -> None:
    path = run / "reports" / "mapping-audit.parquet"
    if not path.exists():
        _notice_missing("گزارش نگاشت", "reports/mapping-audit.parquet")
        return
    frame = _read(path)
    if protocol:
        frame = frame[frame["protocol"].eq(protocol)]
    if frame.empty:
        st.info("برای این پروتکل هنوز نگاشتی وجود ندارد.")
        return
    st.caption("CSV فقط منبع لیبل است؛ همه‌ی فیچرها از PCAP ساخته شده‌اند.")
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("capture:N", sort="-y", title="Capture"),
            y=alt.Y("match_rate:Q", scale=alt.Scale(domain=[0, 1]), title="Match rate"),
            color=alt.condition(alt.datum.accepted, alt.value("#22C55E"), alt.value("#E11D48")),
            tooltip=[
                "protocol",
                "capture",
                "selected_csv",
                "match_rate",
                "mean_match_confidence",
                "selected_offset_seconds",
                "accepted",
            ],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "دانلود جدول نگاشت (CSV)", frame.to_csv(index=False), "mapping-audit.csv", "text/csv"
    )
    distribution_path = run / "reports" / "attack-label-distribution.parquet"
    if distribution_path.exists():
        distribution = _read(distribution_path)
        if protocol:
            distribution = distribution[distribution["protocol"].eq(protocol)]
        st.subheader("توازن برچسب‌های حمله", anchor=False)
        balance = distribution.groupby("label", as_index=False)["records"].sum()
        st.altair_chart(
            alt.Chart(balance)
            .mark_bar(color="#8B1E3F", cornerRadiusEnd=4)
            .encode(
                y=alt.Y("label:N", sort="-x", title=None),
                x=alt.X("records:Q", title="PCAP-derived records"),
                tooltip=["label", "records"],
            ),
            width="stretch",
        )


def _features_view(run: Path, catalog: pd.DataFrame, protocol: str | None) -> None:
    paths = _paths(catalog, run, "feature_overview", protocol)
    if not paths:
        _notice_missing("تحلیل فیچر", "reports/features/<split>/<protocol>-overview.parquet")
        return
    overview = pd.concat(
        [_read(path).assign(source=path.relative_to(run).as_posix()) for path in paths],
        ignore_index=True,
    )
    metrics = ["availability", "iqr", "std", "unique_values", "zero_ratio"]
    metric = st.selectbox("معیار نمودار", metrics, index=0)
    top = overview.groupby("feature", as_index=False)[metric].median().nlargest(25, metric)
    chart = (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=4, color="#7C3AED")
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X(f"{metric}:Q", title=metric),
            tooltip=["feature", metric],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        overview.sort_values(["availability", "feature"], ascending=[False, True]),
        hide_index=True,
        width="stretch",
    )
    record_paths = _paths(catalog, run, "feature_records", protocol)
    if record_paths:
        sample_path = st.selectbox(
            "نمونه‌ی رکورد PCAP",
            record_paths,
            format_func=lambda item: item.relative_to(run).as_posix(),
        )
        st.dataframe(_read(sample_path).head(200), hide_index=True, width="stretch")
    quality_paths = _paths(catalog, run, "feature_quality", protocol)
    if quality_paths:
        quality = pd.concat([_read(path) for path in quality_paths], ignore_index=True)
        st.subheader("کیفیت قابل‌استفاده‌بودن فیچرها", anchor=False)
        st.dataframe(quality, hide_index=True, width="stretch")


def _profile_view(artifact_root: Path, protocol: str | None) -> None:
    profile_dir = artifact_root / "feature_profiles"
    profile_paths = sorted(profile_dir.glob("*.json")) if profile_dir.exists() else []
    st.caption("پروفایل، نسخه‌ی غیرقابل‌تغییر انتخاب فیچر است و همراه مدل ذخیره می‌شود.")
    if profile_paths:
        selected = st.selectbox(
            "پروفایل‌های ذخیره‌شده", profile_paths, format_func=lambda item: item.name
        )
        st.json(load_profile(selected, {"project": {"artifact_dir": str(artifact_root)}}))
    configured = available_features((protocol,)) if protocol else available_features()
    names = [str(item["name"]) for item in configured]
    with st.form("feature-profile"):
        name = st.text_input("نام ساده‌ی پروفایل", placeholder="dns-compact")
        features = st.multiselect("فیچرهای مدل", names, default=names[: min(8, len(names))])
        description = st.text_input("دلیل انتخاب", placeholder="Balanced CPU profile")
        saved = st.form_submit_button("ذخیره‌ی نسخه‌ی پروفایل")
    if saved:
        if not name or not features:
            st.error("نام و حداقل یک فیچر لازم است.")
        else:
            path = create_profile(
                name,
                features,
                {"project": {"artifact_dir": str(artifact_root)}},
                description,
                [protocol] if protocol else None,
            )
            st.success(f"پروفایل ذخیره شد: {path}")


def _training_profiles(artifact_root: Path, protocol: str) -> dict[Path, dict[str, Any]]:
    """Return valid, protocol-scoped immutable profile manifests."""
    profile_dir = artifact_root / "feature_profiles"
    profiles: dict[Path, dict[str, Any]] = {}
    for path in sorted(profile_dir.glob("*.json")) if profile_dir.exists() else []:
        try:
            profile = load_profile(path, {"project": {"artifact_dir": str(artifact_root)}})
        except (OSError, ValueError):
            continue
        if protocol.casefold() in {str(item).casefold() for item in profile["protocols"]}:
            profiles[path] = profile
    return profiles


def _accepted_mapping_counts(run: Path, protocol: str) -> tuple[int, int]:
    """Return accepted and total mapping counts before allowing Stage 2 training."""
    audit_path = run / "reports" / "mapping-audit.parquet"
    if not audit_path.exists():
        return 0, 0
    audit = _read(audit_path)
    scoped = audit[audit["protocol"].astype("string").str.casefold().eq(protocol.casefold())]
    accepted = (
        scoped.get("accepted", pd.Series(False, index=scoped.index)).fillna(False).astype(bool)
    )
    return int(accepted.sum()), len(scoped)


def _project_root() -> Path:
    """Resolve the repository root independently of Streamlit's launch directory."""
    return Path(__file__).resolve().parents[3]


def _tail(path: Path, lines: int = 14) -> str:
    if not path.exists():
        return "فرآیند در حال آماده‌سازی فایل گزارش است…"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _read_live_progress(path: Path) -> dict[str, Any]:
    """Read a durable operation monitor without caching a changing file."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _progress_events_table(progress: dict[str, Any]) -> pd.DataFrame:
    """Return concise, sortable event rows instead of dumping implementation payloads."""
    rows: list[dict[str, Any]] = []
    for event in progress.get("events", []):
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                "زمان UTC": event.get("at"),
                "مرحله": event.get("stage"),
                "رویداد": event.get("event"),
                "پیام": event.get("message"),
                "پروتکل": event.get("protocol"),
                "Capture": event.get("capture_name") or event.get("capture"),
                "ردیف/Packet": event.get("packets_read") or event.get("rows"),
                "Epoch": event.get("epoch"),
                "Train loss": event.get("train_loss"),
                "Validation loss": event.get("validation_loss"),
                "پیشرفت": event.get("progress_ratio")
                if event.get("progress_ratio") is not None
                else event.get("overall_progress_ratio"),
            }
        )
    return pd.DataFrame(rows)


def _render_live_operation_monitor(
    progress_path: Path, log_path: Path, title: str, refresh_seconds: int
) -> None:
    """Render a reusable live monitor from the same durable files used outside the UI."""
    progress = _read_live_progress(progress_path)
    current = progress.get("current", {}) if progress else {}
    status = str(progress.get("status", "running")) if progress else "starting"
    status_text = {"running": "در حال اجرا", "completed": "کامل", "failed": "ناموفق"}.get(
        status, status
    )
    st.subheader(title, anchor=False)
    if current:
        ratio_value = current.get("progress_ratio", current.get("overall_progress_ratio"))
        ratio = pd.to_numeric(pd.Series([ratio_value]), errors="coerce").iloc[0]
        stage = str(current.get("stage", "initialization"))
        message = str(current.get("message", "رویداد جدیدی ثبت نشده است."))
        cards = st.columns(4)
        cards[0].metric("وضعیت", status_text)
        cards[1].metric("مرحلهٔ فعلی", stage.replace("_", " "))
        cards[2].metric(
            "پیشرفت مرحله",
            f"{float(ratio):.0%}" if pd.notna(ratio) else "در حال محاسبه",
        )
        cards[3].metric("آخرین به‌روزرسانی", str(progress.get("updated_at", "—"))[11:19])
        if pd.notna(ratio):
            st.progress(float(min(max(ratio), 1.0)), text=message)
        else:
            st.caption(message)
        st.caption(
            f"رویدادهای زنده از `{progress_path.name}` خوانده می‌شوند؛ هر {refresh_seconds} ثانیه بازخوانی می‌شوند."
        )
    else:
        st.caption("منتظر نخستین رویداد پایدار هستیم؛ لاگ کنسول پایین همچنان قابل مشاهده است.")

    events = _progress_events_table(progress)
    if not events.empty:
        st.dataframe(
            events.tail(18).iloc[::-1],
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                "پیشرفت": st.column_config.NumberColumn(format="percent"),
                "Train loss": st.column_config.NumberColumn(format="%.6f"),
                "Validation loss": st.column_config.NumberColumn(format="%.6f"),
            },
        )
    st.caption("آخرین خط‌های همان اجرای CLI؛ این log خارج از dashboard نیز در کنسول دیده می‌شود.")
    st.code(_tail(log_path, lines=24), language="text")


def _start_dashboard_training(run: Path, protocol: str, profile: Path) -> dict[str, Any]:
    """Start the existing audited CLI trainer without blocking the dashboard session."""
    model_root = run / "models" / f"{protocol}-{profile.stem}"
    model_root.mkdir(parents=True, exist_ok=True)
    log_path = model_root / "dashboard-training.log"
    command = [
        sys.executable,
        "-m",
        "anomdet.cli",
        "two-stage",
        "train",
        str(run.resolve()),
        "--protocol",
        protocol,
        "--profile",
        str(profile.resolve()),
        "--output",
        str(model_root.resolve()),
    ]
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=_project_root(),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
    job: dict[str, Any] = {
        "process": process,
        "pid": process.pid,
        "protocol": protocol,
        "profile": profile.name,
        "model_root": str(model_root),
        "log_path": str(log_path),
        "refreshed": False,
    }
    st.session_state["two_stage_training_job"] = job
    return job


@st.fragment(run_every="3s")
def _training_status() -> None:
    """Show the persisted trainer log while an independently running process is active."""
    job = st.session_state.get("two_stage_training_job")
    if not job:
        return
    process = job.get("process")
    log_path = Path(str(job["log_path"]))
    model_root = Path(str(job["model_root"]))
    return_code = process.poll() if process is not None else None
    if return_code is None:
        st.info(
            f"آموزش {job['protocol']} با PID {job['pid']} در حال اجراست؛ "
            "داشبورد همچنان قابل استفاده است.",
            icon=":material/pending:",
        )
        _render_live_operation_monitor(
            model_root / "training-progress.json",
            log_path,
            "مانیتور زندهٔ آموزش دو مرحله‌ای",
            refresh_seconds=3,
        )
        return
    if return_code == 0 and (model_root / "training-summary.json").exists():
        if not job["refreshed"]:
            job["refreshed"] = True
            _table.clear()
            _json.clear()
            st.rerun()
        st.success("آموزش دو مرحله‌ای کامل شد و نتایج پایین همین صفحه قابل مشاهده‌اند.")
    else:
        st.error(f"آموزش با کد خروج {return_code} متوقف شد. گزارش زیر را بررسی کنید.")
        st.code(_tail(log_path, lines=30), language="text")
    if st.button("پاک‌کردن وضعیت آموزش", key="clear-two-stage-training"):
        st.session_state.pop("two_stage_training_job", None)
        st.rerun()


def _training_panel(run: Path, artifact_root: Path, protocol: str | None) -> None:
    """Offer a guarded UI trigger for the exact same portable CLI training command."""
    st.subheader("آموزش مدل دو مرحله‌ای", anchor=False)
    if protocol is None:
        st.info("برای آموزش، ابتدا یک پروتکل مشخص را از سایدبار انتخاب کنید.")
        return
    profiles = _training_profiles(artifact_root, protocol)
    if not profiles:
        st.info("ابتدا در بخش «پروفایل و پیش‌پردازش» یک پروفایل فیچر برای این پروتکل بسازید.")
        return
    profile_paths = list(profiles)
    profile_path = st.selectbox(
        "پروفایل ورودی مدل",
        profile_paths,
        format_func=lambda item: (
            f"{profiles[item]['name']} · v{profiles[item]['version']} · "
            f"{profiles[item]['feature_count']} فیچر"
        ),
        key=f"training-profile-{run}-{protocol}",
    )
    accepted, total = _accepted_mapping_counts(run, protocol)
    target = run / "models" / f"{protocol}-{profile_path.stem}"
    running = st.session_state.get("two_stage_training_job", {}).get("process")
    is_running = running is not None and running.poll() is None
    if (target / "training-summary.json").exists():
        st.success(f"این مدل از قبل آماده است: {target.relative_to(run)}")
        return
    st.caption(
        f"نگاشت پذیرفته‌شده برای {protocol}: {accepted}/{total}. "
        "Stage 1 با benign و Stage 2 فقط با برچسب‌های CSV پذیرفته‌شده آموزش می‌بیند."
    )
    with st.form(f"two-stage-training-{run}-{protocol}", border=True):
        acknowledged = st.checkbox(
            "نگاشت‌ها و کیفیت فیچرها را بررسی کرده‌ام و شروع آموزش را تأیید می‌کنم."
        )
        submitted = st.form_submit_button(
            "شروع آموزش LSTM-AE + Isolation Forest + Random Forest",
            icon=":material/model_training:",
            type="primary",
            disabled=is_running or accepted == 0,
        )
    if is_running:
        st.warning(
            "یک آموزش در همین نشست در حال اجراست؛ برای جلوگیری از اجرای تکراری، منتظر بمانید."
        )
    elif accepted == 0:
        st.warning("هیچ نگاشت پذیرفته‌شده‌ای برای این پروتکل وجود ندارد؛ Stage 2 نباید آموزش ببیند.")
    elif submitted:
        if not acknowledged:
            st.warning("پیش از شروع، تأیید بررسی نگاشت و فیچرها لازم است.")
        else:
            job = _start_dashboard_training(run, protocol, profile_path)
            st.success(f"آموزش در پس‌زمینه آغاز شد. گزارش: {job['log_path']}")


def _models_view(run: Path, artifact_root: Path, protocol: str | None) -> None:
    _training_panel(run, artifact_root, protocol)
    _training_status()
    models = sorted(run.glob("models/**/training-summary.json"))
    if not models:
        _notice_missing("مدل دو مرحله‌ای", "models/<protocol>-<profile>/training-summary.json")
        return
    selected = st.selectbox(
        "مدل آموزش‌دیده", models, format_func=lambda item: item.parent.relative_to(run).as_posix()
    )
    summary = _read_json(selected)
    first, second, timing = st.columns(3)
    first.metric("Stage 1 FPR", f"{summary['stage1']['false_positive_rate']:.2%}")
    second.metric("Stage 2 F1", f"{summary['stage2']['weighted_f1']:.2%}")
    timing.metric("زمان آموزش", f"{summary['training_seconds']:.1f} s")
    model_root = selected.parent
    score_path = model_root / "stage1" / "scores.parquet"
    evidence_path = model_root / "stage1" / "feature-evidence.parquet"
    history_path = model_root / "stage1" / "lstm-training-history.parquet"
    importance_path = model_root / "stage2" / "feature-importance.parquet"
    confusion_path = model_root / "stage2" / "confusion-matrix.parquet"
    tabs = st.tabs(["Stage 1", "شواهد فیچر", "Stage 2", "پایپ‌لاین"])
    with tabs[0]:
        if score_path.exists():
            scores = _read(score_path)
            chart = (
                alt.Chart(scores)
                .mark_circle(size=28, opacity=0.55)
                .encode(
                    x="lstm_reconstruction_score:Q",
                    y="isolation_forest_score:Q",
                    color=alt.condition(
                        alt.datum.stage1_anomaly, alt.value("#E11D48"), alt.value("#2563EB")
                    ),
                    tooltip=[
                        column
                        for column in [
                            "capture",
                            "timestamp",
                            "label",
                            "stage1_anomaly",
                            "stage1_normalized_score",
                        ]
                        if column in scores
                    ],
                )
            )
            st.altair_chart(chart, width="stretch")
            st.dataframe(scores.head(500), hide_index=True, width="stretch")
        if history_path.exists():
            history = _read(history_path)
            if not history.empty:
                loss = history.melt(
                    id_vars="epoch",
                    value_vars=["train_loss", "validation_loss"],
                    var_name="series",
                    value_name="loss",
                ).dropna()
                st.altair_chart(
                    alt.Chart(loss)
                    .mark_line(point=True)
                    .encode(x="epoch:Q", y="loss:Q", color="series:N"),
                    width="stretch",
                )
    with tabs[1]:
        if evidence_path.exists():
            evidence = _read(evidence_path)
            st.dataframe(evidence, hide_index=True, width="stretch")
    with tabs[2]:
        if importance_path.exists():
            importance = _read(importance_path).head(25)
            st.altair_chart(
                alt.Chart(importance)
                .mark_bar(color="#8B1E3F")
                .encode(
                    y=alt.Y("feature:N", sort="-x"),
                    x="importance:Q",
                    tooltip=["feature", "importance"],
                ),
                width="stretch",
            )
            st.dataframe(importance, hide_index=True, width="stretch")
        if confusion_path.exists():
            st.subheader("Confusion matrix", anchor=False)
            st.dataframe(_read(confusion_path), hide_index=True, width="stretch")
    with tabs[3]:
        predictions = model_root / "pipeline" / "end-to-end-predictions.parquet"
        if predictions.exists():
            st.dataframe(_read(predictions).head(1000), hide_index=True, width="stretch")
        st.json(summary.get("onnx_exports", {}))


def _files_view(run: Path, catalog: pd.DataFrame, protocol: str | None) -> None:
    scoped = catalog if protocol is None else catalog[catalog["protocol"].fillna("").eq(protocol)]
    st.caption("هر جدول و نمودار این داشبورد از همین فایل‌های نسخه‌دار خوانده می‌شود.")
    st.dataframe(scoped.sort_values(["kind", "path"]), hide_index=True, width="stretch")


def render_workbench_legacy() -> None:
    """Render a simple navigation surface over persisted assets, not transient UI state."""
    st.title("سامانه تحلیل ناهنجاری شبکه", anchor=False)
    st.caption("PCAP-first • نگاشت قابل ممیزی • مدل دو مرحله‌ای • اجرای CPU-first")
    with st.sidebar:
        st.header("زمینه‌ی اجرا")
        root_text = st.text_input("پوشه‌ی خروجی‌ها", value="artifacts")
        artifact_root = Path(root_text)
        if st.button("بازخوانی فایل‌ها", icon=":material/refresh:"):
            _table.clear()
            _json.clear()
            st.rerun()
        runs = _runs(artifact_root)
        if not runs:
            st.warning("هنوز dataset run پیدا نشد. ابتدا `anomaly dataset run` را اجرا کنید.")
            return
        run = st.selectbox(
            "اجرای دیتاست",
            runs,
            format_func=lambda item: (
                item.relative_to(artifact_root).as_posix() if item != artifact_root else item.name
            ),
        )
        catalog = _catalog(run)
        protocols = sorted(item for item in catalog["protocol"].dropna().unique() if item)
        protocol = st.selectbox("پروتکل", ["همه", *protocols])
        protocol = None if protocol == "همه" else protocol
    areas = [
        "تابلوی کل سامانه",
        "نمای کلی",
        "داده و نگاشت",
        "فیچرها",
        "پروفایل و پیش‌پردازش",
        "مدل‌ها",
        "فایل‌ها",
    ]
    area = st.segmented_control("بخش", areas, default=areas[0], selection_mode="single")
    if area == "تابلوی کل سامانه":
        _system_observatory(run, catalog)
    elif area == "نمای کلی":
        _overview(run, catalog, protocol)
    elif area == "داده و نگاشت":
        _mapping_view(run, protocol)
    elif area == "فیچرها":
        _features_view(run, catalog, protocol)
    elif area == "پروفایل و پیش‌پردازش":
        _profile_view(artifact_root, protocol)
        report = sorted(run.glob("models/**/reports/preprocessing-comparison.parquet"))
        if report:
            st.subheader("مقایسه‌ی قبل و بعد از پیش‌پردازش", anchor=False)
            comparison = _read(report[0])
            selected_feature = st.selectbox(
                "فیچر برای مقایسه", sorted(comparison["feature"].unique())
            )
            selected = comparison[comparison["feature"].eq(selected_feature)]
            st.altair_chart(
                alt.Chart(selected)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x="stage:N",
                    y="iqr:Q",
                    color="stage:N",
                    tooltip=list(selected.columns),
                ),
                width="stretch",
            )
            st.dataframe(comparison, hide_index=True, width="stretch")
    elif area == "مدل‌ها":
        _models_view(run, artifact_root, protocol)
    else:
        _files_view(run, catalog, protocol)


# The legacy surface above is intentionally retained for backwards-compatible imports.
# The public entry point below is the Persian, Plotly-based workbench.
PLOT_COLORS = {
    "navy": "#0B1530",
    "blue": "#2563EB",
    "sky": "#60A5FA",
    "violet": "#7C3AED",
    "maroon": "#8B1E3F",
    "green": "#22C55E",
    "red": "#E11D48",
    # Plotly needs explicit canvas colours.  Keeping these darker than the app
    # background makes every chart read as a recessed analysis surface instead
    # of a white document embedded in a dark dashboard.
    "surface": "#171C3A",
    "plot_surface": "#25172F",
    "grid": "#4A365B",
    "muted": "#B9C5E2",
    "text": "#F8FAFC",
}
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": False, "responsive": True}
KIND_NAMES = {
    "dataset_inventory": "فهرست ورودی‌ها",
    "feature_records": "فیچرهای خام PCAP",
    "feature_manifest": "شناسنامه استخراج",
    "feature_quality": "کیفیت فیچر",
    "feature_overview": "آمار فیچر",
    "flow_label_mapping": "نگاشت Flow به CSV",
    "mapping_audit": "ممیزی نگاشت",
    "mapping_overview": "خلاصه نگاشت",
    "labelled_feature_records": "فیچرهای برچسب‌خورده",
    "attack_label_distribution": "توازن کلاس حمله",
}


def _enable_rtl() -> None:
    """Apply the requested RTL direction without changing persisted data or chart semantics."""
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { direction: rtl; }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label { text-align: right; }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { text-align: right; }
        [data-testid="stDataFrame"] { direction: rtl; }
        [data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre {
            direction: ltr; text-align: left;
        }
        /* Keep Streamlit's own toolbar LTR. Applying RTL to it shifts its
           controls into each other and was the source of the untidy banner. */
        [data-testid="stHeader"], [data-testid="stToolbar"] { direction: ltr; }
        .st-key-app-banner {
            background: linear-gradient(112deg, rgba(42, 18, 63, .94), rgba(11, 21, 48, .96));
            border: 1px solid rgba(192, 132, 252, .34);
            border-radius: 18px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 12px 28px rgba(0,0,0,.18);
            padding: .35rem 1rem .2rem;
            margin-bottom: .35rem;
        }
        .st-key-app-banner h1 { margin: 0 0 .15rem; }
        .st-key-app-banner p { margin-bottom: .2rem; }
        [data-testid="stPlotlyChart"] {
            background: linear-gradient(135deg, rgba(139, 30, 63, .20), rgba(23, 28, 58, .90));
            border: 1px solid rgba(192, 132, 252, .25);
            border-radius: 16px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.055), inset 0 -12px 28px rgba(0,0,0,.14);
            padding: .3rem .4rem .1rem;
        }
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .plot-container { border-radius: 12px; overflow: hidden; }
        /* The main navigation is one uninterrupted neon rail. Individual
           options deliberately have no boxed outline, so RTL labels do not
           look like disconnected tabs. The requested maroon hover gives a
           clear interaction affordance without changing the dark theme. */
        .st-key-dashboard-area [data-testid="stSegmentedControl"] [role="radiogroup"],
        .st-key-dashboard-area [data-baseweb="button-group"] {
            width: 100%;
            padding: 4px;
            gap: 2px;
            border: 1px solid rgba(192, 132, 252, .88) !important;
            border-radius: 16px;
            background: linear-gradient(110deg, rgba(31, 20, 64, .94), rgba(11, 21, 48, .98));
            box-shadow: 0 0 0 1px rgba(124,58,237,.18), 0 0 18px rgba(124,58,237,.28), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .st-key-dashboard-area [data-testid="stSegmentedControl"] label,
        .st-key-dashboard-area [data-baseweb="button-group"] label {
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            border-radius: 12px !important;
            background: transparent !important;
            transition: background .16s ease, box-shadow .16s ease, color .16s ease;
        }
        .st-key-dashboard-area [data-testid="stSegmentedControl"] label:hover,
        .st-key-dashboard-area [data-baseweb="button-group"] label:hover {
            background: linear-gradient(112deg, rgba(139,30,63,.86), rgba(98,20,66,.86)) !important;
            color: #fff !important;
            box-shadow: inset 0 0 0 1px rgba(251,113,133,.70), 0 0 15px rgba(225,29,72,.35) !important;
        }
        .st-key-dashboard-area [data-testid="stSegmentedControl"] label:has(input:checked),
        .st-key-dashboard-area [data-baseweb="button-group"] label:has(input:checked) {
            background: linear-gradient(112deg, rgba(124,58,237,.92), rgba(74,34,140,.92)) !important;
            color: #fff !important;
            box-shadow: 0 0 14px rgba(167,139,250,.45) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot(figure: go.Figure, title: str | None = None, height: int = 360) -> None:
    """Render an RTL-aware Plotly chart on the dashboard's recessed dark surface."""
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 28, "r": 28, "t": 70, "b": 30},
        font={"family": "sans-serif", "color": PLOT_COLORS["text"], "size": 13},
        paper_bgcolor=PLOT_COLORS["surface"],
        plot_bgcolor=PLOT_COLORS["plot_surface"],
        title={
            "text": title or "",
            "x": 0.98,
            "xanchor": "right",
            "font": {"size": 15, "color": PLOT_COLORS["text"]},
        },
        legend={
            "orientation": "h",
            "y": 1.12,
            "x": 1,
            "xanchor": "right",
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "align": "right",
            "bgcolor": "#101832",
            "bordercolor": PLOT_COLORS["violet"],
            "font": {"color": PLOT_COLORS["text"]},
        },
    )
    figure.update_xaxes(
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor="#6B4B76",
        linecolor="#6B4B76",
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["text"]},
    )
    figure.update_yaxes(
        gridcolor=PLOT_COLORS["grid"],
        zerolinecolor="#6B4B76",
        linecolor="#6B4B76",
        tickfont={"color": PLOT_COLORS["muted"]},
        title_font={"color": PLOT_COLORS["text"]},
    )
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, theme=None)


def _download(frame: pd.DataFrame, label: str, filename: str) -> None:
    """Expose each analysis table outside the UI without duplicating the data source."""
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        icon=":material/download:",
    )


@st.cache_data(max_entries=96, show_spinner=False)
def _sample_table(path_text: str, modified_ns: int, maximum_rows: int) -> pd.DataFrame:
    """Read a bounded, evenly-spread Parquet sample for responsive Plotly views."""
    del modified_ns
    path = Path(path_text)
    if path.suffix.lower() not in {".parquet", ".pq"}:
        return read_table(path).head(maximum_rows)
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    if rows <= maximum_rows:
        return read_table(path)
    groups = np.unique(
        np.linspace(0, parquet.num_row_groups - 1, num=min(12, parquet.num_row_groups), dtype=int)
    )
    per_group = max(1, maximum_rows // len(groups))
    pieces: list[pd.DataFrame] = []
    for group in groups:
        piece = parquet.read_row_group(int(group)).to_pandas()
        if len(piece) > per_group:
            positions = np.linspace(0, len(piece) - 1, num=per_group, dtype=int)
            piece = piece.iloc[positions]
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _read_sample(path: Path, maximum_rows: int = 6000) -> pd.DataFrame:
    return _sample_table(str(path), path.stat().st_mtime_ns, maximum_rows)


def _sample_many(paths: list[Path], maximum_rows: int = 8000) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    per_path = max(250, maximum_rows // len(paths))
    frames = [_read_sample(path, per_path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _kind_name(value: object) -> str:
    return KIND_NAMES.get(str(value), str(value).replace("_", " "))


def _mapping_report(run: Path, protocol: str | None) -> pd.DataFrame:
    path = run / "reports" / "mapping-audit.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = _read(path).copy()
    if protocol:
        frame = frame[frame["protocol"].astype("string").str.casefold().eq(protocol.casefold())]
    return frame.reset_index(drop=True)


def _quality_report(run: Path, protocol: str | None) -> pd.DataFrame:
    paths = sorted((run / "reports" / "features").rglob("*-quality.parquet"))
    frames = [_read(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    quality = pd.concat(frames, ignore_index=True, sort=False)
    if protocol:
        quality = quality[
            quality["protocol"].astype("string").str.casefold().eq(protocol.casefold())
        ]
    return quality.reset_index(drop=True)


def _feature_report(run: Path, protocol: str | None) -> pd.DataFrame:
    paths = sorted((run / "reports" / "features").rglob("*-overview.parquet"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = _read(path).copy()
        relative = path.relative_to(run).parts
        frame["data_group"] = "نرمال" if "benign" in relative else "حمله"
        frame["source"] = path.relative_to(run).as_posix()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    report = pd.concat(frames, ignore_index=True, sort=False)
    if protocol:
        report = report[report["protocol"].astype("string").str.casefold().eq(protocol.casefold())]
    return report.reset_index(drop=True)


def _record_samples(catalog: pd.DataFrame, run: Path, protocol: str | None) -> pd.DataFrame:
    """Return bounded benign/attack feature samples with a display-only group column."""
    benign_paths = _paths(catalog, run, "feature_records", protocol)
    benign = _sample_many(benign_paths, 5000)
    if not benign.empty:
        benign["data_group"] = "نرمال"
    attack_paths = _paths(catalog, run, "labelled_feature_records", protocol)
    attack = _sample_many(attack_paths, 7000)
    if not attack.empty:
        attack["data_group"] = np.where(
            attack.get("is_attack", pd.Series(False, index=attack.index)).fillna(False),
            "حملهٔ برچسب‌خورده",
            "رکورد نامشخص",
        )
    frames = [frame for frame in [benign, attack] if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _numeric_columns(frame: pd.DataFrame, limit: int = 24) -> list[str]:
    ignored = {
        "src_port",
        "dst_port",
        "csv_row",
        "record_id",
        "is_attack",
        "mapping_accepted",
        "stage1_anomaly",
        "true_is_attack",
    }
    candidates = [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in ignored and frame[column].nunique(dropna=True) > 1
    ]
    scored = sorted(
        candidates,
        key=lambda column: float(pd.to_numeric(frame[column], errors="coerce").var() or 0),
        reverse=True,
    )
    return scored[:limit]


def _two_stage_figure() -> go.Figure:
    """Visual model/data contract shared by feature-profile and model pages."""
    labels = [
        "PCAP نرمال",
        "PCAP حمله",
        "CSV لیبل",
        "استخراج‌گر پروتکل",
        "نگاشت ممیزی‌شده",
        "فیچرهای خام جداگانه",
        "پروفایل فیچر نسخه‌دار",
        "پیش‌پردازش مشترک",
        "LSTM Autoencoder",
        "Isolation Forest",
        "گیت تشخیص ناهنجاری",
        "Random Forest نوع حمله",
        "خروجی قابل‌توضیح + ONNX",
    ]
    source = [0, 1, 1, 2, 3, 3, 5, 6, 7, 7, 8, 9, 10, 11]
    target = [3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 10, 11, 12]
    values = [5, 5, 3, 3, 8, 8, 8, 8, 4, 4, 4, 4, 3, 4]
    colors = [
        PLOT_COLORS["blue"],
        PLOT_COLORS["maroon"],
        PLOT_COLORS["sky"],
        PLOT_COLORS["violet"],
        PLOT_COLORS["green"],
        PLOT_COLORS["violet"],
        PLOT_COLORS["blue"],
        PLOT_COLORS["maroon"],
        PLOT_COLORS["violet"],
        PLOT_COLORS["blue"],
        PLOT_COLORS["maroon"],
        PLOT_COLORS["green"],
        PLOT_COLORS["sky"],
    ]
    return go.Figure(
        go.Sankey(
            arrangement="snap",
            node={"label": labels, "pad": 18, "thickness": 18, "color": colors},
            link={
                "source": source,
                "target": target,
                "value": values,
                "color": "rgba(96,165,250,.25)",
            },
        )
    )


def _system_records(catalog: pd.DataFrame, run: Path, maximum_rows: int = 16_000) -> pd.DataFrame:
    """Load a balanced, protocol-separated sample for the all-system observatory.

    Attack rows intentionally come from the labelled copy, while normal rows
    come only from the benign source.  This prevents the raw attack extract
    from being accidentally presented as normal traffic in cross-protocol EDA.
    """
    benign_paths = [
        run / value
        for value in catalog.loc[
            catalog["kind"].eq("feature_records") & catalog["split"].eq("benign"), "path"
        ].astype(str)
        if (run / value).exists()
    ]
    attack_paths = [
        run / value
        for value in catalog.loc[catalog["kind"].eq("labelled_feature_records"), "path"].astype(str)
        if (run / value).exists()
    ]
    normal = _sample_many(benign_paths, maximum_rows // 2)
    if not normal.empty:
        normal["data_group"] = "نرمال"
    attacks = _sample_many(attack_paths, maximum_rows // 2)
    if not attacks.empty:
        attacks["data_group"] = np.where(
            attacks.get("is_attack", pd.Series(True, index=attacks.index)).fillna(False),
            "حملهٔ برچسب‌خورده",
            "رکورد حمله با برچسب نامشخص",
        )
    frames = [frame for frame in (normal, attacks) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    records = pd.concat(frames, ignore_index=True, sort=False)
    if "timestamp" in records:
        records["timestamp"] = pd.to_datetime(records["timestamp"], errors="coerce", utc=True)
    return records


def _time_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a legible UTC bucket whose size follows the observed time span."""
    result = frame.copy()
    if "timestamp" not in result:
        return result
    timestamps = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    valid = timestamps.dropna()
    if valid.empty:
        return result
    span_seconds = (valid.max() - valid.min()).total_seconds()
    frequency = "D" if span_seconds > 60 * 86_400 else "6h" if span_seconds > 7 * 86_400 else "h"
    result["بازهٔ زمانی UTC"] = timestamps.dt.floor(frequency)
    return result


def _system_model_summaries(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect model and persisted resource summaries without assuming every protocol was trained."""
    rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for summary_path in sorted(run.glob("models/**/training-summary.json")):
        try:
            summary = _read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        stage1 = summary.get("stage1", {})
        stage2 = summary.get("stage2", {})
        rows.append(
            {
                "پروتکل": summary.get("protocol", summary_path.parent.name),
                "مدل": summary_path.parent.name,
                "Stage 1 FPR": stage1.get("false_positive_rate"),
                "Stage 1 Recall": stage1.get("recall"),
                "Stage 1 F1": stage1.get("f1"),
                "Stage 2 Weighted F1": stage2.get("weighted_f1"),
                "ورودی transformed": stage1.get("input_features"),
                "زمان آموزش (ثانیه)": summary.get("training_seconds"),
                "رکورد ارزیابی Stage 1": stage1.get("records"),
                "رکورد آزمون Stage 2": stage2.get("test_records"),
                "مسیر": summary_path.parent.relative_to(run).as_posix(),
            }
        )
        resource_path = summary_path.parent / "reports" / "resource-usage.parquet"
        if resource_path.exists():
            resource = _read(resource_path).copy()
            resource["پروتکل"] = summary.get("protocol", summary_path.parent.name)
            resource["مدل"] = summary_path.parent.name
            resource_rows.append(resource)
    return pd.DataFrame(rows), (
        pd.concat(resource_rows, ignore_index=True, sort=False) if resource_rows else pd.DataFrame()
    )


def _system_observatory(run: Path, catalog: pd.DataFrame) -> None:
    """Render the all-protocol control room from persisted, portable artifacts.

    It intentionally does not mix raw protocol files: every aggregate retains a
    protocol/split column and each underlying table remains downloadable from
    the artefact catalogue.  Twenty-four charts and twelve tables are grouped
    into six short sections so a user can audit the whole system without losing
    the separate protocol pages.
    """
    st.subheader("تابلوی کل سامانه؛ همهٔ پروتکل‌ها", anchor=False)
    st.caption(
        "این نما همیشه همهٔ پروتکل‌های اجرای انتخاب‌شده را کنار هم می‌خواند. برای واکاوی عمیق "
        "یک پروتکل، همان پروتکل را از سایدبار انتخاب و بخش‌های دیگر را باز کنید. نمودارهای زمانی "
        "UTC هستند و دادهٔ packet-level روی نمونهٔ یکنواخت و محدود برای پاسخ‌گویی dashboard رسم می‌شود."
    )
    scoped = catalog.copy()
    scoped["protocol"] = scoped["protocol"].fillna("کلی")
    scoped["rows"] = pd.to_numeric(scoped["rows"], errors="coerce").fillna(0)
    record_catalog = scoped[
        scoped["kind"].isin(["feature_records", "labelled_feature_records"])
        & scoped["protocol"].ne("کلی")
    ].copy()
    protocol_health = (
        record_catalog.groupby(["protocol", "split", "kind"], as_index=False)
        .agg(فایل=("path", "count"), رکورد=("rows", "sum"), ستون=("columns", "max"))
        .sort_values(["protocol", "split", "kind"])
    )
    protocol_rollup = (
        protocol_health.groupby("protocol", as_index=False)
        .agg(فایل=("فایل", "sum"), رکورد=("رکورد", "sum"), بیشترین_ستون=("ستون", "max"))
        .sort_values("رکورد", ascending=False)
    )
    mapping = _mapping_report(run, None)
    labels_path = run / "reports" / "attack-label-distribution.parquet"
    labels = _read(labels_path) if labels_path.exists() else pd.DataFrame()
    feature_report = _feature_report(run, None)
    quality = _quality_report(run, None)
    records = _system_records(catalog, run)
    model_summary, resource_summary = _system_model_summaries(run)

    cards = st.columns(4)
    cards[0].metric("پروتکل‌های دیده‌شده", f"{protocol_rollup['protocol'].nunique():,}")
    cards[1].metric("فایل‌های PCAP-derived", f"{int(protocol_rollup['فایل'].sum()):,}")
    cards[2].metric("رکوردهای ثبت‌شده", f"{int(protocol_rollup['رکورد'].sum()):,}")
    cards[3].metric(
        "نگاشت پذیرفته‌شده",
        f"{int(mapping.get('accepted', pd.Series(dtype=bool)).sum()):,}/{len(mapping):,}",
    )

    st.subheader("۱. پوشش داده و دارایی‌های تولیدشده", anchor=False)
    # Table 01: exact cross-protocol source coverage.
    st.dataframe(protocol_rollup, hide_index=True, width="stretch")
    # Table 02: one row per protocol, split and durable output kind.
    st.dataframe(protocol_health, hide_index=True, width="stretch", height=280)
    left, right = st.columns(2)
    with left:
        _plot(
            px.bar(
                protocol_health,
                x="protocol",
                y="رکورد",
                color="split",
                pattern_shape="kind",
                barmode="stack",
                color_discrete_map={"benign": PLOT_COLORS["blue"], "attack": PLOT_COLORS["maroon"]},
            ),
            "۱. پوشش رکوردهای PCAP به تفکیک پروتکل، split و نوع خروجی",
        )
    with right:
        artifact_matrix = (
            scoped[scoped["protocol"].ne("کلی")]
            .groupby(["protocol", "kind"], as_index=False)
            .size()
        )
        _plot(
            px.bar(
                artifact_matrix,
                x="protocol",
                y="size",
                color="kind",
                barmode="stack",
                color_discrete_sequence=[
                    PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]
                ],
            ),
            "۲. تنوع artifactهای تولیدشده در هر پروتکل",
        )
    left, right = st.columns(2)
    with left:
        _plot(
            px.pie(
                protocol_rollup,
                names="protocol",
                values="رکورد",
                hole=0.56,
                color="protocol",
                color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
            ),
            "۳. سهم هر پروتکل از حجم رکوردهای ثبت‌شده",
        )
    with right:
        captures = (
            record_catalog.groupby(["protocol", "split"], as_index=False)
            .agg(Capture=("path", "count"), رکورد=("rows", "sum"))
        )
        _plot(
            px.scatter(
                captures,
                x="Capture",
                y="رکورد",
                color="protocol",
                symbol="split",
                size="رکورد",
                size_max=46,
                color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
            ),
            "۴. تعداد Capture در برابر حجم رکورد هر پروتکل/split",
        )

    st.subheader("۲. رفتار زمانی و جریان‌ها", anchor=False)
    time_summary = pd.DataFrame()
    if not records.empty and "timestamp" in records:
        timed = _time_bucket(records).dropna(subset=["timestamp", "بازهٔ زمانی UTC"]).copy()
        time_summary = (
            timed.groupby("protocol", as_index=False)
            .agg(
                شروع=("timestamp", "min"),
                پایان=("timestamp", "max"),
                ردیف=("timestamp", "size"),
                Flow=("flow_id", "nunique"),
                Capture=("capture", "nunique"),
            )
            .sort_values("شروع")
        )
        # Table 03: exact observed time ranges of the displayed packet sample.
        st.dataframe(time_summary, hide_index=True, width="stretch")
        traffic_time = timed.groupby(["بازهٔ زمانی UTC", "protocol"], as_index=False).size()
        group_time = timed.groupby(["بازهٔ زمانی UTC", "data_group"], as_index=False).size()
        flow_time = timed.groupby(["بازهٔ زمانی UTC", "protocol"], as_index=False).agg(
            Flow=("flow_id", "nunique"), Capture=("capture", "nunique")
        )
        left, right = st.columns(2)
        with left:
            _plot(
                px.line(
                    traffic_time,
                    x="بازهٔ زمانی UTC",
                    y="size",
                    color="protocol",
                    markers=True,
                    color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
                ),
                "۵. حجم ترافیک نمونه در زمان؛ مقایسهٔ همهٔ پروتکل‌ها",
            )
        with right:
            _plot(
                px.area(
                    group_time,
                    x="بازهٔ زمانی UTC",
                    y="size",
                    color="data_group",
                    groupnorm=None,
                    color_discrete_map={"نرمال": PLOT_COLORS["blue"], "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"]},
                ),
                "۶. تغییر زمانی نرمال در برابر حملهٔ برچسب‌خورده",
            )
        left, right = st.columns(2)
        with left:
            _plot(
                px.line(
                    flow_time,
                    x="بازهٔ زمانی UTC",
                    y="Flow",
                    color="protocol",
                    markers=True,
                    color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
                ),
                "۷. تعداد Flow یکتا در هر بازهٔ زمانی",
            )
        with right:
            timeline = time_summary.melt(
                id_vars="protocol", value_vars=["شروع", "پایان"], var_name="کرانه", value_name="زمان"
            )
            _plot(
                px.scatter(
                    timeline,
                    x="زمان",
                    y="protocol",
                    color="کرانه",
                    symbol="کرانه",
                    color_discrete_map={"شروع": PLOT_COLORS["green"], "پایان": PLOT_COLORS["maroon"]},
                ),
                "۸. بازهٔ زمانی مشاهده‌شده برای هر پروتکل؛ شروع تا پایان",
            )
        # Table 04: time buckets make temporal coverage auditable outside charts.
        st.dataframe(traffic_time.sort_values("بازهٔ زمانی UTC", ascending=False).head(200), hide_index=True, width="stretch", height=260)
    else:
        st.info("برای نمودارهای زمانی باید timestamp معتبر در feature records وجود داشته باشد.")

    if not records.empty:
        st.subheader("۳. مشخصات packet و flow در همهٔ پروتکل‌ها", anchor=False)
        numeric_records = records.copy()
        for column in [
            "packet_length", "payload_entropy", "packet_rate", "flow_duration", "source_packet_rate", "burstiness"
        ]:
            if column in numeric_records:
                numeric_records[column] = pd.to_numeric(numeric_records[column], errors="coerce")
        left, right = st.columns(2)
        with left:
            _plot(
                px.violin(
                    numeric_records.dropna(subset=["packet_length"]),
                    x="protocol",
                    y="packet_length",
                    color="data_group",
                    box=True,
                    points="outliers",
                    color_discrete_map={"نرمال": PLOT_COLORS["blue"], "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"]},
                ),
                "۹. توزیع طول packet در پروتکل‌ها و گروه داده",
            )
        with right:
            _plot(
                px.violin(
                    numeric_records.dropna(subset=["payload_entropy"]),
                    x="protocol",
                    y="payload_entropy",
                    color="data_group",
                    box=True,
                    points=False,
                    color_discrete_map={"نرمال": PLOT_COLORS["blue"], "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"]},
                ),
                "۱۰. توزیع آنتروپی payload؛ شاخص ساختار/تصادفی‌بودن محتوا",
            )
        left, right = st.columns(2)
        with left:
            _plot(
                px.box(
                    numeric_records.dropna(subset=["packet_rate"]),
                    x="protocol",
                    y="packet_rate",
                    color="data_group",
                    points="outliers",
                    color_discrete_map={"نرمال": PLOT_COLORS["blue"], "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"]},
                ),
                "۱۱. نرخ packet در Flowها؛ مقایسهٔ فشار ترافیک",
            )
        with right:
            _plot(
                px.box(
                    numeric_records.dropna(subset=["flow_duration"]),
                    x="protocol",
                    y="flow_duration",
                    color="data_group",
                    points="outliers",
                    color_discrete_map={"نرمال": PLOT_COLORS["blue"], "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"]},
                ),
                "۱۲. طول عمر Flowها در پروتکل‌های مختلف",
            )
        left, right = st.columns(2)
        with left:
            _plot(
                px.box(
                    numeric_records.dropna(subset=["source_packet_rate"]),
                    x="protocol",
                    y="source_packet_rate",
                    color="data_group",
                    points="outliers",
                    color_discrete_map={"نرمال": PLOT_COLORS["blue"], "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"]},
                ),
                "۱۳. نرخ ارسال مبدأ؛ نشانهٔ burst و اسکن/سیل احتمالی",
            )
        with right:
            _plot(
                px.scatter(
                    numeric_records.dropna(subset=["packet_length", "payload_entropy"]),
                    x="packet_length",
                    y="payload_entropy",
                    color="protocol",
                    symbol="data_group",
                    size="burstiness" if "burstiness" in numeric_records else None,
                    size_max=26,
                    opacity=0.62,
                    render_mode="svg",
                    color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
                ),
                "۱۴. رابطهٔ طول packet و آنتروپی payload؛ الگوی چندمتغیره",
            )
        transport_counts = numeric_records.groupby(["protocol", "transport"], as_index=False).size()
        direction_counts = numeric_records.assign(
            جهت=np.where(numeric_records.get("direction", 0).fillna(0).astype(int).eq(1), "جهت نرمال‌شده ۱", "جهت نرمال‌شده ۰")
        ).groupby(["protocol", "جهت"], as_index=False).size()
        request_counts = numeric_records.assign(
            سمت_سرویس=np.where(numeric_records.get("is_request_direction", 0).fillna(0).astype(int).eq(1), "به‌سوی سرویس", "خارج از سرویس")
        ).groupby(["protocol", "سمت_سرویس"], as_index=False).size()
        left, middle, right = st.columns(3)
        with left:
            _plot(
                px.bar(transport_counts, x="protocol", y="size", color="transport", barmode="stack"),
                "۱۵. سهم TCP/UDP در هر پروتکل",
                360,
            )
        with middle:
            _plot(
                px.bar(direction_counts, x="protocol", y="size", color="جهت", barmode="group"),
                "۱۶. توازن دو جهت نرمال‌شدهٔ Flow",
                360,
            )
        with right:
            _plot(
                px.bar(request_counts, x="protocol", y="size", color="سمت_سرویس", barmode="group"),
                "۱۷. حرکت packet به‌سوی endpoint سرویس",
                360,
            )
        # Table 05: direct packet/flow profiles complement the distributions.
        packet_profile = numeric_records.groupby(["protocol", "data_group"], as_index=False).agg(
            ردیف=("protocol", "size"),
            Flow=("flow_id", "nunique"),
            طول_میانهٔ_packet=("packet_length", "median"),
            آنتروپی_میانه=("payload_entropy", "median"),
            نرخ_میانهٔ_packet=("packet_rate", "median"),
            طول_میانهٔ_flow=("flow_duration", "median"),
        )
        st.dataframe(packet_profile, hide_index=True, width="stretch")
        # Table 06: source/capture volumes retain the raw evidence behind temporal plots.
        capture_profile = numeric_records.groupby(["protocol", "data_group", "capture"], as_index=False).agg(
            ردیف=("capture", "size"), Flow=("flow_id", "nunique"), مبدأ=("src_ip", "nunique"), مقصد=("dst_ip", "nunique")
        ).sort_values("ردیف", ascending=False)
        st.dataframe(capture_profile.head(250), hide_index=True, width="stretch", height=300)

    st.subheader("۴. نگاشت CSV، کلاس‌های حمله و قابلیت اتکا", anchor=False)
    if not mapping.empty:
        mapping_display = mapping.copy()
        mapping_display["وضعیت"] = np.where(mapping_display["accepted"], "پذیرفته‌شده", "نیازمند بررسی")
        # Table 07: auditable mapping decision per attack capture.
        st.dataframe(mapping_display.sort_values(["protocol", "accepted", "match_rate"]), hide_index=True, width="stretch", height=320)
        left, right = st.columns(2)
        with left:
            acceptance = mapping_display.groupby(["protocol", "وضعیت"], as_index=False).size()
            _plot(
                px.bar(
                    acceptance,
                    x="protocol",
                    y="size",
                    color="وضعیت",
                    barmode="stack",
                    color_discrete_map={"پذیرفته‌شده": PLOT_COLORS["green"], "نیازمند بررسی": PLOT_COLORS["red"]},
                ),
                "۱۸. پذیرش/رد نگاشت CSV به PCAP در هر پروتکل",
            )
        with right:
            _plot(
                px.scatter(
                    mapping_display,
                    x="match_rate",
                    y="mean_match_confidence",
                    color="وضعیت",
                    symbol="protocol",
                    hover_name="capture",
                    hover_data=["selected_csv", "selected_offset_seconds"],
                    color_discrete_map={"پذیرفته‌شده": PLOT_COLORS["green"], "نیازمند بررسی": PLOT_COLORS["red"]},
                ),
                "۱۹. کیفیت هم‌زمان نگاشت: match rate در برابر confidence",
            )
        _plot(
            px.histogram(
                mapping_display,
                x="selected_offset_seconds",
                color="protocol",
                nbins=24,
                barmode="group",
                color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
            ),
            "۲۰. توزیع offset زمانی جبران‌شده بین CSV و PCAP",
        )
    if not labels.empty:
        label_view = labels[labels["label"].astype("string").str.casefold().ne("unknown")].copy()
        if not label_view.empty:
            label_balance = label_view.groupby(["protocol", "label", "mapping_accepted"], as_index=False)["records"].sum()
            # Table 08: class balance uses only PCAP-derived record counts.
            st.dataframe(label_balance.sort_values("records", ascending=False), hide_index=True, width="stretch", height=300)
            _plot(
                px.bar(
                    label_balance,
                    x="records",
                    y="label",
                    color="protocol",
                    facet_col="mapping_accepted",
                    orientation="h",
                    color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
                ),
                "۲۱. توازن نوع حمله به‌ازای پروتکل و پذیرش نگاشت",
                max(410, label_balance["label"].nunique() * 27),
            )

    st.subheader("۵. سلامت featureها در مقیاس کل سامانه", anchor=False)
    if not feature_report.empty:
        feature_aggregate = feature_report.groupby(["protocol", "feature"], as_index=False).agg(
            availability=("availability", "median"),
            iqr=("iqr", "median"),
            std=("std", "median"),
            zero_ratio=("zero_ratio", "median"),
            unique_values=("unique_values", "median"),
        )
        top_features = (
            feature_aggregate.groupby("feature", as_index=False)["availability"].median().nlargest(24, "availability")["feature"]
        )
        heat = feature_aggregate[feature_aggregate["feature"].isin(top_features)]
        availability_matrix = heat.pivot_table(index="feature", columns="protocol", values="availability", aggfunc="median", fill_value=0)
        zero_matrix = heat.pivot_table(index="feature", columns="protocol", values="zero_ratio", aggfunc="median", fill_value=0)
        left, right = st.columns(2)
        with left:
            _plot(
                go.Figure(
                    go.Heatmap(
                        z=availability_matrix.to_numpy(), x=availability_matrix.columns, y=availability_matrix.index,
                        text=availability_matrix.round(2).to_numpy(), texttemplate="%{text}", colorscale="Blues",
                    )
                ),
                "۲۲. heatmap پوشش featureهای مشترک در پروتکل‌ها",
                max(460, len(availability_matrix) * 21),
            )
        with right:
            _plot(
                go.Figure(
                    go.Heatmap(
                        z=zero_matrix.to_numpy(), x=zero_matrix.columns, y=zero_matrix.index,
                        text=zero_matrix.round(2).to_numpy(), texttemplate="%{text}", colorscale="Reds",
                    )
                ),
                "۲۳. heatmap صفر/تنک‌بودن همان featureها",
                max(460, len(zero_matrix) * 21),
            )
        _plot(
            px.scatter(
                feature_aggregate,
                x="availability",
                y="iqr",
                size="unique_values",
                color="protocol",
                hover_name="feature",
                color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
            ),
            "۲۴. پوشش در برابر تغییرپذیری featureها؛ معیار انتخاب برای مدل",
        )
        # Table 09: feature candidates that are observable and variable.
        st.dataframe(
            feature_aggregate.sort_values(["availability", "iqr"], ascending=[False, False]).head(160),
            hide_index=True,
            width="stretch",
            height=320,
        )
        # Table 10: sparse features give a concrete reason to avoid a profile choice.
        st.dataframe(
            feature_aggregate.sort_values(["zero_ratio", "availability"], ascending=[False, True]).head(160),
            hide_index=True,
            width="stretch",
            height=280,
        )
    if not quality.empty:
        quality_summary = quality.groupby(["protocol", "extraction_status", "model_usable"], as_index=False).size()
        left, right = st.columns(2)
        with left:
            _plot(
                px.bar(
                    quality_summary,
                    x="protocol",
                    y="size",
                    color="extraction_status",
                    barmode="stack",
                    color_discrete_sequence=[PLOT_COLORS["green"], PLOT_COLORS["red"], PLOT_COLORS["violet"]],
                ),
                "۲۵. وضعیت استخراج featureها در هر پروتکل",
            )
        with right:
            _plot(
                px.scatter(
                    quality,
                    x="observed_ratio",
                    y="unique_count",
                    size="cost" if pd.api.types.is_numeric_dtype(quality.get("cost")) else None,
                    color="protocol",
                    symbol="model_usable",
                    hover_name="name",
                    color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
                ),
                "۲۶. مشاهده‌پذیری و تنوع featureها؛ علامت شکل = قابلیت استفادهٔ مدل",
            )
        # Table 11: exact quality reasons preserved by the feature pipeline.
        st.dataframe(
            quality.sort_values(["model_usable", "observed_ratio"], ascending=[False, False]),
            hide_index=True,
            width="stretch",
            height=340,
        )

    st.subheader("۶. مدل، منابع و آمادگی استقرار", anchor=False)
    if not model_summary.empty:
        metric_long = model_summary.melt(
            id_vars=["پروتکل", "مدل"],
            value_vars=["Stage 1 FPR", "Stage 1 Recall", "Stage 1 F1", "Stage 2 Weighted F1"],
            var_name="شاخص",
            value_name="مقدار",
        )
        left, right = st.columns(2)
        with left:
            _plot(
                px.bar(
                    metric_long,
                    x="پروتکل",
                    y="مقدار",
                    color="شاخص",
                    barmode="group",
                    range_y=[0, 1],
                    color_discrete_sequence=[PLOT_COLORS["red"], PLOT_COLORS["green"], PLOT_COLORS["blue"], PLOT_COLORS["violet"]],
                ),
                "۲۷. مقایسهٔ معیارهای Stage 1 و Stage 2 میان مدل‌های موجود",
            )
        with right:
            _plot(
                px.scatter(
                    model_summary,
                    x="ورودی transformed",
                    y="زمان آموزش (ثانیه)",
                    size="رکورد ارزیابی Stage 1",
                    color="پروتکل",
                    hover_name="مدل",
                    color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["violet"], PLOT_COLORS["maroon"], PLOT_COLORS["sky"]],
                ),
                "۲۸. هزینهٔ آموزش در برابر ابعاد ورودی و حجم ارزیابی",
            )
        # Table 12: model scorecard is a cross-protocol deployment gate.
        st.dataframe(
            model_summary.sort_values("زمان آموزش (ثانیه)", ascending=False),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("هنوز مدل ذخیره‌شده‌ای برای مقایسهٔ بین‌پروتکلی وجود ندارد؛ پس از آموزش هر پروتکل این بخش خودکار پر می‌شود.")
    if not resource_summary.empty:
        _plot(
            px.bar(
                resource_summary,
                x="پروتکل",
                y="process_rss_mb",
                color="point",
                barmode="group",
                hover_data=["memory_percent", "available_memory_gb"],
                color_discrete_map={"before_training": PLOT_COLORS["blue"], "after_training": PLOT_COLORS["maroon"]},
            ),
            "۲۹. مصرف RAM پردازش قبل و بعد از آموزش مدل‌ها",
        )
    _resource_monitor()


def _overview_enhanced(run: Path, catalog: pd.DataFrame, protocol: str | None) -> None:
    mapping = _mapping_report(run, protocol)
    scoped = catalog if protocol is None else catalog[catalog["protocol"].fillna("").eq(protocol)]
    feature_rows = scoped.loc[scoped["kind"].eq("feature_records"), "rows"].fillna(0).sum()
    labelled_rows = (
        scoped.loc[scoped["kind"].eq("labelled_feature_records"), "rows"].fillna(0).sum()
    )
    accepted = int(mapping.get("accepted", pd.Series(dtype=bool)).sum())
    cards = st.columns(4)
    cards[0].metric("فایل‌های خروجی ثبت‌شده", f"{len(scoped):,}")
    cards[1].metric("رکوردهای فیچر از PCAP", f"{int(feature_rows):,}")
    cards[2].metric("رکوردهای حمله با لیبل", f"{int(labelled_rows):,}")
    cards[3].metric("نگاشت‌های پذیرفته‌شده", f"{accepted}/{len(mapping)}")

    st.subheader("راهنمای سریع این اجرا", anchor=False)
    guide = pd.DataFrame(
        [
            {
                "گام": "۱. ورودی",
                "معنا": "PCAPهای نرمال و حمله، جدا برای هر پروتکل",
                "خروجی قابل‌استفاده": "manifests/input-inventory.json",
            },
            {
                "گام": "۲. استخراج",
                "معنا": "فیچرهای خام فقط از PCAP استخراج شده‌اند",
                "خروجی قابل‌استفاده": "features/<split>/<protocol>/.../records.parquet",
            },
            {
                "گام": "۳. نگاشت",
                "معنا": "CSV فقط با Flow، زمان و endpoint به PCAP وصل شده است",
                "خروجی قابل‌استفاده": "mappings/... و reports/mapping-audit.parquet",
            },
            {
                "گام": "۴. آموزش",
                "معنا": "پس از انتخاب پروفایل، مدل دو مرحله‌ای و گزارش‌ها تولید می‌شود",
                "خروجی قابل‌استفاده": "models/<protocol>-<profile>/",
            },
        ]
    )
    st.dataframe(guide, hide_index=True, width="stretch")

    first, second = st.columns(2)
    with first:
        by_kind = (
            scoped.assign(نوع=scoped["kind"].map(_kind_name))
            .groupby("نوع", as_index=False)
            .agg(فایل=("path", "count"), رکورد=("rows", "sum"))
            .sort_values("فایل", ascending=True)
        )
        _plot(
            px.bar(
                by_kind,
                x="فایل",
                y="نوع",
                orientation="h",
                color="رکورد",
                color_continuous_scale="Purples",
            ),
            "چه چیزهایی تولید شده‌اند؟ تعداد فایل و تعداد رکورد",
        )
    with second:
        records = scoped[
            scoped["kind"].isin(["feature_records", "labelled_feature_records"])
        ].copy()
        if not records.empty:
            records["گروه"] = np.where(
                records["kind"].eq("feature_records"), "فیچر خام PCAP", "فیچر با لیبل"
            )
            aggregate = records.groupby(
                ["protocol", "split", "گروه"], dropna=False, as_index=False
            )["rows"].sum()
            _plot(
                px.bar(
                    aggregate,
                    x="protocol",
                    y="rows",
                    color="گروه",
                    barmode="stack",
                    color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["maroon"]],
                ),
                "پوشش رکوردها به تفکیک پروتکل و مرحله",
            )

    if not mapping.empty:
        left, right = st.columns(2)
        with left:
            acceptance = mapping.groupby(["protocol", "accepted"], as_index=False).size()
            acceptance["وضعیت"] = np.where(acceptance["accepted"], "پذیرفته‌شده", "نیازمند بررسی")
            _plot(
                px.bar(
                    acceptance,
                    x="protocol",
                    y="size",
                    color="وضعیت",
                    barmode="group",
                    color_discrete_map={
                        "پذیرفته‌شده": PLOT_COLORS["green"],
                        "نیازمند بررسی": PLOT_COLORS["red"],
                    },
                ),
                "وضعیت نگاشت در هر پروتکل",
            )
        with right:
            _plot(
                px.pie(
                    mapping.assign(
                        وضعیت=np.where(mapping["accepted"], "پذیرفته‌شده", "نیازمند بررسی")
                    ),
                    names="وضعیت",
                    color="وضعیت",
                    hole=0.58,
                    color_discrete_map={
                        "پذیرفته‌شده": PLOT_COLORS["green"],
                        "نیازمند بررسی": PLOT_COLORS["red"],
                    },
                ),
                "درصد اعتمادپذیری نگاشت‌های این اجرا",
            )

    st.subheader("مانیتور زندهٔ منابع", anchor=False)
    _resource_monitor()


def _mapping_enhanced(run: Path, catalog: pd.DataFrame, protocol: str | None) -> None:
    frame = _mapping_report(run, protocol)
    if frame.empty:
        _notice_missing("گزارش نگاشت", "reports/mapping-audit.parquet")
        return
    st.caption("هر نمودار زیر از گزارش ممیزی ذخیره‌شده ساخته می‌شود؛ CSV هرگز فیچر مدل نیست.")
    cards = st.columns(4)
    cards[0].metric("نرخ پذیرش", f"{frame['accepted'].mean():.1%}")
    cards[1].metric("میانگین Match rate", f"{frame['match_rate'].mean():.1%}")
    cards[2].metric("میانگین Confidence", f"{frame['mean_match_confidence'].mean():.1%}")
    cards[3].metric("Captureهای بررسی‌شده", f"{len(frame):,}")

    one, two = st.columns(2)
    with one:
        ordered = frame.sort_values("match_rate")
        ordered["وضعیت"] = np.where(ordered["accepted"], "پذیرفته‌شده", "نیازمند بررسی")
        _plot(
            px.bar(
                ordered,
                x="match_rate",
                y="capture",
                orientation="h",
                color="وضعیت",
                hover_data=["selected_csv", "mean_match_confidence", "selected_offset_seconds"],
                color_discrete_map={
                    "پذیرفته‌شده": PLOT_COLORS["green"],
                    "نیازمند بررسی": PLOT_COLORS["red"],
                },
            ),
            "نرخ تطبیق هر Capture با CSV منتخب",
            max(360, len(ordered) * 22),
        )
    with two:
        _plot(
            px.scatter(
                frame,
                x="match_rate",
                y="mean_match_confidence",
                color=np.where(frame["accepted"], "پذیرفته‌شده", "نیازمند بررسی"),
                symbol="protocol",
                hover_name="capture",
                hover_data=["selected_csv", "selected_offset_seconds"],
                color_discrete_map={
                    "پذیرفته‌شده": PLOT_COLORS["green"],
                    "نیازمند بررسی": PLOT_COLORS["red"],
                },
            ),
            "کیفیت دوگانهٔ نگاشت: match rate در برابر confidence",
        )
    three, four = st.columns(2)
    with three:
        _plot(
            px.violin(
                frame,
                x="protocol",
                y="mean_match_confidence",
                color="protocol",
                box=True,
                points="all",
                color_discrete_sequence=[
                    PLOT_COLORS["blue"],
                    PLOT_COLORS["violet"],
                    PLOT_COLORS["maroon"],
                    PLOT_COLORS["sky"],
                ],
            ),
            "پراکندگی confidence در پروتکل‌ها",
        )
    with four:
        offsets = frame.assign(
            offset_seconds=pd.to_numeric(frame["selected_offset_seconds"], errors="coerce").fillna(
                0
            )
        )
        _plot(
            px.histogram(
                offsets,
                x="offset_seconds",
                color="protocol",
                barmode="group",
                nbins=16,
                color_discrete_sequence=[
                    PLOT_COLORS["blue"],
                    PLOT_COLORS["violet"],
                    PLOT_COLORS["maroon"],
                    PLOT_COLORS["sky"],
                ],
            ),
            "Offset زمانی انتخاب‌شده برای تطبیق ساعت PCAP و CSV",
        )
    five, six = st.columns(2)
    with five:
        _plot(
            px.bar(
                frame.groupby(["protocol", "pairing_method"], as_index=False).size(),
                x="protocol",
                y="size",
                color="pairing_method",
                barmode="stack",
                color_discrete_sequence=[
                    PLOT_COLORS["blue"],
                    PLOT_COLORS["violet"],
                    PLOT_COLORS["maroon"],
                ],
            ),
            "روش یافتن CSV کاندید؛ این فقط سرنخ اولیه است",
        )
    with six:
        _plot(
            px.histogram(
                frame,
                x="pairing_score",
                color=np.where(frame["accepted"], "پذیرفته‌شده", "نیازمند بررسی"),
                nbins=12,
                barmode="overlay",
                color_discrete_map={
                    "پذیرفته‌شده": PLOT_COLORS["green"],
                    "نیازمند بررسی": PLOT_COLORS["red"],
                },
            ),
            "توزیع شباهت نام فایل در کنار نتیجهٔ واقعی نگاشت",
        )

    distribution_path = run / "reports" / "attack-label-distribution.parquet"
    if distribution_path.exists():
        labels = _read(distribution_path).copy()
        if protocol:
            labels = labels[
                labels["protocol"].astype("string").str.casefold().eq(protocol.casefold())
            ]
        labels = labels[labels["label"].astype("string").str.casefold().ne("unknown")]
        if not labels.empty:
            left, right = st.columns(2)
            with left:
                balance = labels.groupby(["protocol", "label"], as_index=False)["records"].sum()
                _plot(
                    px.bar(
                        balance.sort_values("records"),
                        x="records",
                        y="label",
                        color="protocol",
                        orientation="h",
                        color_discrete_sequence=[
                            PLOT_COLORS["maroon"],
                            PLOT_COLORS["violet"],
                            PLOT_COLORS["blue"],
                        ],
                    ),
                    "توازن کلاس‌های حمله بر پایهٔ رکوردهای PCAP",
                    max(360, len(balance) * 24),
                )
            with right:
                per_capture = labels.groupby(["capture", "mapping_accepted"], as_index=False)[
                    "records"
                ].sum()
                per_capture["وضعیت"] = np.where(
                    per_capture["mapping_accepted"], "برچسب قابل آموزش", "فقط برای بررسی"
                )
                _plot(
                    px.bar(
                        per_capture.sort_values("records"),
                        x="records",
                        y="capture",
                        color="وضعیت",
                        orientation="h",
                        color_discrete_map={
                            "برچسب قابل آموزش": PLOT_COLORS["green"],
                            "فقط برای بررسی": PLOT_COLORS["red"],
                        },
                    ),
                    "حجم دادهٔ برچسب‌خوردهٔ هر Capture",
                    max(360, len(per_capture) * 22),
                )

    mapping_paths = _paths(catalog, run, "flow_label_mapping", protocol)
    details = _sample_many(mapping_paths, 7000)
    if not details.empty:
        st.subheader("جزئیات در سطح Flow", anchor=False)
        left, right = st.columns(2)
        with left:
            _plot(
                px.histogram(
                    details,
                    x="match_confidence",
                    color="match_status",
                    nbins=30,
                    barmode="overlay",
                    color_discrete_map={
                        "matched": PLOT_COLORS["green"],
                        "unmatched": PLOT_COLORS["red"],
                    },
                ),
                "توزیع confidence برای Flowهای matched و unmatched",
            )
        with right:
            _plot(
                px.scatter(
                    details,
                    x="packet_count",
                    y="candidate_count",
                    color="match_status",
                    size="match_confidence",
                    hover_data=["label", "label_source_file"],
                    color_discrete_map={
                        "matched": PLOT_COLORS["green"],
                        "unmatched": PLOT_COLORS["red"],
                    },
                ),
                "پیچیدگی تطبیق Flow: packet count و تعداد کاندید",
            )
    st.subheader("جدول قابل دانلود ممیزی نگاشت", anchor=False)
    st.dataframe(
        frame.sort_values(["accepted", "match_rate"], ascending=[True, True]),
        hide_index=True,
        width="stretch",
    )
    _download(frame, "دانلود ممیزی نگاشت", "mapping-audit.csv")


def _features_enhanced(run: Path, catalog: pd.DataFrame, protocol: str | None) -> None:
    report = _feature_report(run, protocol)
    if report.empty:
        _notice_missing("تحلیل فیچر", "reports/features/...-overview.parquet")
        return
    st.caption(
        "تمام توزیع‌ها از فیچرهای PCAP استخراج‌شده می‌آیند؛ برای اجرای بزرگ، نمایش روی "
        "نمونهٔ محدود اما گسترده در زمان انجام می‌شود."
    )
    aggregate = report.groupby(["protocol", "feature", "data_group"], as_index=False).agg(
        availability=("availability", "median"),
        iqr=("iqr", "median"),
        std=("std", "median"),
        zero_ratio=("zero_ratio", "median"),
        unique_values=("unique_values", "median"),
    )
    cards = st.columns(4)
    cards[0].metric("فیچرهای بررسی‌شده", f"{aggregate['feature'].nunique():,}")
    cards[1].metric("دسترسی میانه", f"{aggregate['availability'].median():.1%}")
    cards[2].metric("فیچرهای با zero بالا", f"{int((aggregate['zero_ratio'] >= 0.99).sum()):,}")
    cards[3].metric("منبع آماری", f"{report['source'].nunique():,} فایل")

    one, two = st.columns(2)
    with one:
        availability = (
            aggregate.groupby("feature", as_index=False)["availability"]
            .median()
            .nlargest(25, "availability")
            .sort_values("availability")
        )
        _plot(
            px.bar(
                availability,
                x="availability",
                y="feature",
                orientation="h",
                color="availability",
                color_continuous_scale="Blues",
            ),
            "۲۵ فیچر با بیشترین پوشش مشاهده‌شده",
        )
    with two:
        variation = aggregate.groupby("feature", as_index=False).agg(
            iqr=("iqr", "median"),
            availability=("availability", "median"),
            unique_values=("unique_values", "median"),
        )
        _plot(
            px.scatter(
                variation,
                x="availability",
                y="iqr",
                size="unique_values",
                hover_name="feature",
                color="iqr",
                color_continuous_scale="Purples",
            ),
            "پوشش در برابر تغییرپذیری؛ فیچرهای مفید فقط پُر نیستند",
        )
    three, four = st.columns(2)
    with three:
        zeros = (
            aggregate.groupby("feature", as_index=False)["zero_ratio"]
            .median()
            .nlargest(25, "zero_ratio")
            .sort_values("zero_ratio")
        )
        _plot(
            px.bar(
                zeros,
                x="zero_ratio",
                y="feature",
                orientation="h",
                color="zero_ratio",
                color_continuous_scale="Reds",
            ),
            "فیچرهای sparse یا تقریباً ثابت",
        )
    with four:
        groups = aggregate.groupby(["data_group", "feature"], as_index=False)[
            "availability"
        ].median()
        common = groups.groupby("feature").filter(lambda item: item["data_group"].nunique() > 1)
        if not common.empty:
            _plot(
                px.bar(
                    common.nlargest(30, "availability"),
                    x="feature",
                    y="availability",
                    color="data_group",
                    barmode="group",
                    color_discrete_map={
                        "نرمال": PLOT_COLORS["blue"],
                        "حمله": PLOT_COLORS["maroon"],
                    },
                ),
                "مقایسهٔ پوشش فیچر در نرمال و حمله",
            )

    records = _record_samples(catalog, run, protocol)
    numerics = _numeric_columns(records)
    if not records.empty and numerics:
        st.subheader("مقایسهٔ مستقیم رفتار نرمال و حمله", anchor=False)
        selected = st.selectbox(
            "فیچر برای توزیع و مقایسه", numerics, key=f"feature-distribution-{run}-{protocol}"
        )
        left, right = st.columns(2)
        with left:
            _plot(
                px.histogram(
                    records,
                    x=selected,
                    color="data_group",
                    nbins=45,
                    histnorm="probability density",
                    barmode="overlay",
                    opacity=0.62,
                    color_discrete_map={
                        "نرمال": PLOT_COLORS["blue"],
                        "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"],
                        "رکورد نامشخص": PLOT_COLORS["muted"],
                    },
                ),
                f"توزیع {selected} در نرمال و حمله",
            )
        with right:
            _plot(
                px.box(
                    records,
                    x="data_group",
                    y=selected,
                    color="data_group",
                    points="outliers",
                    color_discrete_map={
                        "نرمال": PLOT_COLORS["blue"],
                        "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"],
                        "رکورد نامشخص": PLOT_COLORS["muted"],
                    },
                ),
                f"outlierها و چارک‌های {selected}",
            )
        left, right = st.columns(2)
        with left:
            _plot(
                px.ecdf(
                    records,
                    x=selected,
                    color="data_group",
                    color_discrete_map={
                        "نرمال": PLOT_COLORS["blue"],
                        "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"],
                        "رکورد نامشخص": PLOT_COLORS["muted"],
                    },
                ),
                f"ECDF برای تشخیص تفاوت‌های ظریف در {selected}",
            )
        with right:
            x_feature = st.selectbox(
                "محور X برای الگوی دوبعدی", numerics, key=f"feature-x-{run}-{protocol}"
            )
            y_feature = st.selectbox(
                "محور Y برای الگوی دوبعدی",
                numerics,
                index=min(1, len(numerics) - 1),
                key=f"feature-y-{run}-{protocol}",
            )
            hover = [column for column in ["capture", "label", "flow_id"] if column in records]
            _plot(
                px.scatter(
                    records,
                    x=x_feature,
                    y=y_feature,
                    color="data_group",
                    hover_data=hover,
                    opacity=0.55,
                    render_mode="svg",
                    color_discrete_map={
                        "نرمال": PLOT_COLORS["blue"],
                        "حملهٔ برچسب‌خورده": PLOT_COLORS["maroon"],
                        "رکورد نامشخص": PLOT_COLORS["muted"],
                    },
                ),
                "الگوی چندمتغیرهٔ نرمال و حمله",
            )
        correlation_features = numerics[: min(14, len(numerics))]
        if len(correlation_features) >= 2:
            corr = (
                records[correlation_features].apply(pd.to_numeric, errors="coerce").corr().round(2)
            )
            _plot(
                go.Figure(
                    go.Heatmap(
                        z=corr.to_numpy(),
                        x=corr.columns,
                        y=corr.index,
                        colorscale="RdBu",
                        zmid=0,
                        text=corr.to_numpy(),
                        texttemplate="%{text}",
                    )
                ),
                "هم‌بستگی فیچرهای پُرتغییر؛ سرنخ روابط غیرتک‌متغیره",
                540,
            )

    quality = _quality_report(run, protocol)
    if not quality.empty:
        st.subheader("آمادگی فیچر برای مدل", anchor=False)
        status_counts = quality.groupby(
            ["extraction_status", "model_usable"], as_index=False
        ).size()
        left, right = st.columns(2)
        with left:
            _plot(
                px.bar(
                    status_counts,
                    x="extraction_status",
                    y="size",
                    color="model_usable",
                    color_discrete_map={True: PLOT_COLORS["green"], False: PLOT_COLORS["red"]},
                ),
                "وضعیت مشاهده و قابلیت استفادهٔ مدل",
            )
        with right:
            usable = quality.sort_values("observed_ratio", ascending=False).head(25)
            _plot(
                px.bar(
                    usable,
                    x="observed_ratio",
                    y="name",
                    orientation="h",
                    color="cost",
                    color_discrete_map={
                        "low": PLOT_COLORS["blue"],
                        "medium": PLOT_COLORS["violet"],
                        "high": PLOT_COLORS["maroon"],
                    },
                ),
                "پوشش فیچر و هزینهٔ تقریبی استخراج",
            )
        st.dataframe(
            quality.sort_values(["model_usable", "observed_ratio"], ascending=[False, False]),
            hide_index=True,
            width="stretch",
        )
        _download(quality, "دانلود گزارش کیفیت فیچر", "feature-quality.csv")


def _start_dataset_job(
    artifact_root: Path, run_name: str, maximum_packets: int | None
) -> dict[str, Any]:
    target = artifact_root / "runs" / run_name
    target.mkdir(parents=True, exist_ok=False)
    log_path = target / "dashboard-dataset-run.log"
    command = [
        sys.executable,
        "-m",
        "anomdet.cli",
        "dataset",
        "run",
        "--output",
        str(target.resolve()),
    ]
    if maximum_packets is not None:
        command.extend(["--max-packets", str(maximum_packets)])
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, cwd=_project_root(), stdout=output, stderr=subprocess.STDOUT, text=True
        )
    job = {
        "process": process,
        "pid": process.pid,
        "target": str(target),
        "log_path": str(log_path),
        "refreshed": False,
    }
    st.session_state["dataset_build_job"] = job
    return job


@st.fragment(run_every="3s")
def _dataset_job_status() -> None:
    job = st.session_state.get("dataset_build_job")
    if not job:
        return
    process = job.get("process")
    target = Path(str(job["target"]))
    log_path = Path(str(job["log_path"]))
    return_code = process.poll() if process is not None else None
    if return_code is None:
        st.info(
            f"استخراج و نگاشت با PID {job['pid']} در پس‌زمینه در حال اجراست.",
            icon=":material/pending:",
        )
        _render_live_operation_monitor(
            target / "dataset-progress.json",
            log_path,
            "مانیتور زندهٔ استخراج و نگاشت",
            refresh_seconds=3,
        )
        return
    if return_code == 0 and (target / "catalog" / "artifacts.parquet").exists():
        if not job["refreshed"]:
            job["refreshed"] = True
            _table.clear()
            _json.clear()
            _sample_table.clear()
            st.rerun()
        st.success(f"dataset run کامل شد: {target}", icon=":material/check_circle:")
    else:
        st.error(f"dataset run با کد خروج {return_code} متوقف شد.", icon=":material/error:")
        st.code(_tail(log_path, 35), language="text")
    if st.button("پاک‌کردن وضعیت استخراج", key="clear-dataset-build"):
        st.session_state.pop("dataset_build_job", None)
        st.rerun()


def _dataset_build_panel(artifact_root: Path) -> None:
    st.subheader("شروع استخراج فیچر و نگاشت", anchor=False)
    st.info(
        "استخراج از پروفایل فیچر مستقل است: ابتدا extractor همهٔ فیچرهای معتبر PCAP را "
        "جداگانه برای هر پروتکل می‌سازد؛ سپس پروفایل فقط زیرمجموعهٔ ورودی مدل را انتخاب "
        "می‌کند. این دادهٔ خام برای هر دو Stage استفاده می‌شود.",
        icon=":material/info:",
    )
    mode = st.segmented_control(
        "نوع اجرا", ["آزمون محدود", "استخراج کامل"], default="آزمون محدود", key="dataset-mode"
    )
    default_name = "dashboard-test-200" if mode == "آزمون محدود" else "dashboard-full-run"
    running = st.session_state.get("dataset_build_job", {}).get("process")
    with st.form("dataset-build-form", border=True):
        run_name = st.text_input(
            "نام پوشهٔ اجرای جدید",
            value=default_name,
            help="فقط حروف، عدد، خط تیره و underscore؛ این پوشه نباید از قبل وجود داشته باشد.",
        )
        maximum = st.number_input(
            "حداکثر packet برای هر Capture",
            min_value=1,
            value=200,
            step=50,
            disabled=mode != "آزمون محدود",
        )
        confirmed = st.checkbox(
            "می‌دانم اجرای کامل ممکن است زمان‌بر باشد و یک پوشهٔ خروجی جدید می‌سازد."
        )
        submitted = st.form_submit_button(
            "شروع استخراج و نگاشت",
            icon=":material/play_arrow:",
            type="primary",
            disabled=running is not None and running.poll() is None,
        )
    target = artifact_root / "runs" / run_name.strip()
    if submitted:
        valid_name = (
            run_name.strip()
            and Path(run_name.strip()).name == run_name.strip()
            and all(char.isalnum() or char in "-_" for char in run_name.strip())
        )
        if not valid_name:
            st.error("نام اجرا فقط باید از حروف، عدد، - و _ تشکیل شود.")
        elif target.exists():
            st.error("این پوشه از قبل وجود دارد؛ نام جدید انتخاب کنید تا هیچ خروجی بازنویسی نشود.")
        elif not confirmed:
            st.warning("پیش از شروع، تأیید ساخت اجرای جدید لازم است.")
        else:
            job = _start_dataset_job(
                artifact_root, run_name.strip(), int(maximum) if mode == "آزمون محدود" else None
            )
            st.success(f"اجرای داده در پس‌زمینه آغاز شد. log: {job['log_path']}")
    _dataset_job_status()


def _profiles_enhanced(
    run: Path, catalog: pd.DataFrame, artifact_root: Path, protocol: str | None
) -> None:
    st.subheader("نقشهٔ داده تا مدل", anchor=False)
    _plot(_two_stage_figure(), "جریان واقعی داده، قرارداد فیچر و دو مدل", 510)
    st.caption(
        "Stage 1 با ترافیک نرمال یاد می‌گیرد؛ Stage 2 فقط پس از عبور از گیت ناهنجاری و "
        "با لیبل CSV پذیرفته‌شده نوع حمله را پیش‌بینی می‌کند."
    )
    _dataset_build_panel(artifact_root)

    st.subheader("پروفایل فیچر نسخه‌دار", anchor=False)
    st.info(
        "همین پروفایل برای هر دو Stage به‌عنوان ورودی خام مشترک استفاده می‌شود. Stage 2 "
        "علاوه بر آن، دو score از Stage 1 دریافت می‌کند. CSV در هیچ‌کدام از این دو ورودی "
        "فیچر نیست.",
        icon=":material/account_tree:",
    )
    configured = pd.DataFrame(available_features((protocol,)) if protocol else available_features())
    names = configured["name"].tolist()
    profile_dir = artifact_root / "feature_profiles"
    profile_paths = sorted(profile_dir.glob("*.json")) if profile_dir.exists() else []
    selected_profile: dict[str, Any] | None = None
    if profile_paths:
        selected_path = st.selectbox(
            "پروفایل‌های موجود",
            profile_paths,
            format_func=lambda item: item.name,
            key=f"profile-show-{protocol}",
        )
        selected_profile = load_profile(
            selected_path, {"project": {"artifact_dir": str(artifact_root)}}
        )
        st.caption(
            f"نسخه: {selected_profile['version']} · "
            f"تعداد فیچر: {selected_profile['feature_count']} · مسیر: {selected_path}"
        )
    with st.form("feature-profile-v2", border=True):
        name = st.text_input("نام سادهٔ پروفایل", placeholder="dns-compact")
        features = st.multiselect("فیچرهای ورودی مدل", names, default=names[: min(8, len(names))])
        description = st.text_input("دلیل انتخاب", placeholder="پروفایل متوازن برای CPU")
        saved = st.form_submit_button("ذخیرهٔ نسخهٔ پروفایل", icon=":material/save:")
    if saved:
        if not name or not features:
            st.error("نام و حداقل یک فیچر لازم است.")
        else:
            path = create_profile(
                name,
                features,
                {"project": {"artifact_dir": str(artifact_root)}},
                description,
                [protocol] if protocol else None,
            )
            st.success(f"پروفایل immutable ذخیره شد: {path}", icon=":material/check_circle:")
            st.rerun()

    selected_names = selected_profile["features"] if selected_profile else names
    selected_catalogue = configured[configured["name"].isin(selected_names)].copy()
    if not selected_catalogue.empty:
        left, right = st.columns(2)
        with left:
            _plot(
                px.pie(
                    selected_catalogue,
                    names="category",
                    color="category",
                    hole=0.55,
                    color_discrete_sequence=[
                        PLOT_COLORS["blue"],
                        PLOT_COLORS["violet"],
                        PLOT_COLORS["maroon"],
                        PLOT_COLORS["sky"],
                    ],
                ),
                "ترکیب دسته‌های فیچر در پروفایل انتخاب‌شده",
            )
        with right:
            cost = selected_catalogue.groupby("cost", as_index=False).size()
            _plot(
                px.bar(
                    cost,
                    x="cost",
                    y="size",
                    color="cost",
                    color_discrete_map={
                        "low": PLOT_COLORS["blue"],
                        "medium": PLOT_COLORS["violet"],
                        "high": PLOT_COLORS["maroon"],
                    },
                ),
                "هزینهٔ تقریبی استخراج فیچرهای انتخاب‌شده",
            )
        st.dataframe(
            selected_catalogue[["name", "category", "cost", "description", "protocols"]],
            hide_index=True,
            width="stretch",
        )
        _download(
            selected_catalogue,
            "دانلود تعریف فیچرهای پروفایل",
            "selected-feature-profile-details.csv",
        )

    quality = _quality_report(run, protocol)
    if not quality.empty and selected_names:
        selected_quality = quality[quality["name"].isin(selected_names)].copy()
        if not selected_quality.empty:
            _plot(
                px.bar(
                    selected_quality.sort_values("observed_ratio"),
                    x="observed_ratio",
                    y="name",
                    orientation="h",
                    color="extraction_status",
                    hover_data=["reason", "cost"],
                    color_discrete_sequence=[
                        PLOT_COLORS["green"],
                        PLOT_COLORS["red"],
                        PLOT_COLORS["violet"],
                    ],
                ),
                "کیفیت واقعی فیچرهای انتخاب‌شده در دادهٔ فعلی",
                max(360, len(selected_quality) * 22),
            )

    reports = sorted(run.glob("models/**/reports/preprocessing-comparison.parquet"))
    st.subheader("مقایسهٔ قبل و بعد از پیش‌پردازش", anchor=False)
    if not reports:
        st.caption(
            "این گزارش بعد از آموزش یک مدل با پروفایل انتخاب‌شده ایجاد می‌شود. تا آن زمان، "
            "نمودارهای بخش «فیچرها» توزیع خام را نمایش می‌دهند."
        )
        return
    report_path = st.selectbox(
        "گزارش پیش‌پردازش مدل",
        reports,
        format_func=lambda item: item.parent.parent.relative_to(run).as_posix(),
    )
    comparison = _read(report_path)
    if protocol:
        comparison = comparison[
            comparison["protocol"].astype("string").str.casefold().eq(protocol.casefold())
        ]
    if comparison.empty:
        st.info("برای پروتکل انتخابی، گزارش پیش‌پردازش موجود نیست.")
        return
    left, right = st.columns(2)
    with left:
        _plot(
            px.bar(
                comparison,
                x="feature",
                y="iqr",
                color="stage",
                barmode="group",
                color_discrete_map={"raw": PLOT_COLORS["blue"], "prepared": PLOT_COLORS["violet"]},
            ),
            "تغییر IQR در فضای خام و آمادهٔ مدل",
        )
    with right:
        _plot(
            px.bar(
                comparison,
                x="feature",
                y="std",
                color="stage",
                barmode="group",
                color_discrete_map={"raw": PLOT_COLORS["blue"], "prepared": PLOT_COLORS["violet"]},
            ),
            "تغییر پراکندگی پس از imputation، scaling و clipping",
        )
    st.dataframe(comparison, hide_index=True, width="stretch")
    _download(comparison, "دانلود مقایسهٔ پیش‌پردازش", "preprocessing-comparison.csv")


def _model_metrics_table(summary: dict[str, Any]) -> pd.DataFrame:
    stage1 = summary.get("stage1", {})
    stage2 = summary.get("stage2", {})
    return pd.DataFrame(
        [
            {
                "مدل": "Stage 1: LSTM-AE + Isolation Forest",
                "رکورد ارزیابی": stage1.get("records"),
                "دقت": stage1.get("accuracy"),
                "Precision": stage1.get("precision"),
                "Recall": stage1.get("recall"),
                "F1": stage1.get("f1"),
                "FPR": stage1.get("false_positive_rate"),
            },
            {
                "مدل": "Stage 2: Random Forest",
                "رکورد ارزیابی": stage2.get("test_records"),
                "دقت": stage2.get("test_accuracy"),
                "Precision": None,
                "Recall": None,
                "F1": stage2.get("weighted_f1"),
                "FPR": None,
            },
        ]
    )


def _packet_metric(value: object, precision: int = 3) -> str:
    """Format a possibly missing numeric model value for the packet evidence card."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{float(number):.{precision}f}" if pd.notna(number) else "—"


def _packet_option(row: pd.Series) -> str:
    """Give every detected packet a compact, stable label for the evidence selector."""
    record_id = row.get("record_id", "?")
    capture = str(row.get("capture", "بدون نام capture"))
    score = _packet_metric(row.get("stage1_normalized_score"), 2)
    return f"بسته {record_id} · {capture} · score {score}"


@st.cache_resource(max_entries=8, show_spinner=False)
def _isolation_forest(path_text: str, modified_ns: int) -> Any:
    """Load a trained detector once; the model is never fitted or mutated by the UI."""
    del modified_ns
    return joblib.load(path_text)


@st.cache_data(max_entries=12, show_spinner=False)
def _tsne_coordinates(values: np.ndarray) -> np.ndarray:
    """Create a bounded deterministic nonlinear projection for visual analysis only."""
    if len(values) < 3:
        return np.zeros((len(values), 2), dtype=float)
    perplexity = min(30.0, max(2.0, float((len(values) - 1) // 3)))
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        max_iter=600,
        random_state=42,
    ).fit_transform(values)


@st.cache_data(max_entries=12, show_spinner=False)
def _model_visual_inputs(
    run_text: str,
    model_root_text: str,
    protocol: str,
    model_modified_ns: int,
) -> pd.DataFrame:
    """Return the exact model-space inputs used for advanced dashboard diagnostics.

    New training runs persist ``evaluation-prepared.parquet``.  The fallback is
    intentionally read-only and reconstructs those values from PCAP-derived
    feature tables plus the frozen preprocessor, so earlier runs remain useful.
    """
    del model_modified_ns
    run = Path(run_text)
    model_root = Path(model_root_text)
    stage_one = model_root / "stage1"
    persisted = stage_one / "evaluation-prepared.parquet"
    if persisted.exists():
        return read_table(persisted)

    scores_path = stage_one / "scores.parquet"
    contract_path = model_root / "model-contract.json"
    manifest_path = stage_one / "prepared-normal.manifest.json"
    pipeline_path = stage_one / "prepared-normal.pipeline.joblib"
    if not all(
        path.exists() for path in [scores_path, contract_path, manifest_path, pipeline_path]
    ):
        return pd.DataFrame()
    scores = read_table(scores_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = list(contract.get("transformed_model_columns", []))
    if scores.empty or not feature_names:
        return pd.DataFrame()

    paths = [run / "features" / "benign" / protocol / "records.parquet"]
    paths.extend(sorted((run / "labelled" / "attack" / protocol).glob("*/records.parquet")))
    raw_frames = [read_table(path) for path in paths if path.exists()]
    if not raw_frames:
        return pd.DataFrame()
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    if "mapping_accepted" in raw:
        raw = raw[raw["mapping_accepted"].fillna(True)].copy()
    join_keys = [
        column
        for column in [
            "capture",
            "timestamp",
            "protocol",
            "flow_id",
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
        ]
        if column in scores and column in raw
    ]
    if not join_keys or any(feature not in raw for feature in feature_names):
        return pd.DataFrame()

    def join_token(frame: pd.DataFrame) -> pd.Series:
        return frame[join_keys].astype("string").fillna("<missing>").agg("\x1f".join, axis=1)

    raw = raw.copy()
    scores = scores.copy()
    raw["_model_join"] = join_token(raw)
    scores["_model_join"] = join_token(scores)
    raw = raw.drop_duplicates("_model_join")
    matched = scores.merge(
        raw[["_model_join", *feature_names]], on="_model_join", how="inner", validate="many_to_one"
    )
    if matched.empty:
        return pd.DataFrame()
    transformed, transformed_names = transform_with_manifest(matched, pipeline_path, manifest)
    if transformed_names != feature_names:
        return pd.DataFrame()
    output = matched.drop(columns=["_model_join"])
    for position, name in enumerate(feature_names):
        output[f"raw__{name}"] = pd.to_numeric(output[name], errors="coerce")
        output[name] = transformed.iloc[:, position].to_numpy(dtype=float)
    return output


def _balanced_visual_sample(frame: pd.DataFrame, maximum: int = 900) -> pd.DataFrame:
    """Preserve all flagged records while bounding interactive nonlinear plots."""
    if len(frame) <= maximum or "stage1_anomaly" not in frame:
        return frame.copy()
    anomalies = frame[frame["stage1_anomaly"].fillna(False).astype(bool)]
    normal = frame[~frame["stage1_anomaly"].fillna(False).astype(bool)]
    normal_count = max(0, maximum - len(anomalies))
    if normal_count <= 0:
        return anomalies.sample(maximum, random_state=42) if len(anomalies) > maximum else anomalies
    normal_sample = (
        normal.sample(min(len(normal), normal_count), random_state=42)
        if not normal.empty
        else normal
    )
    return pd.concat([anomalies, normal_sample], ignore_index=True, sort=False).sample(
        frac=1, random_state=42
    )


def _population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    """Return a bounded PSI value for a feature; used as a retraining signal, not a diagnosis."""
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 8 or len(current) < 8:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _priority_ai_outputs(run: Path, model_root: Path, protocol: str | None) -> None:
    """Render the high-value outputs that depend on the trained multivariate model.

    Deliberately no synthetic legacy, Geo-IP, analyst response, or throughput
    values appear here.  Each visual is computed from this model run's frozen
    feature contract, detector scores, and PCAP-derived records.
    """
    contract_path = model_root / "model-contract.json"
    score_path = model_root / "stage1" / "scores.parquet"
    normal_path = model_root / "stage1" / "prepared-normal.parquet"
    if not all(path.exists() for path in [contract_path, score_path, normal_path]):
        _notice_missing("تحلیل اولویت‌دار AI", "model-contract.json و stage1 artefacts")
        return
    contract = _read_json(contract_path)
    feature_names = list(contract.get("transformed_model_columns", []))
    resolved_protocol = str(protocol or contract.get("protocol") or "")
    if not feature_names or not resolved_protocol:
        st.info("قرارداد مدل فیچرهای لازم برای تحلیل عمیق را ندارد.")
        return
    inputs = _model_visual_inputs(
        str(run), str(model_root), resolved_protocol, score_path.stat().st_mtime_ns
    )
    if inputs.empty or any(feature not in inputs for feature in feature_names):
        st.warning(
            "ورودی‌های دقیق مدل برای این اجرای قدیمی قابل بازسازی نیستند. با آموزش مجدد، "
            "فایل stage1/evaluation-prepared.parquet خودکار تولید می‌شود.",
            icon=":material/inventory_2:",
        )
        return

    normal = _read(normal_path)
    if any(feature not in normal for feature in feature_names):
        st.warning("فایل ترافیک نرمالِ آماده‌شده با قرارداد مدل هم‌خوان نیست.")
        return
    normal_values = (
        normal[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    normal_median = np.nanmedian(normal_values, axis=0)
    normal_std = np.nanstd(normal_values, axis=0)
    normal_std = np.where(normal_std > 1e-9, normal_std, 1.0)
    normal_iqr = np.nanpercentile(normal_values, 75, axis=0) - np.nanpercentile(
        normal_values, 25, axis=0
    )
    normal_iqr = np.where(normal_iqr > 1e-9, normal_iqr, 1.0)

    visual = inputs.copy()
    values = visual[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    z_scores = np.abs((values - normal_median) / normal_std)
    robust_distance = np.abs((values - normal_median) / normal_iqr)
    visual["max_single_feature_z"] = z_scores.max(axis=1)
    visual["mean_multivariate_deviation"] = robust_distance.mean(axis=1)
    visual["تصمیم مدل"] = np.where(
        visual["stage1_anomaly"].fillna(False).astype(bool), "anomaly", "normal"
    )
    visual["واقعیت"] = np.where(
        visual.get("true_is_attack", pd.Series(False, index=visual.index)).fillna(False),
        "حملهٔ برچسب‌خورده",
        "نرمال/نامشخص",
    )
    lstm_vote = pd.to_numeric(
        visual.get("lstm_reconstruction_score"), errors="coerce"
    ) >= pd.to_numeric(visual.get("lstm_threshold"), errors="coerce")
    forest_vote = pd.to_numeric(
        visual.get("isolation_forest_score"), errors="coerce"
    ) >= pd.to_numeric(visual.get("isolation_forest_threshold"), errors="coerce")
    visual["رأی detectorها"] = lstm_vote.fillna(False).astype(int) + forest_vote.fillna(
        False
    ).astype(int)
    visual["اطمینان ensemble"] = visual["رأی detectorها"] / 2

    st.subheader("خروجی‌های اولویت‌دار؛ مزیت قابل‌نمایش AI", anchor=False)
    st.info(
        "این بخش فقط نمودارهایی را نمایش می‌دهد که از فضای چندبعدیِ مدل، detectorهای آموزش‌دیده "
        "یا خوشه‌بندی بدون‌نظارت ساخته شده‌اند. بنابراین با یک threshold یا گزارش SQL ساده قابل "
        "جایگزینی نیستند.",
        icon=":material/neurology:",
    )

    sampled = _balanced_visual_sample(visual)
    sampled_values = sampled[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    coordinates = _tsne_coordinates(sampled_values.to_numpy(dtype=float))
    embedded = sampled.copy()
    embedded["embedding_1"] = coordinates[:, 0]
    embedded["embedding_2"] = coordinates[:, 1]
    raw_candidates = [name for name in feature_names if f"raw__{name}" in embedded]
    raw_ranked = sorted(
        raw_candidates,
        key=lambda name: float(pd.to_numeric(embedded[f"raw__{name}"], errors="coerce").var() or 0),
        reverse=True,
    )
    raw_x, raw_y = (raw_ranked + feature_names)[:2]
    raw_x_column = f"raw__{raw_x}" if f"raw__{raw_x}" in embedded else raw_x
    raw_y_column = f"raw__{raw_y}" if f"raw__{raw_y}" in embedded else raw_y

    st.markdown("#### ۱. دادهٔ خام در برابر فضای غیرخطیِ یادگرفته‌شده")
    st.caption(
        "سمت چپ فقط دو فیچر خام را می‌بیند؛ سمت راست تمام فیچرهای انتخاب‌شده را با t-SNE در "
        "یک فضای غیرخطی فشرده می‌کند. رنگ در هر دو نمودار، تصمیم واقعی مدل Stage 1 است."
    )
    left, right = st.columns(2)
    with left:
        _plot(
            px.scatter(
                embedded,
                x=raw_x_column,
                y=raw_y_column,
                color="تصمیم مدل",
                symbol="واقعیت",
                hover_data=[
                    column
                    for column in ["record_id", "capture", "stage1_normalized_score"]
                    if column in embedded
                ],
                color_discrete_map={"anomaly": PLOT_COLORS["red"], "normal": PLOT_COLORS["blue"]},
                opacity=0.62,
                render_mode="svg",
            ),
            f"فضای خام: {raw_x} در برابر {raw_y}",
            470,
        )
    with right:
        _plot(
            px.scatter(
                embedded,
                x="embedding_1",
                y="embedding_2",
                color="تصمیم مدل",
                symbol="واقعیت",
                hover_data=[
                    column
                    for column in ["record_id", "capture", "stage1_normalized_score"]
                    if column in embedded
                ],
                color_discrete_map={"anomaly": PLOT_COLORS["red"], "normal": PLOT_COLORS["violet"]},
                opacity=0.70,
                render_mode="svg",
            ),
            "فضای غیرخطی t-SNE از همهٔ فیچرهای پروفایل مدل",
            470,
        )

    detected = visual[visual["stage1_anomaly"].fillna(False).astype(bool)].copy()
    if detected.empty:
        st.warning("مدل در این اجرا anomaly ثبت نکرده است؛ تحلیل تک‌رکورد قابل نمایش نیست.")
        return
    subtle = detected[detected["max_single_feature_z"] < 2].sort_values(
        "stage1_normalized_score", ascending=False
    )
    candidates = (
        subtle
        if not subtle.empty
        else detected.sort_values(
            ["max_single_feature_z", "stage1_normalized_score"], ascending=[True, False]
        )
    )
    st.markdown("#### ۲. anomaly چندبعدی؛ حتی وقتی قانون تک‌فیچر کافی نیست")
    if subtle.empty:
        st.caption(
            "در این اجرا anomaly با z-score تک‌فیچر کمتر از ۲ پیدا نشد؛ نزدیک‌ترین anomaly به این "
            "شرط برای نمایش انتخاب شده است. در اجرای کامل، این فیلتر خودکار دوباره ارزیابی می‌شود."
        )
    selected_id = st.selectbox(
        "رکورد anomaly برای تحلیل چندبعدی",
        candidates["record_id"].tolist(),
        format_func=lambda value: _packet_option(
            candidates.loc[candidates["record_id"].eq(value)].iloc[0]
        ),
        key=f"priority-record-{model_root}",
    )
    selected = candidates.loc[candidates["record_id"].eq(selected_id)].iloc[0]
    selected_position = visual.index.get_loc(selected.name)
    local = pd.DataFrame(
        {
            "فیچر": feature_names,
            "انحراف robust": robust_distance[selected_position],
            "z-score تک‌فیچر": z_scores[selected_position],
            "مقدار مدل": values[selected_position],
            "میانهٔ نرمال": normal_median,
        }
    ).sort_values("انحراف robust", ascending=False)
    metrics = st.columns(4)
    metrics[0].metric("Anomaly score", _packet_metric(selected.get("stage1_normalized_score")))
    metrics[1].metric("بیشترین z-score تک‌فیچر", f"{local['z-score تک‌فیچر'].max():.2f}")
    metrics[2].metric("میانگین انحراف چندبعدی", f"{local['انحراف robust'].mean():.2f}")
    metrics[3].metric("توافق detectorها", f"{int(selected['رأی detectorها'])}/2")
    left, right = st.columns(2)
    with left:
        _plot(
            go.Figure(
                go.Waterfall(
                    orientation="h",
                    measure=["relative"] * len(local),
                    y=local["فیچر"],
                    x=local["انحراف robust"],
                    increasing={"marker": {"color": PLOT_COLORS["red"]}},
                )
            ),
            "Waterfall شواهد چندبعدی؛ این نمودار SHAP نیست",
            430,
        )
    with right:
        radar = pd.concat([local, local.head(1)], ignore_index=True)
        _plot(
            go.Figure(
                go.Scatterpolar(
                    r=radar["انحراف robust"],
                    theta=radar["فیچر"],
                    fill="toself",
                    line={"color": PLOT_COLORS["red"]},
                    fillcolor="rgba(225,29,72,.28)",
                    name="رکورد انتخاب‌شده",
                )
            ).update_layout(polar={"radialaxis": {"visible": True}}),
            "Radar امضای چندفیچریِ anomaly انتخاب‌شده",
            430,
        )

    nearest_index = int(
        np.argmin(np.linalg.norm(normal_values - values[selected_position], axis=1))
    )
    nearest = normal_values[nearest_index]
    nearest_table = local.copy()
    nearest_table["نزدیک‌ترین نرمال در فضای مدل"] = nearest[
        [feature_names.index(name) for name in nearest_table["فیچر"]]
    ]
    nearest_table["فاصله از نزدیک‌ترین نرمال"] = np.abs(
        nearest_table["مقدار مدل"] - nearest_table["نزدیک‌ترین نرمال در فضای مدل"]
    )
    left, right = st.columns(2)
    with left:
        score = float(selected.get("stage1_normalized_score", 0))
        maximum = max(2.0, min(6.0, float(visual["stage1_normalized_score"].max()) * 1.05))
        _plot(
            go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": "× threshold"},
                    gauge={
                        "axis": {"range": [0, maximum]},
                        "bar": {"color": PLOT_COLORS["red"]},
                        "steps": [
                            {"range": [0, 0.8], "color": "#1B6A4A"},
                            {"range": [0.8, 1.0], "color": "#8B6A1E"},
                            {"range": [1.0, maximum], "color": "#5B1E3C"},
                        ],
                        "threshold": {
                            "line": {"color": PLOT_COLORS["text"], "width": 3},
                            "value": 1,
                        },
                    },
                )
            ),
            "Gauge امتیاز anomaly؛ مرز تصمیم برابر ۱ است",
            330,
        )
    with right:
        st.markdown("##### نزدیک‌ترین ترافیک نرمال در فضای مدل")
        st.dataframe(
            nearest_table[
                ["فیچر", "مقدار مدل", "نزدیک‌ترین نرمال در فضای مدل", "فاصله از نزدیک‌ترین نرمال"]
            ],
            hide_index=True,
            width="stretch",
            height=275,
        )

    forest_path = model_root / "stage1" / "isolation-forest.joblib"
    forest: Any | None = None
    if forest_path.exists():
        forest = _isolation_forest(str(forest_path), forest_path.stat().st_mtime_ns)
        forest_threshold = float(selected.get("isolation_forest_threshold", 1.0))
        direction = normal_median - values[selected_position]
        fractions = np.linspace(0, 1, 101)
        counterfactual_values = values[selected_position] + fractions[:, None] * direction
        counterfactual_scores = -forest.score_samples(counterfactual_values) / max(
            forest_threshold, 1e-12
        )
        crossing = np.flatnonzero(counterfactual_scores < 1)
        left, right = st.columns(2)
        with left:
            _plot(
                px.line(
                    pd.DataFrame(
                        {
                            "حرکت به‌سمت مرکز نرمال (%)": fractions * 100,
                            "score IF / threshold": counterfactual_scores,
                        }
                    ),
                    x="حرکت به‌سمت مرکز نرمال (%)",
                    y="score IF / threshold",
                    markers=True,
                ).add_hline(y=1, line_dash="dash", line_color=PLOT_COLORS["red"]),
                "Counterfactual: حداقل حرکت به مرکز نرمال برای عبور از گیت IF",
                400,
            )
        with right:
            end = 100.0
            if len(crossing):
                end = float(fractions[int(crossing[0])] * 100)
            counterfactual_card = pd.DataFrame(
                [
                    {
                        "مؤلفه": "Isolation Forest",
                        "حرکت لازم تا زیر threshold": f"{end:.0f}%",
                        "توضیح": "حرکت هم‌زمان همهٔ فیچرها به‌سمت مرکز نرمال؛ نه تغییر تک‌قانونی.",
                    }
                ]
            )
            st.dataframe(counterfactual_card, hide_index=True, width="stretch", height=150)
            st.caption(
                "این counterfactual فقط برای مؤلفهٔ Isolation Forest محاسبه شده است؛ تصمیم نهایی "
                "Stage 1 همچنان ensemble LSTM-AE + IF است."
            )

        anomaly_positions = visual["stage1_anomaly"].fillna(False).to_numpy(dtype=bool)
        average_deviation = robust_distance[anomaly_positions].mean(axis=0)
        pair_positions = np.argsort(-average_deviation)[:2]
        if len(pair_positions) == 2:
            first, second = (int(value) for value in pair_positions)
            x_reference = normal_values[:, first]
            y_reference = normal_values[:, second]
            x_low, x_high = np.nanquantile(x_reference, [0.01, 0.99])
            y_low, y_high = np.nanquantile(y_reference, [0.01, 0.99])
            if x_low == x_high:
                x_low, x_high = x_low - 1, x_high + 1
            if y_low == y_high:
                y_low, y_high = y_low - 1, y_high + 1
            x_grid = np.linspace(x_low, x_high, 42)
            y_grid = np.linspace(y_low, y_high, 42)
            mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
            interaction_values = np.tile(normal_median, (mesh_x.size, 1))
            interaction_values[:, first] = mesh_x.ravel()
            interaction_values[:, second] = mesh_y.ravel()
            interaction_score = -forest.score_samples(interaction_values).reshape(mesh_x.shape)
            interaction_score /= max(forest_threshold, 1e-12)
            surface = go.Figure(
                go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=interaction_score,
                    colorscale=[
                        [0.0, "#1B6A4A"],
                        [0.5, "#7C3AED"],
                        [1.0, "#E11D48"],
                    ],
                    contours={"showlabels": True},
                    colorbar={"title": "score / threshold"},
                )
            )
            surface.add_trace(
                go.Scatter(
                    x=visual[feature_names[first]],
                    y=visual[feature_names[second]],
                    mode="markers",
                    marker={
                        "color": np.where(
                            visual["stage1_anomaly"], PLOT_COLORS["red"], PLOT_COLORS["sky"]
                        ),
                        "size": 5,
                        "opacity": 0.45,
                    },
                    name="رکورد واقعی",
                )
            )
            _plot(
                surface,
                "تعامل غیرخطی یادگرفته‌شده توسط IF: "
                f"{feature_names[first]} × {feature_names[second]}",
                500,
            )

    st.markdown("#### ۳. خوشه‌های anomaly و اطمینان ensemble")
    embedded_anomalies = embedded[embedded["stage1_anomaly"].fillna(False).astype(bool)].copy()
    if len(embedded_anomalies) >= 3:
        coordinates = embedded_anomalies[["embedding_1", "embedding_2"]].to_numpy(dtype=float)
        neighbor_count = min(5, len(coordinates))
        distances, _ = (
            NearestNeighbors(n_neighbors=neighbor_count).fit(coordinates).kneighbors(coordinates)
        )
        epsilon = max(float(np.quantile(distances[:, -1], 0.82)), 1e-6)
        minimum = min(5, max(2, len(coordinates) // 8))
        embedded_anomalies["cluster"] = DBSCAN(eps=epsilon, min_samples=minimum).fit_predict(
            coordinates
        )
        embedded_anomalies["cluster_name"] = np.where(
            embedded_anomalies["cluster"].eq(-1),
            "نقطهٔ نادر/مجزا",
            "خوشهٔ " + (embedded_anomalies["cluster"] + 1).astype(str),
        )
        clusters = (
            embedded_anomalies.groupby("cluster_name", as_index=False)
            .agg(
                **{
                    "embedding_1": ("embedding_1", "mean"),
                    "embedding_2": ("embedding_2", "mean"),
                    "تعداد رکورد": ("record_id", "size"),
                    "میانگین score": ("stage1_normalized_score", "mean"),
                    "میانگین انحراف": ("mean_multivariate_deviation", "mean"),
                }
            )
            .sort_values("تعداد رکورد", ascending=False)
        )
        left, right = st.columns(2)
        with left:
            _plot(
                px.scatter(
                    clusters,
                    x="embedding_1",
                    y="embedding_2",
                    size="تعداد رکورد",
                    color="میانگین score",
                    text="cluster_name",
                    hover_data=["میانگین انحراف"],
                    color_continuous_scale=[PLOT_COLORS["violet"], PLOT_COLORS["red"]],
                ).update_traces(textposition="top center"),
                "Cluster bubble: خوشه‌های خودکار anomaly در embedding",
                430,
            )
        with right:
            agreement = (
                detected.assign(
                    سطح_توافق=np.where(
                        detected["رأی detectorها"].eq(2), "هر دو detector", "یک detector"
                    )
                )
                .groupby("سطح_توافق", as_index=False)
                .agg(تعداد=("record_id", "size"), میانگین_score=("stage1_normalized_score", "mean"))
            )
            _plot(
                px.bar(
                    agreement,
                    x="سطح_توافق",
                    y="تعداد",
                    color="میانگین_score",
                    color_continuous_scale=[PLOT_COLORS["sky"], PLOT_COLORS["red"]],
                    hover_data=["میانگین_score"],
                ),
                "توافق LSTM-AE و Isolation Forest؛ confidence ensemble",
                430,
            )

    detected["شدت"] = pd.to_numeric(detected["stage1_normalized_score"], errors="coerce")
    detected["اولویت هشدار"] = np.select(
        [
            (detected["شدت"] >= 1.5) & (detected["اطمینان ensemble"] >= 0.5),
            detected["شدت"] >= 1.0,
        ],
        ["بالا", "نیازمند بررسی"],
        default="کم",
    )
    _plot(
        px.scatter(
            detected,
            x="اطمینان ensemble",
            y="شدت",
            color="اولویت هشدار",
            size="mean_multivariate_deviation",
            hover_data=[
                column
                for column in [
                    "record_id",
                    "capture",
                    "predicted_attack_type",
                    "max_single_feature_z",
                ]
                if column in detected
            ],
            color_discrete_map={
                "بالا": PLOT_COLORS["red"],
                "نیازمند بررسی": PLOT_COLORS["violet"],
                "کم": PLOT_COLORS["sky"],
            },
        ),
        "Alert priority matrix: شدت مدل × توافق detectorها",
        440,
    )

    timestamps = pd.to_datetime(visual.get("timestamp"), errors="coerce", utc=True)
    temporal = (
        visual.assign(_timestamp=timestamps).dropna(subset=["_timestamp"]).sort_values("_timestamp")
    )
    if len(temporal) >= 16:
        pivot = len(temporal) // 2
        drift = pd.DataFrame(
            {
                "فیچر": feature_names,
                "PSI دورهٔ اول/دوم": [
                    _population_stability_index(
                        temporal.iloc[:pivot][feature].to_numpy(dtype=float),
                        temporal.iloc[pivot:][feature].to_numpy(dtype=float),
                    )
                    for feature in feature_names
                ],
            }
        ).dropna()
        if not drift.empty:
            _plot(
                px.bar(
                    drift.sort_values("PSI دورهٔ اول/دوم"),
                    x="PSI دورهٔ اول/دوم",
                    y="فیچر",
                    orientation="h",
                    color="PSI دورهٔ اول/دوم",
                    color_continuous_scale=[PLOT_COLORS["sky"], PLOT_COLORS["maroon"]],
                ),
                "Feature drift over time؛ سیگنال بازبینی/بازآموزی، نه ادعای تطبیق خودکار",
                max(360, len(drift) * 34),
            )

    priority_columns = [
        column
        for column in [
            "record_id",
            "timestamp",
            "capture",
            "protocol",
            "src_ip",
            "dst_ip",
            "stage1_normalized_score",
            "رأی detectorها",
            "اطمینان ensemble",
            "max_single_feature_z",
            "mean_multivariate_deviation",
            "اولویت هشدار",
            "predicted_attack_type",
        ]
        if column in detected
    ]
    st.subheader("فهرست اولویت‌بندی‌شدهٔ هشدارهای مدل", anchor=False)
    st.dataframe(
        detected.sort_values(["شدت", "اطمینان ensemble"], ascending=False)[priority_columns],
        hide_index=True,
        width="stretch",
        height=320,
    )
    _download(
        detected[priority_columns], "دانلود دادهٔ خروجی‌های اولویت‌دار", "ai-priority-alerts.csv"
    )
    st.caption(
        "مقایسه با سامانهٔ rule-based قدیمی، حجم هشدار گذشته، Geo-IP، زمان پاسخ تحلیلگر و "
        "throughput "
        "واقعی، به لاگ عملیاتی خارجی نیاز دارند. این داشبورد تا وقتی آن داده‌ها وارد نشوند، عددی "
        "ساختگی برایشان نمایش نمی‌دهد."
    )


def _feature_evidence_enhanced(model_root: Path) -> None:
    """Show aggregate and packet-level, model-derived anomaly evidence.

    The evidence records are deliberately described as *signals behind the model
    decision*: a robust distance from the normal training distribution is useful
    to an analyst, but it is not a claim of network-level causal proof.
    """
    evidence_path = model_root / "stage1" / "feature-evidence.parquet"
    scores_path = model_root / "stage1" / "scores.parquet"
    pipeline_path = model_root / "pipeline" / "end-to-end-predictions.parquet"
    if not evidence_path.exists():
        _notice_missing("شواهد فیچر", "stage1/feature-evidence.parquet")
        return

    evidence = _read_sample(evidence_path, 30000).copy()
    required = {"record_id", "feature", "robust_deviation", "stage1_anomaly"}
    if not required.issubset(evidence.columns):
        st.error("فایل شواهد فیچر ستون‌های لازم برای توضیح تصمیم را ندارد.")
        return
    evidence["stage1_anomaly"] = evidence["stage1_anomaly"].fillna(False).astype(bool)
    evidence["شدت انحراف"] = pd.to_numeric(evidence["robust_deviation"], errors="coerce").abs()
    anomalies = evidence[evidence["stage1_anomaly"]].copy()
    if anomalies.empty:
        st.info(
            "در نمونهٔ شواهد موجود، بسته‌ای از گیت Stage 1 عبور نکرده است؛ بنابراین نمودار "
            "علت‌یابیِ بسته نمایش ندارد.",
            icon=":material/visibility_off:",
        )
        return

    strengths = (
        anomalies.groupby("record_id", as_index=False)
        .agg(
            **{
                "بیشترین شدت انحراف": ("شدت انحراف", "max"),
                "میانگین شدت انحراف": ("شدت انحراف", "mean"),
                "تعداد فیچر مؤثر": ("feature", "nunique"),
            }
        )
        .sort_values("بیشترین شدت انحراف", ascending=False)
    )
    top_feature_per_packet = (
        anomalies.sort_values(["record_id", "rank", "شدت انحراف"], ascending=[True, True, False])[
            ["record_id", "feature"]
        ]
        .drop_duplicates("record_id")
        .rename(columns={"feature": "فیچر رتبهٔ اول"})
    )
    strengths = strengths.merge(top_feature_per_packet, on="record_id", how="left")

    scores = pd.DataFrame()
    if scores_path.exists():
        scores = _read_sample(scores_path, 15000).copy()
        if "stage1_anomaly" in scores:
            scores = scores[scores["stage1_anomaly"].fillna(False).astype(bool)]
        if "record_id" in scores:
            scores = scores.drop_duplicates("record_id")
    if scores.empty:
        packets = strengths.copy()
    else:
        packets = scores.merge(strengths, on="record_id", how="inner")

    if pipeline_path.exists() and "record_id" in packets:
        pipeline = _read_sample(pipeline_path, 15000)
        if {"record_id", "predicted_attack_type"}.issubset(pipeline.columns):
            predicted = pipeline[["record_id", "predicted_attack_type"]].drop_duplicates(
                "record_id"
            )
            packets = packets.merge(predicted, on="record_id", how="left")
    sort_by = (
        "stage1_normalized_score"
        if "stage1_normalized_score" in packets.columns
        else "بیشترین شدت انحراف"
    )
    packets = packets.sort_values(sort_by, ascending=False).reset_index(drop=True)

    feature_impact = (
        anomalies.groupby("feature", as_index=False)
        .agg(
            **{
                "بسته‌های متاثر": ("record_id", "nunique"),
                "میانگین شدت انحراف": ("شدت انحراف", "mean"),
                "بیشترین شدت انحراف": ("شدت انحراف", "max"),
                "میانگین رتبه": ("rank", "mean"),
            }
        )
        .sort_values("میانگین شدت انحراف", ascending=False)
    )
    feature_impact["سهم بسته‌های anomaly"] = feature_impact["بسته‌های متاثر"] / max(len(packets), 1)
    top_features = feature_impact.head(25)

    cards = st.columns(4)
    cards[0].metric("بسته‌های anomaly قابل‌توضیح", f"{len(packets):,}")
    cards[1].metric("فیچرهای مؤثر متمایز", f"{anomalies['feature'].nunique():,}")
    cards[2].metric(
        "میانگین بیشترین انحراف هر بسته",
        f"{strengths['بیشترین شدت انحراف'].mean():.2f}",
    )
    cards[3].metric(
        "قوی‌ترین فیچر در کل نمونه",
        str(top_features.iloc[0]["feature"]) if not top_features.empty else "—",
    )
    st.caption(
        "«فیچر مؤثر» یعنی فاصلهٔ robust آن فیچر از توزیع ترافیک نرمال، در فضای "
        "پیش‌پردازش‌شده. این شواهد توضیح تصمیم مدل‌اند، نه ادعای علت قطعی حمله."
    )

    left, right = st.columns(2)
    with left:
        _plot(
            px.bar(
                top_features.sort_values("میانگین شدت انحراف"),
                x="میانگین شدت انحراف",
                y="feature",
                orientation="h",
                color="بسته‌های متاثر",
                color_continuous_scale=[PLOT_COLORS["violet"], PLOT_COLORS["red"]],
                hover_data=[
                    "بیشترین شدت انحراف",
                    "میانگین رتبه",
                    "سهم بسته‌های anomaly",
                ],
            ),
            "اهمیت کلی فیچرها در تصمیم‌های anomaly",
            max(410, len(top_features) * 25),
        )
    with right:
        _plot(
            px.scatter(
                top_features,
                x="بسته‌های متاثر",
                y="میانگین شدت انحراف",
                size="بیشترین شدت انحراف",
                color="میانگین رتبه",
                text="feature",
                color_continuous_scale=[PLOT_COLORS["sky"], PLOT_COLORS["maroon"]],
                hover_data=["سهم بسته‌های anomaly"],
            ).update_traces(textposition="top center", marker={"opacity": 0.84}),
            "پوشش در بسته‌ها در برابر شدت شواهد هر فیچر",
            460,
        )

    selected_feature_names = top_features["feature"].head(16).tolist()
    selected_packet_ids = strengths["record_id"].head(40).tolist()
    heatmap_data = anomalies[
        anomalies["feature"].isin(selected_feature_names)
        & anomalies["record_id"].isin(selected_packet_ids)
    ]
    matrix = heatmap_data.pivot_table(
        index="feature", columns="record_id", values="شدت انحراف", aggfunc="max", fill_value=0
    )
    left, right = st.columns(2)
    with left:
        if not matrix.empty:
            _plot(
                go.Figure(
                    go.Heatmap(
                        z=matrix.to_numpy(),
                        x=[f"بسته {value}" for value in matrix.columns],
                        y=matrix.index.tolist(),
                        text=matrix.round(2).to_numpy(),
                        texttemplate="%{text}",
                        colorscale=[
                            [0.0, "#171C3A"],
                            [0.30, "#5B216A"],
                            [0.65, "#8B1E3F"],
                            [1.0, "#E11D48"],
                        ],
                        colorbar={"title": "شدت"},
                    )
                ),
                "نقشهٔ حرارتی فیچر × بسته؛ الگوی شواهد در anomalyها",
                max(440, len(matrix.index) * 28),
            )
    with right:
        dense = anomalies[anomalies["feature"].isin(selected_feature_names)]
        _plot(
            px.scatter(
                dense,
                x="feature",
                y="شدت انحراف",
                color="rank",
                size="شدت انحراف",
                hover_data=["record_id", "feature_value", "evidence_method"],
                color_continuous_scale=[PLOT_COLORS["sky"], PLOT_COLORS["violet"]],
                render_mode="svg",
            ),
            "شدت شواهد تمام بسته‌های anomaly به تفکیک rank فیچر",
            460,
        )

    st.subheader("توضیح هر بستهٔ anomaly", anchor=False)
    st.caption(
        "هر بستهٔ عبورکرده از گیت را انتخاب کنید؛ دو نمودار و جدول پایین فقط برای همان "
        "بسته تولید می‌شوند و نشان می‌دهند کدام فیچرها مدل را به تصمیم anomaly رسانده‌اند."
    )
    selected_id = st.selectbox(
        "بستهٔ anomaly برای واکاوی",
        packets["record_id"].tolist(),
        format_func=lambda value: _packet_option(
            packets.loc[packets["record_id"].eq(value)].iloc[0]
        ),
        key=f"anomaly-packet-{model_root}",
    )
    selected_packet = packets.loc[packets["record_id"].eq(selected_id)].iloc[0]
    selected_evidence = anomalies[anomalies["record_id"].eq(selected_id)].sort_values(
        ["rank", "شدت انحراف"], ascending=[True, False]
    )
    score_cards = st.columns(4)
    score_cards[0].metric(
        "score نهایی Stage 1", _packet_metric(selected_packet.get("stage1_normalized_score"))
    )
    score_cards[1].metric(
        "LSTM reconstruction", _packet_metric(selected_packet.get("lstm_reconstruction_score"))
    )
    score_cards[2].metric(
        "Isolation Forest", _packet_metric(selected_packet.get("isolation_forest_score"))
    )
    score_cards[3].metric(
        "نوع حملهٔ خروجی Stage 2", str(selected_packet.get("predicted_attack_type", "—"))
    )

    source_endpoint = f"{selected_packet.get('src_ip', '—')}:{selected_packet.get('src_port', '—')}"
    destination_endpoint = (
        f"{selected_packet.get('dst_ip', '—')}:{selected_packet.get('dst_port', '—')}"
    )
    packet_context = pd.DataFrame(
        [
            {
                "capture": selected_packet.get("capture", "—"),
                "زمان": selected_packet.get("timestamp", "—"),
                "پروتکل": selected_packet.get("protocol", "—"),
                "مبدأ": source_endpoint,
                "مقصد": destination_endpoint,
                "برچسب واقعی": selected_packet.get("label", "—"),
                "فیچر رتبهٔ اول": selected_packet.get("فیچر رتبهٔ اول", "—"),
            }
        ]
    )
    st.dataframe(packet_context, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        _plot(
            px.bar(
                selected_evidence.sort_values("شدت انحراف"),
                x="شدت انحراف",
                y="feature",
                orientation="h",
                color="rank",
                hover_data=["feature_value", "evidence_method"],
                color_continuous_scale=[PLOT_COLORS["sky"], PLOT_COLORS["red"]],
            ),
            "فیچرهای مؤثر برای همین بسته؛ طول میله = شدت انحراف از نرمال",
            390,
        )
    with right:
        _plot(
            px.bar_polar(
                selected_evidence,
                r="شدت انحراف",
                theta="feature",
                color="rank",
                color_continuous_scale=[PLOT_COLORS["sky"], PLOT_COLORS["maroon"]],
            ),
            "امضای شعاعی شواهد فیچری همان بسته",
            390,
        )

    comparison = anomalies[anomalies["feature"].isin(selected_evidence["feature"])].copy()
    comparison["بستهٔ انتخاب‌شده"] = np.where(
        comparison["record_id"].eq(selected_id), "این بسته", "سایر anomalyها"
    )
    comparison_figure = px.box(
        comparison,
        x="feature",
        y="شدت انحراف",
        color="بستهٔ انتخاب‌شده",
        points="outliers",
        color_discrete_map={
            "این بسته": PLOT_COLORS["red"],
            "سایر anomalyها": PLOT_COLORS["violet"],
        },
    )
    _plot(
        comparison_figure,
        "مقایسهٔ شواهد این بسته با سایر anomalyها برای همان فیچرها",
        420,
    )
    selected_display = selected_evidence[
        ["rank", "feature", "feature_value", "robust_deviation", "شدت انحراف", "evidence_method"]
    ].rename(
        columns={
            "rank": "رتبه",
            "feature": "فیچر",
            "feature_value": "مقدار پس از پیش‌پردازش",
            "robust_deviation": "انحراف robust",
            "evidence_method": "روش شواهد",
        }
    )
    st.dataframe(selected_display, hide_index=True, width="stretch")

    st.subheader("فهرست کامل بسته‌های anomaly در این خروجی", anchor=False)
    directory_columns = [
        column
        for column in [
            "record_id",
            "capture",
            "timestamp",
            "protocol",
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "label",
            "predicted_attack_type",
            "stage1_normalized_score",
            "بیشترین شدت انحراف",
            "میانگین شدت انحراف",
            "تعداد فیچر مؤثر",
            "فیچر رتبهٔ اول",
        ]
        if column in packets.columns
    ]
    directory = packets[directory_columns].copy()
    st.dataframe(directory, hide_index=True, width="stretch", height=360)
    _download(directory, "دانلود فهرست بسته‌های anomaly", "anomaly-packet-evidence.csv")
    _download(
        evidence.sort_values(["stage1_anomaly", "شدت انحراف"], ascending=[False, False]),
        "دانلود شواهد نمایش‌داده‌شده",
        "stage1-feature-evidence-visible.csv",
    )


def _models_enhanced(run: Path, artifact_root: Path, protocol: str | None) -> None:
    st.subheader("آموزش و نتایج مدل", anchor=False)
    with st.container(border=True, key="model-flow-card"):
        st.markdown("#### نقشهٔ اجرای مدل دو مرحله‌ای")
        _plot(
            _two_stage_figure(),
            "ورودی‌ها، گیت anomaly، تشخیص نوع حمله و خروجی قابل‌استقرار",
            510,
        )
        st.caption(
            "PCAP منبع فیچر است؛ CSV فقط برای نگاشت قابل ممیزی و برچسب آموزش استفاده می‌شود. "
            "Random Forest فقط رکوردهای عبورکرده از گیت Stage 1 را نوع‌بندی می‌کند."
        )
    _training_panel(run, artifact_root, protocol)
    _training_status()
    models = sorted(run.glob("models/**/training-summary.json"))
    if not models:
        st.info(
            "پس از تأیید نگاشت و انتخاب پروفایل، آموزش را از همین صفحه شروع کنید. تا آن زمان، "
            "نمودارهای فیچر و نگاشت برای تصمیم‌گیری آماده‌اند.",
            icon=":material/model_training:",
        )
        return
    selected = st.selectbox(
        "مدل آموزش‌دیده",
        models,
        format_func=lambda item: item.parent.relative_to(run).as_posix(),
        key=f"model-result-{run}",
    )
    summary = _read_json(selected)
    model_root = selected.parent
    stage1 = summary.get("stage1", {})
    stage2 = summary.get("stage2", {})
    cards = st.columns(4)
    cards[0].metric("Stage 1 — FPR", f"{float(stage1.get('false_positive_rate', 0)):.2%}")
    cards[1].metric("Stage 1 — Recall", f"{float(stage1.get('recall', 0)):.2%}")
    cards[2].metric("Stage 2 — Weighted F1", f"{float(stage2.get('weighted_f1', 0)):.2%}")
    cards[3].metric("زمان آموزش", f"{float(summary.get('training_seconds', 0)):.1f} ثانیه")
    if float(stage1.get("recall", 1)) < 0.30:
        st.warning(
            "Recall مرحلهٔ اول در این اجرا پایین است. این نتیجهٔ واقعی نمونه را نمایش می‌دهد، "
            "نه یک موفقیت ظاهری؛ برای تصمیم استقرار، اجرای کامل و بازبینی پروفایل/threshold "
            "لازم است.",
            icon=":material/warning:",
        )
    metrics = _model_metrics_table(summary)
    st.dataframe(
        metrics,
        hide_index=True,
        width="stretch",
        column_config={
            "دقت": st.column_config.NumberColumn(format="%.2f%%"),
            "Precision": st.column_config.NumberColumn(format="%.2f%%"),
            "Recall": st.column_config.NumberColumn(format="%.2f%%"),
            "F1": st.column_config.NumberColumn(format="%.2f%%"),
            "FPR": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    output_views = [
        "اولویت: مزیت AI",
        "Stage 1: ناهنجاری",
        "شواهد فیچر",
        "Stage 2: نوع حمله",
        "پایپ‌لاین و منابع",
    ]
    output_view = st.segmented_control(
        "نمای خروجی مدل",
        output_views,
        default=output_views[0],
        key=f"model-output-view-{model_root}",
        width="stretch",
    )
    if output_view == "اولویت: مزیت AI":
        _priority_ai_outputs(run, model_root, protocol)
    elif output_view == "Stage 1: ناهنجاری":
        score_path = model_root / "stage1" / "scores.parquet"
        history_path = model_root / "stage1" / "lstm-training-history.parquet"
        if score_path.exists():
            scores = _read_sample(score_path, 12000)
            scores["کلاس واقعی"] = np.where(
                scores.get("true_is_attack", pd.Series(False, index=scores.index)).fillna(False),
                "حمله",
                "نرمال",
            )
            scores["تصمیم مدل"] = np.where(scores["stage1_anomaly"], "ناهنجاری", "نرمال")
            left, right = st.columns(2)
            with left:
                _plot(
                    px.scatter(
                        scores,
                        x="lstm_reconstruction_score",
                        y="isolation_forest_score",
                        color="تصمیم مدل",
                        symbol="کلاس واقعی",
                        hover_data=[
                            column
                            for column in ["capture", "label", "stage1_normalized_score"]
                            if column in scores
                        ],
                        opacity=0.62,
                        render_mode="svg",
                        color_discrete_map={
                            "ناهنجاری": PLOT_COLORS["red"],
                            "نرمال": PLOT_COLORS["blue"],
                        },
                    ),
                    "ترکیب دو detector؛ نقاط قرمز از گیت عبور کرده‌اند",
                )
            with right:
                _plot(
                    px.histogram(
                        scores,
                        x="stage1_normalized_score",
                        color="کلاس واقعی",
                        facet_row="تصمیم مدل",
                        nbins=42,
                        barmode="overlay",
                        color_discrete_map={
                            "نرمال": PLOT_COLORS["blue"],
                            "حمله": PLOT_COLORS["maroon"],
                        },
                    ),
                    "توزیع score نهایی و مرز گیت مرحلهٔ اول",
                    520,
                )
            tn, fp = int(stage1.get("true_negatives", 0)), int(stage1.get("false_positives", 0))
            fn, tp = int(stage1.get("false_negatives", 0)), int(stage1.get("true_positives", 0))
            left, right = st.columns(2)
            with left:
                _plot(
                    go.Figure(
                        go.Heatmap(
                            z=[[tn, fp], [fn, tp]],
                            x=["پیش‌بینی نرمال", "پیش‌بینی ناهنجاری"],
                            y=["واقعی نرمال", "واقعی حمله"],
                            colorscale="Blues",
                            text=[[tn, fp], [fn, tp]],
                            texttemplate="%{text}",
                        )
                    ),
                    "Confusion matrix مرحلهٔ اول",
                )
            with right:
                rates = pd.DataFrame(
                    {
                        "شاخص": ["FPR", "FNR", "Specificity", "Recall"],
                        "مقدار": [
                            stage1.get("false_positive_rate", 0),
                            stage1.get("false_negative_rate", 0),
                            stage1.get("specificity", 0),
                            stage1.get("recall", 0),
                        ],
                    }
                )
                _plot(
                    px.bar(
                        rates,
                        x="شاخص",
                        y="مقدار",
                        color="شاخص",
                        range_y=[0, 1],
                        color_discrete_sequence=[
                            PLOT_COLORS["red"],
                            PLOT_COLORS["maroon"],
                            PLOT_COLORS["blue"],
                            PLOT_COLORS["green"],
                        ],
                    ),
                    "شاخص‌های عملیاتی Stage 1؛ FPR کنار Recall دیده می‌شود",
                )
            truth = scores.get("true_is_attack", pd.Series(False, index=scores.index)).fillna(False)
            anomaly_score = pd.to_numeric(
                scores["stage1_normalized_score"], errors="coerce"
            ).fillna(0)
            if truth.nunique() > 1:
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    truth.astype(int), anomaly_score
                )
                precision, recall, thresholds = precision_recall_curve(
                    truth.astype(int), anomaly_score
                )
                left, middle, right = st.columns(3)
                with left:
                    _plot(
                        px.line(
                            pd.DataFrame(
                                {"FPR": false_positive_rate, "Recall (TPR)": true_positive_rate}
                            ),
                            x="FPR",
                            y="Recall (TPR)",
                            markers=True,
                        ).add_shape(
                            type="line",
                            x0=0,
                            y0=0,
                            x1=1,
                            y1=1,
                            line={"dash": "dash", "color": PLOT_COLORS["muted"]},
                        ),
                        "ROC Stage 1؛ عملکرد روی برچسب واقعی",
                        370,
                    )
                with middle:
                    _plot(
                        px.line(
                            pd.DataFrame({"Recall": recall, "Precision": precision}),
                            x="Recall",
                            y="Precision",
                            markers=True,
                        ),
                        "Precision–Recall؛ مناسب دادهٔ نامتوازن",
                        370,
                    )
                with right:
                    if len(thresholds):
                        threshold_scores = pd.DataFrame(
                            {
                                "Threshold": thresholds,
                                "Precision": precision[:-1],
                                "Recall": recall[:-1],
                            }
                        )
                        threshold_scores["F1"] = (
                            2
                            * threshold_scores["Precision"]
                            * threshold_scores["Recall"]
                            / (threshold_scores["Precision"] + threshold_scores["Recall"]).replace(
                                0, np.nan
                            )
                        ).fillna(0)
                        melted_thresholds = threshold_scores.melt(
                            id_vars="Threshold", var_name="شاخص", value_name="مقدار"
                        )
                        _plot(
                            px.line(
                                melted_thresholds,
                                x="Threshold",
                                y="مقدار",
                                color="شاخص",
                                color_discrete_map={
                                    "Precision": PLOT_COLORS["blue"],
                                    "Recall": PLOT_COLORS["green"],
                                    "F1": PLOT_COLORS["red"],
                                },
                            ).add_vline(x=1, line_dash="dash", line_color=PLOT_COLORS["violet"]),
                            "Sensitivity threshold؛ مرز فعلی = ۱",
                            370,
                        )
            left, right = st.columns(2)
            with left:
                _plot(
                    px.histogram(
                        scores,
                        x="lstm_reconstruction_score",
                        color="کلاس واقعی",
                        barmode="overlay",
                        nbins=45,
                        color_discrete_map={
                            "نرمال": PLOT_COLORS["blue"],
                            "حمله": PLOT_COLORS["red"],
                        },
                    ),
                    "Reconstruction error: نرمال در برابر حمله",
                    400,
                )
            with right:
                _plot(
                    px.histogram(
                        scores,
                        x="stage1_normalized_score",
                        color="تصمیم مدل",
                        barmode="overlay",
                        nbins=45,
                        color_discrete_map={
                            "ناهنجاری": PLOT_COLORS["red"],
                            "نرمال": PLOT_COLORS["blue"],
                        },
                    ).add_vline(x=1, line_dash="dash", line_color=PLOT_COLORS["text"]),
                    "توزیع score و threshold نهایی ensemble",
                    400,
                )
        if history_path.exists():
            history = _read(history_path)
            if not history.empty:
                loss = history.melt(
                    id_vars="epoch",
                    value_vars=["train_loss", "validation_loss"],
                    var_name="مجموعه",
                    value_name="loss",
                )
                _plot(
                    px.line(
                        loss,
                        x="epoch",
                        y="loss",
                        color="مجموعه",
                        markers=True,
                        color_discrete_map={
                            "train_loss": PLOT_COLORS["blue"],
                            "validation_loss": PLOT_COLORS["maroon"],
                        },
                    ),
                    "یادگیری LSTM-AE: loss آموزش در برابر validation",
                )
    elif output_view == "شواهد فیچر":
        _feature_evidence_enhanced(model_root)
    elif output_view == "Stage 2: نوع حمله":
        importance_path = model_root / "stage2" / "feature-importance.parquet"
        confusion_path = model_root / "stage2" / "confusion-matrix.parquet"
        scores_path = model_root / "stage2" / "scores.parquet"
        left, right = st.columns(2)
        with left:
            if importance_path.exists():
                importance = _read(importance_path).sort_values("importance")
                _plot(
                    px.bar(
                        importance,
                        x="importance",
                        y="feature",
                        orientation="h",
                        color="importance",
                        color_continuous_scale="Purples",
                    ),
                    "اهمیت فیچرهای Random Forest؛ scoreهای Stage 1 نیز ممکن است مؤثر باشند",
                )
        with right:
            if confusion_path.exists():
                matrix = _read(confusion_path).set_index("true_attack_type")
                _plot(
                    go.Figure(
                        go.Heatmap(
                            z=matrix.to_numpy(),
                            x=matrix.columns,
                            y=matrix.index,
                            colorscale="Blues",
                            text=matrix.to_numpy(),
                            texttemplate="%{text}",
                        )
                    ),
                    "Confusion matrix تشخیص نوع حمله",
                    480,
                )
        if scores_path.exists():
            stage2_scores = _read_sample(scores_path, 8000)
            per_class = (
                stage2_scores.assign(درست=np.where(stage2_scores["correct"], "درست", "نادرست"))
                .groupby(["label", "درست"], as_index=False)
                .size()
            )
            left, right = st.columns(2)
            with left:
                _plot(
                    px.bar(
                        per_class,
                        x="size",
                        y="label",
                        color="درست",
                        orientation="h",
                        barmode="stack",
                        color_discrete_map={
                            "درست": PLOT_COLORS["green"],
                            "نادرست": PLOT_COLORS["red"],
                        },
                    ),
                    "صحت پیش‌بینی برای هر نوع حمله",
                    max(360, per_class["label"].nunique() * 30),
                )
            with right:
                _plot(
                    px.sunburst(
                        stage2_scores,
                        path=["label", "predicted_attack_type"],
                        color="correct",
                        color_discrete_map={True: PLOT_COLORS["green"], False: PLOT_COLORS["red"]},
                    ),
                    "مسیر خطاهای طبقه‌بندی بین انواع حمله",
                )
            st.dataframe(
                stage2_scores[~stage2_scores["correct"]].head(500), hide_index=True, width="stretch"
            )
    elif output_view == "پایپ‌لاین و منابع":
        pipeline_path = model_root / "pipeline" / "end-to-end-predictions.parquet"
        resource_path = model_root / "reports" / "resource-usage.parquet"
        if pipeline_path.exists():
            predictions = _read_sample(pipeline_path, 12000)
            gate = (
                predictions.assign(
                    گیت=np.where(
                        predictions["stage1_anomaly"], "ارسال به Stage 2", "نرمال در Stage 1"
                    ),
                    نتیجه=predictions["predicted_attack_type"].fillna("normal"),
                )
                .groupby(["گیت", "نتیجه"], as_index=False)
                .size()
            )
            left, right = st.columns(2)
            with left:
                _plot(
                    px.bar(
                        gate,
                        x="گیت",
                        y="size",
                        color="نتیجه",
                        barmode="stack",
                        color_discrete_sequence=[
                            PLOT_COLORS["blue"],
                            PLOT_COLORS["maroon"],
                            PLOT_COLORS["violet"],
                            PLOT_COLORS["sky"],
                        ],
                    ),
                    "خروجی end-to-end؛ گیت Stage 1 تا نوع حمله",
                )
            with right:
                anomalies = predictions[predictions["stage1_anomaly"]]
                if not anomalies.empty:
                    _plot(
                        px.pie(
                            anomalies,
                            names="predicted_attack_type",
                            hole=0.55,
                            color_discrete_sequence=[
                                PLOT_COLORS["maroon"],
                                PLOT_COLORS["violet"],
                                PLOT_COLORS["blue"],
                                PLOT_COLORS["sky"],
                            ],
                        ),
                        "ترکیب نوع حمله در رکوردهای عبورکرده از گیت",
                    )
            st.dataframe(
                predictions.sort_values("stage1_normalized_score", ascending=False).head(1000),
                hide_index=True,
                width="stretch",
            )
            _download(predictions, "دانلود خروجی end-to-end", "two-stage-predictions.csv")
        if resource_path.exists():
            resources = _read(resource_path)
            left, right = st.columns(2)
            with left:
                _plot(
                    px.bar(
                        resources,
                        x="point",
                        y="process_rss_mb",
                        color="point",
                        color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["maroon"]],
                    ),
                    "حافظهٔ process قبل و بعد از آموزش",
                )
            with right:
                _plot(
                    px.bar(
                        resources,
                        x="point",
                        y="memory_percent",
                        color="point",
                        color_discrete_sequence=[PLOT_COLORS["blue"], PLOT_COLORS["maroon"]],
                    ),
                    "درصد RAM سیستم قبل و بعد از آموزش",
                )
        exports = pd.DataFrame(
            [
                {"دارایی قابل‌استقرار": name, "مسیر": path}
                for name, path in summary.get("onnx_exports", {}).items()
            ]
        )
        if not exports.empty:
            st.subheader("فایل‌های ONNX برای استقرار", anchor=False)
            st.dataframe(exports, hide_index=True, width="stretch")


def _files_enhanced(run: Path, catalog: pd.DataFrame, protocol: str | None) -> None:
    scoped = catalog if protocol is None else catalog[catalog["protocol"].fillna("").eq(protocol)]
    display = scoped.copy()
    display["نوع"] = display["kind"].map(_kind_name)
    display = display[
        [
            "نوع",
            "protocol",
            "split",
            "format",
            "rows",
            "columns",
            "path",
            "description",
            "created_at",
        ]
    ]
    st.caption(
        "این فهرست قرارداد حمل‌پذیر تمام خروجی‌هاست. هر جدول یا نمودار dashboard به یکی از "
        "این مسیرهای Parquet/JSON متصل است."
    )
    st.dataframe(display.sort_values(["نوع", "path"]), hide_index=True, width="stretch")
    _download(display, "دانلود catalog خروجی‌ها", "artifact-catalog.csv")


def render_workbench() -> None:
    """Render the Persian, RTL, artifact-first dashboard entry point."""
    st.set_page_config(
        page_title="سامانه تحلیل ناهنجاری شبکه",
        page_icon=":material/network_intelligence:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _enable_rtl()
    with st.container(border=True, key="app-banner"):
        st.title("سامانه تحلیل ناهنجاری شبکه", anchor=False, text_alignment="right")
        st.caption(
            "PCAP-first · نگاشت قابل ممیزی · انتخاب پویای فیچر · مدل دو مرحله‌ای CPU-first",
            text_alignment="right",
        )
    with st.sidebar:
        st.header("زمینهٔ اجرا", anchor=False)
        root_text = st.text_input("پوشهٔ خروجی‌ها", value="artifacts", key="artifact-root")
        artifact_root = Path(root_text)
        if st.button("بازخوانی خروجی‌ها", icon=":material/refresh:"):
            _table.clear()
            _json.clear()
            _sample_table.clear()
            st.rerun()
        runs = _runs(artifact_root)
        if not runs:
            st.warning(
                "هنوز dataset run پیدا نشد. از بخش «پروفایل و پیش‌پردازش» استخراج را شروع کنید.",
                icon=":material/folder_off:",
            )
            return
        run = st.selectbox(
            "اجرای دیتاست",
            runs,
            format_func=lambda item: (
                item.relative_to(artifact_root).as_posix() if item != artifact_root else item.name
            ),
        )
        catalog = _catalog(run)
        protocols = sorted(str(item) for item in catalog["protocol"].dropna().unique() if item)
        selected_protocol = st.selectbox("پروتکل", ["همه", *protocols])
        protocol = None if selected_protocol == "همه" else selected_protocol
        st.caption(f"اجرای فعال: {run.relative_to(artifact_root).as_posix()}")
    areas = [
        "تابلوی کل سامانه",
        "نمای کلی",
        "داده و نگاشت",
        "فیچرها",
        "پروفایل و پیش‌پردازش",
        "مدل‌ها",
        "فایل‌ها",
    ]
    area = st.segmented_control(
        "بخش", areas, default=areas[0], selection_mode="single", key="dashboard-area"
    )
    if area == "تابلوی کل سامانه":
        _system_observatory(run, catalog)
    elif area == "نمای کلی":
        _overview_enhanced(run, catalog, protocol)
    elif area == "داده و نگاشت":
        _mapping_enhanced(run, catalog, protocol)
    elif area == "فیچرها":
        _features_enhanced(run, catalog, protocol)
    elif area == "پروفایل و پیش‌پردازش":
        _profiles_enhanced(run, catalog, artifact_root, protocol)
    elif area == "مدل‌ها":
        _models_enhanced(run, artifact_root, protocol)
    else:
        _files_enhanced(run, catalog, protocol)
