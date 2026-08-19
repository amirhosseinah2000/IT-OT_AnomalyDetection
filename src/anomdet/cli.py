"""Professional command-line interface for the Phase 1 anomaly-detection workflow."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from anomdet.core.config import artifact_root, load_config
from anomdet.core.logging import configure_logging, log_exception
from anomdet.core.paths import resolve_capture_path
from anomdet.core.resources import effective_workers, snapshot
from anomdet.features.extractor import extract_pcap_features
from anomdet.mapping.mapper import map_pcap_to_labels
from anomdet.modelling.training import train_models
from anomdet.orchestration.batch import run_inventory
from anomdet.preprocessing.pipeline import prepare_features
from anomdet.selection.profiles import create_profile, feature_quality_report, load_profile


app = typer.Typer(
    name="anomaly",
    help="PCAP-first anomaly detection for IT and OT network traffic.",
    no_args_is_help=True,
    add_completion=False,
)
select_app = typer.Typer(
    help="Create, inspect, and analyze versioned feature selections.", no_args_is_help=True
)
pipeline_app = typer.Typer(
    help="Run configured multi-PCAP ingestion and mapping workflows.", no_args_is_help=True
)
app.add_typer(select_app, name="select")
app.add_typer(pipeline_app, name="pipeline")
console = Console()


def _context_config(context: typer.Context) -> dict:
    """Return the initialized configuration from the Typer application context."""
    return context.obj["config"]


def _default_output(context: typer.Context, area: str, filename: str) -> Path:
    """Produce a deterministic default output beneath the configured artifact root."""
    return artifact_root(_context_config(context)) / area / filename


def _show_summary(title: str, values: dict[str, object]) -> None:
    """Render compact command results in a readable terminal table."""
    table = Table(title=title, show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config", "-c", exists=True, readable=True, help="YAML configuration override."
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable diagnostic logging.")
    ] = False,
) -> None:
    """Initialize shared configuration, structured logging, and runtime observability."""
    settings = load_config(config)
    logs = artifact_root(settings) / "logs"
    logger = configure_logging(logs, int(verbose))
    current = snapshot()
    logger.info(
        "Runtime resources: %s logical CPUs (%s configured workers), %.2f GB available of %.2f GB, %.1f%% in use",
        current.logical_cpus,
        effective_workers(int(settings["runtime"]["cpu_workers"])),
        current.available_memory_gb,
        current.total_memory_gb,
        current.memory_percent,
    )
    context.obj = {"config": settings, "logger": logger}


@app.command()
def resources(
    context: typer.Context,
    watch: Annotated[bool, typer.Option(help="Print repeated resource snapshots.")] = False,
    samples: Annotated[
        int, typer.Option(min=1, max=3600, help="Number of snapshots when --watch is set.")
    ] = 10,
    interval: Annotated[
        float, typer.Option(min=0.2, max=60.0, help="Seconds between watched snapshots.")
    ] = 2.0,
) -> None:
    """Show CPU and memory capacity before a costly extraction or training operation."""
    config = _context_config(context)
    iterations = samples if watch else 1
    for index in range(iterations):
        current = snapshot()
        _show_summary(
            f"Resource snapshot {index + 1}/{iterations}",
            {
                "logical CPUs": current.logical_cpus,
                "physical CPUs": current.physical_cpus,
                "CPU utilization": f"{current.cpu_percent:.1f}%",
                "total memory": f"{current.total_memory_gb:.2f} GB",
                "available memory": f"{current.available_memory_gb:.2f} GB",
                "memory utilization": f"{current.memory_percent:.1f}%",
                "configured workers": effective_workers(int(config["runtime"]["cpu_workers"])),
                "soft memory limit": f"{config['runtime']['memory_limit_gb']} GB",
            },
        )
        if index < iterations - 1:
            time.sleep(interval)


@app.command()
def extract(
    context: typer.Context,
    capture: Annotated[
        Path, typer.Argument(readable=True, help="Input PCAP/PCAPNG file or protocol folder.")
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Feature output (.parquet, .csv, or .jsonl)."),
    ] = None,
    max_packets: Annotated[
        Optional[int], typer.Option(min=1, help="Cap packets for a controlled trial run.")
    ] = None,
) -> None:
    """Extract protocol-aware features from one capture or all captures in a protocol folder."""
    target = output or _default_output(context, "features", f"{capture.stem}.parquet")
    try:
        config = _context_config(context)
        expected_protocol = (
            capture.name.lower()
            if capture.is_dir() and capture.name.lower() in config["capture"]["supported_protocols"]
            else None
        )
        _, manifest = extract_pcap_features(
            capture, target, config, max_packets, expected_protocol=expected_protocol
        )
        _show_summary(
            "Feature extraction complete",
            {"output": target, "rows": manifest["rows"], "protocols": manifest["protocol_counts"]},
        )
    except Exception:
        log_exception(context.obj["logger"], "extracting features")
        raise typer.Exit(code=1)


@pipeline_app.command("run")
def pipeline_run(
    context: typer.Context,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Batch-run artefact directory.")
    ] = None,
    max_packets: Annotated[
        Optional[int],
        typer.Option(min=1, help="Cap packets per capture for a controlled trial run."),
    ] = None,
) -> None:
    """Extract all configured PCAPs, test label candidates, and produce one run summary."""
    try:
        summary, summary_path = run_inventory(_context_config(context), output, max_packets)
        _show_summary(
            "Batch pipeline complete",
            {
                "output": summary["output_root"],
                "datasets extracted": f"{summary['successful_extractions']}/{summary['dataset_count']}",
                "combined features": summary["combined_features"] or "none",
                "summary": summary_path,
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "running the configured batch pipeline")
        raise typer.Exit(code=1)


@app.command(name="map")
def map_labels(
    context: typer.Context,
    capture: Annotated[Path, typer.Argument(readable=True, help="Input PCAP or PCAPNG capture.")],
    labels: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Labelled CSV, Parquet, or JSONL file."),
    ],
    domain: Annotated[
        str, typer.Option("--domain", case_sensitive=False, help="CSV family: it or ot.")
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Mapping output (.parquet, .csv, or .jsonl)."),
    ] = None,
    max_packets: Annotated[
        Optional[int], typer.Option(min=1, help="Cap packets for a controlled trial run.")
    ] = None,
) -> None:
    """Map labels to PCAP flows using 5-tuple and timestamp evidence."""
    target = output or _default_output(
        context, "mapping", f"{capture.stem}-{domain}-mapping.parquet"
    )
    try:
        resolved_capture = resolve_capture_path(capture)
        _, summary = map_pcap_to_labels(
            resolved_capture, labels, target, domain.lower(), _context_config(context), max_packets
        )
        _show_summary(
            "Mapping complete",
            {
                "output": target,
                "flows": summary["flow_count"],
                "status": summary["match_status_counts"],
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "mapping labels")
        raise typer.Exit(code=1)


@select_app.command("create")
def select_create(
    context: typer.Context,
    name: Annotated[str, typer.Argument(help="Stable, human-readable profile name.")],
    features: Annotated[
        str, typer.Option("--features", "-f", help="Comma-separated catalogue feature names.")
    ],
    description: Annotated[str, typer.Option(help="Why this profile exists.")] = "",
    protocols: Annotated[str, typer.Option(help="Optional comma-separated protocol scope.")] = "",
) -> None:
    """Create an immutable feature-selection profile from catalogue names."""
    try:
        selected = [item.strip() for item in features.split(",")]
        scoped_protocols = [
            item.strip().lower() for item in protocols.split(",") if item.strip()
        ] or None
        path = create_profile(
            name, selected, _context_config(context), description, scoped_protocols
        )
        _show_summary(
            "Feature profile created",
            {"path": path, "features": len(selected), "scope": scoped_protocols or "all"},
        )
    except Exception:
        log_exception(context.obj["logger"], "creating a feature profile")
        raise typer.Exit(code=1)


@select_app.command("list")
def select_list(context: typer.Context) -> None:
    """List versioned feature profiles available for preprocessing and training."""
    profile_dir = artifact_root(_context_config(context)) / "feature_profiles"
    profiles = sorted(profile_dir.glob("*.json")) if profile_dir.exists() else []
    if not profiles:
        console.print("No feature profiles have been created.")
        return
    table = Table(title="Feature profiles")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Features", justify="right")
    table.add_column("Protocols")
    table.add_column("Created")
    for path in profiles:
        item = load_profile(path, _context_config(context))
        table.add_row(
            item["name"],
            item["version"],
            str(item["feature_count"]),
            ", ".join(item["protocols"]),
            item["created_at"],
        )
    console.print(table)


@select_app.command("analyze")
def select_analyze(
    context: typer.Context,
    features: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Extracted feature table.")
    ],
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Quality report output.")
    ] = None,
) -> None:
    """Create a data-quality and computational-cost report for every catalogued feature."""
    target = output or _default_output(context, "reports", "feature-quality.parquet")
    try:
        report = feature_quality_report(features, target)
        _show_summary(
            "Feature analysis complete",
            {
                "output": target,
                "features": len(report),
                "model usable": int(report["model_usable"].sum()),
                "requires review": int((~report["model_usable"]).sum()),
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "analyzing feature quality")
        raise typer.Exit(code=1)


@app.command()
def prepare(
    context: typer.Context,
    features: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Extracted feature table.")
    ],
    profile: Annotated[
        Optional[str],
        typer.Option(help="Profile path or profile name. Omit to use all catalogue features."),
    ] = None,
    labels: Annotated[
        Optional[Path],
        typer.Option(exists=True, readable=True, help="Optional mapping output with flow labels."),
    ] = None,
    protocols: Annotated[str, typer.Option(help="Optional comma-separated protocol filter.")] = "",
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Prepared matrix output.")
    ] = None,
) -> None:
    """Transform selected feature data into a reusable model-ready numeric matrix."""
    suffix = profile or "all-features"
    target = output or _default_output(context, "prepared", f"{features.stem}-{suffix}.parquet")
    selected_protocols = [
        item.strip().lower() for item in protocols.split(",") if item.strip()
    ] or None
    try:
        _, manifest, pipeline = prepare_features(
            features, target, _context_config(context), profile, selected_protocols, labels
        )
        _show_summary(
            "Preparation complete",
            {
                "output": target,
                "pipeline": pipeline,
                "rows": manifest["prepared_rows"],
                "model columns": manifest["transformed_feature_count"],
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "preparing features")
        raise typer.Exit(code=1)


@app.command()
def train(
    context: typer.Context,
    features: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Extracted feature table.")
    ],
    strategy: Annotated[str, typer.Option(help="per_protocol or grouped.")] = "per_protocol",
    group: Annotated[str, typer.Option(help="For grouped strategy: it, ot, or all.")] = "all",
    profile: Annotated[Optional[str], typer.Option(help="Feature profile path or name.")] = None,
    labels: Annotated[
        Optional[Path],
        typer.Option(exists=True, readable=True, help="Optional mapped-flow labels."),
    ] = None,
    models: Annotated[
        str,
        typer.Option(
            help="Optional comma-separated detector names; omit for configured candidates."
        ),
    ] = "",
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Experiment directory.")
    ] = None,
) -> None:
    """Train and compare configured unsupervised detectors, including LSTM AE when selected."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = output or _default_output(context, "experiments", f"{strategy}-{group}-{run_id}")
    try:
        candidates = [item.strip() for item in models.split(",") if item.strip()] or None
        comparison, summary = train_models(
            features,
            _context_config(context),
            strategy,
            group.lower(),
            profile,
            target,
            labels,
            candidates,
        )
        _show_summary(
            "Training complete",
            {"output": target, "model runs": len(comparison), "comparison": summary["comparison"]},
        )
    except Exception:
        log_exception(context.obj["logger"], "training anomaly models")
        raise typer.Exit(code=1)


@app.command()
def dashboard(
    context: typer.Context,
    host: Annotated[str, typer.Option(help="Local bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1024, max=65535, help="Local dashboard port.")] = 8501,
) -> None:
    """Start the local Streamlit EDA dashboard using the active uv environment."""
    del context
    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    console.print(f"Starting dashboard at http://{host}:{port}")
    raise typer.Exit(subprocess.call(command))


@app.command()
def replay(
    context: typer.Context,
    capture: Annotated[Path, typer.Argument(readable=True, help="PCAP file to replay.")],
    interface: Annotated[
        Optional[str], typer.Option(help="Target interface; otherwise use configured interface.")
    ] = None,
    multiplier: Annotated[
        Optional[float], typer.Option(min=0.01, help="Replay speed multiplier.")
    ] = None,
    confirm: Annotated[
        bool, typer.Option(help="Required to send packets to the target interface.")
    ] = False,
) -> None:
    """Build or explicitly execute a configured tcpreplay command for a controlled lab."""
    config = _context_config(context)
    replay_config = config["capture"]["tcpreplay"]
    target_interface = interface or replay_config.get("interface")
    if not target_interface:
        raise typer.BadParameter("Set --interface or capture.tcpreplay.interface in configuration.")
    speed = multiplier if multiplier is not None else float(replay_config["multiplier"])
    resolved_capture = resolve_capture_path(capture)
    command = [
        str(replay_config["binary"]),
        "--intf1",
        target_interface,
        "--multiplier",
        str(speed),
        str(resolved_capture),
    ]
    console.print("Prepared replay command:")
    console.print(" ".join(command))
    if not confirm:
        console.print(
            "Dry run only. Add --confirm after validating the target interface and lab isolation."
        )
        return
    if not replay_config.get("enabled", False):
        raise typer.BadParameter(
            "Set capture.tcpreplay.enabled: true in configuration before execution."
        )
    result = subprocess.run(command, check=False)
    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
