"""Professional command-line interface for the Phase 1 anomaly-detection workflow."""

from __future__ import annotations

import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from anomdet.core.config import artifact_root, load_config
from anomdet.core.io import read_table
from anomdet.core.logging import configure_logging, log_exception
from anomdet.core.paths import resolve_capture_path
from anomdet.core.resources import effective_workers, snapshot
from anomdet.datasets.inventory import discover_dataset
from anomdet.features.extractor import extract_pcap_features
from anomdet.features.protocols import extractor_for
from anomdet.mapping.mapper import map_pcap_to_labels
from anomdet.modelling.training import train_models
from anomdet.modelling.two_stage import (
    discover_dataset_run_sources,
    score_two_stage,
    train_two_stage,
)
from anomdet.orchestration.batch import run_inventory
from anomdet.orchestration.dataset_pipeline import (
    run_dataset_extraction,
    run_dataset_mapping,
    run_dataset_pipeline,
)
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
dataset_app = typer.Typer(
    help="Discover the protocol-first dataset and create auditable model assets.",
    no_args_is_help=True,
)
two_stage_app = typer.Typer(
    help="Train and run the deployable LSTM-AE/Isolation Forest to Random Forest pipeline.",
    no_args_is_help=True,
)
protocol_app = typer.Typer(
    help="Run one explicit protocol feature-extraction script at a time.",
    no_args_is_help=True,
)
app.add_typer(select_app, name="select")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(dataset_app, name="dataset")
app.add_typer(two_stage_app, name="two-stage")
app.add_typer(protocol_app, name="protocol")
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
        Path | None,
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
        Path | None,
        typer.Option("--output", "-o", help="Feature output (.parquet, .csv, or .jsonl)."),
    ] = None,
    max_packets: Annotated[
        int | None, typer.Option(min=1, help="Cap packets for a controlled trial run.")
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


@protocol_app.command("extract")
def extract_protocol(
    context: typer.Context,
    protocol: Annotated[str, typer.Argument(help="dns, http, modbus, s7comm, or ssh.")],
    capture: Annotated[
        Path, typer.Argument(readable=True, help="PCAP/PCAPNG file or a folder for this protocol.")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Protocol-only feature table.")
    ] = None,
    max_packets: Annotated[
        int | None, typer.Option(min=1, help="Cap packets per capture for a trial run.")
    ] = None,
) -> None:
    """Use the dedicated feature extractor for one protocol only."""
    try:
        selected_protocol = protocol.casefold()
        target = output or _default_output(
            context, "features", f"{selected_protocol}/{capture.stem}/records.parquet"
        )
        _, manifest = extractor_for(selected_protocol)(
            capture, target, _context_config(context), max_packets
        )
        _show_summary(
            "Protocol feature extraction complete",
            {
                "protocol": selected_protocol,
                "output": target,
                "rows": manifest["rows"],
                "flows": manifest["flow_count"],
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "extracting protocol-specific features")
        raise typer.Exit(code=1)


@pipeline_app.command("run")
def pipeline_run(
    context: typer.Context,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Batch-run artefact directory.")
    ] = None,
    max_packets: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Override the configured safe packet cap per capture for this dataset run.",
        ),
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


@dataset_app.command("inspect")
def dataset_inspect(context: typer.Context) -> None:
    """Show the current benign/attack protocol layout before starting extraction."""
    inventory = discover_dataset(_context_config(context))
    _show_summary(
        "Protocol-first dataset inventory",
        {
            "dataset root": inventory["dataset_root"],
            "benign captures": inventory["counts"]["benign_captures"],
            "attack captures": inventory["counts"]["attack_captures"],
            "attack PCAP/CSV candidates": inventory["counts"]["attack_pairs"],
        },
    )
    for pair in inventory["attack_pairs"]:
        console.print(
            f"[{pair['protocol']}] {Path(pair['capture']).name} -> "
            f"{Path(pair['labels']).name} ({pair['pairing_method']}; {pair['pairing_score']:.2f})"
        )


@dataset_app.command("run")
def dataset_run(
    context: typer.Context,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Self-contained dataset-run directory.")
    ] = None,
    max_packets: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Cap packets per capture; large values automatically use disk-backed streaming.",
        ),
    ] = None,
    all_packets: Annotated[
        bool,
        typer.Option(
            "--all-packets",
            help=(
                "Process every packet with the disk-backed streaming extractor; "
                "requires disk space."
            ),
        ),
    ] = False,
    remap: Annotated[
        bool,
        typer.Option(
            "--remap",
            help="Ignore a matching packet-to-CSV cache and build a fresh exhaustive mapping.",
        ),
    ] = False,
) -> None:
    """Extract protocol-separated PCAP features and verify all CSV label mappings."""
    try:
        if all_packets and max_packets is not None:
            raise typer.BadParameter("Choose either --max-packets or --all-packets, not both.")
        summary, summary_path = run_dataset_pipeline(
            _context_config(context),
            output,
            max_packets,
            allow_unbounded_streaming=all_packets,
            force_remap=remap,
        )
        _show_summary(
            "Dataset run complete",
            {
                "output": summary["output_root"],
                "benign protocol files": summary["protocol_feature_files"]["benign"],
                "attack capture files": summary["protocol_feature_files"]["attack"],
                "accepted mappings": f"{summary['accepted_mappings']}/{summary['mapping_count']}",
                "summary": summary_path,
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "running the protocol-first dataset pipeline")
        raise typer.Exit(code=1)


@dataset_app.command("extract")
def dataset_extract(
    context: typer.Context,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Self-contained dataset-run directory.")
    ] = None,
    max_packets: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Cap packets per capture; large values automatically use disk-backed streaming.",
        ),
    ] = None,
    all_packets: Annotated[
        bool,
        typer.Option(
            "--all-packets",
            help=(
                "Process every packet with the disk-backed streaming extractor; "
                "requires disk space."
            ),
        ),
    ] = False,
) -> None:
    """Extract PCAP features only; run `dataset map` later as a separate one-time step."""
    try:
        if all_packets and max_packets is not None:
            raise typer.BadParameter("Choose either --max-packets or --all-packets, not both.")
        summary, summary_path = run_dataset_extraction(
            _context_config(context),
            output,
            max_packets,
            allow_unbounded_streaming=all_packets,
        )
        _show_summary(
            "Dataset feature extraction complete",
            {
                "output": summary["output_root"],
                "benign protocol files": summary["protocol_feature_files"]["benign"],
                "attack capture files": summary["protocol_feature_files"]["attack"],
                "mapping": "not started; run `anomaly dataset map <dataset-run>`",
                "summary": summary_path,
            },
        )
    except Exception as error:
        log_exception(context.obj["logger"], "extracting protocol-first dataset features")
        raise typer.Exit(code=1) from error


@dataset_app.command("map")
def dataset_map(
    context: typer.Context,
    dataset_run: Annotated[
        Path,
        typer.Argument(
            exists=True, readable=True, help="Dataset-run created by `dataset extract`."
        ),
    ],
    protocol: Annotated[
        str | None,
        typer.Option(help="Map one protocol only; omit to map every saved attack capture."),
    ] = None,
    remap: Annotated[
        bool,
        typer.Option(
            "--remap",
            help=(
                "Ignore a matching saved relation and rebuild the exhaustive "
                "packet-to-CSV mapping."
            ),
        ),
    ] = False,
) -> None:
    """Build or reuse packet-to-CSV mappings from saved features without reopening PCAP files."""
    try:
        summary, summary_path = run_dataset_mapping(
            _context_config(context), dataset_run, protocol=protocol, force_remap=remap
        )
        _show_summary(
            "Dataset mapping complete",
            {
                "dataset run": summary["output_root"],
                "mapped captures": summary["mapped_captures"],
                "accepted mappings": summary["accepted_mappings"],
                "cache reused": summary["mapping_cache_reused"],
                "cache created": summary["mapping_cache_created"],
                "summary": summary_path,
            },
        )
    except Exception as error:
        log_exception(context.obj["logger"], "mapping saved PCAP feature records")
        raise typer.Exit(code=1) from error


@two_stage_app.command("train")
def two_stage_train(
    context: typer.Context,
    dataset_run: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Completed dataset-run directory.")
    ],
    protocol: Annotated[
        str | None, typer.Option(help="Train one protocol only; recommended for deployment.")
    ] = None,
    profile: Annotated[
        str | None, typer.Option(help="Frozen feature-profile path or name.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Model artifact directory.")
    ] = None,
    stage_one_only: Annotated[
        bool,
        typer.Option(
            "--stage-one-only",
            help="Train and evaluate LSTM-AE + Isolation Forest only; skip attack-type RF.",
        ),
    ] = False,
) -> None:
    """Train the deployable pipeline, or independently validate Stage 1 first."""
    try:
        benign, attacks = discover_dataset_run_sources(dataset_run, protocol)
        profile_id = Path(profile).stem if profile else "all-features"
        scope = protocol or "all-protocols"
        target = output or dataset_run / "models" / f"{scope}-{profile_id}"
        config = deepcopy(_context_config(context))
        if stage_one_only:
            config.setdefault("stage_two", {})["enabled"] = False
        summary = train_two_stage(
            benign, attacks, config, target, profile, protocol
        )
        _show_summary(
            "Two-stage training complete",
            {
                "output": target,
                "stage-one FPR": summary["stage1"]["false_positive_rate"],
                "stage-one recall": summary["stage1"]["recall"],
                "stage-two status": summary["stage2"].get("status", "trained"),
                "stage-two weighted F1": (
                    summary["stage2"].get("weighted_f1")
                    if summary["stage2"].get("status", "trained") == "trained"
                    else "not trained (see stage2/readiness.json)"
                ),
                "ONNX": summary["onnx_exports"],
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "training the two-stage model")
        raise typer.Exit(code=1)


@two_stage_app.command("score")
def two_stage_score(
    context: typer.Context,
    features: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="PCAP-derived feature records to score."),
    ],
    contract: Annotated[
        Path,
        typer.Option(
            exists=True, readable=True, help="model-contract.json from two-stage training."
        ),
    ],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Prediction table.")] = None,
) -> None:
    """Run stage one and stage two inference using the frozen training feature contract."""
    try:
        target = output or _default_output(
            context, "predictions", f"{features.stem}-predictions.parquet"
        )
        predictions = score_two_stage(read_table(features), contract)
        from anomdet.core.io import write_table

        write_table(predictions, target)
        _show_summary(
            "Two-stage scoring complete",
            {
                "output": target,
                "records": len(predictions),
                "anomalies": int(predictions["stage1_anomaly"].sum()),
            },
        )
    except Exception:
        log_exception(context.obj["logger"], "scoring the two-stage model")
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
        Path | None,
        typer.Option("--output", "-o", help="Mapping output (.parquet, .csv, or .jsonl)."),
    ] = None,
    max_packets: Annotated[
        int | None, typer.Option(min=1, help="Cap packets for a controlled trial run.")
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
        Path | None, typer.Option("--output", "-o", help="Quality report output.")
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
        str | None,
        typer.Option(help="Profile path or profile name. Omit to use all catalogue features."),
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option(exists=True, readable=True, help="Optional mapping output with flow labels."),
    ] = None,
    protocols: Annotated[str, typer.Option(help="Optional comma-separated protocol filter.")] = "",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Prepared matrix output.")
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
    profile: Annotated[str | None, typer.Option(help="Feature profile path or name.")] = None,
    labels: Annotated[
        Path | None,
        typer.Option(exists=True, readable=True, help="Optional mapped-flow labels."),
    ] = None,
    models: Annotated[
        str,
        typer.Option(
            help="Optional comma-separated detector names; omit for configured candidates."
        ),
    ] = "",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Experiment directory.")
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
        str | None, typer.Option(help="Target interface; otherwise use configured interface.")
    ] = None,
    multiplier: Annotated[
        float | None, typer.Option(min=0.01, help="Replay speed multiplier.")
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
