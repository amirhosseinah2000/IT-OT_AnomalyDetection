"""Simple service interface for anomaly detection - clean API for external use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomdet.core.config import load_config
from anomdet.features.extractor import extract_pcap_features
from anomdet.modelling.lstm_autoencoder import LSTMAutoencoder
from anomdet.modelling.detectors import MLPAutoencoder
from anomdet.modelling.training import train_models
from anomdet.preprocessing.pipeline import prepare_features
from anomdet.selection.profiles import create_profile, load_profile


@dataclass
class ModelArtifacts:
    """Paths to all saved model artifacts."""
    model_dir: Path
    scope_dir: Path           # parent scope directory (has pipeline)
    model_file: Path          # model.pt or model.joblib
    pipeline_file: Path       # preprocessing pipeline
    scores_file: Path         # anomaly scores on test set
    metrics_file: Path        # metrics.json
    manifest_file: Path       # feature preparation manifest
    comparison_file: Path | None = None  # aggregate comparison (if multiple profiles)


@dataclass
class TrainResult:
    """Result of training one or more models."""
    model: str
    scope: str
    artifacts: ModelArtifacts
    metrics: dict[str, float | None]
    input_features: int
    training_rows: int


class AnomalyService:
    """
    Simple service interface for anomaly detection.
    
    Usage:
        service = AnomalyService()
        
        # 1. Extract features from PCAP
        features_path = service.extract_features("my_capture.pcap")
        
        # 2. Create a feature profile (select which features to use)
        profile_path = service.create_profile("my_profile", ["packet_length", "flow_duration", ...])
        
        # 3. Train models
        results = service.train(features_path, profile_path, models=["isolation_forest", "mlp_autoencoder"])
        
        # 4. Score new data
        scores = service.score(features_path, results[0].artifacts)
    """
    
    def __init__(self, config_path: Path | str | None = None, artifact_dir: Path | str | None = None):
        """
        Initialize the service.
        
        Args:
            config_path: Path to YAML config (optional, uses default)
            artifact_dir: Where to store all outputs (default: ./artifacts)
        """
        self.config = load_config(Path(config_path) if config_path else None)
        if artifact_dir:
            self.config["project"]["artifact_dir"] = str(artifact_dir)
        self.artifact_root = Path(self.config["project"]["artifact_dir"])
        self.artifact_root.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: Feature Extraction
    # -------------------------------------------------------------------------
    def extract_features(
        self,
        pcap_path: Path | str,
        output_name: str | None = None,
        max_packets: int | None = None,
    ) -> Path:
        """
        Extract features from a PCAP/PCAPNG file or protocol folder.
        
        Args:
            pcap_path: Path to .pcap/.pcapng file OR directory of PCAPs
            output_name: Custom output filename (default: <input>.parquet)
            max_packets: Limit packets for testing
            
        Returns:
            Path to extracted features Parquet file
        """
        pcap_path = Path(pcap_path)
        output_name = output_name or f"{pcap_path.stem}.parquet"
        output_path = self.artifact_root / "features" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        _, manifest = extract_pcap_features(
            pcap_path, output_path, self.config, max_packets
        )
        return output_path
    
    def extract_features_batch(
        self,
        pcap_folder: Path | str,
        protocol: str | None = None,
    ) -> dict[str, Path]:
        """
        Extract features from all PCAPs in a protocol folder.
        
        Returns:
            Dict of {protocol: feature_parquet_path}
        """
        # This uses the pipeline internally - simplified for common case
        from anomdet.orchestration.batch import run_inventory
        
        # Create temporary config for single folder
        temp_config = self.config.copy()
        temp_config["data"]["protocol_folders"] = [
            {"protocol": protocol or "modbus", "folder": Path(pcap_folder).name}
        ]
        
        summary, _ = run_inventory(temp_config, max_packets=None)
        return {
            d["dataset_id"]: Path(d["features"])
            for d in summary["datasets"]
            if d["status"] == "extracted"
        }
    
    # -------------------------------------------------------------------------
    # STEP 2: Feature Profiles (Select which features to use)
    # -------------------------------------------------------------------------
    def create_profile(
        self,
        name: str,
        features: list[str],
        description: str = "",
        protocols: list[str] | None = None,
    ) -> Path:
        """
        Create a reusable feature selection profile.
        
        Args:
            name: Profile name (alphanumeric, underscore, hyphen)
            features: List of feature names from catalogue
            description: Why this profile exists
            protocols: Optional protocol scope (default: all)
            
        Returns:
            Path to saved profile JSON
        """
        profile_path = create_profile(name, features, self.config, description, protocols)
        return profile_path
    
    def list_profiles(self) -> list[dict[str, Any]]:
        """List all available feature profiles."""
        profile_dir = self.artifact_root / "feature_profiles"
        if not profile_dir.exists():
            return []
        
        profiles = []
        for path in sorted(profile_dir.glob("*.json")):
            try:
                p = load_profile(path, self.config)
                profiles.append({
                    "name": p["name"],
                    "version": p["version"],
                    "feature_count": p["feature_count"],
                    "protocols": p["protocols"],
                    "features": p["features"],
                    "path": str(path),
                })
            except Exception:
                continue
        return profiles
    
    def get_profile(self, name: str) -> dict[str, Any]:
        """Load a profile by name (gets latest version)."""
        return load_profile(name, self.config)
    
    # -------------------------------------------------------------------------
    # STEP 3: Training
    # -------------------------------------------------------------------------
    def train(
        self,
        features_path: Path | str,
        profile: Path | str | None = None,
        strategy: str = "grouped",
        group: str = "all",
        models: list[str] | None = None,
        labels_path: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> list[TrainResult]:
        """
        Train anomaly detection models.
        
        Args:
            features_path: Path to extracted features Parquet
            profile: Feature profile path/name (None = all catalogue features)
            strategy: "per_protocol" or "grouped"
            group: For grouped: "it", "ot", or "all"
            models: List of model names (default: all candidates from config)
            labels_path: Optional mapped labels for supervised metrics
            output_dir: Custom output directory (default: auto-generated)
            
        Returns:
            List of TrainResult (one per model)
        """
        features_path = Path(features_path)
        if output_dir is None:
            from datetime import UTC, datetime
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_dir = self.artifact_root / "experiments" / f"{strategy}-{group}-{run_id}"
        else:
            output_dir = Path(output_dir)
        
        if labels_path:
            labels_path = Path(labels_path)
        
        candidates = models or self.config["models"]["candidates"]
        
        comparison, summary = train_models(
            features_path,
            self.config,
            strategy,
            group.lower(),
            profile,
            output_dir,
            labels_path,
            candidates,
        )
        
        results = []
        for _, row in comparison.iterrows():
            model_name = row["model"]
            scope_name = row["scope"]
            model_dir = output_dir / scope_name / model_name
            scope_dir = output_dir / scope_name
            
            # Determine model file extension
            model_file = model_dir / ("model.pt" if model_name in {"lstm_autoencoder", "mlp_autoencoder"} else "model.joblib")
            
            artifacts = ModelArtifacts(
                model_dir=model_dir,
                scope_dir=scope_dir,
                model_file=model_file,
                pipeline_file=scope_dir / "prepared.pipeline.joblib",
                scores_file=model_dir / "scores.parquet",
                metrics_file=model_dir / "metrics.json",
                manifest_file=scope_dir / "prepared.manifest.json",
            )
            
            metrics = {
                "score_mean": row.get("score_mean"),
                "score_p95": row.get("score_p95"),
                "threshold": row.get("threshold"),
                "predicted_anomaly_rate": row.get("predicted_anomaly_rate"),
                "fit_seconds": row.get("fit_seconds"),
                "roc_auc": row.get("roc_auc"),
                "average_precision": row.get("average_precision"),
                "reconstruction_mse_mean": row.get("reconstruction_mse_mean"),
                "epochs_completed": row.get("epochs_completed"),
            }
            
            results.append(TrainResult(
                model=model_name,
                scope=scope_name,
                artifacts=artifacts,
                metrics=metrics,
                input_features=int(row["input_features"]),
                training_rows=int(row["training_rows"]),
            ))
        
        return results
    
    def train_multiple_profiles(
        self,
        features_path: Path | str,
        profiles: list[Path | str],
        strategy: str = "grouped",
        group: str = "all",
        models: list[str] | None = None,
    ) -> dict[str, list[TrainResult]]:
        """
        Train across multiple feature profiles (baseline + selected).
        
        Returns:
            Dict of {profile_name: [TrainResult, ...]}
        """
        from anomdet.modelling.training import run_feature_experiments
        
        features_path = Path(features_path)
        output_dir = self.artifact_root / "experiments" / "multi_profile"
        
        comparison, summary = run_feature_experiments(
            features_path,
            self.config,
            strategy,
            group.lower(),
            profiles,
            output_dir,
            candidates=models,
        )
        
        # Group by profile
        results: dict[str, list[TrainResult]] = {}
        for profile_name, group_df in comparison.groupby("feature_profile"):
            profile_results = []
            for _, row in group_df.iterrows():
                model_name = row["model"]
                scope_name = row["scope"]
                model_dir = output_dir / profile_name / scope_name / model_name
                
                artifacts = ModelArtifacts(
                    model_dir=model_dir,
                    model_file=model_dir / ("model.pt" if model_name in {"lstm_autoencoder", "mlp_autoencoder"} else "model.joblib"),
                    pipeline_file=model_dir / "pipeline.joblib",
                    scores_file=model_dir / "scores.parquet",
                    metrics_file=model_dir / "metrics.json",
                    manifest_file=model_dir / "manifest.json",
                )
                
                profile_results.append(TrainResult(
                    model=model_name,
                    scope=scope_name,
                    artifacts=artifacts,
                    metrics=row.to_dict(),
                    input_features=int(row["input_features"]),
                    training_rows=int(row["training_rows"]),
                ))
            results[profile_name] = profile_results
        
        return results
    
    # -------------------------------------------------------------------------
    # STEP 4: Scoring / Inference
    # -------------------------------------------------------------------------
    def score(
        self,
        features_path: Path | str,
        artifacts: ModelArtifacts,
        batch_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Score new feature data using a trained model.
        
        Args:
            features_path: Path to features Parquet to score
            artifacts: ModelArtifacts from TrainResult
            batch_size: Batch size for scoring
            
        Returns:
            DataFrame with anomaly_score, is_anomaly, score_percentile columns
        """
        # Load pipeline from scope directory
        import joblib
        pipeline = joblib.load(artifacts.pipeline_file)
        
        # Load model
        if artifacts.model_file.suffix == ".pt":
            import torch
            model = self._load_torch_model(artifacts.model_file)
        else:
            model = joblib.load(artifacts.model_file)["model"]
        
        # Load metrics for threshold
        import json
        with open(artifacts.metrics_file) as f:
            metrics = json.load(f)
        threshold = metrics.get("threshold", 0.0)
        
        # Load and prepare features using the SAME pipeline
        features_path = Path(features_path)
        # Read raw features
        raw_df = pd.read_parquet(features_path)
        
        # Apply the saved pipeline directly to new data
        # We need to select the same columns and transform
        metadata_cols = {"row_id", "label", "protocol", "flow_id", "timestamp", "capture"}
        
        # The pipeline expects the same input columns it was fitted on
        # Get input columns from manifest
        import json
        with open(artifacts.manifest_file) as f:
            manifest = json.load(f)
        selected_features = manifest.get("selected_input_features", [])
        
        # Ensure all selected features exist
        available = [c for c in selected_features if c in raw_df.columns]
        missing = set(selected_features) - set(available)
        if missing:
            print(f"Warning: missing features {missing}, filling with NaN")
            for m in missing:
                raw_df[m] = np.nan
        
        # Prepare matrix
        matrix = raw_df[selected_features].copy()
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        
        # Transform using saved pipeline
        transformed = pipeline.transform(matrix)
        feature_names = pipeline.get_feature_names_out().tolist()
        
        # Score
        values = transformed.astype(float)
        
        # Score in batches
        scores_list = []
        for i in range(0, len(values), batch_size):
            batch = values[i:i+batch_size]
            if hasattr(model, "score_samples"):
                batch_scores = model.score_samples(batch)
            else:
                batch_scores = -model.score_samples(batch)
            scores_list.append(batch_scores)
        
        all_scores = np.concatenate(scores_list)
        
        # Build result
        result = pd.DataFrame(index=range(len(all_scores)))
        result["anomaly_score"] = all_scores
        result["score_percentile"] = pd.Series(all_scores).rank(pct=True).values
        result["is_anomaly"] = all_scores >= threshold
        
        # Add metadata if available
        for col in ["protocol", "flow_id", "timestamp", "capture"]:
            if col in raw_df.columns:
                result[col] = raw_df[col].values
        
        return result
    
    def _load_torch_model(self, model_path: Path):
        """Load a saved PyTorch model."""
        import torch
        import inspect
        from anomdet.modelling.lstm_autoencoder import LSTMAutoencoder, _build_network as _build_lstm
        from anomdet.modelling.detectors import MLPAutoencoder
        
        payload = torch.load(model_path, map_location="cpu")
        arch = payload.get("architecture", "lstm_autoencoder")
        params = payload.get("parameters", {})
        input_size = payload.get("input_size")
        
        if arch == "lstm_autoencoder":
            # Filter params to only include constructor arguments
            sig = inspect.signature(LSTMAutoencoder.__init__)
            valid_params = {k: v for k, v in params.items() if k in sig.parameters}
            model = LSTMAutoencoder(**valid_params)
            # Build the network manually since we're not calling fit
            hidden = params.get("resolved_hidden_size") or valid_params.get("hidden_size") or 32
            latent = params.get("resolved_latent_size") or valid_params.get("latent_size") or 16
            layers = params.get("resolved_num_layers") or valid_params.get("num_layers") or 1
            dropout = valid_params.get("dropout", 0.1)
            model.model_ = _build_lstm(input_size, hidden, latent, layers, dropout)
        elif arch == "mlp_autoencoder":
            ModelClass = MLPAutoencoder
            sig = inspect.signature(ModelClass.__init__)
            valid_params = {k: v for k, v in params.items() if k in sig.parameters}
            model = ModelClass(**valid_params)
            # Build the network manually
            model.device_ = "cpu"
            model.model_ = model._build_network(input_size)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        model.model_.load_state_dict(payload["state_dict"])
        model.input_size_ = input_size
        model.device_ = "cpu"
        model.model_.eval()
        return model
    
    # -------------------------------------------------------------------------
    # Utility: Quick end-to-end
    # -------------------------------------------------------------------------
    def quick_train(
        self,
        pcap_path: Path | str,
        features: list[str] | None = None,
        models: list[str] | None = None,
    ) -> list[TrainResult]:
        """
        One-call end-to-end: extract -> profile -> train.
        
        Args:
            pcap_path: PCAP file or folder
            features: Feature names for profile (None = use all)
            models: Models to train
            
        Returns:
            List of TrainResult
        """
        # Extract
        features_path = self.extract_features(pcap_path)
        
        # Create profile if features specified
        profile_path = None
        if features:
            profile_path = self.create_profile(
                "auto_profile", features, "Auto-created for quick_train"
            )
        
        # Train
        return self.train(features_path, profile_path, models=models)
    
    # -------------------------------------------------------------------------
    # Utility: List artifacts
    # -------------------------------------------------------------------------
    def list_runs(self) -> list[dict[str, Any]]:
        """List all experiment runs."""
        runs = []
        for run_dir in sorted((self.artifact_root / "runs").glob("*")):
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "run-summary.json"
            if summary_file.exists():
                import json
                with open(summary_file) as f:
                    s = json.load(f)
                runs.append({
                    "run_id": s.get("run_id"),
                    "created": s.get("created_at"),
                    "datasets": s.get("dataset_count"),
                    "path": str(run_dir),
                })
        return runs
    
    def list_experiments(self) -> list[dict[str, Any]]:
        """List all model experiments."""
        exps = []
        for exp_dir in sorted((self.artifact_root / "experiments").glob("*")):
            if not exp_dir.is_dir():
                continue
            comparison_file = exp_dir / "comparison.parquet"
            if comparison_file.exists():
                df = pd.read_parquet(comparison_file)
                exps.append({
                    "name": exp_dir.name,
                    "path": str(exp_dir),
                    "models": df["model"].nunique() if "model" in df.columns else 0,
                    "profiles": df["feature_profile"].nunique() if "feature_profile" in df.columns else 1,
                    "protocols": df["protocols"].iloc[0] if "protocols" in df.columns else "unknown",
                })
        return exps


# Convenience function for simplest usage
def quick_anomaly_detection(
    pcap_path: Path | str,
    features: list[str] | None = None,
    models: list[str] | None = None,
    artifact_dir: Path | str = "./artifacts",
) -> list[TrainResult]:
    """
    Simplest possible interface - one function call.
    
    Example:
        results = quick_anomaly_detection(
            "traffic.pcap",
            features=["packet_length", "flow_duration", "packet_rate", "payload_entropy"],
            models=["isolation_forest", "mlp_autoencoder"]
        )
        # results[0].artifacts.model_file has the saved model
        # results[0].metrics has performance numbers
    """
    service = AnomalyService(artifact_dir=artifact_dir)
    return service.quick_train(pcap_path, features, models)


# Example usage
if __name__ == "__main__":
    # This demonstrates the intended usage pattern
    service = AnomalyService()
    
    # 1. See available profiles
    print("Available profiles:")
    for p in service.list_profiles():
        print(f"  {p['name']} v{p['version']}: {p['feature_count']} features")
    
    # 2. Create custom profile
    # my_profile = service.create_profile("my_modbus", 
    #     ["packet_length", "flow_duration", "packet_rate", "payload_entropy"],
    #     protocols=["modbus"])
    
    # 3. Train
    # results = service.train("artifacts/features/my_capture.parquet", my_profile)
    
    # 4. Score new data
    # scores = service.score("artifacts/features/new_capture.parquet", results[0].artifacts)
    
    print("\nService ready. Use AnomalyService() or quick_anomaly_detection()")