"""Network anomaly detection platform package."""

__version__ = "0.1.0"

from anomdet.service import AnomalyService, ModelArtifacts, TrainResult, quick_anomaly_detection

__all__ = [
    "AnomalyService",
    "ModelArtifacts", 
    "TrainResult",
    "quick_anomaly_detection",
]
