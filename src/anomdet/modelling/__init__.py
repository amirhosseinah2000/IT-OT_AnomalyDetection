"""CPU-first unsupervised anomaly model comparison."""

from .training import train_models
from .random_forest import train_attack_random_forest

__all__ = ["train_attack_random_forest", "train_models"]
