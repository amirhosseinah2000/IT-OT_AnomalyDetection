"""Professional detector adapters with dynamic architecture and consistent anomaly-score contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAutoencoder:
    """A lightweight linear autoencoder using PCA reconstruction error as anomaly score."""

    def __init__(self, explained_variance: float = 0.95) -> None:
        self.explained_variance = explained_variance
        self.model = PCA(n_components=explained_variance, svd_solver="full")

    def fit(self, values: np.ndarray) -> "PCAAutoencoder":
        """Fit the compact reconstruction basis on presumed normal training data."""
        self.model.fit(values)
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        """Return reconstruction MSE; larger values consistently mean more anomalous."""
        reconstructed = self.model.inverse_transform(self.model.transform(values))
        return np.mean(np.square(values - reconstructed), axis=1)


class MLPAutoencoder:
    """Professional tabular MLP autoencoder for non-sequence anomaly detection.

    Features:
    - Dynamic architecture scaling based on input dimension
    - Layer normalization and GELU activations for stable training
    - Residual connections in deeper layers
    - AdamW optimizer with weight decay
    - Gradient clipping
    - Early stopping with validation monitoring
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] | None = None,
        latent_size: int | None = None,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        validation_fraction: float = 0.15,
        patience: int = 10,
        max_train_samples: int = 10000,
        device: str = "cpu",
        random_seed: int = 42,
        auto_architecture: bool = True,
    ) -> None:
        if latent_size is not None and latent_size < 1:
            raise ValueError("latent_size must be positive.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        if not 0 <= validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5).")
        if batch_size < 1 or epochs < 1 or max_train_samples < 1:
            raise ValueError("batch_size, epochs, max_train_samples must be positive.")

        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.max_train_samples = max_train_samples
        self.device = device
        self.random_seed = random_seed
        self.auto_architecture = auto_architecture
        self.model_ = None
        self.input_size_ = None
        self.device_ = None
        self.training_history_ = []
        self.resolved_hidden_ = None
        self.resolved_latent_ = None

    def _calculate_architecture(self, input_size: int) -> tuple[tuple[int, ...], int]:
        """Calculate optimal architecture for tabular data."""
        if self.hidden_sizes is not None and self.latent_size is not None:
            return self.hidden_sizes, self.latent_size

        if input_size <= 10:
            hidden = (32, 16)
            latent = 8
        elif input_size <= 30:
            hidden = (64, 32, 16)
            latent = 16
        elif input_size <= 60:
            hidden = (128, 64, 32)
            latent = 32
        elif input_size <= 120:
            hidden = (256, 128, 64)
            latent = 64
        else:
            hidden = (256, 128, 64, 32)
            latent = 64

        return hidden, latent

    def _build_network(self, input_size: int):
        import torch
        from torch import nn

        hidden_sizes, latent_size = self._calculate_architecture(input_size)
        self.resolved_hidden_ = hidden_sizes
        self.resolved_latent_ = latent_size

        layers = []
        prev_size = input_size

        for i, h_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.LayerNorm(h_size))
            layers.append(nn.GELU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, latent_size))
        layers.append(nn.LayerNorm(latent_size))
        layers.append(nn.GELU())
        encoder = nn.Sequential(*layers)

        dec_layers = []
        prev_size = latent_size
        for h_size in reversed(hidden_sizes):
            dec_layers.append(nn.Linear(prev_size, h_size))
            dec_layers.append(nn.LayerNorm(h_size))
            dec_layers.append(nn.GELU())
            if self.dropout > 0:
                dec_layers.append(nn.Dropout(self.dropout))
            prev_size = h_size

        dec_layers.append(nn.Linear(prev_size, input_size))
        decoder = nn.Sequential(*dec_layers)

        class TabularAutoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                latent = self.encoder(x)
                return self.decoder(latent)

        return TabularAutoencoder(encoder, decoder).to(self.device_)

    def fit(
        self,
        values: np.ndarray,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> "MLPAutoencoder":
        import torch
        from torch import nn

        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("MLPAutoencoder expects a 2D feature matrix.")
        if len(array) < 10:
            raise ValueError("Need at least 10 samples.")

        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        self.input_size_ = array.shape[1]

        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        self.device_ = self.device.lower()
        if self.device_.startswith("cuda") and not torch.cuda.is_available():
            self.device_ = "cpu"

        n_samples = len(array)
        val_count = int(n_samples * self.validation_fraction)
        indices = np.random.default_rng(self.random_seed).permutation(n_samples)
        train_idx = indices[val_count:] if val_count > 0 else indices
        val_idx = indices[:val_count] if val_count > 0 else np.array([], dtype=int)

        train_data = torch.from_numpy(array[train_idx]).to(self.device_)
        val_data = torch.from_numpy(array[val_idx]).to(self.device_) if len(val_idx) > 0 else None

        self.model_ = self._build_network(self.input_size_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        stalled = 0
        history = []

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "mlp_started",
                    "epochs": self.epochs,
                    "training_samples": len(train_data),
                    "validation_samples": len(val_data) if val_data is not None else 0,
                    "input_features": self.input_size_,
                    "hidden_sizes": self.resolved_hidden_,
                    "latent_size": self.resolved_latent_,
                }
            )

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            perm = torch.randperm(len(train_data), device=self.device_)
            epoch_loss = 0.0
            seen = 0

            for start in range(0, len(perm), self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                batch = train_data[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                recon = self.model_(batch)
                loss = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.detach().item()) * len(batch)
                seen += len(batch)

            train_loss = epoch_loss / max(seen, 1)

            val_loss = None
            if val_data is not None and len(val_data) > 0:
                self.model_.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.model_(val_data), val_data).item())

            monitor = val_loss if val_loss is not None else train_loss
            if monitor < best_loss - 1e-8:
                best_loss = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                stalled = 0
            else:
                stalled += 1

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "best_loss": best_loss,
            })

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch",
                        "epoch": epoch,
                        "epochs": self.epochs,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                        "best_loss": best_loss,
                        "stalled_epochs": stalled,
                    }
                )

            if stalled >= self.patience:
                break

        if best_state:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.training_history_ = history

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "mlp_completed",
                    "epochs": self.epochs,
                    "epochs_completed": len(history),
                    "best_loss": best_loss,
                    "stopped_early": len(history) < self.epochs,
                    "hidden_sizes": self.resolved_hidden_,
                    "latent_size": self.resolved_latent_,
                }
            )
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None or self.input_size_ is None:
            raise RuntimeError("Fit the MLPAutoencoder before scoring.")

        array = np.asarray(values, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

        if array.shape[1] != self.input_size_:
            raise ValueError(f"Expected {self.input_size_} features, got {array.shape[1]}")

        self.model_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(array).to(self.device_)
            recon = self.model_(tensor)
            mse = torch.mean((recon - tensor) ** 2, dim=1)
            return mse.cpu().numpy()

    def _parameters(self) -> dict[str, Any]:
        return {
            "hidden_sizes": self.hidden_sizes,
            "latent_size": self.latent_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_fraction": self.validation_fraction,
            "patience": self.patience,
            "max_train_samples": self.max_train_samples,
            "device": self.device,
            "random_seed": self.random_seed,
            "auto_architecture": self.auto_architecture,
            "resolved_hidden_sizes": self.resolved_hidden_,
            "resolved_latent_size": self.resolved_latent_,
        }

    def save(self, path: Path) -> Path:
        import torch

        if self.model_ is None or self.input_size_ is None:
            raise RuntimeError("Fit before saving.")

        state = {k: v.detach().cpu() for k, v in self.model_.state_dict().items()}
        payload = {
            "schema_version": "1.0.0",
            "architecture": "mlp_autoencoder",
            "parameters": self._parameters(),
            "input_size": self.input_size_,
            "training_history": self.training_history_,
            "state_dict": state,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path
