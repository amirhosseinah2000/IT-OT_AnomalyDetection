"""CPU-first sequence LSTM autoencoder used for temporal anomaly scoring."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def _torch_modules() -> tuple[Any, Any]:
    """Import PyTorch only when this optional detector is actually used."""
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - depends on the local environment.
        raise RuntimeError(
            "The LSTM autoencoder requires PyTorch. Run `uv sync` after updating the project."
        ) from error
    return torch, nn


def _build_network(
    input_size: int,
    hidden_size: int,
    latent_size: int,
    num_layers: int,
    dropout: float,
) -> Any:
    """Build a compact sequence-to-sequence LSTM reconstruction network."""
    _torch, nn = _torch_modules()

    class SequenceAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.to_latent = nn.Linear(hidden_size, latent_size)
            self.decoder = nn.LSTM(
                input_size=latent_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.to_features = nn.Linear(hidden_size, input_size)

        def forward(self, values: Any) -> Any:
            _encoded, (hidden, _cell) = self.encoder(values)
            latent = self.to_latent(hidden[-1])
            repeated = latent.unsqueeze(1).expand(-1, values.shape[1], -1)
            decoded, _state = self.decoder(repeated)
            return self.to_features(decoded)

    return SequenceAutoencoder()


class LSTMAutoencoder:
    """Reconstruct contiguous feature windows and score their per-record error.

    The input remains the numeric matrix used by every detector. Unlike the
    tabular detectors, this model first groups adjacent records into windows so
    it can learn short-term traffic behaviour. Its score is reconstruction MSE
    aggregated back to one value for every input record.
    """

    def __init__(
        self,
        sequence_length: int = 16,
        sequence_stride: int = 16,
        hidden_size: int = 32,
        latent_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        validation_fraction: float = 0.15,
        patience: int = 5,
        max_train_windows: int = 5000,
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")
        if sequence_stride < 1 or sequence_stride > sequence_length:
            raise ValueError("sequence_stride must be between 1 and sequence_length.")
        if min(hidden_size, latent_size, num_layers, batch_size, epochs, max_train_windows) < 1:
            raise ValueError(
                "LSTM dimensions, batch size, epochs, and window cap must be positive."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1).")
        if not 0 <= validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in the interval [0, 0.5).")

        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.max_train_windows = max_train_windows
        self.device = device
        self.random_seed = random_seed
        self.model_: Any | None = None
        self.input_size_: int | None = None
        self.effective_sequence_length_: int | None = None
        self.device_: str | None = None
        self.training_history_: list[dict[str, float | int | None]] = []

    def _parameters(self) -> dict[str, Any]:
        """Return serialisable constructor settings for metrics and persistence."""
        return {
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_fraction": self.validation_fraction,
            "patience": self.patience,
            "max_train_windows": self.max_train_windows,
            "device": self.device,
            "random_seed": self.random_seed,
        }

    @staticmethod
    def _clean(values: np.ndarray) -> np.ndarray:
        """Normalize numerical edge cases before the tensor conversion."""
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("LSTMAutoencoder expects a two-dimensional feature matrix.")
        if len(array) < 2 or array.shape[1] < 1:
            raise ValueError("LSTMAutoencoder needs at least two rows and one feature.")
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def _windows(self, values: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Create contiguous, tail-complete windows and retain their source starts."""
        clean = self._clean(values)
        length = min(self.sequence_length, len(clean))
        stride = min(self.sequence_stride, length)
        starts = list(range(0, len(clean) - length + 1, stride))
        final_start = len(clean) - length
        if starts[-1] != final_start:
            starts.append(final_start)
        windows = np.stack([clean[start : start + length] for start in starts]).astype(np.float32)
        return windows, starts

    @staticmethod
    def _evenly_sample(windows: np.ndarray, maximum: int) -> np.ndarray:
        """Cap training cost without biasing the sequence sample to early traffic."""
        if len(windows) <= maximum:
            return windows
        positions = np.linspace(0, len(windows) - 1, num=maximum, dtype=int)
        return windows[positions]

    def _resolved_device(self, torch: Any) -> str:
        """Use an explicitly requested accelerator only when it is available."""
        requested = self.device.lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
        return "cpu"

    def fit(
        self,
        values: np.ndarray,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LSTMAutoencoder:
        """Fit the reconstruction model and optionally report epoch-level live progress."""
        torch, _nn = _torch_modules()
        windows, _starts = self._windows(values)
        sampled = self._evenly_sample(windows, self.max_train_windows)
        validation_count = int(round(len(sampled) * self.validation_fraction))
        if validation_count >= len(sampled) - 1:
            validation_count = 0
        training = sampled[:-validation_count] if validation_count else sampled
        validation = sampled[-validation_count:] if validation_count else None
        if len(training) < 2:
            raise ValueError("The LSTM autoencoder needs at least two training windows.")

        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        resolved_device = self._resolved_device(torch)
        model = _build_network(
            input_size=training.shape[2],
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(resolved_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()
        training_tensor = torch.from_numpy(training).to(resolved_device)
        validation_tensor = (
            torch.from_numpy(validation).to(resolved_device) if validation is not None else None
        )
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stalled_epochs = 0
        history: list[dict[str, float | int | None]] = []
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "lstm_started",
                    "epochs": self.epochs,
                    "training_windows": len(training),
                    "validation_windows": len(validation) if validation is not None else 0,
                    "sequence_length": int(training.shape[1]),
                    "input_features": int(training.shape[2]),
                }
            )

        for epoch in range(1, self.epochs + 1):
            model.train()
            order = torch.randperm(len(training_tensor), device=resolved_device)
            running_loss = 0.0
            seen = 0
            for start in range(0, len(order), self.batch_size):
                batch = training_tensor[order[start : start + self.batch_size]]
                optimizer.zero_grad(set_to_none=True)
                reconstructed = model(batch)
                loss = loss_function(reconstructed, batch)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().item()) * len(batch)
                seen += len(batch)
            training_loss = running_loss / max(seen, 1)

            validation_loss: float | None = None
            if validation_tensor is not None and len(validation_tensor):
                model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        loss_function(model(validation_tensor), validation_tensor).item()
                    )
            monitor = validation_loss if validation_loss is not None else training_loss
            if monitor < best_loss - 1e-8:
                best_loss = monitor
                best_state = deepcopy(model.state_dict())
                stalled_epochs = 0
            else:
                stalled_epochs += 1
            history_entry = {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": self.learning_rate,
                "best_loss": best_loss,
                "stalled_epochs": stalled_epochs,
            }
            history.append(history_entry)
            if progress_callback is not None:
                progress_callback({"event": "epoch", "epochs": self.epochs, **history_entry})
            if stalled_epochs >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.input_size_ = int(training.shape[2])
        self.effective_sequence_length_ = int(training.shape[1])
        self.device_ = resolved_device
        self.training_history_ = history
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "lstm_completed",
                    "epochs": self.epochs,
                    "epochs_completed": len(history),
                    "best_loss": best_loss,
                    "stopped_early": len(history) < self.epochs,
                }
            )
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        """Return one reconstruction-MSE anomaly score per source record."""
        if self.model_ is None or self.input_size_ is None or self.device_ is None:
            raise RuntimeError("Fit the LSTM autoencoder before scoring samples.")
        torch, _nn = _torch_modules()
        clean = self._clean(values)
        if clean.shape[1] != self.input_size_:
            raise ValueError("Input feature count does not match the fitted LSTM autoencoder.")
        windows, starts = self._windows(clean)
        errors: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = windows[start : start + self.batch_size]
                tensor = torch.from_numpy(batch).to(self.device_)
                reconstructed = self.model_(tensor).detach().cpu().numpy()
                errors.append(np.mean(np.square(reconstructed - batch), axis=2))
        window_errors = np.concatenate(errors, axis=0)
        total = np.zeros(len(clean), dtype=float)
        counts = np.zeros(len(clean), dtype=float)
        sequence_length = window_errors.shape[1]
        for start, error in zip(starts, window_errors, strict=True):
            total[start : start + sequence_length] += error
            counts[start : start + sequence_length] += 1
        return total / np.maximum(counts, 1.0)

    def save(self, path: Path) -> Path:
        """Persist only state and explicit metadata, never a pickled module object."""
        if self.model_ is None or self.input_size_ is None:
            raise RuntimeError("Fit the LSTM autoencoder before saving it.")
        torch, _nn = _torch_modules()
        state = {name: value.detach().cpu() for name, value in self.model_.state_dict().items()}
        payload = {
            "schema_version": "1.0.0",
            "architecture": "lstm_autoencoder",
            "parameters": self._parameters(),
            "input_size": self.input_size_,
            "effective_sequence_length": self.effective_sequence_length_,
            "training_history": self.training_history_,
            "state_dict": state,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path
