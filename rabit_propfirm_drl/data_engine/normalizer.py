"""
Running Normalizer — Welford's Online Algorithm for streaming normalization.

Key properties:
- Numerically stable (no catastrophic cancellation)
- Online: updates incrementally without storing all data
- Serializable: save/load state for live inference
- Clips extreme values at ±5σ to prevent outlier damage
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RunningNormalizer:
    """
    Online running normalization using Welford's algorithm.

    Tracks per-feature mean and variance incrementally.
    Normalizes to approximately N(0, 1) distribution.
    Clips at ±clip_sigma standard deviations to prevent extreme outliers.
    """

    def __init__(
        self,
        n_features: int,
        clip_sigma: float = 5.0,
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            n_features: Number of features to normalize
            clip_sigma: Clip normalized values at ±clip_sigma
            eps: Small constant for numerical stability in division
        """
        self.n_features = n_features
        self.clip_sigma = clip_sigma
        self.eps = eps

        # Welford's state
        self._count: int = 0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._m2 = np.zeros(n_features, dtype=np.float64)  # Sum of squared diffs

    @property
    def count(self) -> int:
        """Number of samples seen so far."""
        return self._count

    @property
    def mean(self) -> np.ndarray:
        """Current running mean."""
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        """Current running variance (population variance)."""
        if self._count < 2:
            return np.ones(self.n_features, dtype=np.float64)
        return self._m2 / self._count

    @property
    def std(self) -> np.ndarray:
        """Current running standard deviation."""
        return np.sqrt(self.var + self.eps)

    def update(self, batch: np.ndarray) -> None:
        """
        Update running statistics with a new batch of data.

        Uses Welford's online algorithm for numerical stability.

        Args:
            batch: Array of shape (batch_size, n_features) or (n_features,)
        """
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        if batch.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {batch.shape[1]}"
            )

        for x in batch:
            self._count += 1
            delta = x - self._mean
            self._mean += delta / self._count
            delta2 = x - self._mean
            self._m2 += delta * delta2

    def update_batch(self, batch: np.ndarray) -> None:
        """
        Batch update — more efficient than row-by-row for large batches.

        Uses Chan's parallel algorithm to merge batch statistics.

        Args:
            batch: Array of shape (batch_size, n_features)
        """
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        n_b = batch.shape[0]
        if n_b == 0:
            return

        mean_b = batch.mean(axis=0)
        # Population variance
        var_b = batch.var(axis=0)
        m2_b = var_b * n_b

        n_a = self._count
        mean_a = self._mean

        # Combined count
        n_combined = n_a + n_b
        if n_combined == 0:
            return

        # Chan's parallel merge
        delta = mean_b - mean_a
        self._mean = (n_a * mean_a + n_b * mean_b) / n_combined
        self._m2 = self._m2 + m2_b + delta ** 2 * n_a * n_b / n_combined
        self._count = n_combined

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize input using current running statistics.

        Args:
            x: Array of shape (batch_size, n_features) or (n_features,)

        Returns:
            Normalized array, clipped to ±clip_sigma
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x.reshape(1, -1)

        normalized = (x - self._mean) / self.std
        clipped = np.clip(normalized, -self.clip_sigma, self.clip_sigma)

        if squeeze:
            return clipped.squeeze(0)
        return clipped

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization (for debugging/logging)."""
        return x_norm * self.std + self._mean

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def state_dict(self) -> dict:
        """Export state for serialization."""
        return {
            "n_features": self.n_features,
            "clip_sigma": self.clip_sigma,
            "eps": self.eps,
            "count": self._count,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningNormalizer":
        """Create normalizer from saved state."""
        norm = cls(
            n_features=state["n_features"],
            clip_sigma=state["clip_sigma"],
            eps=state["eps"],
        )
        norm._count = state["count"]
        norm._mean = np.array(state["mean"], dtype=np.float64)
        norm._m2 = np.array(state["m2"], dtype=np.float64)
        return norm

    def save(self, path: Path | str) -> None:
        """Save normalizer state to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)
        logger.info("Saved normalizer state to %s (count=%d)", path, self._count)

    @classmethod
    def load(cls, path: Path | str) -> "RunningNormalizer":
        """Load normalizer state from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalizer state not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        norm = cls.from_state_dict(state)
        logger.info("Loaded normalizer state from %s (count=%d)", path, norm._count)
        return norm

    def __repr__(self) -> str:
        return (
            f"RunningNormalizer(n_features={self.n_features}, "
            f"count={self._count}, clip_sigma={self.clip_sigma})"
        )
