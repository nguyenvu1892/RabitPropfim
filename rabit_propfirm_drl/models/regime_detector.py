"""
RegimeDetector — Market regime classification using SMC-derived features.

Purpose:
    Identifies the current market regime so the AI can ADAPT its strategy:
    - Trending-up:   Follow BOS breakouts, use pullbacks into FVG/OB as entries
    - Trending-down:  Same logic but short direction
    - Ranging:        Trade between liquidity zones, mean-revert from OB extremes
    - Volatile:       Reduce position size, widen SL, or sit flat

    A trader who applies the same strategy in all conditions WILL fail.
    This module gives the AI "market awareness."

Architecture:
    ┌─────────────────────────────────────────────────┐
    │  SMC-Derived Feature Extraction                 │
    │  (NO ATR/ADX/RSI — only from our 28 features)  │
    │                                                  │
    │  1. trend_strength  ← swing_trend mean          │
    │  2. vol_percentile  ← log_return std percentile │
    │  3. range_width     ← (high-low)/close range    │
    │  4. bos_frequency   ← BOS count / window        │
    │  5. volume_intensity← climax_vol frequency      │
    └─────────────────────────────────────────────────┘
                        │
               ┌────────▼────────┐
               │  GMM Clustering  │  ← fit() offline on historical data
               │  4 components    │     Learns: "what does trending look like?"
               │  (sklearn-free)  │     Uses K-means initialization + EM
               └────────▼────────┘
                        │
               ┌────────▼────────┐
               │  Neural Head     │  ← For end-to-end DRL integration
               │  MLP → softmax   │     Input: raw features (128-dim from TF)
               │  → regime_probs  │     Output: (batch, 4) probabilities
               └─────────────────┘

Dual-mode operation:
    1. Statistical mode (fit/predict): Offline clustering on SMC features
    2. Neural mode (forward): End-to-end trainable with DRL agent

All logic derived from SMC features — ZERO traditional indicators.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Part 1: Statistical Regime Detector (fit/predict with GMM)
# ═══════════════════════════════════════════════════════════════════

class SMCFeatureExtractor:
    """
    Extracts regime-relevant features from SMC data.

    HARD RULE: NO traditional indicators (ATR, ADX, RSI, MA).
    All features derived from existing SMC/PA/Volume columns.

    Features extracted (5-dimensional):
        1. trend_strength:  Mean of swing_trend over window
           → +1 = strong uptrend, -1 = strong downtrend, ~0 = no trend
        2. volatility_pctl: Percentile of log_return std within rolling history
           → 0.0 = very calm, 1.0 = extremely volatile
        3. range_ratio:     Avg (high-low)/close normalized by historical
           → Small = tight range, Large = wide swings
        4. bos_frequency:   Count of BOS signals / window length
           → High = strong directional structure breaks
        5. volume_climax:   Frequency of climax volume events
           → High = institutional activity / exhaustion

    These 5 features naturally separate the 4 regimes:
        trend_up:    trend > 0, bos_freq high, vol moderate
        trend_down:  trend < 0, bos_freq high, vol moderate
        ranging:     trend ≈ 0, bos_freq low, range small
        volatile:    vol_pctl high, climax_vol high, range large
    """

    REGIME_FEATURE_NAMES: list[str] = [
        "trend_strength", "volatility_pctl", "range_ratio",
        "bos_frequency", "volume_climax_freq",
    ]

    @staticmethod
    def extract(
        swing_trend: np.ndarray,
        log_return: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        bos: np.ndarray,
        climax_vol: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """
        Extract 5 regime features from raw SMC data.

        Args:
            swing_trend: Swing trend values (-1/0/+1 series)
            log_return: Log return series
            high: High price series
            low: Low price series
            close: Close price series
            bos: BOS signal series (0/1)
            climax_vol: Climax volume signal series (0/1)
            window: Rolling window size for feature calculation

        Returns:
            (n_windows, 5) — one 5-dim feature vector per window
        """
        n = len(swing_trend)
        n_windows = (n - window) // window
        if n_windows < 1:
            n_windows = 1

        features = np.zeros((n_windows, 5), dtype=np.float32)

        for i in range(n_windows):
            start = i * window
            end = start + window

            # 1. Trend strength: mean swing_trend in window
            #    → +1 = all bars bullish trend, 0 = mixed, -1 = all bearish
            features[i, 0] = np.mean(swing_trend[start:end])

            # 2. Volatility percentile: std of log_returns, normalized
            #    → ranked vs all windows (done post-extraction via percentile)
            features[i, 1] = np.std(log_return[start:end])

            # 3. Range ratio: avg (high-low)/close
            #    → narrow range = ranging, wide range = volatile/trending
            range_pcts = (high[start:end] - low[start:end]) / np.clip(close[start:end], 1e-8, None)
            features[i, 2] = np.mean(range_pcts)

            # 4. BOS frequency: structure breaks per bar
            #    → high = aggressive directional structure
            features[i, 3] = np.sum(bos[start:end]) / window

            # 5. Volume climax frequency: climax events per bar
            #    → high = institutional activity, potential reversals
            features[i, 4] = np.sum(climax_vol[start:end]) / window

        # Normalize volatility to percentile (rank within dataset)
        if n_windows > 1:
            vol_values = features[:, 1]
            ranks = np.argsort(np.argsort(vol_values)).astype(np.float32)
            features[:, 1] = ranks / max(n_windows - 1, 1)

        return features


class GaussianMixtureRegime:
    """
    Lightweight Gaussian Mixture Model for regime clustering.

    Implemented from scratch (no sklearn dependency) using EM algorithm.

    Clustering method:
        1. Initialize centroids with K-means++ inspired logic
        2. Run Expectation-Maximization (EM):
           - E-step: Compute posterior P(regime | features) for each sample
           - M-step: Update mean, covariance, mixing weights for each regime
        3. After convergence, predict() returns posterior probabilities

    Why GMM (not K-means)?
        - K-means gives hard assignments (0 or 1)
        - GMM gives SOFT probabilities — e.g., "70% trending-up, 30% volatile"
        - Soft probabilities are much more useful for the DRL agent
    """

    def __init__(self, n_regimes: int = 4, max_iter: int = 100, tol: float = 1e-4) -> None:
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol

        # GMM parameters (set during fit)
        self.means: np.ndarray | None = None          # (K, D)
        self.covariances: np.ndarray | None = None     # (K, D, D)
        self.weights: np.ndarray | None = None         # (K,)
        self.fitted: bool = False

    def fit(self, X: np.ndarray) -> "GaussianMixtureRegime":
        """
        Fit GMM on feature data using EM algorithm.

        Args:
            X: (n_samples, n_features) — regime feature vectors

        Returns:
            self (for chaining)
        """
        n_samples, n_features = X.shape
        K = self.n_regimes

        # ── Initialization (K-means++ inspired) ──
        # Pick first centroid randomly, then pick subsequent centroids
        # with probability proportional to distance from nearest centroid.
        means = np.zeros((K, n_features), dtype=np.float64)
        idx = np.random.randint(0, n_samples)
        means[0] = X[idx]

        for k in range(1, K):
            dists = np.array([
                np.min([np.sum((x - means[j]) ** 2) for j in range(k)])
                for x in X
            ])
            probs = dists / (dists.sum() + 1e-10)
            idx = np.random.choice(n_samples, p=probs)
            means[k] = X[idx]

        # Initialize covariances as identity × data variance
        data_var = np.var(X, axis=0).mean() + 1e-6
        covariances = np.array([np.eye(n_features) * data_var for _ in range(K)])

        # Initialize weights uniformly
        weights = np.ones(K) / K

        # ── EM Algorithm ──
        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities γ(k|x)
            # γ(k|x) = π_k × N(x|μ_k, Σ_k) / Σ_j π_j × N(x|μ_j, Σ_j)
            responsibilities = np.zeros((n_samples, K))
            for k in range(K):
                responsibilities[:, k] = weights[k] * self._gaussian_pdf(
                    X, means[k], covariances[k]
                )

            # Normalize (each sample's responsibilities sum to 1)
            resp_sum = responsibilities.sum(axis=1, keepdims=True) + 1e-300
            responsibilities /= resp_sum

            # Log likelihood for convergence check
            log_likelihood = np.sum(np.log(resp_sum.squeeze() + 1e-300))

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

            # M-step: Update parameters
            Nk = responsibilities.sum(axis=0) + 1e-10  # effective count per regime

            for k in range(K):
                # Updated mean
                means[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]

                # Updated covariance
                diff = X - means[k]  # (N, D)
                weighted_diff = responsibilities[:, k:k+1] * diff  # (N, D)
                covariances[k] = (weighted_diff.T @ diff) / Nk[k]
                # Add regularization to prevent singular covariance
                covariances[k] += np.eye(n_features) * 1e-6

            # Updated mixing weights
            weights = Nk / n_samples

        self.means = means.astype(np.float32)
        self.covariances = covariances.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.fitted = True

        # ── Auto-label regimes based on cluster centroids ──
        # Sort regimes by trend_strength (feature 0) to get consistent labeling:
        #   Highest trend → trend_up, lowest → trend_down
        #   Among remaining: higher volatility → volatile, lower → ranging
        self._assign_regime_labels()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for new data.

        Args:
            X: (n_samples, n_features) — regime feature vectors

        Returns:
            (n_samples, n_regimes) — regime probabilities (sum = 1 per sample)
        """
        assert self.fitted, "Must call fit() before predict()"

        n_samples = X.shape[0]
        K = self.n_regimes
        responsibilities = np.zeros((n_samples, K), dtype=np.float32)

        for k in range(K):
            responsibilities[:, k] = self.weights[k] * self._gaussian_pdf(
                X, self.means[k], self.covariances[k]
            )

        # Normalize
        resp_sum = responsibilities.sum(axis=1, keepdims=True) + 1e-300
        responsibilities /= resp_sum

        # Reorder to [trend_up, trend_down, ranging, volatile]
        return responsibilities[:, self._label_order]

    def _gaussian_pdf(
        self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        """Multivariate Gaussian probability density."""
        n_features = X.shape[1]
        diff = X - mean  # (N, D)

        # Regularize covariance
        cov_reg = cov + np.eye(n_features) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov_reg)
            det = max(np.linalg.det(cov_reg), 1e-300)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(n_features)
            det = 1.0

        # Mahalanobis distance
        mahal = np.sum(diff @ cov_inv * diff, axis=1)  # (N,)

        # PDF
        norm_const = np.sqrt((2 * np.pi) ** n_features * det)
        return np.exp(-0.5 * mahal) / (norm_const + 1e-300)

    def _assign_regime_labels(self) -> None:
        """
        Auto-assign regime labels based on cluster centroids.

        Strategy: use feature 0 (trend_strength) and feature 1 (volatility)
        to map clusters to semantic regime names.
        """
        K = self.n_regimes
        trend_values = self.means[:, 0]  # trend_strength per cluster
        vol_values = self.means[:, 1]    # volatility per cluster

        # Sort by trend: highest = trend_up, lowest = trend_down
        sorted_by_trend = np.argsort(trend_values)
        trend_down_idx = sorted_by_trend[0]
        trend_up_idx = sorted_by_trend[-1]

        # Remaining 2: higher vol = volatile, lower = ranging
        remaining = [i for i in range(K) if i not in [trend_up_idx, trend_down_idx]]
        if len(remaining) >= 2:
            if vol_values[remaining[0]] > vol_values[remaining[1]]:
                volatile_idx, ranging_idx = remaining[0], remaining[1]
            else:
                volatile_idx, ranging_idx = remaining[1], remaining[0]
        elif len(remaining) == 1:
            volatile_idx = remaining[0]
            ranging_idx = remaining[0]
        else:
            volatile_idx, ranging_idx = 0, 1

        # Order: [trend_up, trend_down, ranging, volatile]
        self._label_order = np.array([trend_up_idx, trend_down_idx, ranging_idx, volatile_idx])


# ═══════════════════════════════════════════════════════════════════
# Part 2: Neural Regime Detector (for end-to-end DRL training)
# ═══════════════════════════════════════════════════════════════════

class RegimeDetector(nn.Module):
    """
    Neural regime classifier for end-to-end DRL integration.

    Dual-mode:
    1. Can be initialized from GMM clusters (transfer statistical knowledge)
    2. Can be trained end-to-end with actor-critic (learned regime awareness)

    Input:  (batch, input_dim) — encoded features from Transformer/CrossAttention
    Output: (regime_probs, regime_embedding)
            - regime_probs: (batch, n_regimes=4)
            - regime_embedding: (batch, input_dim) — weighted regime representation

    Regime names (in order):
        [0] trend_up, [1] trend_down, [2] ranging, [3] volatile
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            input_dim: Dimension of input features
            n_regimes: Number of market regimes (default: 4)
            hidden_dim: Hidden layer size for classification head
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.n_regimes = n_regimes
        self.input_dim = input_dim

        # Classification head: features → regime logits
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_regimes),
        )

        # Regime embeddings: each regime has a learnable vector
        # These are mixed by regime_probs to produce a regime-aware embedding
        self.regime_embeddings = nn.Embedding(n_regimes, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: features → regime probabilities + embedding.

        Args:
            x: (batch, input_dim) — encoded features

        Returns:
            (regime_probs, regime_embedding)
            - regime_probs: (batch, n_regimes) — soft probabilities (sum=1)
            - regime_embedding: (batch, input_dim) — weighted regime representation
        """
        # Classify → probabilities
        logits = self.classifier(x)               # (B, n_regimes)
        probs = F.softmax(logits, dim=-1)         # (B, n_regimes) — sum = 1

        # Weighted sum of regime embeddings
        all_emb = self.regime_embeddings.weight.unsqueeze(0)  # (1, K, D)
        weighted = probs.unsqueeze(-1) * all_emb              # (B, K, D)
        regime_emb = weighted.sum(dim=1)                      # (B, D)

        return probs, regime_emb

    def predict_regime(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hard regime prediction.

        Args:
            x: (batch, input_dim)

        Returns:
            (batch,) — regime indices [0=trend_up, 1=trend_down, 2=ranging, 3=volatile]
        """
        logits = self.classifier(x)
        return logits.argmax(dim=-1)

    @property
    def regime_names(self) -> list[str]:
        """Human-readable regime names in order."""
        names = ["trend_up", "trend_down", "ranging", "volatile"]
        return names[:self.n_regimes]

    def init_from_gmm(self, gmm: GaussianMixtureRegime, feature_projection: nn.Module) -> None:
        """
        Transfer knowledge from fitted GMM to neural classifier.

        Can be used to "warm start" the neural regime detector with
        statistical cluster knowledge before end-to-end DRL training.

        Args:
            gmm: Fitted GaussianMixtureRegime instance
            feature_projection: Module that projects raw features to regime features
        """
        # This is a hook for future integration —
        # the GMM centroids could initialize the classifier biases
        pass
