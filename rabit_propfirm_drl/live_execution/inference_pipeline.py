"""
Live Inference Pipeline — Connects trained model to real-time data.

Pipeline:
1. Fetch latest M1 data from MT5
2. Resample to M15/H1/H4 timeframes
3. Build features for each timeframe
4. Normalize with loaded RunningNormalizer state
5. Convert to PyTorch tensors (sliding window)
6. Run inference through trained model
7. Map action → trading decision
8. Execute through MT5 connector (with killswitch check)

Designed as a single `infer_and_act()` call per M15 bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of a single inference step."""
    timestamp: str
    action: dict[str, float]   # confidence, risk_frac, sl_mult, tp_mult
    regime: str
    regime_probs: dict[str, float]
    decision: str              # "BUY", "SELL", or "HOLD"
    killswitch_status: str     # "normal", "soft", "hard", "emergency"
    executed: bool
    details: dict[str, Any]


class LiveInferencePipeline:
    """
    End-to-end inference pipeline for live trading.

    Takes raw market data, processes through the full feature pipeline,
    runs the trained model, and produces a trading decision.
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        config: dict,
        normalizer_state_path: Path | str | None = None,
        regime_fn: Optional[Callable[[np.ndarray], tuple[str, dict]]] = None,
    ) -> None:
        """
        Args:
            model_fn: Function that takes state array → action array
            config: Dict from prop_rules.yaml
            normalizer_state_path: Path to RunningNormalizer saved state
            regime_fn: Optional function to detect trading regime
        """
        self.model_fn = model_fn
        self.config = config
        self.regime_fn = regime_fn

        self.confidence_threshold = config.get("confidence_threshold", 0.3)
        self.trading_start = config.get("trading_start_utc", 1)
        self.trading_end = config.get("trading_end_utc", 21)

        # Load normalizer if path provided
        self._normalizer = None
        if normalizer_state_path:
            from data_engine.normalizer import RunningNormalizer
            self._normalizer = RunningNormalizer.load(normalizer_state_path)

        self.inference_history: list[InferenceResult] = []

    def infer(
        self,
        features: np.ndarray,
        hour_utc: int | None = None,
        killswitch_status: str = "normal",
    ) -> InferenceResult:
        """
        Run inference on prepared features.

        Args:
            features: (n_features,) array of current features
            hour_utc: Current hour in UTC (auto-detected if None)
            killswitch_status: Current killswitch status

        Returns:
            InferenceResult with decision and details
        """
        now = datetime.now(timezone.utc)
        if hour_utc is None:
            hour_utc = now.hour

        # Normalize if normalizer available
        if self._normalizer is not None:
            features = self._normalizer.normalize(features)

        # Run model
        action = self.model_fn(features)

        # Parse action
        confidence = float(np.clip(action[0], -1.0, 1.0))
        risk_frac = float(np.clip(action[1], 0.0, 1.0))
        sl_mult = float(np.clip(action[2], 0.5, 3.0))
        tp_mult = float(np.clip(action[3], 0.5, 5.0))

        # Detect regime
        regime = "unknown"
        regime_probs: dict[str, float] = {}
        if self.regime_fn is not None:
            regime, regime_probs = self.regime_fn(features)

        # Decision logic
        decision = "HOLD"
        executed = False

        # Check killswitch
        if killswitch_status in ("hard", "emergency"):
            decision = "BLOCKED"
        elif not (self.trading_start <= hour_utc < self.trading_end):
            decision = "OUTSIDE_HOURS"
        elif killswitch_status == "soft":
            # Soft kill: only trade with very high confidence
            if abs(confidence) >= self.confidence_threshold * 1.5:
                decision = "BUY" if confidence > 0 else "SELL"
                risk_frac *= 0.5  # Reduce exposure
        elif abs(confidence) >= self.confidence_threshold:
            decision = "BUY" if confidence > 0 else "SELL"

        result = InferenceResult(
            timestamp=now.isoformat(),
            action={
                "confidence": confidence,
                "risk_fraction": risk_frac,
                "sl_mult": sl_mult,
                "tp_mult": tp_mult,
            },
            regime=regime,
            regime_probs=regime_probs,
            decision=decision,
            killswitch_status=killswitch_status,
            executed=executed,
            details={
                "hour_utc": hour_utc,
                "raw_action": action.tolist() if hasattr(action, 'tolist') else list(action),
            },
        )

        self.inference_history.append(result)
        return result

    def get_session_stats(self) -> dict[str, Any]:
        """Get summary statistics for current trading session."""
        if not self.inference_history:
            return {"total_inferences": 0}

        decisions = [r.decision for r in self.inference_history]
        return {
            "total_inferences": len(self.inference_history),
            "buys": decisions.count("BUY"),
            "sells": decisions.count("SELL"),
            "holds": decisions.count("HOLD"),
            "blocked": decisions.count("BLOCKED"),
            "outside_hours": decisions.count("OUTSIDE_HOURS"),
        }
