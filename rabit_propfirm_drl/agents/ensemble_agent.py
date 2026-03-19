"""
EnsembleAgent — Regime-Aware Confidence-Weighted Multi-Agent Voting.

Architecture:
    3 specialist agents (Trend, Range, Volatility) each produce an action.
    RegimeDetector boosts the weight of the most relevant specialist.
    Final direction = weighted sum of all agent directions.
    SL/TP/Risk = from the TOP-SCORING agent (highest weight * confidence).

    ┌─────────────┐
    │ RegimeDetect │──► regime_id + conf ──► boost_weights
    └─────────────┘
           │
    ┌──────┼──────┐
    │      │      │
    ▼      ▼      ▼
  Trend  Range  Vol     (3 SACTransformerActors)
    │      │      │
    └──────┼──────┘
           ▼
    Weighted Score Vote
    direction = Σ(a[0] * conf * weight)
    SL/TP    = from argmax(scores)
           │
           ▼
    ActionGating (final safety)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Protocol for type hints (duck-typing friendly)
# ------------------------------------------------------------------

class ActorProtocol(Protocol):
    """Any model with forward(m5, h1, h4, deterministic) -> (action, log_prob)."""
    def __call__(
        self, m5: torch.Tensor, h1: torch.Tensor, h4: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class GatingProtocol(Protocol):
    """Any object with gate_single(action) -> GatedAction."""
    def gate_single(self, action: torch.Tensor) -> object: ...


# ------------------------------------------------------------------
# Regime boost lookup table (precomputed for speed)
# ------------------------------------------------------------------

# regime_id: 0=trend_up, 1=trend_down, 2=ranging, 3=volatile
# Columns: [TrendAgent, RangeAgent, VolAgent]
REGIME_BOOST_TABLE = np.array([
    [1.5, 0.5, 1.0],  # 0: trend_up   -> boost Trend
    [1.5, 0.5, 1.0],  # 1: trend_down -> boost Trend
    [0.5, 1.5, 1.0],  # 2: ranging    -> boost Range
    [0.5, 1.0, 1.5],  # 3: volatile   -> boost Vol
], dtype=np.float32)

# Weight clamps
MAX_WEIGHT = 0.6
MIN_WEIGHT = 0.1


# ------------------------------------------------------------------
# EnsembleAgent
# ------------------------------------------------------------------

class EnsembleAgent:
    """
    Regime-Aware Confidence-Weighted Multi-Agent Voting System.

    Combines 3 specialist agents using dynamic regime-based weights.
    Direction is determined by weighted score sum across all agents.
    SL/TP/Risk parameters come from the highest-scoring specialist.

    Usage:
        ensemble = EnsembleAgent(
            agents=[trend_actor, range_actor, vol_actor],
            regime_detector=regime_detector_module,
            action_gating=gating,
        )
        gated = ensemble.get_action(m5, h1, h4)
    """

    def __init__(
        self,
        agents: list[nn.Module],
        regime_detector: nn.Module | None = None,
        action_gating: GatingProtocol | None = None,
        base_weights: list[float] | None = None,
    ) -> None:
        """
        Args:
            agents: List of 3 SACTransformerActor specialists
                    [TrendAgent, RangeAgent, VolAgent]
            regime_detector: RegimeDetector module (or None for equal weights)
            action_gating: ActionGating instance (or None to skip gating)
            base_weights: Base weights before regime boost [0.4, 0.3, 0.3]
        """
        assert len(agents) >= 2, f"Need at least 2 agents, got {len(agents)}"

        self.agents = agents
        self.regime_detector = regime_detector
        self.action_gating = action_gating
        self.n_agents = len(agents)

        # Base weights (default: TrendAgent slightly favored)
        if base_weights is not None:
            self.base_weights = np.array(base_weights, dtype=np.float32)
        else:
            self.base_weights = np.array([0.4, 0.3, 0.3], dtype=np.float32)

        assert len(self.base_weights) == self.n_agents, (
            f"base_weights length ({len(self.base_weights)}) != "
            f"agents count ({self.n_agents})"
        )

        # Pre-allocate for speed
        self._actions_buf = np.zeros((self.n_agents, 4), dtype=np.float32)
        self._scores_buf = np.zeros(self.n_agents, dtype=np.float32)

    @torch.no_grad()
    def get_action(
        self,
        m5: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Get ensemble action from all specialists.

        Args:
            m5: (1, seq_m5, n_features) — M5 candle features
            h1: (1, seq_h1, n_features) — H1 context features
            h4: (1, seq_h4, n_features) — H4 context features
            deterministic: Use mean action (no sampling noise)

        Returns:
            action: (4,) numpy array [direction, risk, SL, TP]
        """
        # --- Step 1: Detect regime ---
        weights = self._compute_weights(m5)

        # --- Step 2: Get each agent's action + confidence ---
        for i, agent in enumerate(self.agents):
            action, _ = agent(m5, h1, h4, deterministic=deterministic)
            a_np = action.squeeze(0).cpu().numpy()  # (4,)
            self._actions_buf[i] = a_np

            # Confidence = |action[0]| (direction conviction)
            confidence = abs(a_np[0])
            self._scores_buf[i] = confidence * weights[i]

        # --- Step 3: Weighted direction (score-based, no majority) ---
        weighted_direction = 0.0
        for i in range(self.n_agents):
            weighted_direction += (
                self._actions_buf[i, 0]    # direction sign+magnitude
                * self._scores_buf[i]       # confidence * weight
            )

        # --- Step 4: SL/TP/Risk from TOP-SCORING agent ---
        top_idx = int(np.argmax(self._scores_buf))
        final_action = self._actions_buf[top_idx].copy()
        final_action[0] = weighted_direction  # Override direction

        return final_action

    def get_gated_action(
        self,
        m5: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
        deterministic: bool = True,
    ):
        """
        Get ensemble action with ActionGating applied.

        Returns:
            GatedAction object (signal, risk_fraction, sl_multiplier, tp_multiplier)
        """
        raw_action = self.get_action(m5, h1, h4, deterministic)
        if self.action_gating is not None:
            action_t = torch.from_numpy(raw_action).unsqueeze(0).float()
            return self.action_gating.gate_single(action_t)
        return raw_action

    def _compute_weights(self, m5: torch.Tensor) -> np.ndarray:
        """
        Compute regime-aware weights for each agent.

        Returns:
            (n_agents,) normalized weights
        """
        if self.regime_detector is None:
            return self.base_weights / self.base_weights.sum()

        # Get regime from the first agent's feature extractor
        # (RegimeDetector is inside TransformerFeatureExtractor)
        agent0 = self.agents[0]
        if hasattr(agent0, 'feature_extractor'):
            extractor = agent0.feature_extractor
            smc_latent = extractor.transformer_smc(m5)
            regime_probs, _ = extractor.regime_detector(smc_latent)
            # regime_probs: (1, 4) -> get top regime
            probs_np = regime_probs.squeeze(0).cpu().numpy()
            regime_id = int(np.argmax(probs_np))
            regime_conf = float(probs_np[regime_id])
        else:
            # Fallback: use external regime_detector directly
            regime_probs, _ = self.regime_detector(m5)
            probs_np = regime_probs.squeeze(0).cpu().numpy()
            regime_id = int(np.argmax(probs_np))
            regime_conf = float(probs_np[regime_id])

        # Lookup boost from table
        if regime_id < len(REGIME_BOOST_TABLE) and self.n_agents <= 3:
            boost = REGIME_BOOST_TABLE[regime_id, :self.n_agents].copy()
        else:
            boost = np.ones(self.n_agents, dtype=np.float32)

        # Scale boost by regime confidence (higher conf = stronger boost)
        boost = 1.0 + (boost - 1.0) * regime_conf

        # Apply to base weights
        raw = self.base_weights * boost

        # Clamp individual weights
        raw = np.clip(raw, MIN_WEIGHT, MAX_WEIGHT)

        # Normalize to sum=1
        total = raw.sum()
        if total > 0:
            raw /= total
        else:
            raw = np.ones(self.n_agents, dtype=np.float32) / self.n_agents

        return raw

    def get_agent_diagnostics(
        self,
        m5: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
    ) -> dict:
        """
        Get detailed diagnostics for debugging/logging.

        Returns dict with per-agent actions, scores, weights, and final decision.
        """
        weights = self._compute_weights(m5)
        actions = []
        scores = []

        for i, agent in enumerate(self.agents):
            with torch.no_grad():
                action, _ = agent(m5, h1, h4, deterministic=True)
            a_np = action.squeeze(0).cpu().numpy()
            conf = abs(a_np[0])
            score = conf * weights[i]
            actions.append(a_np.tolist())
            scores.append(float(score))

        top_idx = int(np.argmax(scores))
        weighted_dir = sum(
            actions[i][0] * scores[i] for i in range(self.n_agents)
        )

        return {
            "weights": weights.tolist(),
            "actions": actions,
            "confidences": [abs(a[0]) for a in actions],
            "scores": scores,
            "top_agent_idx": top_idx,
            "weighted_direction": weighted_dir,
            "final_action": [weighted_dir] + actions[top_idx][1:],
        }
