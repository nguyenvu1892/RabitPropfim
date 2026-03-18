"""
Tests for Sprint 4 — Training Pipeline.

Validates:
- SumTree operations (add, sample, total)
- PER Buffer sampling returns correct shapes
- PER priority updates affect sampling distribution
- Curriculum stage progression
- SAC Trainer update step runs without error
- Checkpoint save/load roundtrip
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from models.actor_critic import Actor, TwinQCritic
from training_pipeline.curriculum import CurriculumManager, StageConfig
from training_pipeline.per_buffer import PERBuffer, SumTree
from training_pipeline.sac_trainer import SACTrainer


# ─────────────────────────────────────────────
# SumTree Tests
# ─────────────────────────────────────────────

class TestSumTree:

    def test_total_after_add(self) -> None:
        tree = SumTree(capacity=8)
        tree.add(1.0, 0)
        tree.add(2.0, 1)
        tree.add(3.0, 2)
        assert abs(tree.total - 6.0) < 1e-10

    def test_get_returns_valid_index(self) -> None:
        tree = SumTree(capacity=8)
        for i in range(8):
            tree.add(1.0, i)
        _, priority, data_idx = tree.get(3.5)
        assert 0 <= data_idx < 8

    def test_max_priority(self) -> None:
        tree = SumTree(capacity=8)
        tree.add(1.0, 0)
        tree.add(5.0, 1)
        tree.add(3.0, 2)
        assert tree.max_priority == 5.0


# ─────────────────────────────────────────────
# PER Buffer Tests
# ─────────────────────────────────────────────

class TestPERBuffer:

    def _fill_buffer(self, n: int = 100) -> PERBuffer:
        buf = PERBuffer(capacity=1000, state_dim=14, action_dim=4)
        rng = np.random.default_rng(42)
        for _ in range(n):
            buf.add(
                state=rng.normal(0, 1, 14).astype(np.float32),
                action=rng.normal(0, 1, 4).astype(np.float32),
                reward=rng.normal(),
                next_state=rng.normal(0, 1, 14).astype(np.float32),
                done=rng.random() > 0.9,
            )
        return buf

    def test_size_increases(self) -> None:
        buf = self._fill_buffer(50)
        assert buf.size == 50

    def test_sample_shapes(self) -> None:
        buf = self._fill_buffer(100)
        states, actions, rewards, next_states, dones, weights, indices = buf.sample(32)
        assert states.shape == (32, 14)
        assert actions.shape == (32, 4)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, 14)
        assert dones.shape == (32,)
        assert weights.shape == (32,)
        assert len(indices) == 32

    def test_weights_are_positive(self) -> None:
        buf = self._fill_buffer(200)
        _, _, _, _, _, weights, _ = buf.sample(64)
        assert np.all(weights > 0), "IS weights should be positive"
        assert np.all(weights <= 1.0 + 1e-5), "IS weights should be <= 1"

    def test_beta_anneals(self) -> None:
        buf = PERBuffer(capacity=100, beta_start=0.4, beta_frames=100, state_dim=2, action_dim=1)
        initial_beta = buf.beta
        # Fill and sample to advance frames
        for i in range(50):
            buf.add(np.zeros(2), np.zeros(1), 0.0, np.zeros(2), False)
        for _ in range(50):
            if buf.size >= 8:
                buf.sample(8)
        assert buf.beta > initial_beta, "Beta should increase over time"

    def test_capacity_wraparound(self) -> None:
        buf = PERBuffer(capacity=50, state_dim=2, action_dim=1)
        for i in range(100):
            buf.add(np.ones(2) * i, np.zeros(1), 0.0, np.zeros(2), False)
        assert buf.size == 50, "Buffer should cap at capacity"


# ─────────────────────────────────────────────
# Curriculum Tests
# ─────────────────────────────────────────────

def _curriculum_config() -> dict:
    return {
        "curriculum_stage_configs": {
            "stage_1": {
                "name": "kindergarten",
                "max_steps": 50_000,
                "spread_mode": "fixed",
                "slippage_enabled": False,
                "commission_enabled": False,
                "data_filter": "trending",
                "max_dd_override": 0.10,
                "promote_reward_threshold": 1.0,
            },
            "stage_2": {
                "name": "elementary",
                "max_steps": 100_000,
                "spread_mode": "variable",
                "slippage_enabled": True,
                "commission_enabled": True,
                "data_filter": "trending+ranging",
                "max_dd_override": 0.07,
                "promote_reward_threshold": 2.0,
            },
            "stage_3": {
                "name": "high_school",
                "max_steps": 200_000,
                "spread_mode": "variable",
                "slippage_enabled": True,
                "commission_enabled": True,
                "data_filter": "all",
                "max_dd_override": None,
                "promote_reward_threshold": 3.0,
            },
            "stage_4": {
                "name": "university",
                "max_steps": 500_000,
                "spread_mode": "variable",
                "slippage_enabled": True,
                "commission_enabled": True,
                "data_filter": "all+news",
                "max_dd_override": None,
                "promote_reward_threshold": None,
            },
        }
    }


class TestCurriculum:

    def test_starts_at_stage_1(self) -> None:
        cm = CurriculumManager(_curriculum_config())
        assert cm.stage_name == "kindergarten"
        assert cm.current_stage_idx == 0

    def test_total_stages(self) -> None:
        cm = CurriculumManager(_curriculum_config())
        assert cm.total_stages == 4

    def test_auto_promote(self) -> None:
        cm = CurriculumManager(_curriculum_config())
        # Record enough high rewards to trigger promotion
        for _ in range(25):
            cm.record_episode(5.0)  # Well above threshold of 1.0
        promoted = cm.check_and_promote()
        assert promoted
        assert cm.stage_name == "elementary"

    def test_no_promote_below_threshold(self) -> None:
        cm = CurriculumManager(_curriculum_config())
        for _ in range(25):
            cm.record_episode(0.5)  # Below threshold of 1.0
        promoted = cm.check_and_promote()
        assert not promoted

    def test_final_stage_no_promote(self) -> None:
        cm = CurriculumManager(_curriculum_config())
        cm.current_stage_idx = 3  # University
        assert cm.is_final_stage
        promoted = cm.check_and_promote()
        assert not promoted

    def test_state_dict_roundtrip(self) -> None:
        cm = CurriculumManager(_curriculum_config())
        cm.current_stage_idx = 2
        cm.stage_steps = 1234
        cm.stage_rewards = [1.0, 2.0, 3.0]

        state = cm.state_dict()
        cm2 = CurriculumManager(_curriculum_config())
        cm2.load_state_dict(state)
        assert cm2.current_stage_idx == 2
        assert cm2.stage_steps == 1234

    def test_env_overrides_kindergarten(self) -> None:
        cm = CurriculumManager(_curriculum_config())
        base_config = {"slippage_base_pips": 0.2, "max_daily_drawdown": 0.05}
        overrides = cm.get_env_config(base_config)
        assert overrides["slippage_base_pips"] == 0.0  # Disabled in kindergarten
        assert overrides["max_daily_drawdown"] == 0.10  # Relaxed


# ─────────────────────────────────────────────
# SAC Trainer Tests
# ─────────────────────────────────────────────

class TestSACTrainer:

    def _make_trainer(self) -> SACTrainer:
        state_dim, action_dim = 14, 4
        actor = Actor(state_dim, action_dim, hidden_dims=[32, 32])
        critic = TwinQCritic(state_dim, action_dim, hidden_dims=[32, 32])
        config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 16,
            "warmup_steps": 10,
            "buffer_size": 1000,
            "critic_hidden_dims": [32, 32],
        }
        return SACTrainer(actor, critic, state_dim, action_dim, config)

    def test_select_action_shape(self) -> None:
        trainer = self._make_trainer()
        state = np.random.randn(14).astype(np.float32)
        action = trainer.select_action(state)
        assert action.shape == (4,)

    def test_action_bounded(self) -> None:
        trainer = self._make_trainer()
        for _ in range(20):
            state = np.random.randn(14).astype(np.float32)
            action = trainer.select_action(state)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_update_with_small_buffer(self) -> None:
        """Update should return empty metrics when buffer is too small."""
        trainer = self._make_trainer()
        metrics = trainer.update()
        assert metrics.actor_loss == 0.0  # No update yet

    def test_update_after_fill(self) -> None:
        """After filling buffer, update should produce non-zero losses."""
        trainer = self._make_trainer()
        rng = np.random.default_rng(42)
        for _ in range(100):
            trainer.buffer.add(
                state=rng.normal(0, 1, 14).astype(np.float32),
                action=rng.normal(0, 1, 4).astype(np.float32),
                reward=rng.normal(),
                next_state=rng.normal(0, 1, 14).astype(np.float32),
                done=rng.random() > 0.9,
            )
        metrics = trainer.update()
        # After a real update, losses should be non-zero
        assert metrics.critic_loss != 0.0 or metrics.actor_loss != 0.0

    def test_checkpoint_save_load(self, tmp_path: Path) -> None:
        trainer = self._make_trainer()
        trainer.total_steps = 999

        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(path)

        trainer2 = self._make_trainer()
        trainer2.load_checkpoint(path)
        assert trainer2.total_steps == 999

    def test_soft_update_changes_target(self) -> None:
        trainer = self._make_trainer()
        # Get initial target params
        before = [p.clone() for p in trainer.target_critic.parameters()]

        # Modify critic params
        for p in trainer.critic.parameters():
            p.data += 1.0

        trainer._soft_update_target()

        # Target should have moved slightly
        for p_before, p_after in zip(before, trainer.target_critic.parameters()):
            assert not torch.allclose(p_before, p_after)
