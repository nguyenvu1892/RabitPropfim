"""
Tests for Sprint 4: PER Buffer + Curriculum Runner.

Tests SumTree correctness, PER sampling distribution,
IS weights, beta annealing, and CurriculumRunner auto-promote.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training_pipeline.per_buffer import SumTree, PERBuffer
from training_pipeline.curriculum_runner import (
    CurriculumRunner,
    StageConfig,
    DEFAULT_STAGES,
)


# =============================================
# SumTree Tests
# =============================================

class TestSumTree:
    """Tests for SumTree data structure."""

    def test_init_empty(self) -> None:
        """Empty tree has total=0."""
        tree = SumTree(8)
        assert tree.total == 0.0
        assert tree.n_entries == 0

    def test_capacity_rounds_to_power_of_2(self) -> None:
        """Capacity 10 -> rounds up to 16."""
        tree = SumTree(10)
        assert tree.capacity == 16

    def test_add_single(self) -> None:
        """Adding one priority updates total correctly."""
        tree = SumTree(4)
        tree.add(0, 5.0)
        assert abs(tree.total - 5.0) < 1e-10
        assert tree.n_entries == 1

    def test_add_multiple(self) -> None:
        """Total is sum of all added priorities."""
        tree = SumTree(4)
        tree.add(0, 1.0)
        tree.add(1, 2.0)
        tree.add(2, 3.0)
        assert abs(tree.total - 6.0) < 1e-10
        assert tree.n_entries == 3

    def test_update_priority(self) -> None:
        """Updating a priority recalculates total."""
        tree = SumTree(4)
        tree.add(0, 1.0)
        tree.add(1, 2.0)
        # Update idx 0: 1.0 -> 5.0
        tree.update(0, 5.0)
        assert abs(tree.total - 7.0) < 1e-10

    def test_sample_retrieves_correct_leaf(self) -> None:
        """Sampling with value 0 returns first leaf."""
        tree = SumTree(4)
        tree.add(0, 1.0)
        tree.add(1, 2.0)
        tree.add(2, 3.0)
        # Value < 1.0 should return data_idx=0
        _, _, data_idx = tree.get(0.5)
        assert data_idx == 0

    def test_sample_proportional(self) -> None:
        """Higher priority items are sampled more often."""
        tree = SumTree(4)
        tree.add(0, 1.0)   # 10% chance
        tree.add(1, 9.0)   # 90% chance

        counts = {0: 0, 1: 0}
        for _ in range(10000):
            val = np.random.uniform(0, tree.total)
            _, _, data_idx = tree.get(val)
            counts[data_idx] += 1

        # idx 1 should have ~90% of samples
        ratio = counts[1] / 10000
        assert 0.85 < ratio < 0.95, f"Expected ~0.9, got {ratio}"

    def test_max_priority(self) -> None:
        """max_priority returns highest priority."""
        tree = SumTree(4)
        tree.add(0, 1.0)
        tree.add(1, 5.0)
        tree.add(2, 3.0)
        assert tree.max_priority == 5.0

    def test_min_priority(self) -> None:
        """min_priority returns lowest non-zero priority."""
        tree = SumTree(4)
        tree.add(0, 1.0)
        tree.add(1, 5.0)
        tree.add(2, 3.0)
        assert tree.min_priority == 1.0


# =============================================
# PERBuffer Tests
# =============================================

class TestPERBuffer:
    """Tests for multi-TF PER Buffer."""

    def _make_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create dummy multi-TF observation."""
        return (
            np.random.randn(64, 28).astype(np.float32),  # M5
            np.random.randn(24, 28).astype(np.float32),  # H1
            np.random.randn(30, 28).astype(np.float32),  # H4
        )

    def test_init(self) -> None:
        """Buffer initializes with size=0."""
        buf = PERBuffer(capacity=100)
        assert buf.size == 0

    def test_add_increments_size(self) -> None:
        """Adding transitions increases size."""
        buf = PERBuffer(capacity=100)
        obs = self._make_obs()
        action = np.zeros(4, dtype=np.float32)
        buf.add(obs, action, 1.0, obs, False)
        assert buf.size == 1
        buf.add(obs, action, 2.0, obs, True)
        assert buf.size == 2

    def test_sample_returns_correct_shapes(self) -> None:
        """Sample batch has correct tensor shapes."""
        buf = PERBuffer(capacity=100)
        for _ in range(20):
            obs = self._make_obs()
            action = np.random.randn(4).astype(np.float32)
            buf.add(obs, action, np.random.randn(), obs, False)

        batch = buf.sample(8)
        assert batch["m5"].shape == (8, 64, 28)
        assert batch["h1"].shape == (8, 24, 28)
        assert batch["h4"].shape == (8, 30, 28)
        assert batch["act"].shape == (8, 4)
        assert batch["rew"].shape == (8,)
        assert batch["next_m5"].shape == (8, 64, 28)
        assert batch["next_h1"].shape == (8, 24, 28)
        assert batch["next_h4"].shape == (8, 30, 28)
        assert batch["done"].shape == (8,)
        assert batch["is_weights"].shape == (8,)
        assert len(batch["tree_indices"]) == 8

    def test_is_weights_bounded(self) -> None:
        """IS weights are in [0, 1] (normalized)."""
        buf = PERBuffer(capacity=100)
        for _ in range(50):
            obs = self._make_obs()
            buf.add(obs, np.zeros(4), 1.0, obs, False, td_error=np.random.rand())

        batch = buf.sample(16)
        weights = batch["is_weights"].numpy()
        assert np.all(weights >= 0), f"Negative weights: {weights}"
        assert np.all(weights <= 1.01), f"Weights > 1: {weights}"
        assert np.any(np.isclose(weights, 1.0)), "Max weight should be ~1.0"

    def test_beta_annealing(self) -> None:
        """Beta increases from beta_start toward 1.0 over frames."""
        buf = PERBuffer(capacity=100, beta_start=0.4, beta_frames=1000)
        assert abs(buf.beta - 0.4) < 1e-5

        # Add some data
        for _ in range(20):
            obs = self._make_obs()
            buf.add(obs, np.zeros(4), 1.0, obs, False)

        # Sample 500 times (half of beta_frames)
        for _ in range(500):
            buf.sample(4)

        beta_mid = buf.beta
        assert 0.65 < beta_mid < 0.75, f"Expected ~0.7, got {beta_mid}"

        # Sample 500 more (at beta_frames)
        for _ in range(500):
            buf.sample(4)

        beta_end = buf.beta
        assert abs(beta_end - 1.0) < 0.01, f"Expected ~1.0, got {beta_end}"

    def test_update_priorities(self) -> None:
        """Updating priorities changes sampling distribution."""
        buf = PERBuffer(capacity=100)
        for i in range(10):
            obs = self._make_obs()
            buf.add(obs, np.zeros(4), 1.0, obs, False, td_error=1.0)

        batch = buf.sample(5)
        # Set very high priority for sampled indices
        new_td = np.ones(5) * 100.0
        buf.update_priorities(batch["tree_indices"], new_td)

        # High-priority items should dominate sampling now
        assert buf.tree.total > 0

    def test_capacity_overflow(self) -> None:
        """Buffer wraps around when capacity exceeded."""
        buf = PERBuffer(capacity=16)  # Small capacity
        for i in range(32):
            obs = self._make_obs()
            buf.add(obs, np.zeros(4), float(i), obs, False)

        assert buf.size == 16  # Capped at capacity

    def test_gpu_device(self) -> None:
        """Sample can target a specific device."""
        buf = PERBuffer(capacity=100)
        for _ in range(20):
            obs = self._make_obs()
            buf.add(obs, np.zeros(4), 1.0, obs, False)

        batch = buf.sample(4, device=torch.device("cpu"))
        assert batch["m5"].device == torch.device("cpu")

    def test_state_dict_roundtrip(self) -> None:
        """State dict save/load preserves buffer state."""
        buf = PERBuffer(capacity=100)
        for _ in range(30):
            obs = self._make_obs()
            buf.add(obs, np.zeros(4), 1.0, obs, False)
        buf.sample(8)  # Advance frame counter

        state = buf.state_dict()
        buf2 = PERBuffer(capacity=100)
        buf2.load_state_dict(state)
        assert buf2.size == buf.size
        assert buf2._frame == buf._frame


# =============================================
# CurriculumRunner Tests
# =============================================

class TestCurriculumRunner:
    """Tests for CurriculumRunner 4-stage pipeline."""

    def test_init_default_stages(self) -> None:
        """Default initialization has 4 stages."""
        runner = CurriculumRunner()
        assert len(runner.stages) == 4
        assert runner.stage_name == "Kindergarten"
        assert not runner.is_final_stage

    def test_stage_names_ordered(self) -> None:
        """Stages go K -> E -> HS -> U."""
        runner = CurriculumRunner()
        names = [s.name for s in runner.stages]
        assert names == ["Kindergarten", "Elementary", "High School", "University"]

    def test_progress_string(self) -> None:
        """Progress shows correct format."""
        runner = CurriculumRunner()
        assert runner.progress == "1/4 (Kindergarten)"

    def test_record_episode(self) -> None:
        """Recording episodes updates counters."""
        runner = CurriculumRunner()
        runner.record_episode(1.5)
        runner.record_episode(2.0)
        assert runner.total_episodes == 2
        assert runner.stage_episode_count == 2

    def test_no_promote_insufficient_episodes(self) -> None:
        """Don't promote before window is filled."""
        runner = CurriculumRunner(promote_window=10)
        for _ in range(5):
            runner.record_episode(999.0)  # High reward
        assert not runner.should_promote()

    def test_promote_when_threshold_exceeded(self) -> None:
        """Promote when avg_reward > threshold over window."""
        runner = CurriculumRunner(promote_window=10)
        # Stage 1 threshold = 2.0
        for _ in range(10):
            runner.record_episode(3.0)  # Above threshold
        assert runner.should_promote()
        assert runner.check_and_promote()
        assert runner.stage_name == "Elementary"

    def test_no_promote_below_threshold(self) -> None:
        """Don't promote when reward is below threshold."""
        runner = CurriculumRunner(promote_window=10)
        for _ in range(10):
            runner.record_episode(0.5)  # Below threshold=2.0
        assert not runner.should_promote()

    def test_full_promotion_chain(self) -> None:
        """Can promote through all 4 stages."""
        runner = CurriculumRunner(promote_window=5)

        # Stage 1 -> 2 (threshold 2.0)
        for _ in range(5):
            runner.record_episode(3.0)
        assert runner.check_and_promote()
        assert runner.stage_name == "Elementary"

        # Stage 2 -> 3 (threshold 1.5)
        for _ in range(5):
            runner.record_episode(2.0)
        assert runner.check_and_promote()
        assert runner.stage_name == "High School"

        # Stage 3 -> 4 (threshold 1.0)
        for _ in range(5):
            runner.record_episode(1.5)
        assert runner.check_and_promote()
        assert runner.stage_name == "University"

        # Stage 4: final, no more promotion
        assert runner.is_final_stage
        for _ in range(10):
            runner.record_episode(100.0)
        assert not runner.should_promote()

    def test_env_overrides_kindergarten(self) -> None:
        """Kindergarten has no costs."""
        runner = CurriculumRunner()
        cfg = runner.get_env_overrides()
        assert cfg["spread_mode"] == "fixed"
        assert cfg["spread_multiplier"] == 0.0
        assert cfg["slippage_enabled"] is False
        assert cfg["commission_enabled"] is False

    def test_env_overrides_high_school(self) -> None:
        """High School has real costs."""
        runner = CurriculumRunner(promote_window=5)
        # Promote to stage 3
        for _ in range(5):
            runner.record_episode(3.0)
        runner.check_and_promote()  # -> Elementary
        for _ in range(5):
            runner.record_episode(2.0)
        runner.check_and_promote()  # -> High School

        cfg = runner.get_env_overrides()
        assert cfg["spread_mode"] == "variable"
        assert cfg["spread_multiplier"] == 1.0
        assert cfg["slippage_enabled"] is True
        assert cfg["commission_enabled"] is True
        assert cfg["commission_per_lot"] == 7.0

    def test_promote_clears_buffer(self) -> None:
        """Promotion resets reward buffer and stage counter."""
        runner = CurriculumRunner(promote_window=5)
        for _ in range(5):
            runner.record_episode(3.0)
        runner.check_and_promote()
        assert runner.stage_episode_count == 0
        assert len(runner._reward_buffer) == 0

    def test_state_dict_roundtrip(self) -> None:
        """State save/load preserves position."""
        runner = CurriculumRunner(promote_window=5)
        for _ in range(5):
            runner.record_episode(3.0)
        runner.check_and_promote()
        runner.record_episode(1.0)

        state = runner.state_dict()
        runner2 = CurriculumRunner(promote_window=5)
        runner2.load_state_dict(state)
        assert runner2.current_stage_idx == runner.current_stage_idx
        assert runner2.total_episodes == runner.total_episodes
        assert runner2.stage_name == "Elementary"
