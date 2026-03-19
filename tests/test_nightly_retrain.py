"""
Tests for Sprint 4.5: Safe Nightly Retrain.

Uses unittest.mock to mock backtest results for the Validation Gate.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from training_pipeline.safe_retrain import (
    MixedSampler,
    RetrainMetrics,
    SafeNightlyRetrainer,
)


# =============================================
# MixedSampler Tests
# =============================================

class TestMixedSampler:
    """Tests for 20/80 data split sampling."""

    def test_20_80_ratio(self) -> None:
        """Batch respects ~20% new + ~80% old split."""
        new_idx = list(range(100))
        old_idx = list(range(100, 1000))
        sampler = MixedSampler(new_idx, old_idx, new_ratio=0.2)

        # Sample many batches and count
        new_count = 0
        total = 0
        for _ in range(1000):
            batch = sampler.sample_indices(64)
            assert len(batch) == 64
            for idx in batch:
                if idx < 100:
                    new_count += 1
                total += 1

        ratio = new_count / total
        assert 0.15 < ratio < 0.25, f"Expected ~0.2, got {ratio:.3f}"

    def test_no_new_data_raises(self) -> None:
        """Empty new_data_indices raises ValueError."""
        with pytest.raises(ValueError, match="No new data"):
            MixedSampler([], [1, 2, 3])

    def test_batch_size_respected(self) -> None:
        """Output has exactly batch_size elements."""
        sampler = MixedSampler([0, 1, 2], [3, 4, 5, 6, 7])
        batch = sampler.sample_indices(32)
        assert len(batch) == 32

    def test_small_old_data_works(self) -> None:
        """Works even if old data is smaller than n_old."""
        sampler = MixedSampler([0, 1], [2], new_ratio=0.2)
        batch = sampler.sample_indices(10)
        assert len(batch) == 10


# =============================================
# RetrainMetrics Tests
# =============================================

class TestRetrainMetrics:
    """Tests for validation gate logic."""

    def test_passes_gate_both_ok(self) -> None:
        """Passes when both Sharpe and DD conditions met."""
        old = RetrainMetrics(sharpe_ratio=2.0, max_drawdown=0.05,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        new = RetrainMetrics(sharpe_ratio=1.9, max_drawdown=0.05,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        # 1.9 >= 2.0*0.9=1.8 OK, 0.05 <= 0.05*1.1=0.055 OK
        assert new.passes_gate(old) is True

    def test_fails_gate_sharpe_too_low(self) -> None:
        """Fails when Sharpe drops below 90% of old."""
        old = RetrainMetrics(sharpe_ratio=2.0, max_drawdown=0.05,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        new = RetrainMetrics(sharpe_ratio=1.5, max_drawdown=0.04,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        # 1.5 < 2.0*0.9=1.8 FAIL
        assert new.passes_gate(old) is False

    def test_fails_gate_dd_too_high(self) -> None:
        """Fails when DD exceeds 110% of old."""
        old = RetrainMetrics(sharpe_ratio=2.0, max_drawdown=0.05,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        new = RetrainMetrics(sharpe_ratio=2.5, max_drawdown=0.06,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        # 0.06 > 0.05*1.1=0.055 FAIL
        assert new.passes_gate(old) is False

    def test_fails_gate_both_bad(self) -> None:
        """Fails when both conditions fail."""
        old = RetrainMetrics(sharpe_ratio=2.0, max_drawdown=0.05,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        new = RetrainMetrics(sharpe_ratio=1.0, max_drawdown=0.10,
                             win_rate=0.3, profit_factor=0.8,
                             total_trades=50, total_return=-0.05)
        assert new.passes_gate(old) is False

    def test_passes_gate_exact_boundary(self) -> None:
        """Passes at exact boundary values (>=, <=)."""
        old = RetrainMetrics(sharpe_ratio=2.0, max_drawdown=0.05,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        new = RetrainMetrics(sharpe_ratio=1.8, max_drawdown=0.055,
                             win_rate=0.5, profit_factor=1.2,
                             total_trades=100, total_return=0.1)
        # 1.8 >= 1.8 OK, 0.055 <= 0.055 OK (exact boundaries)
        assert new.passes_gate(old) is True


# =============================================
# SafeNightlyRetrainer Tests
# =============================================

class TestSafeNightlyRetrainer:
    """Tests for full retrain pipeline with mocked backtest."""

    @pytest.fixture
    def tmp_model(self, tmp_path) -> tuple[Path, dict]:
        """Create a temporary model file."""
        model_path = tmp_path / "best_model.pt"
        state = {"actor_state": {"weight": torch.randn(10, 10)}, "step": 1000}
        torch.save(state, model_path)
        return model_path, state

    def test_deploy_better_model(self, tmp_model) -> None:
        """Mock better backtest -> model should be deployed."""
        model_path, old_state = tmp_model

        retrainer = SafeNightlyRetrainer(
            model_path=model_path,
            backup_dir=model_path.parent / "backups",
        )

        old_metrics = RetrainMetrics(
            sharpe_ratio=2.0, max_drawdown=0.05,
            win_rate=0.45, profit_factor=1.2,
            total_trades=100, total_return=0.1,
        )

        # Mock backtest returns BETTER metrics
        new_metrics = RetrainMetrics(
            sharpe_ratio=2.5, max_drawdown=0.04,   # Better on both!
            win_rate=0.50, profit_factor=1.4,
            total_trades=120, total_return=0.15,
        )
        mock_backtest = MagicMock(return_value=new_metrics)

        new_state = {"actor_state": {"weight": torch.randn(10, 10)}, "step": 2000}
        result = retrainer.validate_and_deploy(
            new_model_state=new_state,
            backtest_fn=mock_backtest,
            old_metrics=old_metrics,
        )

        assert result.accepted is True
        assert "ACCEPTED" in result.reason
        assert result.backup_path is not None
        # Verify new model is saved
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        assert loaded["step"] == 2000

    def test_reject_worse_model(self, tmp_model) -> None:
        """Mock worse DD -> model should be rejected."""
        model_path, old_state = tmp_model

        retrainer = SafeNightlyRetrainer(
            model_path=model_path,
            backup_dir=model_path.parent / "backups",
        )

        old_metrics = RetrainMetrics(
            sharpe_ratio=2.0, max_drawdown=0.05,
            win_rate=0.45, profit_factor=1.2,
            total_trades=100, total_return=0.1,
        )

        # Mock backtest returns WORSE DD
        new_metrics = RetrainMetrics(
            sharpe_ratio=2.1, max_drawdown=0.08,   # DD too high!
            win_rate=0.48, profit_factor=1.3,
            total_trades=110, total_return=0.12,
        )
        mock_backtest = MagicMock(return_value=new_metrics)

        new_state = {"actor_state": {"weight": torch.randn(10, 10)}, "step": 2000}
        result = retrainer.validate_and_deploy(
            new_model_state=new_state,
            backtest_fn=mock_backtest,
            old_metrics=old_metrics,
        )

        assert result.accepted is False
        assert "REJECTED" in result.reason
        assert "DD" in result.reason
        # Verify old model preserved (step=1000)
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        assert loaded["step"] == 1000

    def test_backup_creation(self, tmp_model) -> None:
        """Backup file must be created BEFORE overwrite."""
        model_path, old_state = tmp_model
        backup_dir = model_path.parent / "backups"

        retrainer = SafeNightlyRetrainer(
            model_path=model_path,
            backup_dir=backup_dir,
        )

        old_metrics = RetrainMetrics(
            sharpe_ratio=1.0, max_drawdown=0.05,
            win_rate=0.4, profit_factor=1.0,
            total_trades=50, total_return=0.05,
        )
        new_metrics = RetrainMetrics(
            sharpe_ratio=1.5, max_drawdown=0.04,
            win_rate=0.5, profit_factor=1.3,
            total_trades=80, total_return=0.10,
        )
        mock_backtest = MagicMock(return_value=new_metrics)

        new_state = {"actor_state": {"weight": torch.randn(10, 10)}, "step": 5000}
        result = retrainer.validate_and_deploy(
            new_model_state=new_state,
            backtest_fn=mock_backtest,
            old_metrics=old_metrics,
        )

        assert result.accepted is True
        assert result.backup_path is not None
        # Verify backup file exists and contains OLD model
        backup_path = Path(result.backup_path)
        assert backup_path.exists()
        backup_loaded = torch.load(backup_path, map_location="cpu", weights_only=False)
        assert backup_loaded["step"] == 1000  # Old model's step

    def test_rollback(self, tmp_model) -> None:
        """Rollback restores the most recent backup."""
        model_path, _ = tmp_model
        backup_dir = model_path.parent / "backups"

        retrainer = SafeNightlyRetrainer(
            model_path=model_path,
            backup_dir=backup_dir,
        )

        # Deploy a new model (creates backup)
        old_metrics = RetrainMetrics(1.0, 0.05, 0.4, 1.0, 50, 0.05)
        new_metrics = RetrainMetrics(1.5, 0.04, 0.5, 1.3, 80, 0.10)
        mock_backtest = MagicMock(return_value=new_metrics)
        new_state = {"step": 9999}
        retrainer.validate_and_deploy(new_state, mock_backtest, old_metrics)

        # Verify new model deployed
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        assert loaded["step"] == 9999

        # Rollback
        success = retrainer.rollback()
        assert success is True
        # Verify old model restored
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        assert loaded["step"] == 1000

    def test_no_backup_rollback_fails(self, tmp_path) -> None:
        """Rollback with no backups returns False."""
        model_path = tmp_path / "model.pt"
        torch.save({"step": 1}, model_path)

        retrainer = SafeNightlyRetrainer(
            model_path=model_path,
            backup_dir=tmp_path / "empty_backups",
        )
        assert retrainer.rollback() is False

    def test_reject_preserves_old_model_exactly(self, tmp_model) -> None:
        """Rejected model should not modify old model file at all."""
        model_path, old_state = tmp_model
        old_size = model_path.stat().st_size
        old_mtime = model_path.stat().st_mtime

        retrainer = SafeNightlyRetrainer(
            model_path=model_path,
            backup_dir=model_path.parent / "backups",
        )

        old_metrics = RetrainMetrics(3.0, 0.02, 0.6, 1.5, 200, 0.2)
        # Terrible new model
        new_metrics = RetrainMetrics(0.5, 0.15, 0.2, 0.5, 30, -0.1)
        mock_backtest = MagicMock(return_value=new_metrics)

        result = retrainer.validate_and_deploy(
            {"step": 9999}, mock_backtest, old_metrics,
        )

        assert result.accepted is False
        # File should be UNTOUCHED
        assert model_path.stat().st_size == old_size
        assert model_path.stat().st_mtime == old_mtime

    def test_history_tracking(self, tmp_model) -> None:
        """Each retrain attempt is recorded in history."""
        model_path, _ = tmp_model

        retrainer = SafeNightlyRetrainer(
            model_path=model_path,
            backup_dir=model_path.parent / "backups",
        )

        old_m = RetrainMetrics(1.0, 0.05, 0.4, 1.0, 50, 0.05)

        # Accept one
        good = RetrainMetrics(1.5, 0.04, 0.5, 1.3, 80, 0.10)
        retrainer.validate_and_deploy({"step": 2}, MagicMock(return_value=good), old_m)

        # Reject one
        bad = RetrainMetrics(0.1, 0.20, 0.1, 0.3, 10, -0.1)
        retrainer.validate_and_deploy({"step": 3}, MagicMock(return_value=bad), old_m)

        assert len(retrainer.history) == 2
        assert retrainer.history[0].accepted is True
        assert retrainer.history[1].accepted is False
