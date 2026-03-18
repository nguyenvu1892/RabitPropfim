"""
Tests for Sprint 5 — Safety Layer & Self-Evolution.

Validates:
- Killswitch triggers at correct DD thresholds
- Killswitch daily reset works
- Watchdog respects check interval
- Model Registry saves, loads, and rolls back versions
- Safe Retrainer accepts/rejects based on improvement threshold
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from live_execution.killswitch import EquityWatchdog, Killswitch
from model_registry.registry import ModelRegistry
from training_pipeline.safe_retrain import SafeRetrainer


# ─────────────────────────────────────────────
# Killswitch Tests
# ─────────────────────────────────────────────

class TestKillswitch:

    def _config(self) -> dict:
        return {
            "killswitch_dd_threshold": 0.045,
            "max_daily_drawdown": 0.05,
            "max_total_drawdown": 0.10,
        }

    def test_normal_state(self) -> None:
        ks = Killswitch(self._config())
        status = ks.check(daily_dd=0.01, total_dd=0.02, equity=9800)
        assert status == "normal"

    def test_soft_killswitch(self) -> None:
        ks = Killswitch(self._config())
        status = ks.check(daily_dd=0.046, total_dd=0.046, equity=9540)
        assert status == "soft"
        assert ks.is_soft_triggered

    def test_hard_killswitch(self) -> None:
        ks = Killswitch(self._config())
        status = ks.check(daily_dd=0.055, total_dd=0.055, equity=9450)
        assert status == "hard"
        assert ks.is_hard_triggered

    def test_emergency_shutdown(self) -> None:
        ks = Killswitch(self._config())
        status = ks.check(daily_dd=0.08, total_dd=0.11, equity=8900)
        assert status == "emergency"
        assert ks.is_emergency

    def test_daily_reset(self) -> None:
        ks = Killswitch(self._config())
        ks.check(daily_dd=0.046, total_dd=0.046, equity=9540)
        assert ks.is_soft_triggered
        ks.reset_daily()
        assert not ks.is_soft_triggered
        assert not ks.is_hard_triggered

    def test_alert_callback(self) -> None:
        ks = Killswitch(self._config())
        alerts: list[str] = []
        ks.set_alert_callback(lambda title, details: alerts.append(title))
        ks.check(daily_dd=0.046, total_dd=0.046, equity=9540)
        assert len(alerts) == 1
        assert "WARNING" in alerts[0]

    def test_events_logged(self) -> None:
        ks = Killswitch(self._config())
        ks.check(daily_dd=0.055, total_dd=0.055, equity=9450)
        assert len(ks.events) == 1
        assert ks.events[0].event_type == "hard_killswitch"


# ─────────────────────────────────────────────
# Watchdog Tests
# ─────────────────────────────────────────────

class TestWatchdog:

    def test_equity_history_recorded(self) -> None:
        ks = Killswitch({"killswitch_dd_threshold": 0.045,
                         "max_daily_drawdown": 0.05, "max_total_drawdown": 0.10})
        wd = EquityWatchdog(ks, check_interval_seconds=0)
        for _ in range(10):
            wd.tick(0.01, 0.02, 9800.0)
        assert len(wd.equity_history) == 10


# ─────────────────────────────────────────────
# Model Registry Tests
# ─────────────────────────────────────────────

class TestModelRegistry:

    def test_register_and_load(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        state = {"weights": torch.tensor([1.0, 2.0, 3.0])}
        version = reg.register(
            checkpoint_state=state,
            metrics={"eval_reward": 5.0},
            curriculum_stage="elementary",
            training_steps=50_000,
        )
        assert version.version_id == 1
        loaded = reg.load_version(1)
        assert torch.allclose(loaded["weights"], state["weights"])

    def test_best_model_tracking(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 3.0})
        reg.register({"w": torch.tensor([2.0])}, {"eval_reward": 7.0})
        reg.register({"w": torch.tensor([3.0])}, {"eval_reward": 5.0})
        assert reg.best_version == 2  # Version 2 had highest reward

    def test_load_best(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 3.0})
        reg.register({"w": torch.tensor([2.0])}, {"eval_reward": 7.0})
        best = reg.load_best()
        assert torch.allclose(best["w"], torch.tensor([2.0]))

    def test_rollback(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 3.0})
        reg.register({"w": torch.tensor([2.0])}, {"eval_reward": 7.0})
        assert reg.best_version == 2

        reg.rollback(1)
        assert reg.best_version == 1
        best = reg.load_best()
        assert torch.allclose(best["w"], torch.tensor([1.0]))

    def test_manifest_persistence(self, tmp_path: Path) -> None:
        reg_dir = tmp_path / "registry"
        reg = ModelRegistry(reg_dir)
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 5.0})

        # Reload registry
        reg2 = ModelRegistry(reg_dir)
        assert reg2.latest_version == 1

    def test_list_versions(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 3.0}, notes="first")
        reg.register({"w": torch.tensor([2.0])}, {"eval_reward": 5.0}, notes="second")
        versions = reg.list_versions()
        assert len(versions) == 2
        assert versions[0]["notes"] == "first"


# ─────────────────────────────────────────────
# Safe Retrainer Tests
# ─────────────────────────────────────────────

class TestSafeRetrainer:

    def test_accepts_above_threshold(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 5.0})

        retrainer = SafeRetrainer(
            config={"improvement_threshold": 0.05},
            registry=reg,
        )
        result = retrainer.try_retrain(
            old_metrics={"eval_reward": 5.0},
            new_metrics={"eval_reward": 6.0},  # 20% improvement
            new_checkpoint={"w": torch.tensor([2.0])},
        )
        assert result.accepted
        assert result.improvement_pct > 5.0

    def test_rejects_below_threshold(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 5.0})

        retrainer = SafeRetrainer(
            config={"improvement_threshold": 0.05},
            registry=reg,
        )
        result = retrainer.try_retrain(
            old_metrics={"eval_reward": 5.0},
            new_metrics={"eval_reward": 5.01},  # Only 0.2% improvement
        )
        assert not result.accepted

    def test_rejects_worse_model(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 5.0})

        retrainer = SafeRetrainer(
            config={"improvement_threshold": 0.05},
            registry=reg,
        )
        result = retrainer.try_retrain(
            old_metrics={"eval_reward": 5.0},
            new_metrics={"eval_reward": 4.0},  # Worse
        )
        assert not result.accepted
        assert result.improvement_pct < 0

    def test_history_tracked(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry")
        reg.register({"w": torch.tensor([1.0])}, {"eval_reward": 5.0})

        retrainer = SafeRetrainer(
            config={"improvement_threshold": 0.05},
            registry=reg,
        )
        retrainer.try_retrain({"eval_reward": 5.0}, {"eval_reward": 6.0},
                               new_checkpoint={"w": torch.tensor([2.0])})
        retrainer.try_retrain({"eval_reward": 6.0}, {"eval_reward": 5.5})
        assert len(retrainer.history) == 2
