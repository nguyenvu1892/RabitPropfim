"""Tests for config validation (T1.1.5).

Verifies that:
- Valid YAML configs parse successfully
- Invalid values raise clear errors
- Cross-field validation catches inconsistencies
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from configs.validator import (
    PropRulesConfig,
    TrainHyperparamsConfig,
    load_prop_rules,
    load_train_hyperparams,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _write_yaml(data: dict, tmp_path: Path, filename: str = "test.yaml") -> Path:
    """Write a dict to a temporary YAML file."""
    fpath = tmp_path / filename
    with open(fpath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return fpath


def _valid_prop_rules() -> dict:
    """Return a valid prop_rules config dict."""
    return {
        "max_daily_drawdown": 0.05,
        "max_total_drawdown": 0.10,
        "trading_start_utc": 1,
        "trading_end_utc": 21,
        "max_lots_per_trade": 10.0,
        "max_open_positions": 5,
        "max_trades_per_day": 15,
        "overnight_penalty": -5.0,
        "unrealized_shaping_weight": 0.1,
        "rr_bonus_threshold": 1.5,
        "rr_bonus_coefficient": 0.3,
        "overtrading_penalty": -0.5,
        "inaction_nudge": -0.01,
        "inaction_threshold_steps": 500,
        "dd_penalty_alpha": 2.0,
        "dd_penalty_beta": 3.0,
        "dd_penalty_start": 0.02,
        "confidence_threshold": 0.3,
        "killswitch_dd_threshold": 0.045,
        "base_spread_pips": 1.5,
        "news_spread_multiplier": 8.0,
        "low_liquidity_multiplier": 2.5,
        "slippage_base_pips": 0.2,
        "slippage_lot_coefficient": 0.1,
        "execution_delay_min_ms": 50,
        "execution_delay_max_ms": 150,
        "partial_fill_probability": 0.05,
        "requote_probability": 0.02,
    }


def _valid_train_hyperparams() -> dict:
    """Return a valid train_hyperparams config dict."""
    return {
        "algo": "sac",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha_lr": 3e-4,
        "target_entropy": "auto",
        "buffer_size": 1_000_000,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 500_000,
        "transformer_embed_dim": 128,
        "transformer_n_heads": 4,
        "transformer_n_layers": 2,
        "transformer_dropout": 0.1,
        "transformer_lookback": 96,
        "context_lookback_h4": 30,
        "context_lookback_h1": 48,
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "regime_n_states": 4,
        "n_ensemble_models": 3,
        "ensemble_consensus": 0.67,
        "ensemble_random_seeds": [42, 123, 789],
        "curriculum_stages": 4,
        "curriculum_stage_configs": {
            "stage_1": {
                "name": "kindergarten",
                "max_steps": 50_000,
                "spread_mode": "fixed",
                "slippage_enabled": False,
                "commission_enabled": False,
                "data_filter": "trending_only",
                "max_dd_override": 0.20,
                "promote_reward_threshold": 0.5,
            },
            "stage_2": {
                "name": "elementary",
                "max_steps": 200_000,
                "spread_mode": "variable",
                "slippage_enabled": True,
                "commission_enabled": False,
                "data_filter": "trending_and_ranging",
                "max_dd_override": 0.15,
                "promote_reward_threshold": 0.3,
            },
            "stage_3": {
                "name": "highschool",
                "max_steps": 500_000,
                "spread_mode": "variable",
                "slippage_enabled": True,
                "commission_enabled": True,
                "data_filter": "all_regimes",
                "max_dd_override": 0.10,
                "promote_reward_threshold": 0.2,
            },
            "stage_4": {
                "name": "university",
                "max_steps": -1,
                "spread_mode": "variable",
                "slippage_enabled": True,
                "commission_enabled": True,
                "data_filter": "all_regimes_with_events",
                "max_dd_override": None,
                "promote_reward_threshold": None,
            },
        },
        "total_timesteps": 2_000_000,
        "eval_frequency": 10_000,
        "checkpoint_frequency": 50_000,
        "n_eval_episodes": 20,
        "warmup_steps": 10_000,
        "gradient_clip_norm": 1.0,
        "retrain_lr": 1e-5,
        "retrain_max_epochs": 5,
        "retrain_new_data_ratio": 0.2,
        "retrain_gradient_clip": 0.5,
        "retrain_validation_days": 30,
        "retrain_sharpe_tolerance": 0.9,
        "retrain_dd_tolerance": 1.1,
        "wandb_project": "rabit-propfirm-drl",
        "wandb_entity": None,
        "wandb_log_frequency": 100,
    }


# ─────────────────────────────────────────────
# TEST: Valid configs parse successfully
# ─────────────────────────────────────────────

class TestValidConfigs:
    """Verify that known-good configs pass validation."""

    def test_valid_prop_rules(self, tmp_path: Path) -> None:
        fpath = _write_yaml(_valid_prop_rules(), tmp_path)
        config = load_prop_rules(fpath)
        assert config.max_daily_drawdown == 0.05
        assert config.max_total_drawdown == 0.10
        assert config.confidence_threshold == 0.3

    def test_valid_train_hyperparams(self, tmp_path: Path) -> None:
        fpath = _write_yaml(_valid_train_hyperparams(), tmp_path)
        config = load_train_hyperparams(fpath)
        assert config.algo == "sac"
        assert config.learning_rate == 3e-4
        assert config.n_ensemble_models == 3

    def test_load_actual_prop_rules(self) -> None:
        """Test loading the actual project config file."""
        config_path = (
            Path(__file__).parent.parent
            / "rabit_propfirm_drl"
            / "configs"
            / "prop_rules.yaml"
        )
        if config_path.exists():
            config = load_prop_rules(config_path)
            assert isinstance(config, PropRulesConfig)

    def test_load_actual_train_hyperparams(self) -> None:
        """Test loading the actual project config file."""
        config_path = (
            Path(__file__).parent.parent
            / "rabit_propfirm_drl"
            / "configs"
            / "train_hyperparams.yaml"
        )
        if config_path.exists():
            config = load_train_hyperparams(config_path)
            assert isinstance(config, TrainHyperparamsConfig)


# ─────────────────────────────────────────────
# TEST: Invalid values raise errors
# ─────────────────────────────────────────────

class TestInvalidPropRules:
    """Verify that bad prop_rules values are caught."""

    def test_daily_dd_as_percentage_not_decimal(self, tmp_path: Path) -> None:
        """max_daily_dd: 50 instead of 0.05 → should fail."""
        data = _valid_prop_rules()
        data["max_daily_drawdown"] = 50
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):  # Pydantic ValidationError
            load_prop_rules(fpath)

    def test_daily_dd_too_large(self, tmp_path: Path) -> None:
        """max_daily_dd: 0.5 (50%) → should fail (le=0.10)."""
        data = _valid_prop_rules()
        data["max_daily_drawdown"] = 0.5
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_prop_rules(fpath)

    def test_total_dd_less_than_daily(self, tmp_path: Path) -> None:
        """total_dd < daily_dd → should fail cross-validation."""
        data = _valid_prop_rules()
        data["max_daily_drawdown"] = 0.05
        data["max_total_drawdown"] = 0.03  # Less than daily!
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_prop_rules(fpath)

    def test_killswitch_above_daily_dd(self, tmp_path: Path) -> None:
        """killswitch threshold > daily DD → should fail."""
        data = _valid_prop_rules()
        data["killswitch_dd_threshold"] = 0.06  # > 0.05 daily DD
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_prop_rules(fpath)

    def test_negative_spread(self, tmp_path: Path) -> None:
        data = _valid_prop_rules()
        data["base_spread_pips"] = -1.0
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_prop_rules(fpath)

    def test_confidence_threshold_out_of_range(self, tmp_path: Path) -> None:
        data = _valid_prop_rules()
        data["confidence_threshold"] = 1.5  # Must be < 1.0
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_prop_rules(fpath)

    def test_execution_delay_min_greater_than_max(self, tmp_path: Path) -> None:
        data = _valid_prop_rules()
        data["execution_delay_min_ms"] = 200
        data["execution_delay_max_ms"] = 50  # Min > Max
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_prop_rules(fpath)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        data = _valid_prop_rules()
        del data["max_daily_drawdown"]
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_prop_rules(fpath)


class TestInvalidTrainHyperparams:
    """Verify that bad training configs are caught."""

    def test_invalid_algo(self, tmp_path: Path) -> None:
        """algo: 'random_algo' → should fail."""
        data = _valid_train_hyperparams()
        data["algo"] = "random_algo"
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_train_hyperparams(fpath)

    def test_learning_rate_too_high(self, tmp_path: Path) -> None:
        data = _valid_train_hyperparams()
        data["learning_rate"] = 1.0  # Way too high
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_train_hyperparams(fpath)

    def test_embed_dim_not_divisible_by_heads(self, tmp_path: Path) -> None:
        """embed_dim=100, n_heads=3 → 100%3≠0 → should fail."""
        data = _valid_train_hyperparams()
        data["transformer_embed_dim"] = 100
        data["transformer_n_heads"] = 3
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_train_hyperparams(fpath)

    def test_ensemble_seeds_mismatch(self, tmp_path: Path) -> None:
        """3 models but only 2 seeds → should fail."""
        data = _valid_train_hyperparams()
        data["n_ensemble_models"] = 3
        data["ensemble_random_seeds"] = [42, 123]  # Only 2 seeds!
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_train_hyperparams(fpath)

    def test_batch_size_too_small(self, tmp_path: Path) -> None:
        data = _valid_train_hyperparams()
        data["batch_size"] = 8  # Min is 32
        fpath = _write_yaml(data, tmp_path)
        with pytest.raises(Exception):
            load_train_hyperparams(fpath)


# ─────────────────────────────────────────────
# TEST: File not found
# ─────────────────────────────────────────────

class TestFileErrors:
    """Verify file access errors."""

    def test_prop_rules_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prop_rules("/nonexistent/path/config.yaml")

    def test_train_hyperparams_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_train_hyperparams("/nonexistent/path/config.yaml")
