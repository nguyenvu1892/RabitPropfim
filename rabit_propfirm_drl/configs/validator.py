"""
Pydantic config validation for all YAML configuration files.
Ensures type safety and catches invalid config values BEFORE the system runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# Prop Firm Rules Schema
# ─────────────────────────────────────────────

class SymbolConfig(BaseModel):
    """Per-symbol execution config."""
    base_spread_pips: float = Field(..., gt=0)
    pip_value: float = Field(0.01, gt=0)
    fixed_lot: Optional[float] = None


class PropRulesConfig(BaseModel):
    """Schema for prop_rules.yaml — validates all trading rules and reward params."""

    # Target symbols
    target_symbols: list[str] = Field(default_factory=list)

    # Drawdown limits
    max_daily_drawdown: float = Field(..., gt=0, le=0.10, description="Max daily DD as decimal")
    max_total_drawdown: float = Field(..., gt=0, le=0.15, description="Max total DD as decimal")
    profit_target: float = Field(0.10, ge=0, le=1.0)
    min_trading_days: int = Field(4, ge=1, le=30)

    # Trading hours
    trading_start_utc: int = Field(..., ge=0, le=23)
    trading_end_utc: int = Field(..., ge=0, le=23)

    # Position limits
    max_lots_per_trade: float = Field(..., gt=0, le=100.0)
    max_open_positions: int = Field(..., ge=1, le=20)
    max_trades_per_day: int = Field(..., ge=1, le=100)

    # Reward engine
    overnight_penalty: float = Field(..., le=0)
    unrealized_shaping_weight: float = Field(..., ge=0, le=1.0)
    rr_bonus_threshold: float = Field(..., gt=0)
    rr_bonus_coefficient: float = Field(..., ge=0)
    overtrading_penalty: float = Field(..., le=0)
    inaction_nudge: float = Field(..., le=0)
    inaction_threshold_steps: int = Field(..., ge=1)

    # Exponential DD penalty
    dd_penalty_alpha: float = Field(..., gt=0)
    dd_penalty_beta: float = Field(..., gt=0)
    dd_penalty_start: float = Field(..., gt=0, le=0.10)

    # Action gating
    confidence_threshold: float = Field(..., gt=0, lt=1.0)

    # Killswitch
    killswitch_dd_threshold: float = Field(..., gt=0, le=0.10)

    # Strict per-trade / daily loss limits
    max_loss_per_trade_pct: float = Field(0.003, ge=0.001, le=0.05)
    daily_loss_cooldown_pct: float = Field(0.03, ge=0.005, le=0.10)

    # SMC Exit Rules
    h1_inside_bar_exit: bool = Field(True)

    # Per-symbol execution config
    symbol_configs: dict[str, SymbolConfig] = Field(default_factory=dict)

    # Execution simulation (global defaults)
    news_spread_multiplier: float = Field(..., ge=1.0)
    low_liquidity_multiplier: float = Field(..., ge=1.0)
    slippage_base_pips: float = Field(..., ge=0)
    slippage_lot_coefficient: float = Field(..., ge=0)
    execution_delay_min_ms: int = Field(..., ge=0)
    execution_delay_max_ms: int = Field(..., ge=0)
    partial_fill_probability: float = Field(..., ge=0, le=1.0)
    requote_probability: float = Field(..., ge=0, le=1.0)

    @field_validator("max_daily_drawdown")
    @classmethod
    def dd_must_be_decimal(cls, v: float) -> float:
        if v > 0.10:
            raise ValueError(
                f"max_daily_drawdown={v} looks like a percentage, not a decimal. "
                f"Use 0.05 for 5%, not 5 or 50."
            )
        return v

    @model_validator(mode="after")
    def cross_field_validation(self) -> "PropRulesConfig":
        if self.max_total_drawdown < self.max_daily_drawdown:
            raise ValueError(
                f"max_total_drawdown ({self.max_total_drawdown}) must be >= "
                f"max_daily_drawdown ({self.max_daily_drawdown})"
            )
        if self.killswitch_dd_threshold > self.max_daily_drawdown:
            raise ValueError(
                f"killswitch_dd_threshold ({self.killswitch_dd_threshold}) must be <= "
                f"max_daily_drawdown ({self.max_daily_drawdown})"
            )
        if self.execution_delay_min_ms > self.execution_delay_max_ms:
            raise ValueError(
                f"execution_delay_min_ms ({self.execution_delay_min_ms}) must be <= "
                f"execution_delay_max_ms ({self.execution_delay_max_ms})"
            )
        return self


# ─────────────────────────────────────────────
# Curriculum Stage Schema
# ─────────────────────────────────────────────

class CurriculumStageConfig(BaseModel):
    """Schema for a single curriculum training stage."""
    name: str
    max_steps: int
    spread_mode: Literal["fixed", "variable"]
    slippage_enabled: bool
    commission_enabled: bool
    data_filter: str
    max_dd_override: Optional[float] = None
    promote_reward_threshold: Optional[float] = None


# ─────────────────────────────────────────────
# Training Hyperparameters Schema
# ─────────────────────────────────────────────

class TrainHyperparamsConfig(BaseModel):
    """Schema for train_hyperparams.yaml — validates all training settings."""

    # Algorithm
    algo: Literal["sac", "ppo", "td3"]
    learning_rate: float = Field(..., gt=1e-7, lt=1e-1)
    batch_size: int = Field(..., ge=32, le=8192)
    gamma: float = Field(..., ge=0.9, le=0.9999)
    tau: float = Field(..., gt=0, lt=1.0)
    alpha_lr: float = Field(..., gt=1e-7, lt=1e-1)
    target_entropy: Any = "auto"  # "auto" or float

    # Replay buffer
    buffer_size: int = Field(..., ge=10_000, le=50_000_000)
    per_alpha: float = Field(..., ge=0, le=1.0)
    per_beta_start: float = Field(..., ge=0, le=1.0)
    per_beta_frames: int = Field(..., ge=1)

    # Network architecture
    transformer_embed_dim: int = Field(..., ge=32, le=1024)
    transformer_n_heads: int = Field(..., ge=1, le=16)
    transformer_n_layers: int = Field(..., ge=1, le=12)
    transformer_dropout: float = Field(..., ge=0, lt=1.0)
    transformer_lookback: int = Field(..., ge=10, le=500)
    context_lookback_h4: int = Field(..., ge=5, le=200)
    context_lookback_h1: int = Field(..., ge=5, le=200)
    actor_hidden_dims: list[int]
    critic_hidden_dims: list[int]

    # Regime
    regime_n_states: int = Field(..., ge=2, le=10)

    # Ensemble
    n_ensemble_models: int = Field(..., ge=1, le=10)
    ensemble_consensus: float = Field(..., gt=0, le=1.0)
    ensemble_random_seeds: list[int]

    # Curriculum
    curriculum_stages: int = Field(..., ge=1, le=10)
    curriculum_stage_configs: dict[str, CurriculumStageConfig]

    # Training
    total_timesteps: int = Field(..., ge=1_000)
    eval_frequency: int = Field(..., ge=100)
    checkpoint_frequency: int = Field(..., ge=1_000)
    n_eval_episodes: int = Field(..., ge=1)
    warmup_steps: int = Field(..., ge=0)
    gradient_clip_norm: float = Field(..., gt=0)

    # Nightly retrain
    retrain_lr: float = Field(..., gt=1e-8, lt=1e-2)
    retrain_max_epochs: int = Field(..., ge=1, le=50)
    retrain_new_data_ratio: float = Field(..., gt=0, lt=1.0)
    retrain_gradient_clip: float = Field(..., gt=0)
    retrain_validation_days: int = Field(..., ge=5)
    retrain_sharpe_tolerance: float = Field(..., gt=0, le=1.0)
    retrain_dd_tolerance: float = Field(..., ge=1.0, le=2.0)

    # W&B
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = Field(..., ge=1)

    @field_validator("transformer_embed_dim")
    @classmethod
    def embed_dim_divisible_by_heads(cls, v: int, info: Any) -> int:
        # This will be checked in model_validator below
        return v

    @model_validator(mode="after")
    def cross_validate(self) -> "TrainHyperparamsConfig":
        if self.transformer_embed_dim % self.transformer_n_heads != 0:
            raise ValueError(
                f"transformer_embed_dim ({self.transformer_embed_dim}) must be divisible by "
                f"transformer_n_heads ({self.transformer_n_heads})"
            )
        if len(self.ensemble_random_seeds) != self.n_ensemble_models:
            raise ValueError(
                f"ensemble_random_seeds length ({len(self.ensemble_random_seeds)}) must match "
                f"n_ensemble_models ({self.n_ensemble_models})"
            )
        return self


# ─────────────────────────────────────────────
# Config Loaders
# ─────────────────────────────────────────────

def _find_config_dir() -> Path:
    """Find the configs directory relative to this file."""
    return Path(__file__).parent


def load_prop_rules(path: Path | str | None = None) -> PropRulesConfig:
    """Load and validate prop_rules.yaml. Raises ValidationError if invalid."""
    if path is None:
        path = _find_config_dir() / "prop_rules.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PropRulesConfig(**raw)


def load_train_hyperparams(path: Path | str | None = None) -> TrainHyperparamsConfig:
    """Load and validate train_hyperparams.yaml. Raises ValidationError if invalid."""
    if path is None:
        path = _find_config_dir() / "train_hyperparams.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainHyperparamsConfig(**raw)
