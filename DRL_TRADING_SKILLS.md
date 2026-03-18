# 🧠 DRL Trading System — Consolidated Skills Reference

> **Project:** RabitPropfim — Zero-Hardcoded Institutional-Grade DRL Trading System
> **Source:** Curated from 1,265 skills in `.agent/skills/CATALOG.md`

---

## 🎯 TIER 1 — CRITICAL CORE SKILLS (Must-Have)

### 1. `quant-analyst`
**Role:** Strategy design, backtesting engine, market data pipeline
- Trading strategy development & backtesting with transaction costs + slippage
- Risk metrics: VaR, Sharpe, max drawdown
- Time series analysis & forecasting
- Statistical arbitrage & pairs trading
- Out-of-sample testing to prevent overfitting
- **Libraries:** pandas, numpy, scipy
```
Path: .agent/skills/skills/quant-analyst/SKILL.md
```

---

### 2. `risk-manager`
**Role:** Position sizing, drawdown enforcement, Prop Firm rule compliance
- Position sizing via Kelly Criterion
- R-multiple tracking & expectancy calculation
- VaR calculations & correlation analysis
- Stress testing via Monte Carlo simulation
- Stop-loss & take-profit management
- Maximum drawdown analysis (5% daily / 10% total enforcement)
```
Path: .agent/skills/skills/risk-manager/SKILL.md
```

---

### 3. `risk-metrics-calculation`
**Role:** Core risk math engine for reward function
- VaR (Value at Risk) & CVaR (Expected Shortfall)
- Sharpe & Sortino ratios
- Drawdown analysis & risk limits
- Position sizing from risk parameters
```
Path: .agent/skills/skills/risk-metrics-calculation/SKILL.md
```

---

### 4. `ml-engineer`
**Role:** PyTorch DRL architecture, Transformer networks, model training
- **PyTorch 2.x** with torch.compile, FSDP, distributed training
- Ray/Ray Train for distributed computing & hyperparameter tuning
- Optuna for hyperparameter optimization
- Model serving: TorchServe, FastAPI, gRPC
- Real-time inference with Redis/Kafka
- Model optimization: quantization, pruning, distillation
- Reinforcement learning: policy optimization
- Time series forecasting: deep learning approaches
```
Path: .agent/skills/skills/ml-engineer/SKILL.md
```

---

### 5. `mlops-engineer`
**Role:** Self-evolving pipeline, nightly retraining, model registry
- ML pipeline orchestration (Airflow, Kubeflow, Prefect)
- Experiment tracking: MLflow, Weights & Biases
- Model registry & versioning
- Continuous training triggered by performance degradation
- Model performance monitoring & drift detection
- Cloud GPU training infrastructure (AWS, GCP)
- Auto-scaling for training/inference workloads
```
Path: .agent/skills/skills/mlops-engineer/SKILL.md
```

---

### 6. `ml-pipeline-workflow`
**Role:** End-to-end pipeline orchestration (data → train → validate → deploy)
- DAG-based orchestration patterns
- Data validation & quality checks
- Feature engineering pipelines
- Model validation & comparison workflows
- Deployment automation with canary/blue-green strategies
- Continuous training triggered by data drift
```
Path: .agent/skills/skills/ml-pipeline-workflow/SKILL.md
```

---

## 🔧 TIER 2 — ESSENTIAL SUPPORT SKILLS

### 7. `python-pro`
**Role:** Core language mastery for the entire system
- Python 3.12+ with async/await patterns (asyncio)
- Performance optimization & profiling (cProfile, py-spy)
- NumPy/Pandas optimization for data processing
- Type hints, Pydantic models for data validation
- Docker containerization for deployment
- Modern tooling: uv, ruff, pytest
```
Path: .agent/skills/skills/python-pro/SKILL.md
```

---

### 8. `data-scientist`
**Role:** Feature engineering, statistical analysis, time series
- Time series analysis: ARIMA, Prophet, seasonal decomposition
- Bayesian statistics with PyMC3
- Deep learning: CNNs, RNNs, LSTMs, Transformers (PyTorch/TF)
- Anomaly detection (isolation forests, autoencoders)
- Reinforcement learning for optimization
- Financial analytics: credit risk, volatility modeling, algorithmic trading
```
Path: .agent/skills/skills/data-scientist/SKILL.md
```

---

### 9. `alpha-vantage`
**Role:** Market data ingestion (forex, crypto, commodities, indicators)
- Real-time & historical OHLCV data
- Forex rates (FX_INTRADAY, FX_DAILY)
- 50+ technical indicators (SMA, EMA, MACD, RSI, BBANDS, ATR, VWAP)
- Commodities: GOLD, BRENT, NATURAL_GAS
- Economic indicators: GDP, CPI, INFLATION
```
Path: .agent/skills/skills/alpha-vantage/SKILL.md
```

---

## 🏗️ TIER 3 — INFRASTRUCTURE & DEPLOYMENT

### 10. `docker-expert`
**Role:** Containerization for training & inference environments
```
Path: .agent/skills/skills/docker-expert/SKILL.md
```

### 11. `cloud-architect`
**Role:** Cloud GPU training (AWS/GCP), VPS for live trading
```
Catalog: .agent/skills/CATALOG.md → cloud-architect
```

### 12. `performance-optimizer`
**Role:** Latency optimization for real-time inference
```
Catalog: .agent/skills/CATALOG.md → performance-optimizer
```

### 13. `ai-engineer`
**Role:** AI system design patterns, prompt engineering for agent architecture
```
Path: .agent/skills/skills/ai-engineer/SKILL.md
```

---

## 📊 SKILL → BLUEPRINT MAPPING

| Blueprint Component | Primary Skill(s) | Secondary Skill(s) |
|---|---|---|
| **State Space (Price Action, Volume, SMC)** | `quant-analyst`, `alpha-vantage` | `data-scientist` |
| **Transformer Self-Attention (Order Blocks, FVG)** | `ml-engineer` | `data-scientist` |
| **DRL Engine (SAC/PPO)** | `ml-engineer` | `ai-engineer` |
| **Continuous Action Space** | `ml-engineer`, `risk-manager` | — |
| **Reward Function (Prop Firm Rules)** | `risk-metrics-calculation`, `risk-manager` | `quant-analyst` |
| **Drawdown Penalty (5%/10%)** | `risk-manager`, `risk-metrics-calculation` | — |
| **Intraday Rule (Sin/Cos Time Encoding)** | `data-scientist` | `ml-engineer` |
| **Custom Gymnasium Environment** | `ml-engineer` | `python-pro` |
| **Nightly Retraining Pipeline** | `mlops-engineer`, `ml-pipeline-workflow` | `docker-expert` |
| **Live Inference via Broker API** | `python-pro` | `performance-optimizer` |
| **Cloud GPU Training** | `cloud-architect` | `docker-expert` |
| **Model Registry & Versioning** | `mlops-engineer` | — |

---

## 🛠️ TECH STACK ALIGNMENT

| Tech Requirement | Skill Coverage |
|---|---|
| Python 3.10+ | `python-pro` |
| PyTorch | `ml-engineer` |
| Ray RLlib / Stable-Baselines3 | `ml-engineer` |
| OpenAI Gymnasium | `ml-engineer` |
| MetaTrader5 API / cTrader FIX | `python-pro` (async), `quant-analyst` |
| MLflow / W&B | `mlops-engineer` |
| Docker | `docker-expert` |
| Cloud GPU (AWS/Runpod) | `cloud-architect` |

---

## 📋 HOW TO INVOKE SKILLS

To activate any skill during development, read the full SKILL.md:
```
.agent/skills/skills/<skill-name>/SKILL.md
```

For implementation playbooks and templates, check:
```
.agent/skills/skills/<skill-name>/resources/implementation-playbook.md
```

---

> **Total Skills Selected:** 13 out of 1,265
> **Coverage:** 100% of all blueprint components mapped
