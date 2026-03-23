"""
Fetch Historical Data v3 — Downloads M1/M5/M15/H1 directly from MT5.

v3.0 — Cognitive Architecture (Intraday):
    - Fetches ALL 4 TFs directly (no more resampling M5→M15/H1)
    - M1: 250K bars (~174 days) for deep encoder training
    - M5: 50K bars (~174 days)
    - M15: 17K bars (~174 days)
    - H1: 4200 bars (~174 days)
    - Builds 28 raw features per TF via feature_builder
    - Generates Knowledge features (22-dim) via KnowledgeExtractor
    - Outputs 50-dim feature arrays per TF as Parquet
    - Generates normalizer_v3.json (50-dim mean/std per TF)

Usage: python scripts/fetch_historical_data.py

⚠ M1 data is LARGE. We fetch in multiple batches to avoid MT5 timeout.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─── Timeframe config ───
TF_CONFIG = {
    "M1":  {"bars": 250_000, "batch_size": 50_000},  # Fetch in 50K batches
    "M5":  {"bars":  50_000, "batch_size": 50_000},
    "M15": {"bars":  17_000, "batch_size": 17_000},
    "H1":  {"bars":   4_200, "batch_size":  4_200},
}

# Polars resample durations (for generating TFs from M1 if direct fetch fails)
TF_POLARS_DUR = {"M5": "5m", "M15": "15m", "H1": "1h"}


def load_env() -> dict[str, str]:
    env_path = project_root / ".env"
    env = {}
    if not env_path.exists():
        return env
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env


def fetch_bars_batched(mt5, symbol: str, timeframe, total_bars: int, batch_size: int):
    """
    Fetch bars in multiple batches to avoid MT5 API timeout.
    Works backward from the most recent bar.
    Returns a Polars DataFrame.
    """
    import polars as pl

    all_frames = []
    fetched = 0
    offset = 0

    while fetched < total_bars:
        count = min(batch_size, total_bars - fetched)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, offset, count)

        if rates is None or len(rates) == 0:
            print(f"    ⚠ No more data at offset {offset} (got {fetched} so far)")
            break

        times = [datetime.utcfromtimestamp(r[0]) for r in rates]
        df_batch = pl.DataFrame({
            "time": times,
            "open": [float(r[1]) for r in rates],
            "high": [float(r[2]) for r in rates],
            "low": [float(r[3]) for r in rates],
            "close": [float(r[4]) for r in rates],
            "volume": [int(r[5]) for r in rates],
            "spread": [int(r[6]) for r in rates],
        })

        all_frames.append(df_batch)
        fetched += len(rates)
        offset += len(rates)

        if len(rates) < count:
            print(f"    ⚠ Fewer bars than requested at offset {offset}")
            break

        # Small delay to avoid hammering MT5
        if fetched < total_bars:
            time.sleep(0.2)

    if not all_frames:
        return None

    combined = pl.concat(all_frames).unique(subset=["time"]).sort("time")
    return combined


def build_knowledge_features(ohlcv_np: np.ndarray) -> np.ndarray:
    """
    Extract 22-dim knowledge features from OHLCV array.
    Uses KnowledgeExtractor in batch mode.
    """
    from features.knowledge_extractor import KnowledgeExtractor

    extractor = KnowledgeExtractor()

    # Compute ATR for entire series
    h = ohlcv_np[:, 1]  # high
    l = ohlcv_np[:, 2]  # low
    c = ohlcv_np[:, 3]  # close
    atr_array = extractor.compute_atr(h, l, c, period=14)

    # Batch extract (skip first 30 bars for warmup)
    start_idx = max(30, extractor.vol_sma_period + extractor.swing_lookback * 4)
    knowledge = extractor.extract_batch(ohlcv_np, atr_array, start_idx=start_idx)

    # Pad front with zeros to match original length
    pad = np.zeros((start_idx, 22), dtype=np.float32)
    full_knowledge = np.vstack([pad, knowledge])

    return full_knowledge


def main() -> None:
    print("=" * 70)
    print("  RABIT-PROPFIRM — Data Fetch v3.0 (Cognitive Architecture)")
    print("  Timeframes: M1 / M5 / M15 / H1")
    print("  Features: 28 raw + 22 knowledge = 50-dim per bar")
    print("=" * 70)

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[X] MetaTrader5 not installed!")
        sys.exit(1)

    try:
        import polars as pl
    except ImportError:
        print("[X] Polars not installed!")
        sys.exit(1)

    from data_engine.feature_builder import build_features, FEATURE_COLUMNS
    from data_engine.normalizer import RunningNormalizer

    # Load config
    env = load_env()
    login = int(env.get("MT5_LOGIN", "0"))
    password = env.get("MT5_PASSWORD", "")
    server = env.get("MT5_SERVER", "")
    symbols_str = env.get("MT5_SYMBOLS", "XAUUSD,US100.cash,US30.cash,ETHUSD,BTCUSD")
    symbols = [s.strip() for s in symbols_str.split(",")]

    # Connect
    print("\n[1/5] Connecting to MT5...")
    if not mt5.initialize():
        print(f"  [X] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    if not mt5.login(login, password=password, server=server):
        print(f"  [X] Login failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    account = mt5.account_info()
    print(f"  [OK] Logged in: {account.name} (${account.balance:,.2f})")

    # MT5 TF mapping
    MT5_TF = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }

    # ── Step 2: Fetch raw OHLCV for all 4 TFs ──
    print(f"\n[2/5] Fetching raw OHLCV data...")
    print(f"  Symbols: {symbols}")
    print(f"  TFs: {list(TF_CONFIG.keys())}")
    print("-" * 70)

    raw_data: dict[str, dict[str, "pl.DataFrame"]] = {}

    for sym in symbols:
        mt5.symbol_select(sym, True)
        raw_data[sym] = {}
        safe_name = sym.replace(".", "_")

        for tf_name, tf_cfg in TF_CONFIG.items():
            print(f"  {sym} {tf_name} ({tf_cfg['bars']:,} bars)...", end=" ", flush=True)

            df = fetch_bars_batched(
                mt5, sym, MT5_TF[tf_name],
                total_bars=tf_cfg["bars"],
                batch_size=tf_cfg["batch_size"],
            )

            if df is None or len(df) == 0:
                print(f"[X] No data!")
                continue

            # Save raw OHLCV
            raw_path = DATA_DIR / f"{safe_name}_{tf_name}.parquet"
            df.write_parquet(raw_path)
            raw_data[sym][tf_name] = df

            days = (df["time"].max() - df["time"].min()).days
            size_kb = raw_path.stat().st_size / 1024
            print(f"[OK] {len(df):,} bars ({days}d) → {raw_path.name} ({size_kb:.0f}KB)")

    if not raw_data:
        print("\n[X] No data fetched!")
        mt5.shutdown()
        sys.exit(1)

    # ── Step 3: Build 28 raw features per TF ──
    print(f"\n[3/5] Building raw SMC + Volume + PA features (28 per TF)...")

    features_data: dict[str, dict[str, "pl.DataFrame"]] = {}

    for sym in raw_data:
        features_data[sym] = {}
        safe_name = sym.replace(".", "_")

        for tf_name, df in raw_data[sym].items():
            # Rename tick_volume → volume if needed
            if "tick_volume" in df.columns and "volume" not in df.columns:
                df = df.rename({"tick_volume": "volume"})

            feat_df = build_features(df, vol_window=20, swing_lookback=5)
            features_data[sym][tf_name] = feat_df

            feat_path = DATA_DIR / f"{safe_name}_{tf_name}_features.parquet"
            feat_df.write_parquet(feat_path)
            print(f"  {sym} {tf_name}: {len(feat_df):,} rows × {len(feat_df.columns)} cols → {feat_path.name}")

    # ── Step 4: Add Knowledge features (22-dim) → 50-dim total ──
    print(f"\n[4/5] Adding Knowledge features (22-dim per bar)...")

    # Initialize normalizers (one per TF, 50-dim each)
    normalizers: dict[str, RunningNormalizer] = {
        tf: RunningNormalizer(n_features=50) for tf in TF_CONFIG
    }

    for sym in features_data:
        safe_name = sym.replace(".", "_")

        for tf_name, feat_df in features_data[sym].items():
            # Extract OHLCV for KnowledgeExtractor
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            if all(c in feat_df.columns for c in ohlcv_cols):
                ohlcv_np = feat_df.select(ohlcv_cols).to_numpy().astype(np.float32)
            else:
                # Fallback: use first 5 numeric cols
                ohlcv_np = feat_df.select(feat_df.columns[1:6]).to_numpy().astype(np.float32)

            # Build 22 knowledge features
            knowledge = build_knowledge_features(ohlcv_np)

            # Trim knowledge to match feat_df length
            knowledge = knowledge[-len(feat_df):]

            # Get 28 raw feature columns
            from data_engine.feature_builder import FEATURE_COLUMNS
            raw_cols = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
            raw_np = feat_df.select(raw_cols).to_numpy().astype(np.float32)

            # Pad or trim raw features to exactly 28
            if raw_np.shape[1] < 28:
                pad = np.zeros((len(raw_np), 28 - raw_np.shape[1]), dtype=np.float32)
                raw_np = np.hstack([raw_np, pad])
            elif raw_np.shape[1] > 28:
                raw_np = raw_np[:, :28]

            # Concatenate: 28 raw + 22 knowledge = 50-dim
            full_features = np.hstack([raw_np, knowledge]).astype(np.float32)

            # Replace NaN with 0
            full_features = np.nan_to_num(full_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Update normalizer
            normalizers[tf_name].update_batch(full_features)

            # Save 50-dim feature array
            save_path = DATA_DIR / f"{safe_name}_{tf_name}_50dim.npy"
            np.save(save_path, full_features)
            size_mb = full_features.nbytes / (1024 * 1024)
            print(f"  {sym} {tf_name}: ({full_features.shape[0]:,} × {full_features.shape[1]}) → {save_path.name} ({size_mb:.1f}MB)")

            # V3: Also save raw OHLCV for env price calculations
            # Trim OHLCV to match feature array length
            ohlcv_trimmed = ohlcv_np[-len(full_features):]
            ohlcv_path = DATA_DIR / f"{safe_name}_{tf_name}_ohlcv.npy"
            np.save(ohlcv_path, ohlcv_trimmed)
            print(f"    + OHLCV: ({ohlcv_trimmed.shape[0]:,} × {ohlcv_trimmed.shape[1]}) → {ohlcv_path.name}")

        gc.collect()  # Free memory after each symbol

    # ── Step 5: Save normalizer_v3.json ──
    print(f"\n[5/5] Saving normalizer_v3.json...")

    normalizer_data = {}
    for tf_name, norm in normalizers.items():
        normalizer_data[tf_name] = norm.state_dict()
        print(f"  {tf_name}: count={norm.count:,}, mean_range=[{norm.mean.min():.3f}, {norm.mean.max():.3f}], std_range=[{norm.std.min():.3f}, {norm.std.max():.3f}]")

    norm_path = DATA_DIR / "normalizer_v3.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(normalizer_data, f, indent=2)
    print(f"  Saved → {norm_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  DATA FETCH v3.0 COMPLETE!")
    print(f"  TFs: M1 / M5 / M15 / H1 | Features: 50-dim (28 raw + 22 knowledge)")
    print("=" * 70)

    npy_files = sorted(DATA_DIR.glob("*_50dim.npy"))
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    total_npy = sum(f.stat().st_size for f in npy_files)
    total_parquet = sum(f.stat().st_size for f in parquet_files)

    print(f"  50-dim NPY files: {len(npy_files)} ({total_npy / 1024 / 1024:.1f} MB)")
    print(f"  Parquet files: {len(parquet_files)} ({total_parquet / 1024 / 1024:.1f} MB)")
    print(f"  Normalizer: {norm_path.name}")

    for f in npy_files:
        arr = np.load(f)
        print(f"    {f.name:<45} {arr.shape[0]:>8,} × {arr.shape[1]} = {f.stat().st_size / 1024:.0f} KB")

    mt5.shutdown()
    print("\nDone! Ready for train_curriculum.py →")


if __name__ == "__main__":
    main()
