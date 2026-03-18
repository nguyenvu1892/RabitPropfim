"""
Fetch Historical Data — Downloads M1 candles from MT5 for all target symbols.

Usage: py -3.11 scripts/fetch_historical_data.py

Fetches ~6 months of M1 data using copy_rates_from_pos (most reliable API),
resamples to M15/H1/H4, builds 11 features, saves all as Parquet.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)

# M1 bars in 6 months: ~180 days * 24h * 60m = ~260,000 (markets ~5 days/week)
# Actual: ~130,000 bars for forex (5d/7d * 260000)
MAX_BARS = 200_000


def load_env() -> dict[str, str]:
    env_path = project_root / ".env"
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env


def main() -> None:
    print("=" * 60)
    print("  RABIT-PROPFIRM -- Historical Data Fetch")
    print("=" * 60)

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[X] MetaTrader5 not installed!")
        sys.exit(1)

    try:
        import polars as pl
    except ImportError:
        print("[X] Polars not installed! Run: pip install polars")
        sys.exit(1)

    # Load config
    env = load_env()
    login = int(env.get("MT5_LOGIN", "0"))
    password = env.get("MT5_PASSWORD", "")
    server = env.get("MT5_SERVER", "")
    symbols_str = env.get("MT5_SYMBOLS", "XAUUSD,US100.cash,US30.cash,ETHUSD,BTCUSD")
    symbols = [s.strip() for s in symbols_str.split(",")]

    # Connect
    print("\n[1/4] Connecting to MT5...")
    if not mt5.initialize():
        print(f"  [X] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    if not mt5.login(login, password=password, server=server):
        print(f"  [X] Login failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    account = mt5.account_info()
    print(f"  [OK] Logged in: {account.name} (${account.balance:,.2f})")

    # Fetch M1 data
    print(f"\n[2/4] Fetching M1 data (up to {MAX_BARS:,} bars per symbol)...")
    print(f"  Symbols: {symbols}")
    print("-" * 60)

    all_data: dict[str, "pl.DataFrame"] = {}
    for sym in symbols:
        print(f"  Fetching {sym}...", end=" ", flush=True)
        mt5.symbol_select(sym, True)

        # Batch fetch: MT5 limit is ~50K bars per call
        BATCH = 50_000
        all_rates = []
        for start_pos in range(0, 150_001, BATCH):
            batch = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, start_pos, BATCH)
            if batch is not None and len(batch) > 0:
                all_rates.extend(batch)
            else:
                break

        if not all_rates:
            print(f"[X] No data! ({mt5.last_error()})")
            continue

        # Convert to Polars, deduplicate by timestamp, sort
        times = [datetime.utcfromtimestamp(r[0]) for r in all_rates]
        df = pl.DataFrame({
            "time": times,
            "open": [float(r[1]) for r in all_rates],
            "high": [float(r[2]) for r in all_rates],
            "low": [float(r[3]) for r in all_rates],
            "close": [float(r[4]) for r in all_rates],
            "tick_volume": [int(r[5]) for r in all_rates],
            "spread": [int(r[6]) for r in all_rates],
            "real_volume": [int(r[7]) for r in all_rates],
        }).unique(subset=["time"]).sort("time")

        # Save M1
        safe_name = sym.replace(".", "_")
        m1_path = DATA_DIR / f"{safe_name}_M1.parquet"
        df.write_parquet(m1_path)

        all_data[sym] = df
        days = (df["time"].max() - df["time"].min()).days
        size_kb = m1_path.stat().st_size / 1024
        print(f"[OK] {len(df):,} bars ({days} days) -> {m1_path.name} ({size_kb:.0f} KB)")

    if not all_data:
        print("\n[X] No data fetched for any symbol!")
        mt5.shutdown()
        sys.exit(1)

    # Step 3: Resample to M15, H1, H4
    print(f"\n[3/4] Resampling to M15 / H1 / H4...")
    for sym, df in all_data.items():
        safe_name = sym.replace(".", "_")

        for tf_name, tf_dur in [("M15", "15m"), ("H1", "1h"), ("H4", "4h")]:
            resampled = df.group_by_dynamic(
                "time", every=tf_dur
            ).agg([
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("tick_volume").sum().alias("tick_volume"),
                pl.col("spread").mean().alias("spread"),
            ]).sort("time")

            tf_path = DATA_DIR / f"{safe_name}_{tf_name}.parquet"
            resampled.write_parquet(tf_path)
            print(f"  {sym} {tf_name}: {len(resampled):>6,} bars -> {tf_path.name}")

    # Step 4: Build features for M15
    print(f"\n[4/4] Building features for M15...")

    for sym in all_data:
        safe_name = sym.replace(".", "_")
        m15_path = DATA_DIR / f"{safe_name}_M15.parquet"
        df = pl.read_parquet(m15_path)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["tick_volume"].to_numpy().astype(np.float64)

        # 1) Log returns
        log_ret = np.zeros_like(close)
        log_ret[1:] = np.log(close[1:] / (close[:-1] + 1e-10))

        # 2) ATR-14
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = np.zeros_like(tr)
        if len(tr) >= 14:
            atr[:14] = np.mean(tr[:14])
            for i in range(14, len(tr)):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14

        # 3) Volatility ratio
        vol_ratio = (high - low) / (atr + 1e-10)

        # 4) RSI-14
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        if len(close) >= 15:
            avg_gain[14] = np.mean(gains[1:15])
            avg_loss[14] = np.mean(losses[1:15])
            for i in range(15, len(close)):
                avg_gain[i] = (avg_gain[i-1] * 13 + gains[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + losses[i]) / 14
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)

        # 5) Body ratio
        body = np.abs(close - open_)
        candle_range = high - low + 1e-10
        body_ratio = body / candle_range

        # 6) Relative volume
        vol_ma = np.convolve(volume, np.ones(20)/20, mode="same")
        rvol = volume / (vol_ma + 1e-10)

        # 7-10) Time encoding
        times_list = df["time"].to_list()
        hours = np.array([t.hour for t in times_list])
        dow = np.array([t.weekday() for t in times_list])
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        dow_sin = np.sin(2 * np.pi * dow / 5)
        dow_cos = np.cos(2 * np.pi * dow / 5)

        # Build feature DF
        features_df = pl.DataFrame({
            "time": df["time"],
            "open": df["open"], "high": df["high"],
            "low": df["low"], "close": df["close"],
            "tick_volume": df["tick_volume"],
            "log_return": log_ret,
            "atr": atr,
            "vol_ratio": vol_ratio,
            "rsi": rsi,
            "body_ratio": body_ratio,
            "rvol": rvol,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
        })

        feat_path = DATA_DIR / f"{safe_name}_M15_features.parquet"
        features_df.write_parquet(feat_path)
        print(f"  {sym}: {len(features_df):>6,} rows x {len(features_df.columns)} cols -> {feat_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("  DATA FETCH COMPLETE!")
    print("=" * 60)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    print(f"  Files: {len(parquet_files)}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

    for f in parquet_files:
        size = f.stat().st_size / 1024
        print(f"    {f.name:<40} {size:>8.1f} KB")

    mt5.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
