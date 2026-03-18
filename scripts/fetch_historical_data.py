"""
Fetch Historical Data — Downloads M5 candles from MT5, resamples, builds SMC features.

Usage: py -3.11 scripts/fetch_historical_data.py

Pipeline:
1. Fetch M5 data (primary TF) for 5 target symbols → 50K bars each
2. Resample to M15, H1, H4
3. Build SMC + Volume + Price Action features for M5
4. Build inside_bar feature for H1 (for exit rule)
5. Save all as Parquet
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "rabit_propfirm_drl"))

DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)


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
    print("  RABIT-PROPFIRM -- Data Fetch (M5 Primary TF)")
    print("  Features: SMC + Volume + Price Action")
    print("=" * 60)

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

    from data_engine.feature_builder import build_features, inside_bar, FEATURE_COLUMNS

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

    # Fetch M5 data
    BATCH = 50_000
    print(f"\n[2/5] Fetching M5 data (up to {BATCH:,} bars per symbol)...")
    print(f"  Symbols: {symbols}")
    print("-" * 60)

    all_data: dict[str, "pl.DataFrame"] = {}
    for sym in symbols:
        print(f"  Fetching {sym} M5...", end=" ", flush=True)
        mt5.symbol_select(sym, True)

        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, BATCH)

        if rates is None or len(rates) == 0:
            print(f"[X] No data! ({mt5.last_error()})")
            continue

        times = [datetime.utcfromtimestamp(r[0]) for r in rates]
        df = pl.DataFrame({
            "time": times,
            "open": [float(r[1]) for r in rates],
            "high": [float(r[2]) for r in rates],
            "low": [float(r[3]) for r in rates],
            "close": [float(r[4]) for r in rates],
            "tick_volume": [int(r[5]) for r in rates],
            "spread": [int(r[6]) for r in rates],
            "real_volume": [int(r[7]) for r in rates],
        }).unique(subset=["time"]).sort("time")

        safe_name = sym.replace(".", "_")
        m5_path = DATA_DIR / f"{safe_name}_M5.parquet"
        df.write_parquet(m5_path)

        all_data[sym] = df
        days = (df["time"].max() - df["time"].min()).days
        size_kb = m5_path.stat().st_size / 1024
        print(f"[OK] {len(df):,} bars ({days} days) -> {m5_path.name} ({size_kb:.0f} KB)")

    if not all_data:
        print("\n[X] No data fetched!")
        mt5.shutdown()
        sys.exit(1)

    # Resample to M15, H1, H4
    print(f"\n[3/5] Resampling to M15 / H1 / H4...")
    h1_data: dict[str, "pl.DataFrame"] = {}

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

            if tf_name == "H1":
                h1_data[sym] = resampled

    # Build SMC + Volume + PA features for M5
    print(f"\n[4/5] Building SMC + Volume + Price Action features for M5...")
    for sym in all_data:
        safe_name = sym.replace(".", "_")
        m5_path = DATA_DIR / f"{safe_name}_M5.parquet"
        df = pl.read_parquet(m5_path)

        # Rename tick_volume → volume for feature_builder compatibility
        if "volume" not in df.columns and "tick_volume" in df.columns:
            df = df.rename({"tick_volume": "volume"})

        features_df = build_features(df, vol_window=20, swing_lookback=5)
        feat_path = DATA_DIR / f"{safe_name}_M5_features.parquet"
        features_df.write_parquet(feat_path)
        print(f"  {sym}: {len(features_df):>6,} rows x {len(features_df.columns)} cols -> {feat_path.name}")

    # Build H1 inside bar feature (for exit rule)
    print(f"\n[5/5] Building H1 inside bar feature...")
    for sym, h1_df in h1_data.items():
        safe_name = sym.replace(".", "_")
        h1_ib = inside_bar(h1_df)
        ib_path = DATA_DIR / f"{safe_name}_H1_insidebar.parquet"
        h1_ib.select(["time", "inside_bar"]).write_parquet(ib_path)
        ib_count = h1_ib["inside_bar"].sum()
        print(f"  {sym} H1: {ib_count:.0f} inside bars detected -> {ib_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("  DATA FETCH COMPLETE!")
    print("  Primary TF: M5 | Features: SMC + Volume + PA")
    print("=" * 60)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    print(f"  Files: {len(parquet_files)}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

    for f in parquet_files:
        size = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size:>8.1f} KB")

    mt5.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
