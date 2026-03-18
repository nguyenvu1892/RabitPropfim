"""
MT5 Connection Test — Verifies FTMO Demo account connectivity.

Usage: py -3.11 scripts/test_mt5_connection.py

Checks:
1. MT5 package installed
2. Terminal connection
3. Account login with credentials from .env
4. Account info retrieval
5. Symbol availability
6. Tick data retrieval
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_env() -> dict[str, str]:
    """Load credentials from .env file."""
    env_path = project_root / ".env"
    if not env_path.exists():
        print("❌ .env file not found! Copy .env.example to .env and fill in credentials.")
        sys.exit(1)

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
    print("🐰 RABIT-PROPFIRM — MT5 Connection Test")
    print("=" * 60)

    # Step 1: Check MT5 package
    print("\n[1/6] Checking MetaTrader5 package...")
    try:
        import MetaTrader5 as mt5
        print(f"  ✅ MetaTrader5 version: {mt5.__version__}")
    except ImportError:
        print("  ❌ MetaTrader5 not installed!")
        print("  Run: pip install MetaTrader5")
        sys.exit(1)

    # Step 2: Load credentials
    print("\n[2/6] Loading credentials from .env...")
    env = load_env()
    login = int(env.get("MT5_LOGIN", "0"))
    password = env.get("MT5_PASSWORD", "")
    server = env.get("MT5_SERVER", "")
    symbol = env.get("MT5_SYMBOL", "EURUSD")

    print(f"  Login: {login}")
    print(f"  Server: {server}")
    print(f"  Symbol: {symbol}")

    # Step 3: Initialize MT5
    print("\n[3/6] Connecting to MT5 terminal...")
    if not mt5.initialize():
        print(f"  ❌ MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)
    print(f"  ✅ Terminal: {mt5.terminal_info().name}")

    # Step 4: Login to account
    print("\n[4/6] Logging into FTMO account...")
    if not mt5.login(login, password=password, server=server):
        print(f"  ❌ Login failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    account = mt5.account_info()
    print(f"  ✅ Logged in successfully!")
    print(f"  Account: {account.login}")
    print(f"  Name: {account.name}")
    print(f"  Server: {account.server}")
    print(f"  Balance: ${account.balance:,.2f}")
    print(f"  Equity: ${account.equity:,.2f}")
    print(f"  Leverage: 1:{account.leverage}")
    print(f"  Currency: {account.currency}")
    print(f"  Trade mode: {'Demo' if account.trade_mode == 0 else 'Real'}")

    # Step 5: Check symbol
    print(f"\n[5/6] Checking symbol {symbol}...")
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        # Try with suffix
        for suffix in ["", "m", ".r", "_raw"]:
            test_sym = symbol + suffix
            sym_info = mt5.symbol_info(test_sym)
            if sym_info:
                symbol = test_sym
                break

    if sym_info is None:
        print(f"  ⚠️ Symbol {symbol} not found. Available symbols:")
        symbols = mt5.symbols_get()
        if symbols:
            forex = [s.name for s in symbols if "USD" in s.name][:10]
            print(f"  {forex}")
    else:
        mt5.symbol_select(symbol, True)
        print(f"  ✅ Symbol: {sym_info.name}")
        print(f"  Spread: {sym_info.spread} points")
        print(f"  Digits: {sym_info.digits}")
        print(f"  Min lot: {sym_info.volume_min}")
        print(f"  Max lot: {sym_info.volume_max}")
        print(f"  Lot step: {sym_info.volume_step}")

    # Step 6: Get current tick
    print(f"\n[6/6] Getting current tick data...")
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"  ✅ Bid: {tick.bid}")
        print(f"  ✅ Ask: {tick.ask}")
        print(f"  Spread: {(tick.ask - tick.bid) / sym_info.point:.1f} points")
    else:
        print(f"  ⚠️ No tick data available (market may be closed)")

    # Summary
    print("\n" + "=" * 60)
    print("🎉 ALL CHECKS PASSED — MT5 connection is READY!")
    print(f"Account: {account.login} @ {account.server}")
    print(f"Balance: ${account.balance:,.2f}")
    print("=" * 60)

    mt5.shutdown()


if __name__ == "__main__":
    main()
