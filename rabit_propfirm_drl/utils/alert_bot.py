"""
Telegram Alert Bot — Push notifications for all critical system events.

Events:
- Trade opened/closed
- Drawdown warning
- Killswitch activated
- Nightly retrain result
- System errors

Uses httpx for async HTTP calls to Telegram Bot API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Graceful import
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class AlertBot:
    """
    Telegram alert sender. Reads token/chat_id from env vars or constructor.

    Env vars:
        TELEGRAM_BOT_TOKEN: Bot API token
        TELEGRAM_CHAT_ID: Target chat ID
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.token and self.chat_id)

        if not self._enabled:
            logger.warning(
                "AlertBot disabled: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID not set"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _format_message(self, level: str, title: str, body: str) -> str:
        """Format alert message with emoji and timestamp."""
        emojis = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "🚨",
            "trade": "💹",
        }
        emoji = emojis.get(level, "📌")
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"{emoji} *{title}*\n{body}\n\n_{timestamp}_"

    async def send_async(
        self,
        level: str,
        title: str,
        body: str,
    ) -> bool:
        """Send alert asynchronously via httpx."""
        if not self._enabled:
            logger.debug("Alert skipped (bot disabled): %s", title)
            return False

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed, cannot send Telegram alert")
            return False

        message = self._format_message(level, title, body)
        url = TELEGRAM_API_URL.format(token=self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                logger.info("Alert sent: %s", title)
                return True
        except Exception as e:
            logger.error("Failed to send alert '%s': %s", title, e)
            return False

    def send(self, level: str, title: str, body: str) -> bool:
        """Send alert synchronously (creates event loop if needed)."""
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule it
            future = asyncio.ensure_future(self.send_async(level, title, body))
            return False  # Can't block, return False
        except RuntimeError:
            # No running loop — safe to run synchronously
            return asyncio.run(self.send_async(level, title, body))

    # ─── Convenience methods ─────────────────

    def trade_opened(
        self, symbol: str, direction: str, lots: float, price: float,
        sl: float, tp: float, confidence: float,
    ) -> bool:
        return self.send(
            "trade",
            f"Trade Opened: {direction} {symbol}",
            f"Lots: {lots}\nEntry: {price}\nSL: {sl} | TP: {tp}\n"
            f"Confidence: {confidence:.2f}",
        )

    def trade_closed(
        self, symbol: str, pnl: float, duration_min: float,
    ) -> bool:
        level = "success" if pnl >= 0 else "warning"
        return self.send(
            level,
            f"Trade Closed: {symbol}",
            f"PnL: {pnl:+.2f}\nDuration: {duration_min:.0f} min",
        )

    def dd_warning(self, current_dd: float, limit: float) -> bool:
        return self.send(
            "warning",
            "Drawdown Warning",
            f"Current DD: {current_dd:.2%}\nLimit: {limit:.2%}\n"
            f"Headroom: {limit - current_dd:.2%}",
        )

    def killswitch_activated(self, dd: float, positions_closed: int) -> bool:
        return self.send(
            "error",
            "🛑 KILLSWITCH ACTIVATED",
            f"DD: {dd:.2%} exceeded threshold\n"
            f"Positions force-closed: {positions_closed}\n"
            f"Trading HALTED",
        )

    def retrain_result(
        self, deployed: bool, new_sharpe: float, old_sharpe: float,
        version: str,
    ) -> bool:
        if deployed:
            return self.send(
                "success",
                "Nightly Retrain: Deployed",
                f"New model: {version}\n"
                f"Sharpe: {old_sharpe:.2f} → {new_sharpe:.2f}",
            )
        else:
            return self.send(
                "warning",
                "Nightly Retrain: Rejected",
                f"New Sharpe ({new_sharpe:.2f}) worse than old ({old_sharpe:.2f})\n"
                f"Keeping current model",
            )

    def system_error(self, error: str, module: str) -> bool:
        return self.send(
            "error",
            f"System Error: {module}",
            f"```\n{error[:500]}\n```",
        )
