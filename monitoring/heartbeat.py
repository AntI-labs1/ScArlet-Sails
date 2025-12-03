"""Lightweight heartbeat utilities for live/backtest monitoring."""

import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


def send_heartbeat(strategy_name: str, last_ts: datetime, last_price: float, extra: Dict[str, Any] | None = None):
    """Emit a heartbeat log line with minimal strategy context."""

    payload = {
        "strategy": strategy_name,
        "timestamp": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
        "price": float(last_price) if last_price is not None else None,
    }
    if extra:
        payload.update(extra)

    logger.info("HEARTBEAT: %s", payload)

