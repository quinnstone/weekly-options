"""
Universe module for the Zero-DTE Options Trading Analysis System.

Provides a curated universe of ~100 stocks known to have liquid weekly
options and sufficient intraday movement for 0DTE trading on Robinhood.

Previous approach scraped 520+ S&P 500 / Nasdaq-100 tickers from Wikipedia
and checked each for Friday options -- too broad, most lacked the liquidity
needed.  This module instead maintains a hardcoded list of tickers selected
for:
  - Weekly options with high volume (thousands of contracts daily)
  - Intraday movement (ATR% > 1%) making 0DTE viable
  - Availability on Robinhood

A validation function is still provided to confirm Friday-expiry
availability via yfinance, but it is a *validation* step, not discovery.

Results are cached to disk for 24 hours to avoid redundant network calls.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yfinance as yf

from config import CANDIDATES_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_FILE = CANDIDATES_DIR / "universe_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# ---------------------------------------------------------------------------
# Curated universe -- ~100 tickers with liquid weekly options, organized
# by sector so the pipeline can enforce sector diversity.
# ---------------------------------------------------------------------------

LIQUID_WEEKLY_OPTIONS: Dict[str, List[str]] = {
    "indices_etfs": [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
        "GLD", "SLV", "TLT", "EEM", "HYG", "ARKK",
    ],
    "mega_cap_tech": [
        "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    ],
    "semiconductors": [
        "AMD", "INTC", "MU", "QCOM", "AVGO", "MRVL", "ON", "SMCI",
    ],
    "software_cloud": [
        "CRM", "SNOW", "PLTR", "NET", "DDOG", "ZS", "CRWD", "PANW",
    ],
    "fintech_payments": [
        "XYZ", "PYPL", "COIN", "HOOD", "SOFI", "V", "MA",
    ],
    "banks_finance": [
        "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW",
    ],
    "energy": [
        "XOM", "CVX", "OXY", "SLB", "HAL", "DVN", "MPC",
    ],
    "retail_consumer": [
        "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
    ],
    "healthcare_biotech": [
        "UNH", "JNJ", "PFE", "MRNA", "ABBV", "LLY", "AMGN",
    ],
    "industrials": [
        "BA", "CAT", "DE", "GE", "RTX", "LMT", "UPS", "FDX",
    ],
    "media_entertainment": [
        "NFLX", "DIS", "CMCSA", "WBD", "PSKY", "ROKU",
    ],
    "travel_leisure": [
        "DAL", "UAL", "LUV", "AAL", "ABNB", "MAR", "WYNN",
    ],
    "auto_ev": [
        "F", "GM", "RIVN", "LCID", "NIO", "LI",
    ],
    "telecom": [
        "T", "VZ", "TMUS",
    ],
}

# ETFs with true daily (Mon-Fri) 0-DTE expirations
_ZERO_DTE_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

# Reverse lookup: ticker -> sector (built once at import time)
_TICKER_TO_SECTOR: Dict[str, str] = {}
for _sector, _tickers in LIQUID_WEEKLY_OPTIONS.items():
    for _t in _tickers:
        _TICKER_TO_SECTOR[_t] = _sector


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _read_cache() -> Optional[dict]:
    """Return cached data if the cache file exists and is fresh, else None."""
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at < CACHE_TTL_SECONDS:
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _write_cache(payload: dict) -> None:
    """Persist *payload* to the cache file with a timestamp."""
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    payload["cached_at"] = time.time()
    CACHE_FILE.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_full_universe() -> List[str]:
    """Return all ~100 curated tickers as a flat sorted list."""
    tickers: set[str] = set()
    for sector_tickers in LIQUID_WEEKLY_OPTIONS.values():
        tickers.update(sector_tickers)
    return sorted(tickers)


def get_zero_dte_etfs() -> List[str]:
    """Return the ETFs that have true daily (Mon-Fri) 0-DTE expirations."""
    return list(_ZERO_DTE_ETFS)


def get_universe_by_sector() -> Dict[str, List[str]]:
    """Return the curated universe organized by sector.

    Returns a copy so callers cannot mutate the module-level dict.
    """
    return {sector: list(tickers) for sector, tickers in LIQUID_WEEKLY_OPTIONS.items()}


def get_sector_for_ticker(ticker: str) -> str:
    """Return the sector name for *ticker*, or ``'unknown'`` if not found."""
    return _TICKER_TO_SECTOR.get(ticker.upper(), "unknown")


def validate_options_availability(
    tickers: List[str],
    use_cache: bool = True,
) -> List[str]:
    """Check which *tickers* have a Friday-expiry options chain via yfinance.

    This is a **validation** step, not a discovery step -- we expect the vast
    majority of tickers in our curated list to pass.  Results are cached for
    24 hours so repeated calls are cheap.

    Parameters
    ----------
    tickers : list[str]
        Tickers to validate.
    use_cache : bool
        When True (default), return cached results if fresh.

    Returns
    -------
    list[str]
        Sorted subset of *tickers* confirmed to have a Friday expiration.
    """
    # --- Try the cache first ---
    if use_cache:
        cached = _read_cache()
        if cached and "validated_tickers" in cached:
            cached_set = set(cached["validated_tickers"])
            requested = set(t.upper() for t in tickers)
            # If every requested ticker was already validated, return the
            # intersection immediately.
            if requested.issubset(cached_set | set(cached.get("failed_tickers", []))):
                result = sorted(requested & cached_set)
                logger.info(
                    "Returning %d validated tickers from cache (age %.1f hrs).",
                    len(result),
                    (time.time() - cached["cached_at"]) / 3600,
                )
                return result

    logger.info("Validating options availability for %d tickers...", len(tickers))

    today = datetime.now().date()
    # Calculate next Friday (weekday 4)
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        next_friday = today
    else:
        next_friday = today + timedelta(days=days_until_friday)
    next_friday_str = next_friday.strftime("%Y-%m-%d")

    validated: List[str] = []
    failed: List[str] = []

    for i, ticker in enumerate(tickers, 1):
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options  # tuple of "YYYY-MM-DD" strings
            if expirations and next_friday_str in expirations:
                validated.append(ticker)
            else:
                failed.append(ticker)
        except Exception:
            failed.append(ticker)

        if i % 25 == 0:
            logger.info("Validation progress: %d / %d checked", i, len(tickers))

    validated = sorted(set(validated))
    logger.info(
        "Validation complete: %d / %d tickers have Friday expiry.",
        len(validated),
        len(tickers),
    )
    if failed:
        logger.info("Tickers without Friday expiry: %s", ", ".join(sorted(failed)))

    # --- Persist to cache ---
    _write_cache({"validated_tickers": validated, "failed_tickers": sorted(failed)})

    return validated
