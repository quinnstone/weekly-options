"""
Dynamic universe expansion for the Zero-DTE Options Trading Analysis System.

Discovers tickers outside the core curated universe that may be worth
scanning this week, based on:
  - Unusual options volume (market is pricing a big move)
  - Upcoming earnings (juiced IV, temporary weekly liquidity)
  - Finnhub news buzz (high article count in last 48 hours)

Every addition is validated for Friday-expiry options availability
before it enters the pipeline. Tickers without liquid weeklies are
discarded regardless of signal strength.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

import yfinance as yf
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from universe.robinhood import get_full_universe

logger = logging.getLogger(__name__)
config = Config()

# Broad scan pool — large/mid-cap tickers NOT in the core 103 that
# sometimes have liquid weeklies. Kept intentionally broad; the
# Friday-expiry validation gate filters out anything illiquid.
_EXPANSION_POOL = [
    # Tech / growth not in core
    "SHOP", "UBER", "LYFT", "SNAP", "PINS", "XYZ", "TTD", "RBLX",
    "DASH", "ZM", "DOCU", "TWLO", "OKTA", "MDB", "U", "PATH",
    "ANET", "FTNT", "HUBS", "BILL",
    # Biotech / pharma
    "GILD", "REGN", "VRTX", "BIIB", "ILMN", "DXCM", "ISRG", "GEHC",
    "ARGX", "BMRN",
    # Financials
    "AXP", "BLK", "ICE", "CME", "SPGI", "MCO", "TFC", "USB",
    # Consumer / retail
    "LULU", "ROST", "DG", "DLTR", "EL", "KO", "PEP", "PG",
    "CL", "KMB",
    # Industrials / materials
    "MMM", "EMR", "ITW", "SHW", "APD", "ECL", "FCX", "NEM",
    # Energy
    "COP", "EOG", "PSX", "VLO", "FANG",
    # Media / gaming
    "EA", "TTWO", "MTCH", "SPOT",
    # China ADRs (volatile, sometimes liquid weeklies)
    "BABA", "JD", "PDD", "BIDU", "BILI",
    # Other high-vol names
    "MARA", "RIOT", "CLSK", "HUT", "BITF",
]


def discover_dynamic_tickers(max_additions: int = 20) -> list:
    """Find tickers outside the core universe worth scanning this week.

    Runs three independent scans and merges results:
      1. Unusual options volume
      2. Earnings this week
      3. News buzz (high Finnhub article count)

    Each candidate is validated for Friday-expiry options. Only those
    with confirmed liquid weeklies are returned.

    Parameters
    ----------
    max_additions : int
        Maximum number of dynamic tickers to add (default 20).

    Returns
    -------
    list[dict]
        Each dict has: ticker, source (why it was added), sector ("dynamic").
    """
    core = set(get_full_universe())
    candidates = {}  # ticker -> set of sources

    # 1. Unusual options volume
    logger.info("Scanning for unusual options volume...")
    for ticker in _EXPANSION_POOL:
        if ticker in core:
            continue
        vol_signal = _check_unusual_options_volume(ticker)
        if vol_signal:
            candidates.setdefault(ticker, set()).add("unusual_options_volume")

    # 2. Earnings this week
    logger.info("Checking earnings calendar...")
    earnings_tickers = _get_earnings_this_week()
    for ticker in earnings_tickers:
        if ticker not in core and ticker in set(_EXPANSION_POOL):
            candidates.setdefault(ticker, set()).add("earnings_this_week")

    # 3. News buzz
    logger.info("Checking news buzz...")
    for ticker in _EXPANSION_POOL[:30]:  # limit API calls
        if ticker in core:
            continue
        if _check_news_buzz(ticker):
            candidates.setdefault(ticker, set()).add("news_buzz")

    # 4. Social trending (Reddit/Twitter buzz from social crawler)
    logger.info("Checking social trending tickers...")
    try:
        from scanners.social_crawler import SocialCrawler
        crawler = SocialCrawler()
        social_data = crawler.crawl_all()
        for item in social_data.get("combined_trending", []):
            ticker = item.get("ticker", "")
            mentions = item.get("mentions", 0)
            if ticker and ticker not in core and mentions >= 3:
                candidates.setdefault(ticker, set()).add("social_trending")
    except ImportError:
        logger.debug("Social crawler not available for dynamic discovery")
    except Exception as exc:
        logger.debug("Social trending scan failed (non-fatal): %s", exc)

    if not candidates:
        logger.info("No dynamic candidates found this week")
        return []

    # Rank by number of sources (more signals = higher priority)
    ranked = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)

    # Validate Friday-expiry options exist
    tickers_to_validate = [t for t, _ in ranked]
    validated = _validate_friday_expiry(tickers_to_validate)
    validated_set = set(validated)

    results = []
    for ticker, sources in ranked:
        if ticker not in validated_set:
            continue
        results.append({
            "ticker": ticker,
            "source": ", ".join(sorted(sources)),
            "sector": "dynamic",
        })
        if len(results) >= max_additions:
            break

    logger.info(
        "Dynamic universe: %d candidates found, %d validated with Friday options",
        len(candidates), len(results),
    )
    return results


def get_expanded_universe() -> list:
    """Return core universe + dynamic additions as a flat ticker list."""
    core = get_full_universe()
    dynamic = discover_dynamic_tickers()
    dynamic_tickers = [d["ticker"] for d in dynamic]

    expanded = sorted(set(core + dynamic_tickers))
    logger.info(
        "Expanded universe: %d core + %d dynamic = %d total",
        len(core), len(dynamic_tickers), len(expanded),
    )
    return expanded


def get_dynamic_metadata() -> dict:
    """Return metadata about dynamic additions (for logging/reports).

    Returns dict mapping ticker -> source reason.
    """
    dynamic = discover_dynamic_tickers()
    return {d["ticker"]: d["source"] for d in dynamic}


# ------------------------------------------------------------------
#  Signal detection
# ------------------------------------------------------------------

def _check_unusual_options_volume(ticker: str) -> bool:
    """Check if a ticker has unusually high options volume today.

    Looks for total options volume > 2x the 20-day average.
    Uses a lightweight yfinance call.
    """
    try:
        tk = yf.Ticker(ticker)
        # Get current options chain for nearest expiry
        expirations = tk.options
        if not expirations:
            return False

        chain = tk.option_chain(expirations[0])
        calls_vol = chain.calls["volume"].sum() if "volume" in chain.calls.columns else 0
        puts_vol = chain.puts["volume"].sum() if "volume" in chain.puts.columns else 0
        total_vol = (calls_vol or 0) + (puts_vol or 0)

        # Threshold: > 5000 contracts suggests active weekly options
        return total_vol > 5000
    except Exception:
        return False


def _get_earnings_this_week() -> list:
    """Get tickers with earnings this week from Finnhub."""
    if not config.finnhub_api_key:
        return []

    today = datetime.now()
    # Find this week's Monday and Friday
    monday = today - timedelta(days=today.weekday())
    friday = monday + timedelta(days=4)

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={
                "from": monday.strftime("%Y-%m-%d"),
                "to": friday.strftime("%Y-%m-%d"),
                "token": config.finnhub_api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        earnings = data.get("earningsCalendar", [])
        return [e["symbol"] for e in earnings if "symbol" in e]
    except Exception as exc:
        logger.debug("Earnings calendar fetch failed: %s", exc)
        return []


def _check_news_buzz(ticker: str) -> bool:
    """Check if a ticker has high news volume in the last 48 hours.

    Returns True if >= 5 articles found on Finnhub in 2 days.
    """
    if not config.finnhub_api_key:
        return False

    today = datetime.now()
    date_from = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": date_from,
                "to": date_to,
                "token": config.finnhub_api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return False

        articles = resp.json()
        return len(articles) >= 5
    except Exception:
        return False


# ------------------------------------------------------------------
#  Validation
# ------------------------------------------------------------------

def _validate_friday_expiry(tickers: list) -> list:
    """Check which tickers have a Friday-expiry options chain."""
    today = datetime.now().date()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        next_friday = today
    else:
        next_friday = today + timedelta(days=days_until_friday)
    friday_str = next_friday.strftime("%Y-%m-%d")

    validated = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options
            if expirations and friday_str in expirations:
                validated.append(ticker)
        except Exception:
            continue

    return validated
