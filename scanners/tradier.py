"""
Tradier API client for real-time options and equity data.

Provides fresh intraday options chains, quotes, and Greeks that yfinance
cannot deliver reliably. Used as the primary data source when available,
with yfinance as fallback.

Tradier sandbox (free tier) supports:
- Real-time equity quotes
- Options chains with Greeks
- Options expirations
- Historical price data

Setup: Get a free sandbox API key at https://developer.tradier.com/
Set TRADIER_API_KEY in .env (and optionally TRADIER_BASE_URL for production).
"""

import logging
import time
from datetime import datetime, timedelta

import requests

from config import Config

logger = logging.getLogger(__name__)
config = Config()


class TradierClient:
    """Thin wrapper around the Tradier API for options and equity data."""

    def __init__(self):
        self.api_key = config.tradier_api_key
        self.base_url = config.tradier_base_url.rstrip("/")
        self.enabled = config.has_tradier()
        self._session = requests.Session()
        if self.enabled:
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            })
            logger.info("Tradier client initialized (%s)", self.base_url)
        else:
            logger.info("Tradier not configured — will fall back to yfinance")

    def _get(self, path: str, params: dict = None) -> "dict | None":
        """Make a GET request to the Tradier API."""
        if not self.enabled:
            return None
        try:
            url = f"{self.base_url}{path}"
            resp = self._session.get(url, params=params or {}, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Tradier %s returned %d: %s", path, resp.status_code, resp.text[:200])
            return None
        except requests.RequestException as exc:
            logger.warning("Tradier request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    #  Equity quotes
    # ------------------------------------------------------------------

    def get_quote(self, ticker: str) -> "dict | None":
        """Get real-time quote for a single ticker.

        Returns dict with last, bid, ask, open, high, low, close,
        volume, change, change_pct, etc.
        """
        data = self._get("/v1/markets/quotes", {"symbols": ticker})
        if not data:
            return None

        quotes = data.get("quotes", {})
        quote = quotes.get("quote")
        if not quote:
            return None

        # Handle single vs list response
        if isinstance(quote, list):
            quote = quote[0]

        return {
            "ticker": ticker,
            "last": quote.get("last"),
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "prev_close": quote.get("prevclose"),
            "volume": quote.get("volume"),
            "change": quote.get("change"),
            "change_pct": quote.get("change_percentage"),
            "trade_date": quote.get("trade_date"),
        }

    def get_quotes_batch(self, tickers: list) -> dict:
        """Get quotes for multiple tickers in a single call."""
        if not tickers:
            return {}
        data = self._get("/v1/markets/quotes", {"symbols": ",".join(tickers)})
        if not data:
            return {}

        quotes = data.get("quotes", {}).get("quote", [])
        if isinstance(quotes, dict):
            quotes = [quotes]

        result = {}
        for q in quotes:
            sym = q.get("symbol", "")
            result[sym] = {
                "last": q.get("last"),
                "bid": q.get("bid"),
                "ask": q.get("ask"),
                "volume": q.get("volume"),
                "change_pct": q.get("change_percentage"),
            }
        return result

    # ------------------------------------------------------------------
    #  Options expirations
    # ------------------------------------------------------------------

    def get_expirations(self, ticker: str) -> list:
        """Get available options expiration dates for a ticker.

        Returns list of date strings (YYYY-MM-DD).
        """
        data = self._get(f"/v1/markets/options/expirations", {
            "symbol": ticker,
            "includeAllRoots": "true",
        })
        if not data:
            return []

        expirations = data.get("expirations", {})
        dates = expirations.get("date", [])
        if isinstance(dates, str):
            dates = [dates]
        return dates or []

    # ------------------------------------------------------------------
    #  Options chains
    # ------------------------------------------------------------------

    def get_options_chain(self, ticker: str, expiry: str, greeks: bool = True) -> dict:
        """Fetch a full options chain for a given expiry with real-time Greeks.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        expiry : str
            Expiration date (YYYY-MM-DD).
        greeks : bool
            Include Greeks in the response (default True).

        Returns
        -------
        dict
            Keys 'calls' and 'puts', each a list of option dicts with
            strike, bid, ask, last, volume, oi, iv, delta, gamma, theta, vega.
        """
        data = self._get("/v1/markets/options/chains", {
            "symbol": ticker,
            "expiration": expiry,
            "greeks": "true" if greeks else "false",
        })
        if not data:
            return {}

        options = data.get("options", {})
        option_list = options.get("option", [])
        if isinstance(option_list, dict):
            option_list = [option_list]

        calls = []
        puts = []

        for opt in option_list:
            greeks_data = opt.get("greeks", {}) or {}
            parsed = {
                "strike": opt.get("strike"),
                "bid": opt.get("bid", 0),
                "ask": opt.get("ask", 0),
                "mid": round((opt.get("bid", 0) + opt.get("ask", 0)) / 2, 4) if opt.get("bid") and opt.get("ask") else None,
                "last": opt.get("last"),
                "volume": opt.get("volume", 0),
                "open_interest": opt.get("open_interest", 0),
                "iv": greeks_data.get("mid_iv"),
                "delta": greeks_data.get("delta"),
                "gamma": greeks_data.get("gamma"),
                "theta": greeks_data.get("theta"),
                "vega": greeks_data.get("vega"),
                "rho": greeks_data.get("rho"),
                "expiry": expiry,
                "option_type": opt.get("option_type"),
                "symbol": opt.get("symbol"),
            }

            if opt.get("option_type") == "call":
                calls.append(parsed)
            else:
                puts.append(parsed)

        return {"calls": calls, "puts": puts}

    def get_weekly_expiry(self, ticker: str, target_days: int = 5) -> "str | None":
        """Find the best weekly expiry, same logic as OptionsScanner but using Tradier data."""
        expirations = self.get_expirations(ticker)
        if not expirations:
            return None

        today = datetime.now().date()
        target_date = today + timedelta(days=target_days)
        min_date = today + timedelta(days=3)

        # Prefer Friday expiries
        friday_candidates = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= min_date and exp_date.weekday() == 4:
                    friday_candidates.append((exp_str, exp_date))
            except ValueError:
                continue

        if friday_candidates:
            best = min(friday_candidates, key=lambda x: abs((x[1] - target_date).days))
            return best[0]

        # Fallback: nearest valid expiry
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= min_date:
                    return exp_str
            except ValueError:
                continue

        return None

    # ------------------------------------------------------------------
    #  Real-time strike selection helpers
    # ------------------------------------------------------------------

    def find_best_strike(
        self,
        ticker: str,
        direction: str,
        target_delta: float = 0.35,
        max_spread_pct: float = 15.0,
    ) -> "dict | None":
        """Find the optimal strike using real-time Tradier Greeks.

        Unlike BSM-estimated delta from yfinance, Tradier provides
        market-maker-computed Greeks that account for skew, term structure,
        and real-time supply/demand.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        direction : str
            'bullish' or 'bearish'.
        target_delta : float
            Target absolute delta (default 0.35).
        max_spread_pct : float
            Max bid-ask spread as % of mid (skip illiquid strikes).

        Returns
        -------
        dict or None
            Best strike data with real Greeks, or None if unavailable.
        """
        expiry = self.get_weekly_expiry(ticker)
        if not expiry:
            return None

        chain = self.get_options_chain(ticker, expiry)
        if not chain:
            return None

        quote = self.get_quote(ticker)
        current_price = quote.get("last") if quote else None
        if not current_price:
            return None

        option_type = "call" if direction == "bullish" else "put"
        candidates = chain.get("calls" if option_type == "call" else "puts", [])

        if not candidates:
            return None

        best = None
        best_delta_dist = float("inf")

        for opt in candidates:
            delta = opt.get("delta")
            if delta is None:
                continue

            bid = opt.get("bid", 0) or 0
            ask = opt.get("ask", 0) or 0
            mid = opt.get("mid") or ((bid + ask) / 2 if bid > 0 and ask > 0 else 0)

            # Skip: no real market
            if bid <= 0.05 or mid <= 0:
                continue

            # Skip: too wide spread
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999
            if spread_pct > max_spread_pct:
                continue

            # Delta distance from target
            abs_delta = abs(delta)
            delta_dist = abs(abs_delta - target_delta)

            if delta_dist < best_delta_dist:
                best_delta_dist = delta_dist
                strike = opt.get("strike")

                # Breakeven
                if option_type == "call":
                    breakeven = strike + mid
                    breakeven_move_pct = ((breakeven - current_price) / current_price) * 100
                else:
                    breakeven = strike - mid
                    breakeven_move_pct = ((current_price - breakeven) / current_price) * 100

                best = {
                    "ticker": ticker,
                    "option_type": option_type,
                    "direction": direction,
                    "expiry": expiry,
                    "strike": strike,
                    "premium": round(mid, 2),
                    "bid": bid,
                    "ask": ask,
                    "spread_pct": round(spread_pct, 2),
                    "iv": opt.get("iv"),
                    # Real Greeks from Tradier (not BSM estimates)
                    "greeks": {
                        "delta": opt.get("delta"),
                        "gamma": opt.get("gamma"),
                        "theta": opt.get("theta"),
                        "vega": opt.get("vega"),
                    },
                    "estimated_delta": round(abs_delta, 3),
                    "volume": opt.get("volume", 0),
                    "oi": opt.get("open_interest", 0),
                    "current_price": round(current_price, 2),
                    "breakeven": round(breakeven, 2),
                    "breakeven_move_pct": round(breakeven_move_pct, 2),
                    "data_source": "tradier_realtime",
                }

        return best

    # ------------------------------------------------------------------
    #  Opening range / intraday context
    # ------------------------------------------------------------------

    def get_opening_range(self, ticker: str) -> "dict | None":
        """Get the opening range (first 30 min high/low) for entry timing.

        Uses the current day's quote data to compute:
        - Opening range high/low
        - Current price position within the range
        - Whether the stock gapped up/down from previous close

        Returns None if market hasn't been open long enough.
        """
        quote = self.get_quote(ticker)
        if not quote:
            return None

        open_price = quote.get("open")
        high = quote.get("high")
        low = quote.get("low")
        last = quote.get("last")
        prev_close = quote.get("prev_close")

        if not all([open_price, high, low, last]):
            return None

        # Gap calculation
        gap_pct = ((open_price - prev_close) / prev_close * 100) if prev_close else 0

        # Opening range position (0 = at low, 1 = at high)
        range_size = high - low
        if range_size > 0:
            position_in_range = (last - low) / range_size
        else:
            position_in_range = 0.5

        return {
            "ticker": ticker,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "last": round(last, 2),
            "prev_close": round(prev_close, 2) if prev_close else None,
            "gap_pct": round(gap_pct, 2),
            "range_size": round(range_size, 2),
            "range_pct": round(range_size / open_price * 100, 2) if open_price > 0 else 0,
            "position_in_range": round(position_in_range, 3),
            "above_open": last > open_price,
        }
