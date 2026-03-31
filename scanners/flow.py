"""
Options-flow scanner for unusual activity detection.

Identifies unusual options volume, high volume-to-OI ratios on individual
strikes, and attempts to scrape public unusual-activity feeds.  Falls back
to yfinance-based analysis when scraping is unavailable.
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import Config

logger = logging.getLogger(__name__)
config = Config()


class FlowScanner:
    """Detect unusual options flow that may signal institutional positioning."""

    # ------------------------------------------------------------------ #
    #  Unusual activity scraping (with fallback)
    # ------------------------------------------------------------------ #

    def scrape_unusual_activity(self) -> list[dict]:
        """Attempt to fetch unusual options activity from public sources.

        Tries Barchart's unusual options activity page first.  If scraping
        fails (blocked, rate-limited, etc.), falls back to scanning a
        default watchlist via yfinance volume/OI analysis.

        Returns
        -------
        list[dict]
            Each dict: ticker, option_type, strike, expiry, volume, oi,
            vol_oi_ratio, signal_type.
        """
        # --- Attempt web scrape ---
        results = self._try_barchart_scrape()
        if results:
            return results

        logger.info("Barchart scrape unavailable; falling back to yfinance analysis")

        # --- Fallback: analyze popular tickers ---
        fallback_tickers = [
            "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA",
            "META", "GOOG", "AMD", "IWM", "DIA",
        ]

        all_signals = []
        for ticker in fallback_tickers:
            try:
                signals = self._analyze_ticker_flow(ticker)
                all_signals.extend(signals)
                time.sleep(0.3)
            except Exception as exc:
                logger.error("Fallback flow analysis failed for %s: %s", ticker, exc)

        # Sort by vol/OI ratio descending
        all_signals.sort(key=lambda s: s.get("vol_oi_ratio", 0), reverse=True)
        return all_signals[:50]  # Top 50 signals

    def _try_barchart_scrape(self) -> list[dict]:
        """Try scraping Barchart unusual options activity."""
        try:
            url = "https://www.barchart.com/options/unusual-activity/stocks"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            }

            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.info("Barchart returned status %d", resp.status_code)
                return []

            # Barchart typically blocks scraping or requires JS rendering.
            # Check if we got usable content.
            if "unusual-activity" not in resp.text.lower() or len(resp.text) < 5000:
                logger.info("Barchart page appears blocked or empty")
                return []

            # Attempt basic HTML table parsing
            try:
                tables = pd.read_html(resp.text)
                if not tables:
                    return []

                df = tables[0]
                results = []
                for _, row in df.head(50).iterrows():
                    results.append({
                        "ticker": str(row.get("Symbol", row.iloc[0] if len(row) > 0 else "")),
                        "option_type": str(row.get("Type", "unknown")).lower(),
                        "strike": float(row.get("Strike", 0)) if "Strike" in row.index else 0,
                        "expiry": str(row.get("Exp Date", "")),
                        "volume": int(row.get("Volume", 0)) if "Volume" in row.index else 0,
                        "oi": int(row.get("Open Int", 0)) if "Open Int" in row.index else 0,
                        "vol_oi_ratio": float(row.get("Vol/OI", 0)) if "Vol/OI" in row.index else 0,
                        "signal_type": "unusual_activity_scraped",
                    })
                return results

            except Exception as parse_exc:
                logger.info("HTML table parsing failed: %s", parse_exc)
                return []

        except Exception as exc:
            logger.info("Barchart scrape failed: %s", exc)
            return []

    def _analyze_ticker_flow(self, ticker: str) -> list[dict]:
        """Analyze a single ticker's options for unusual volume vs OI.

        Returns strikes where today's volume significantly exceeds open
        interest, suggesting new positions are being established.
        """
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options

            if not expirations:
                return []

            # Focus on nearest expiration
            expiry = expirations[0]
            chain = tk.option_chain(expiry)

            signals = []

            for option_type, opts in [("call", chain.calls), ("put", chain.puts)]:
                if opts.empty:
                    continue

                for _, row in opts.iterrows():
                    volume = int(row.get("volume", 0) or 0)
                    oi = int(row.get("openInterest", 0) or 0)

                    if volume == 0:
                        continue

                    vol_oi = volume / oi if oi > 0 else volume  # high ratio if no prior OI
                    bid = float(row.get("bid", 0) or 0)

                    # Flag if volume/OI > 3 and minimum liquidity met
                    if vol_oi > 3.0 and bid > 0.05 and volume >= 100:
                        signals.append({
                            "ticker": ticker,
                            "option_type": option_type,
                            "strike": float(row["strike"]),
                            "expiry": expiry,
                            "volume": volume,
                            "oi": oi,
                            "vol_oi_ratio": round(vol_oi, 2),
                            "iv": round(float(row.get("impliedVolatility", 0) or 0), 4),
                            "signal_type": "high_vol_oi_ratio",
                        })

            return signals

        except Exception as exc:
            logger.error("Flow analysis failed for %s: %s", ticker, exc)
            return []

    # ------------------------------------------------------------------ #
    #  Unusual volume detection
    # ------------------------------------------------------------------ #

    def detect_unusual_volume(self, ticker: str) -> dict:
        """Compare current total options volume to historical average.

        Flags the ticker if today's total options volume exceeds 2x
        the recent average.

        Returns
        -------
        dict
            Keys: ticker, current_call_vol, current_put_vol, total_vol,
            avg_daily_vol (estimated), vol_ratio, is_unusual, direction_bias.
        """
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options

            if not expirations:
                return {}

            # Sum volume across all near-term expirations (first 3)
            total_call_vol = 0
            total_put_vol = 0
            total_call_oi = 0
            total_put_oi = 0

            for expiry in expirations[:3]:
                try:
                    chain = tk.option_chain(expiry)
                    total_call_vol += int(chain.calls["volume"].sum() or 0)
                    total_put_vol += int(chain.puts["volume"].sum() or 0)
                    total_call_oi += int(chain.calls["openInterest"].sum() or 0)
                    total_put_oi += int(chain.puts["openInterest"].sum() or 0)
                except Exception:
                    continue

            total_vol = total_call_vol + total_put_vol
            total_oi = total_call_oi + total_put_oi

            # Use total OI as a rough proxy for "average daily volume"
            # Typically, daily volume is a fraction of total OI.
            # A ratio > 0.5 of vol/OI across all strikes is noteworthy.
            vol_ratio = total_vol / total_oi if total_oi > 0 else 0
            is_unusual = vol_ratio > 0.5  # More than 50% of OI traded today

            # Direction bias
            if total_call_vol > total_put_vol * 1.5:
                direction_bias = "bullish"
            elif total_put_vol > total_call_vol * 1.5:
                direction_bias = "bearish"
            else:
                direction_bias = "neutral"

            return {
                "ticker": ticker,
                "current_call_vol": total_call_vol,
                "current_put_vol": total_put_vol,
                "total_vol": total_vol,
                "total_oi": total_oi,
                "vol_oi_ratio": round(vol_ratio, 3),
                "is_unusual": is_unusual,
                "direction_bias": direction_bias,
            }

        except Exception as exc:
            logger.error("Unusual volume detection failed for %s: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Analyze flow signals across tickers
    # ------------------------------------------------------------------ #

    def analyze_flow_signals(self, tickers: list[str]) -> list[dict]:
        """Check volume/OI on individual strikes for each ticker.

        High volume relative to open interest on a single strike
        suggests new positions being opened (directional bet or hedge).

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to scan.

        Returns
        -------
        list[dict]
            Each dict: ticker, signal_type, direction, strike, volume, oi,
            vol_oi_ratio, expiry, option_type.
        """
        all_signals = []

        for ticker in tickers:
            logger.info("Scanning flow for %s ...", ticker)
            try:
                tk = yf.Ticker(ticker)
                expirations = tk.options

                if not expirations:
                    continue

                # Current price for direction inference
                hist = tk.history(period="1d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0

                # Scan nearest 2 expirations
                for expiry in expirations[:2]:
                    try:
                        chain = tk.option_chain(expiry)
                    except Exception:
                        continue

                    for option_type, opts in [("call", chain.calls), ("put", chain.puts)]:
                        if opts.empty:
                            continue

                        for _, row in opts.iterrows():
                            volume = int(row.get("volume", 0) or 0)
                            oi = int(row.get("openInterest", 0) or 0)
                            bid = float(row.get("bid", 0) or 0)
                            strike = float(row["strike"])

                            if volume < 50 or bid < 0.05:
                                continue

                            vol_oi = volume / oi if oi > 0 else float(volume)

                            if vol_oi < 2.0:
                                continue

                            # Infer direction
                            if option_type == "call":
                                direction = "bullish"
                            else:
                                direction = "bearish"

                            # Signal type
                            if oi == 0:
                                signal_type = "new_strike_activity"
                            elif vol_oi > 10:
                                signal_type = "aggressive_opening"
                            elif vol_oi > 5:
                                signal_type = "notable_flow"
                            else:
                                signal_type = "elevated_activity"

                            all_signals.append({
                                "ticker": ticker,
                                "signal_type": signal_type,
                                "direction": direction,
                                "strike": strike,
                                "volume": volume,
                                "oi": oi,
                                "vol_oi_ratio": round(vol_oi, 2),
                                "expiry": expiry,
                                "option_type": option_type,
                                "iv": round(float(row.get("impliedVolatility", 0) or 0), 4),
                            })

                time.sleep(0.3)  # Rate-limit between tickers

            except Exception as exc:
                logger.error("Flow signal analysis failed for %s: %s", ticker, exc)

        # Sort by vol/OI ratio descending
        all_signals.sort(key=lambda s: s["vol_oi_ratio"], reverse=True)
        return all_signals
