"""
Technical analysis scanner for individual tickers.

Computes standard indicators (RSI, MACD, Bollinger Bands, ATR, volume,
moving averages) plus weekly-specific signals (multi-day momentum, ADX,
trend strength, SMA slopes, 52-week context, weekly expected move) via
the ``ta`` library and identifies actionable setups suitable for weekly
options trades (5-day hold, Monday entry, Friday expiry).
"""

import logging
import math
import time

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from config import Config

logger = logging.getLogger(__name__)
config = Config()


class TechnicalScanner:
    """Compute technical indicators for one or many tickers and flag setups
    suitable for weekly options trades."""

    # ------------------------------------------------------------------ #
    #  Single-ticker scan
    # ------------------------------------------------------------------ #

    def scan_ticker(self, ticker: str, period: str = "3mo") -> dict:
        """Fetch price data for *ticker* and compute technical indicators.

        Designed for weekly options analysis: includes daily indicators,
        multi-day momentum, trend strength, and weekly expected move
        alongside intraday execution signals for Monday-morning entry.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        period : str
            yfinance history period (default ``'3mo'`` to support 50-day
            SMA and momentum look-backs).

        Returns
        -------
        dict
            Indicator values keyed by name, or empty dict on failure.
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            data = yf_ticker.history(period=period)

            if data.empty or len(data) < 20:
                logger.warning(
                    "Insufficient data for %s (got %d rows, need >=20)",
                    ticker,
                    len(data),
                )
                return {}

            close = data["Close"]
            high = data["High"]
            low = data["Low"]
            volume = data["Volume"]

            # --- RSI (14) ---
            rsi_ind = RSIIndicator(close=close, window=14)
            rsi_series = rsi_ind.rsi()
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
            rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else rsi

            # --- MACD (12, 26, 9) ---
            macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            macd_line = macd_ind.macd()
            macd_signal = macd_ind.macd_signal()
            macd_hist = macd_ind.macd_diff()

            macd_val = float(macd_line.iloc[-1]) if not macd_line.empty else None
            macd_sig_val = float(macd_signal.iloc[-1]) if not macd_signal.empty else None
            macd_hist_val = float(macd_hist.iloc[-1]) if not macd_hist.empty else None
            macd_hist_prev = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else macd_hist_val

            # --- Bollinger Bands (20, 2) ---
            bb = BollingerBands(close=close, window=20, window_dev=2)
            bb_upper = float(bb.bollinger_hband().iloc[-1])
            bb_middle = float(bb.bollinger_mavg().iloc[-1])
            bb_lower = float(bb.bollinger_lband().iloc[-1])
            bb_width = float(bb.bollinger_wband().iloc[-1])
            bb_pct = float(bb.bollinger_pband().iloc[-1])

            # Bandwidth trend (contracting = squeeze)
            bw_series = bb.bollinger_wband()
            bw_sma = float(bw_series.rolling(10).mean().iloc[-1]) if len(bw_series) >= 10 else bb_width
            bb_squeeze = bb_width < bw_sma

            # --- ATR (14) ---
            atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
            atr = float(atr_ind.average_true_range().iloc[-1])

            # --- Volume ---
            current_vol = float(volume.iloc[-1])
            vol_sma_20 = float(volume.tail(20).mean())
            volume_ratio = current_vol / vol_sma_20 if vol_sma_20 > 0 else 0.0

            # --- SMAs ---
            sma_20_series = SMAIndicator(close=close, window=20).sma_indicator()
            sma_20 = float(sma_20_series.iloc[-1])
            sma_50_series = SMAIndicator(close=close, window=50).sma_indicator()
            sma_50 = float(sma_50_series.iloc[-1]) if not sma_50_series.isna().iloc[-1] else None

            current_price = float(close.iloc[-1])
            price_vs_sma20 = ((current_price / sma_20) - 1) * 100 if sma_20 else None
            price_vs_sma50 = ((current_price / sma_50) - 1) * 100 if sma_50 else None

            # --- Multi-day momentum returns (weekly-specific) ---
            return_1d = self._pct_return(close, 1)
            return_5d = self._pct_return(close, 5)
            return_10d = self._pct_return(close, 10)
            return_21d = self._pct_return(close, 21)

            # --- ADX (14) ---
            adx_val, plus_di, minus_di = self._compute_adx(high, low, close)

            # --- SMA slopes (trend persistence) ---
            sma20_slope = self._sma_slope(sma_20_series, periods=10, price=current_price)
            sma50_slope = self._sma_slope(sma_50_series, periods=10, price=current_price)

            # --- Trend strength classification ---
            trend_strength = self._classify_trend(
                adx=adx_val,
                price_vs_sma20=price_vs_sma20,
                sma20_slope=sma20_slope,
            )

            # --- Weekly expected move ---
            atr_pct = round((atr / current_price) * 100, 2)
            weekly_atr_pct = round(atr_pct * math.sqrt(5), 2)

            # --- 52-week context ---
            pct_from_52w_high, pct_from_52w_low = self._compute_52w_context(
                yf_ticker, current_price
            )

            # --- Execution signals (intraday, for Monday entry timing) ---
            intraday = self._scan_execution_signals(ticker, current_price)

            result = {
                "ticker": ticker,
                "price": round(current_price, 2),
                # Daily indicators (unchanged field names)
                "rsi": round(rsi, 2) if rsi is not None else None,
                "rsi_prev": round(rsi_prev, 2) if rsi_prev is not None else None,
                "macd": round(macd_val, 4) if macd_val is not None else None,
                "macd_signal": round(macd_sig_val, 4) if macd_sig_val is not None else None,
                "macd_histogram": round(macd_hist_val, 4) if macd_hist_val is not None else None,
                "macd_histogram_prev": round(macd_hist_prev, 4) if macd_hist_prev is not None else None,
                "bb_upper": round(bb_upper, 2),
                "bb_middle": round(bb_middle, 2),
                "bb_lower": round(bb_lower, 2),
                "bb_width": round(bb_width, 4),
                "bb_pct": round(bb_pct, 4),
                "bb_squeeze": bb_squeeze,
                "atr": round(atr, 2),
                "atr_pct": atr_pct,
                "volume": int(current_vol),
                "volume_sma20": int(vol_sma_20),
                "volume_ratio": round(volume_ratio, 2),
                "sma_20": round(sma_20, 2),
                "sma_50": round(sma_50, 2) if sma_50 is not None else None,
                "price_vs_sma20_pct": round(price_vs_sma20, 2) if price_vs_sma20 is not None else None,
                "price_vs_sma50_pct": round(price_vs_sma50, 2) if price_vs_sma50 is not None else None,
                # Weekly signals (new)
                "return_1d": return_1d,
                "return_5d": return_5d,
                "return_10d": return_10d,
                "return_21d": return_21d,
                "adx": round(adx_val, 2) if adx_val is not None else None,
                "plus_di": round(plus_di, 2) if plus_di is not None else None,
                "minus_di": round(minus_di, 2) if minus_di is not None else None,
                "trend_strength": trend_strength,
                "sma20_slope": round(sma20_slope, 4) if sma20_slope is not None else None,
                "sma50_slope": round(sma50_slope, 4) if sma50_slope is not None else None,
                "pct_from_52w_high": round(pct_from_52w_high, 2) if pct_from_52w_high is not None else None,
                "pct_from_52w_low": round(pct_from_52w_low, 2) if pct_from_52w_low is not None else None,
                "weekly_atr_pct": weekly_atr_pct,
            }
            result.update(intraday)
            return result

        except Exception as exc:
            logger.error("Technical scan failed for %s: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Helper: multi-day percentage return
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pct_return(close: pd.Series, days: int) -> float | None:
        """Return the percentage change over the last *days* trading days."""
        if len(close) <= days:
            return None
        current = float(close.iloc[-1])
        prior = float(close.iloc[-1 - days])
        if prior == 0:
            return None
        return round(((current / prior) - 1) * 100, 2)

    # ------------------------------------------------------------------ #
    #  Helper: ADX calculation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_adx(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> tuple[float | None, float | None, float | None]:
        """Compute ADX, +DI, and -DI using a 14-period window.

        Returns (adx, plus_di, minus_di); all None if insufficient data.
        """
        try:
            adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
            adx = adx_ind.adx()
            plus_di = adx_ind.adx_pos()
            minus_di = adx_ind.adx_neg()

            adx_val = float(adx.iloc[-1]) if not adx.isna().iloc[-1] else None
            plus_val = float(plus_di.iloc[-1]) if not plus_di.isna().iloc[-1] else None
            minus_val = float(minus_di.iloc[-1]) if not minus_di.isna().iloc[-1] else None

            return adx_val, plus_val, minus_val
        except Exception:
            return None, None, None

    # ------------------------------------------------------------------ #
    #  Helper: SMA slope (normalized % per day)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sma_slope(sma_series: pd.Series, periods: int = 10, price: float = 1.0) -> float | None:
        """Linear-regression slope of *sma_series* over the last *periods*
        data points, normalized as percent-of-price per day.

        Returns None if the series has insufficient non-NaN data.
        """
        tail = sma_series.dropna().tail(periods)
        if len(tail) < max(periods, 5):
            return None
        y = tail.values.astype(float)
        x = np.arange(len(y), dtype=float)
        # slope via least-squares: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
        # Normalize as % of current price per day
        if price == 0:
            return None
        return (slope / price) * 100

    # ------------------------------------------------------------------ #
    #  Helper: trend strength classification
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_trend(
        adx: float | None,
        price_vs_sma20: float | None,
        sma20_slope: float | None,
    ) -> str:
        """Classify the current trend into one of five buckets.

        Logic:
        - ADX > 25 means a meaningful trend is present.
        - Direction determined by price vs SMA20 and SMA20 slope.
        - "strong" variants require both ADX > 25 and agreement between
          price position and slope direction.
        """
        if adx is None or price_vs_sma20 is None:
            return "flat"

        trending = adx > 25
        above_sma = price_vs_sma20 > 0
        slope_up = (sma20_slope is not None and sma20_slope > 0)
        slope_down = (sma20_slope is not None and sma20_slope < 0)

        if trending and above_sma and slope_up:
            return "strong_up"
        if above_sma or slope_up:
            return "up"
        if trending and (not above_sma) and slope_down:
            return "strong_down"
        if (not above_sma) or slope_down:
            return "down"
        return "flat"

    # ------------------------------------------------------------------ #
    #  Helper: 52-week context
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_52w_context(
        yf_ticker, current_price: float
    ) -> tuple[float | None, float | None]:
        """Fetch 1-year data and compute distance from 52-week high/low.

        Returns (pct_from_52w_high, pct_from_52w_low) where:
        - pct_from_52w_high is negative (how far below the high)
        - pct_from_52w_low is positive (how far above the low)
        """
        try:
            yearly = yf_ticker.history(period="1y")
            if yearly.empty:
                return None, None
            high_52w = float(yearly["High"].max())
            low_52w = float(yearly["Low"].min())

            pct_from_high = ((current_price / high_52w) - 1) * 100 if high_52w > 0 else None
            pct_from_low = ((current_price / low_52w) - 1) * 100 if low_52w > 0 else None

            return pct_from_high, pct_from_low
        except Exception:
            return None, None

    # ------------------------------------------------------------------ #
    #  Execution signals (intraday, for Monday entry timing)
    # ------------------------------------------------------------------ #

    def _scan_execution_signals(self, ticker: str, current_price: float) -> dict:
        """Compute intraday execution signals for weekly options entry timing.

        Fetches 5-minute bars for the last 5 days to calculate:
        - Pre-market gap from previous close
        - Previous day's high, low, and VWAP (key support/resistance)
        - Intraday ATR (actual intraday movement vs daily ATR)

        These help optimize ENTRY TIMING on Monday morning.

        Returns neutral defaults if intraday data is unavailable.
        """
        defaults = {
            "gap_pct": 0.0,
            "prev_day_high": None,
            "prev_day_low": None,
            "prev_day_vwap": None,
            "price_vs_prev_vwap": None,
            "intraday_atr_pct": None,
        }

        try:
            data = yf.Ticker(ticker).history(period="5d", interval="5m")
            if data.empty or len(data) < 20:
                return defaults

            # Group by trading day
            data.index = pd.to_datetime(data.index)
            data["date"] = data.index.date
            days = sorted(data["date"].unique())

            if len(days) < 2:
                return defaults

            # Previous trading day's bars
            prev_day = data[data["date"] == days[-2]]
            today = data[data["date"] == days[-1]]

            if prev_day.empty:
                return defaults

            prev_close = float(prev_day["Close"].iloc[-1])
            prev_high = float(prev_day["High"].max())
            prev_low = float(prev_day["Low"].min())

            # VWAP = sum(price * volume) / sum(volume) for previous day
            typical_price = (prev_day["High"] + prev_day["Low"] + prev_day["Close"]) / 3
            prev_volume = prev_day["Volume"]
            total_vol = prev_volume.sum()
            prev_vwap = float((typical_price * prev_volume).sum() / total_vol) if total_vol > 0 else prev_close

            # Gap: current price vs previous close
            gap_pct = round(((current_price - prev_close) / prev_close) * 100, 2)

            # Price position relative to previous day's VWAP
            price_vs_vwap = round(((current_price - prev_vwap) / prev_vwap) * 100, 2)

            # Intraday ATR: average of each day's (high - low) / close
            intraday_ranges = []
            for day in days[-5:]:
                day_data = data[data["date"] == day]
                if day_data.empty:
                    continue
                day_high = day_data["High"].max()
                day_low = day_data["Low"].min()
                day_close = day_data["Close"].iloc[-1]
                if day_close > 0:
                    intraday_ranges.append((day_high - day_low) / day_close * 100)

            intraday_atr_pct = round(np.mean(intraday_ranges), 2) if intraday_ranges else None

            return {
                "gap_pct": gap_pct,
                "prev_day_high": round(prev_high, 2),
                "prev_day_low": round(prev_low, 2),
                "prev_day_vwap": round(prev_vwap, 2),
                "price_vs_prev_vwap": price_vs_vwap,
                "intraday_atr_pct": intraday_atr_pct,
            }

        except Exception as exc:
            logger.debug("Execution signal scan failed for %s: %s", ticker, exc)
            return defaults

    # ------------------------------------------------------------------ #
    #  Batch scan
    # ------------------------------------------------------------------ #

    def scan_batch(self, tickers: list, period: str = "3mo") -> pd.DataFrame:
        """Scan multiple tickers and return a combined DataFrame.

        Designed for weekly options screening across a watchlist.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols.
        period : str
            yfinance period string (default ``'3mo'``).

        Returns
        -------
        pd.DataFrame
            One row per ticker with all indicator columns.
        """
        results = []

        for ticker in tickers:
            logger.info("Scanning %s ...", ticker)
            row = self.scan_ticker(ticker, period=period)
            if row:
                results.append(row)
            time.sleep(0.2)  # Gentle rate-limit

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df

    # ------------------------------------------------------------------ #
    #  Setup identification
    # ------------------------------------------------------------------ #

    def identify_setups(self, df: pd.DataFrame) -> list[dict]:
        """Screen scan results for actionable technical setups.

        Evaluates weekly options setups: mean-reversion, trend
        continuation, squeeze breakouts, volume breakouts, and MACD
        crossovers.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`scan_batch` (indexed by ticker).

        Returns
        -------
        list[dict]
            Each dict has: ticker, setup_type, direction, strength (1-10),
            description.
        """
        setups = []

        for _, row in df.iterrows():
            try:
                ticker = row.get("ticker", "UNKNOWN")
                rsi = row.get("rsi")
                rsi_prev = row.get("rsi_prev")
                macd_hist = row.get("macd_histogram")
                macd_hist_prev = row.get("macd_histogram_prev")
                bb_squeeze = row.get("bb_squeeze")
                bb_width = row.get("bb_width")
                bb_pct = row.get("bb_pct")
                volume_ratio = row.get("volume_ratio")

                # -- Oversold bounce --
                if rsi is not None and rsi_prev is not None:
                    if rsi_prev < 30 and rsi > rsi_prev:
                        strength = min(10, int((30 - rsi_prev) / 3) + 4)
                        setups.append({
                            "ticker": ticker,
                            "setup_type": "oversold_bounce",
                            "direction": "bullish",
                            "strength": strength,
                            "description": f"RSI turning up from {rsi_prev:.1f} to {rsi:.1f}",
                        })

                # -- Overbought reversal --
                if rsi is not None and rsi_prev is not None:
                    if rsi_prev > 70 and rsi < rsi_prev:
                        strength = min(10, int((rsi_prev - 70) / 3) + 4)
                        setups.append({
                            "ticker": ticker,
                            "setup_type": "overbought_reversal",
                            "direction": "bearish",
                            "strength": strength,
                            "description": f"RSI turning down from {rsi_prev:.1f} to {rsi:.1f}",
                        })

                # -- Bollinger squeeze --
                if bb_squeeze and bb_width is not None:
                    # Direction hint from price position within bands
                    if bb_pct is not None and bb_pct > 0.5:
                        direction = "bullish"
                    elif bb_pct is not None and bb_pct < 0.5:
                        direction = "bearish"
                    else:
                        direction = "bullish"  # slight bullish bias on squeeze
                    setups.append({
                        "ticker": ticker,
                        "setup_type": "bollinger_squeeze",
                        "direction": direction,
                        "strength": 6,
                        "description": f"Bandwidth contracting ({bb_width:.4f}), breakout imminent",
                    })

                # -- Volume breakout --
                if volume_ratio is not None and volume_ratio > 2.0:
                    price_vs_sma = row.get("price_vs_sma20_pct", 0) or 0
                    direction = "bullish" if price_vs_sma > 0 else "bearish"
                    strength = min(10, int(volume_ratio) + 3)
                    setups.append({
                        "ticker": ticker,
                        "setup_type": "volume_breakout",
                        "direction": direction,
                        "strength": strength,
                        "description": f"Volume {volume_ratio:.1f}x average",
                    })

                # -- MACD crossover (bullish) --
                if macd_hist is not None and macd_hist_prev is not None:
                    if macd_hist_prev < 0 and macd_hist > 0:
                        setups.append({
                            "ticker": ticker,
                            "setup_type": "macd_crossover",
                            "direction": "bullish",
                            "strength": 5,
                            "description": "MACD histogram crossed above zero",
                        })
                    elif macd_hist_prev > 0 and macd_hist < 0:
                        setups.append({
                            "ticker": ticker,
                            "setup_type": "macd_crossover",
                            "direction": "bearish",
                            "strength": 5,
                            "description": "MACD histogram crossed below zero",
                        })

            except Exception as exc:
                logger.error("Setup identification failed for %s: %s", ticker, exc)

        # Sort by strength descending
        setups.sort(key=lambda s: s["strength"], reverse=True)
        return setups
