"""
Market-wide scanner for macro conditions, sector rotation, breadth, and yields.

Provides the broad market context that informs zero-DTE trade selection:
VIX regime, sector momentum, market breadth, and Treasury yield curves.
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

from config import Config

logger = logging.getLogger(__name__)
config = Config()

# Sector ETFs used for rotation analysis
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

VIX_REGIMES = {
    "low": (0, 15),
    "normal": (15, 20),
    "elevated": (20, 25),
    "high": (25, 30),
    "extreme": (30, float("inf")),
}

RECURRING_HIGH_IMPACT = {
    "FOMC": "Federal Reserve interest rate decision — dominates all markets",
    "CPI": "Consumer Price Index — major inflation indicator",
    "PPI": "Producer Price Index — inflation leading indicator",
    "NFP": "Non-Farm Payrolls (first Friday of month) — major jobs data",
    "PCE": "Personal Consumption Expenditures — Fed's preferred inflation gauge",
    "GDP": "Gross Domestic Product — quarterly economic growth",
    "JOLTS": "Job Openings — labor market health",
    "Retail Sales": "Consumer spending indicator",
    "Jobless Claims": "Weekly unemployment claims (every Thursday)",
}


class MarketScanner:
    """Scans broad market conditions to establish the macro backdrop for trading."""

    # ------------------------------------------------------------------ #
    #  VIX
    # ------------------------------------------------------------------ #

    def get_vix_data(self) -> dict:
        """Fetch VIX current level and 20-day trend.

        Returns
        -------
        dict
            Keys: current, previous_close, change, change_pct, trend_20d (DataFrame),
            sma_20, trend_direction ('rising', 'falling', 'flat').
        """
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="2mo")

            if hist.empty:
                logger.warning("VIX history returned empty")
                return {}

            current = float(hist["Close"].iloc[-1])
            previous_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
            change = current - previous_close
            change_pct = (change / previous_close * 100) if previous_close else 0.0

            trend_20d = hist["Close"].tail(20)
            sma_20 = float(trend_20d.mean())

            # Simple linear regression slope on last 20 days
            if len(trend_20d) >= 5:
                y = trend_20d.values
                x = np.arange(len(y))
                slope = float(np.polyfit(x, y, 1)[0])
                if slope > 0.1:
                    trend_direction = "rising"
                elif slope < -0.1:
                    trend_direction = "falling"
                else:
                    trend_direction = "flat"
            else:
                trend_direction = "unknown"

            return {
                "current": current,
                "previous_close": previous_close,
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "trend_20d": trend_20d,
                "sma_20": round(sma_20, 2),
                "trend_direction": trend_direction,
            }

        except Exception as exc:
            logger.error("Failed to fetch VIX data: %s", exc)
            return {}

    def get_vix_regime(self) -> dict:
        """Classify VIX into a named regime.

        Returns
        -------
        dict
            Keys: regime (str), vix_level (float), description (str).
        """
        try:
            vix_data = self.get_vix_data()
            if not vix_data:
                return {"regime": "unknown", "vix_level": None, "description": "Data unavailable"}

            level = vix_data["current"]
            regime = "unknown"
            for name, (low, high) in VIX_REGIMES.items():
                if low <= level < high:
                    regime = name
                    break

            descriptions = {
                "low": "Low volatility - favor selling premium, smaller moves expected",
                "normal": "Normal volatility - standard setups work well",
                "elevated": "Elevated volatility - wider strikes, larger premiums available",
                "high": "High volatility - large swings likely, reduce position size",
                "extreme": "Extreme volatility - crisis conditions, be very cautious",
                "unknown": "Unable to classify VIX regime",
            }

            return {
                "regime": regime,
                "vix_level": level,
                "description": descriptions.get(regime, ""),
                "trend_direction": vix_data.get("trend_direction", "unknown"),
            }

        except Exception as exc:
            logger.error("Failed to classify VIX regime: %s", exc)
            return {"regime": "unknown", "vix_level": None, "description": str(exc)}

    # ------------------------------------------------------------------ #
    #  Sector Performance
    # ------------------------------------------------------------------ #

    def get_sector_performance(self) -> pd.DataFrame:
        """Fetch 1-day, 5-day, and 1-month returns for sector ETFs.

        Returns
        -------
        pd.DataFrame
            Indexed by ticker with columns: sector, return_1d, return_5d, return_1mo.
        """
        records = []

        for ticker, sector_name in SECTOR_ETFS.items():
            try:
                etf = yf.Ticker(ticker)
                hist = etf.history(period="2mo")

                if hist.empty or len(hist) < 2:
                    logger.warning("No history for sector ETF %s", ticker)
                    continue

                close = hist["Close"]
                current = float(close.iloc[-1])

                ret_1d = ((current / float(close.iloc[-2])) - 1) * 100 if len(close) >= 2 else 0.0
                ret_5d = ((current / float(close.iloc[-6])) - 1) * 100 if len(close) >= 6 else 0.0
                ret_1mo = ((current / float(close.iloc[-22])) - 1) * 100 if len(close) >= 22 else 0.0

                records.append({
                    "ticker": ticker,
                    "sector": sector_name,
                    "price": round(current, 2),
                    "return_1d": round(ret_1d, 2),
                    "return_5d": round(ret_5d, 2),
                    "return_1mo": round(ret_1mo, 2),
                })

                time.sleep(0.15)  # Rate-limit yfinance calls

            except Exception as exc:
                logger.error("Failed to fetch sector ETF %s: %s", ticker, exc)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("ticker")
        return df

    # ------------------------------------------------------------------ #
    #  Market Breadth
    # ------------------------------------------------------------------ #

    def get_market_breadth(self) -> dict:
        """Compare SPY vs RSP (equal-weight S&P 500) for breadth signal.

        When SPY outperforms RSP, leadership is narrow (mega-cap driven).
        When RSP outperforms SPY, breadth is broad (healthier rally).

        Returns
        -------
        dict
            Keys: spy_return_1d, rsp_return_1d, breadth_spread, breadth_signal,
            spy_return_5d, rsp_return_5d, breadth_spread_5d.
        """
        try:
            spy = yf.Ticker("SPY").history(period="2mo")
            rsp = yf.Ticker("RSP").history(period="2mo")

            if spy.empty or rsp.empty:
                logger.warning("SPY or RSP history returned empty")
                return {}

            def calc_return(series, periods):
                if len(series) < periods + 1:
                    return 0.0
                return ((float(series.iloc[-1]) / float(series.iloc[-1 - periods])) - 1) * 100

            spy_1d = calc_return(spy["Close"], 1)
            rsp_1d = calc_return(rsp["Close"], 1)
            spy_5d = calc_return(spy["Close"], 5)
            rsp_5d = calc_return(rsp["Close"], 5)

            spread_1d = rsp_1d - spy_1d  # Positive = broad breadth
            spread_5d = rsp_5d - spy_5d

            if spread_5d > 0.5:
                signal = "broad_rally"
            elif spread_5d < -0.5:
                signal = "narrow_leadership"
            else:
                signal = "neutral"

            return {
                "spy_return_1d": round(spy_1d, 2),
                "rsp_return_1d": round(rsp_1d, 2),
                "breadth_spread_1d": round(spread_1d, 2),
                "spy_return_5d": round(spy_5d, 2),
                "rsp_return_5d": round(rsp_5d, 2),
                "breadth_spread_5d": round(spread_5d, 2),
                "breadth_signal": signal,
            }

        except Exception as exc:
            logger.error("Failed to compute market breadth: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Treasury Yields
    # ------------------------------------------------------------------ #

    def get_treasury_yields(self) -> dict:
        """Fetch 2-year and 10-year Treasury yields and the 2s10s spread from FRED.

        Requires FRED_API_KEY to be configured.

        Returns
        -------
        dict
            Keys: yield_2y, yield_10y, spread_2s10s, curve_signal.
        """
        if not config.has_fred():
            logger.info("FRED API key not configured; skipping treasury yields")
            return {}

        try:
            fred = Fred(api_key=config.fred_api_key)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            gs2 = fred.get_series("GS2", start_date, end_date)
            gs10 = fred.get_series("GS10", start_date, end_date)

            if gs2.empty or gs10.empty:
                logger.warning("FRED returned empty yield data")
                return {}

            yield_2y = float(gs2.dropna().iloc[-1])
            yield_10y = float(gs10.dropna().iloc[-1])
            spread = yield_10y - yield_2y

            if spread < 0:
                curve_signal = "inverted"
            elif spread < 0.25:
                curve_signal = "flat"
            elif spread < 1.0:
                curve_signal = "normal"
            else:
                curve_signal = "steep"

            return {
                "yield_2y": round(yield_2y, 3),
                "yield_10y": round(yield_10y, 3),
                "spread_2s10s": round(spread, 3),
                "curve_signal": curve_signal,
            }

        except Exception as exc:
            logger.error("Failed to fetch treasury yields from FRED: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Economic Calendar
    # ------------------------------------------------------------------ #

    def get_economic_calendar(self) -> list[dict]:
        """Check for major economic events this week that could dominate 0DTE outcomes.

        Tries the Finnhub economic calendar endpoint first. If that fails or
        returns empty, falls back to matching a hardcoded list of recurring
        high-impact events.

        Returns
        -------
        list[dict]
            Each dict has keys: event, date, impact, description.
        """
        today = datetime.now()
        # Monday of this week through Friday
        monday = today - timedelta(days=today.weekday())
        friday = monday + timedelta(days=4)
        from_date = monday.strftime("%Y-%m-%d")
        to_date = friday.strftime("%Y-%m-%d")

        events: list[dict] = []

        # --- Try Finnhub API first ---
        if config.has_finnhub():
            try:
                url = "https://finnhub.io/api/v1/calendar/economic"
                params = {
                    "from": from_date,
                    "to": to_date,
                    "token": config.finnhub_api_key,
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                raw_events = data.get("economicCalendar", [])
                if raw_events:
                    for item in raw_events:
                        event_name = item.get("event", "")
                        impact = item.get("impact", "low")
                        # Finnhub uses numeric impact: 3=high, 2=medium, 1=low
                        if isinstance(impact, (int, float)):
                            if impact >= 3:
                                impact = "high"
                            elif impact >= 2:
                                impact = "medium"
                            else:
                                impact = "low"

                        # Check if this matches a known high-impact event
                        matched_description = ""
                        for keyword, desc in RECURRING_HIGH_IMPACT.items():
                            if keyword.lower() in event_name.lower():
                                impact = "high"
                                matched_description = desc
                                break

                        events.append({
                            "event": event_name,
                            "date": item.get("date", from_date),
                            "impact": impact,
                            "description": matched_description or event_name,
                        })

                    logger.info(
                        "Fetched %d economic events from Finnhub (%s to %s)",
                        len(events), from_date, to_date,
                    )
                    return events

            except Exception as exc:
                logger.warning("Finnhub economic calendar request failed: %s", exc)

        # --- Fallback: hardcoded recurring high-impact events ---
        logger.info(
            "Using fallback recurring high-impact event list (Finnhub unavailable or empty)"
        )
        for keyword, description in RECURRING_HIGH_IMPACT.items():
            events.append({
                "event": keyword,
                "date": "",
                "impact": "high",
                "description": description,
            })

        return events

    # ------------------------------------------------------------------ #
    #  Credit Spread & Financial Conditions (FRED)
    # ------------------------------------------------------------------ #

    def get_credit_spread(self) -> dict:
        """Fetch ICE BofA High Yield OAS from FRED.

        Backtest finding: model accuracy is 57.1% when HY OAS is 3-4.5%
        (normal) vs 50.4% when < 3% (tight/complacent).

        Returns
        -------
        dict
            Keys: hy_oas, credit_state (tight/normal/wide/stressed).
        """
        if not config.has_fred():
            return {}

        try:
            fred = Fred(api_key=config.fred_api_key)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = fred.get_series("BAMLH0A0HYM2", start_date, end_date)
            if data is None or data.empty:
                return {}

            hy_oas = float(data.dropna().iloc[-1])
            if hy_oas < 3.0:
                state = "tight"
            elif hy_oas < 4.5:
                state = "normal"
            elif hy_oas < 6.0:
                state = "wide"
            else:
                state = "stressed"

            return {"hy_oas": round(hy_oas, 3), "credit_state": state}
        except Exception as exc:
            logger.error("Failed to fetch credit spread from FRED: %s", exc)
            return {}

    def get_financial_conditions(self) -> dict:
        """Fetch Chicago Fed National Financial Conditions Index from FRED.

        Backtest finding: model accuracy is 59.4% when NFCI is normal
        (0 to -0.5) vs 50.8% when loose (< -0.5).

        Returns
        -------
        dict
            Keys: nfci, conditions_state (loose/normal/tightening/tight).
        """
        if not config.has_fred():
            return {}

        try:
            fred = Fred(api_key=config.fred_api_key)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # NFCI is weekly
            data = fred.get_series("NFCI", start_date, end_date)
            if data is None or data.empty:
                return {}

            nfci = float(data.dropna().iloc[-1])
            if nfci < -0.5:
                state = "loose"
            elif nfci < 0.0:
                state = "normal"
            elif nfci < 0.5:
                state = "tightening"
            else:
                state = "tight"

            return {"nfci": round(nfci, 4), "conditions_state": state}
        except Exception as exc:
            logger.error("Failed to fetch NFCI from FRED: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Combined Summary
    # ------------------------------------------------------------------ #

    def get_market_summary(self) -> dict:
        """Aggregate all market scanner data into a single summary dict.

        Returns
        -------
        dict
            Keys: vix, vix_regime, sectors (DataFrame), breadth, yields, timestamp.
        """
        logger.info("Building full market summary ...")

        vix_data = self.get_vix_data()
        vix_regime = self.get_vix_regime()
        sectors = self.get_sector_performance()
        breadth = self.get_market_breadth()
        yields = self.get_treasury_yields()
        credit = self.get_credit_spread()
        conditions = self.get_financial_conditions()

        economic_events = self.get_economic_calendar()

        summary = {
            "vix": vix_data,
            "vix_regime": vix_regime,
            "sectors": sectors,
            "breadth": breadth,
            "yields": yields,
            "credit_spread": credit,
            "financial_conditions": conditions,
            "economic_events": economic_events,
            "has_high_impact_event": any(
                e["impact"] == "high" for e in economic_events
            ),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            "Market summary complete — VIX %.2f (%s), breadth: %s",
            vix_data.get("current", 0),
            vix_regime.get("regime", "unknown"),
            breadth.get("breadth_signal", "unknown"),
        )

        return summary
