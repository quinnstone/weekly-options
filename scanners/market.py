"""
Market-wide scanner for macro conditions, sector rotation, breadth, and yields.

Provides the broad market context that informs weekly options trade selection:
VIX regime, sector momentum, market breadth, Treasury yield curves,
holding-window event analysis, and regime persistence assessment.
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
    """Scans broad market conditions to establish the macro backdrop for weekly options trading."""

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

        # Add momentum rank (1 = best) and relative strength vs SPY
        if not df.empty and "return_5d" in df.columns:
            df["momentum_rank"] = df["return_5d"].rank(ascending=False, method="min").astype(int)
            spy_5d = df.loc["XLK", "return_5d"] if "XLK" in df.index else df["return_5d"].mean()
            # Use SPY if we have breadth data, otherwise use sector average
            try:
                spy = yf.Ticker("SPY").history(period="2mo")
                if len(spy) >= 6:
                    spy_ret_5d = ((float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[-6])) - 1) * 100
                else:
                    spy_ret_5d = df["return_5d"].mean()
            except Exception:
                spy_ret_5d = df["return_5d"].mean()
            df["relative_strength"] = round(df["return_5d"] - spy_ret_5d, 2)

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
        """Check for major economic events in the next 7 days.

        Covers the full holding window for weekly options (Mon-Fri + buffer).
        Tries the Finnhub economic calendar endpoint first. If that fails or
        returns empty, falls back to matching a hardcoded list of recurring
        high-impact events.

        Returns
        -------
        list[dict]
            Each dict has keys: event, date, impact, description.
        """
        today = datetime.now()
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")

        events: list[dict] = []

        # --- Try Finnhub API first ---
        if config.has_finnhub():
            try:
                from scanners.sentiment import _cached_finnhub_get
                url = "https://finnhub.io/api/v1/calendar/economic"
                params = {
                    "from": from_date,
                    "to": to_date,
                    "token": config.finnhub_api_key,
                }
                data = _cached_finnhub_get(url, params)
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
    #  Holding-window event analysis (weekly-specific)
    # ------------------------------------------------------------------ #

    def get_holding_window_events(self, entry_date=None, expiry_date=None) -> dict:
        """Check for high-impact events during the Mon-Fri holding window.

        Unlike the general calendar check, this specifically flags events
        that fall WITHIN our holding period, which changes the trade thesis.

        Parameters
        ----------
        entry_date : datetime or None
            Defaults to next Monday.
        expiry_date : datetime or None
            Defaults to the Friday of entry_date's week.

        Returns
        -------
        dict
            Keys: has_earnings_risk, has_fomc, has_cpi_nfp, high_impact_count,
            events, risk_level, event_days.
        """
        today = datetime.now()
        if entry_date is None:
            # Next Monday
            days_until_monday = (7 - today.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7 if today.weekday() != 0 else 0
            entry_date = today + timedelta(days=days_until_monday)
        if expiry_date is None:
            # Friday of entry_date's week
            days_to_friday = (4 - entry_date.weekday()) % 7
            expiry_date = entry_date + timedelta(days=days_to_friday)

        all_events = self.get_economic_calendar()

        # Filter to holding window
        entry_str = entry_date.strftime("%Y-%m-%d")
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        window_events = []
        for ev in all_events:
            ev_date = ev.get("date", "")
            if ev_date and entry_str <= ev_date <= expiry_str:
                window_events.append(ev)

        # Classify
        high_impact_keywords = {"FOMC", "CPI", "NFP", "PPI", "PCE", "GDP"}
        has_fomc = False
        has_cpi_nfp = False
        high_impact_count = 0
        event_days = set()

        for ev in window_events:
            name = ev.get("event", "").upper()
            if ev.get("impact") == "high":
                high_impact_count += 1
                if ev.get("date"):
                    try:
                        day_name = datetime.strptime(ev["date"], "%Y-%m-%d").strftime("%A")
                        event_days.add(day_name)
                    except ValueError:
                        pass
            if "FOMC" in name or "FEDERAL RESERVE" in name:
                has_fomc = True
            if any(k in name for k in ("CPI", "NFP", "NON-FARM", "NONFARM")):
                has_cpi_nfp = True

        if high_impact_count >= 2 or has_fomc:
            risk_level = "high"
        elif high_impact_count >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "has_earnings_risk": False,  # populated per-ticker in pipeline
            "has_fomc": has_fomc,
            "has_cpi_nfp": has_cpi_nfp,
            "high_impact_count": high_impact_count,
            "events": window_events,
            "risk_level": risk_level,
            "event_days": sorted(event_days),
        }

    # ------------------------------------------------------------------ #
    #  Regime persistence assessment (weekly-specific)
    # ------------------------------------------------------------------ #

    def assess_regime_persistence(self) -> dict:
        """Assess whether the current VIX regime is stable or transitioning.

        For weekly options, regime stability matters: a regime that's
        transitioning means our entry-day signals may not hold for 5 days.

        Returns
        -------
        dict
            Keys: is_stable, transition_direction, vix_5d_change_pct,
            regime_days, persistence_score.
        """
        defaults = {
            "is_stable": True,
            "transition_direction": None,
            "vix_5d_change_pct": 0.0,
            "regime_days": 0,
            "persistence_score": 0.5,
        }
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1mo")
            if hist.empty or len(hist) < 10:
                return defaults

            closes = hist["Close"]

            # Current regime
            current_vix = float(closes.iloc[-1])
            current_regime = "normal"
            for name, (lo, hi) in VIX_REGIMES.items():
                if lo <= current_vix < hi:
                    current_regime = name
                    break

            # Check regime for each of the last 10 days
            regime_streak = 0
            for i in range(1, min(11, len(closes))):
                level = float(closes.iloc[-i])
                regime = "normal"
                for name, (lo, hi) in VIX_REGIMES.items():
                    if lo <= level < hi:
                        regime = name
                        break
                if regime == current_regime:
                    regime_streak += 1
                else:
                    break

            # 5-day change
            vix_5d_ago = float(closes.iloc[-6]) if len(closes) >= 6 else current_vix
            vix_5d_change = ((current_vix / vix_5d_ago) - 1) * 100 if vix_5d_ago > 0 else 0

            # Stability assessment
            is_stable = regime_streak >= 5
            transition_direction = None
            if not is_stable:
                if vix_5d_change > 10:
                    transition_direction = "escalating"
                elif vix_5d_change < -10:
                    transition_direction = "calming"
                else:
                    transition_direction = "choppy"

            # Persistence score: 1.0 if stable 10+ days, scales down
            persistence_score = min(1.0, regime_streak / 10.0)

            return {
                "is_stable": is_stable,
                "transition_direction": transition_direction,
                "vix_5d_change_pct": round(vix_5d_change, 2),
                "regime_days": regime_streak,
                "persistence_score": round(persistence_score, 2),
            }

        except Exception as exc:
            logger.error("Failed to assess regime persistence: %s", exc)
            return defaults

    # ------------------------------------------------------------------ #
    #  CBOE Skew, Market P/C Ratio, VIX Term Structure
    # ------------------------------------------------------------------ #

    def get_cboe_skew(self) -> dict:
        """Fetch CBOE Skew Index — measures tail risk demand.

        Skew > 130 = heavy put hedging (institutional fear)
        Skew < 115 = complacency
        Divergence from VIX signals regime shifts.
        """
        try:
            skew = yf.Ticker("^SKEW")
            hist = skew.history(period="1mo")
            if hist.empty:
                return {"skew": None, "skew_state": "unknown"}

            current = float(hist["Close"].iloc[-1])
            sma_20 = float(hist["Close"].tail(20).mean()) if len(hist) >= 20 else current

            if current > 145:
                state = "extreme_fear"
            elif current > 130:
                state = "elevated"
            elif current > 115:
                state = "normal"
            else:
                state = "complacent"

            return {
                "skew": round(current, 2),
                "skew_sma20": round(sma_20, 2),
                "skew_state": state,
                "skew_vs_sma": round(current - sma_20, 2),
            }
        except Exception as exc:
            logger.warning("Failed to fetch CBOE Skew: %s", exc)
            return {"skew": None, "skew_state": "unknown"}

    def get_market_put_call_ratio(self) -> dict:
        """Compute market-wide put/call ratio from index options.

        Uses SPY options volume as a proxy for market-wide sentiment.
        P/C > 1.2 = heavy put buying (bearish)
        P/C < 0.7 = heavy call buying (bullish/complacent)
        """
        try:
            spy = yf.Ticker("SPY")
            exp_dates = spy.options
            if not exp_dates:
                return {"market_pc_ratio": None, "market_pc_state": "unknown"}

            # Use the nearest weekly expiry
            nearest = exp_dates[0]
            chain = spy.option_chain(nearest)

            total_call_vol = int(chain.calls["volume"].sum()) if "volume" in chain.calls.columns else 0
            total_put_vol = int(chain.puts["volume"].sum()) if "volume" in chain.puts.columns else 0

            if total_call_vol > 0:
                pc_ratio = total_put_vol / total_call_vol
            else:
                pc_ratio = 1.0

            if pc_ratio > 1.3:
                state = "bearish"
            elif pc_ratio > 1.0:
                state = "cautious"
            elif pc_ratio > 0.7:
                state = "neutral"
            else:
                state = "bullish"

            return {
                "market_pc_ratio": round(pc_ratio, 3),
                "market_pc_state": state,
                "spy_call_volume": total_call_vol,
                "spy_put_volume": total_put_vol,
            }
        except Exception as exc:
            logger.warning("Failed to compute market P/C ratio: %s", exc)
            return {"market_pc_ratio": None, "market_pc_state": "unknown"}

    def get_vix_term_structure(self) -> dict:
        """Assess VIX term structure (contango vs backwardation).

        VIX9D (9-day) vs VIX (30-day) vs VIX3M (3-month).
        Backwardation (short-term > long-term) = stress / fear.
        Contango (short-term < long-term) = complacency / normal.
        """
        try:
            vix_data = {}
            for sym, label in [("^VIX9D", "vix9d"), ("^VIX", "vix"), ("^VIX3M", "vix3m")]:
                try:
                    tk = yf.Ticker(sym)
                    hist = tk.history(period="5d")
                    if not hist.empty:
                        vix_data[label] = float(hist["Close"].iloc[-1])
                except Exception:
                    pass

            if "vix" not in vix_data:
                return {"term_structure": "unknown", "contango": None}

            vix = vix_data["vix"]
            vix9d = vix_data.get("vix9d")
            vix3m = vix_data.get("vix3m")

            # Near-term vs VIX
            if vix9d and vix9d > vix * 1.05:
                near_term = "backwardation"
            elif vix9d and vix9d < vix * 0.95:
                near_term = "contango"
            else:
                near_term = "flat"

            # VIX vs 3-month
            if vix3m and vix > vix3m * 1.05:
                far_term = "backwardation"
            elif vix3m and vix < vix3m * 0.95:
                far_term = "contango"
            else:
                far_term = "flat"

            # Overall structure assessment
            if near_term == "backwardation" or far_term == "backwardation":
                structure = "backwardation"  # stress signal
            elif near_term == "contango" and far_term == "contango":
                structure = "contango"  # normal/complacent
            else:
                structure = "mixed"

            result = {
                "term_structure": structure,
                "near_term": near_term,
                "far_term": far_term,
            }
            result.update(vix_data)

            return result
        except Exception as exc:
            logger.warning("Failed to fetch VIX term structure: %s", exc)
            return {"term_structure": "unknown"}

    # ------------------------------------------------------------------ #
    #  CFTC Commitment of Traders (COT) — speculator positioning
    # ------------------------------------------------------------------ #

    def get_cot_positioning(self) -> dict:
        """Fetch CFTC Commitment of Traders data for S&P 500 futures.

        Large speculator net positioning in equity index futures is a
        well-studied weekly contrarian signal:
        - Extreme net-long → market vulnerable to pullbacks (bearish lean)
        - Extreme net-short → contrarian bullish
        - Neutral → no signal

        Data source: CFTC public reports (free, no API key needed).
        Released every Friday at 3:30 PM ET with Tuesday's data.

        Returns
        -------
        dict
            net_position, signal (extreme_long/long/neutral/short/extreme_short),
            percentile (0-100 vs 52-week range).
        """
        try:
            # CFTC Traders in Financial Futures report — S&P 500 (contract code 13874A)
            # We use the Quandl-style CFTC API endpoint (free, no key)
            url = (
                "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
                "?$where=cftc_contract_market_code='13874A'"
                "&$order=report_date_as_yyyy_mm_dd DESC"
                "&$limit=52"
            )
            resp = requests.get(url, timeout=15)

            if resp.status_code != 200:
                logger.warning("CFTC COT request failed: %d", resp.status_code)
                return {"signal": "unavailable"}

            data = resp.json()
            if not data:
                return {"signal": "unavailable"}

            # Parse net speculator positioning (non-commercial long - short)
            net_positions = []
            for row in data:
                try:
                    long_pos = int(row.get("noncomm_positions_long_all", 0))
                    short_pos = int(row.get("noncomm_positions_short_all", 0))
                    net_positions.append(long_pos - short_pos)
                except (ValueError, TypeError):
                    continue

            if len(net_positions) < 4:
                return {"signal": "insufficient_data"}

            current_net = net_positions[0]  # most recent week

            # Percentile rank vs 52-week history
            sorted_pos = sorted(net_positions)
            rank = sorted_pos.index(current_net) if current_net in sorted_pos else len(sorted_pos) // 2
            percentile = round(rank / max(len(sorted_pos) - 1, 1) * 100, 1)

            # Classify signal
            if percentile >= 90:
                signal = "extreme_long"   # contrarian bearish
            elif percentile >= 70:
                signal = "long"
            elif percentile <= 10:
                signal = "extreme_short"  # contrarian bullish
            elif percentile <= 30:
                signal = "short"
            else:
                signal = "neutral"

            # Trend: is positioning building or unwinding?
            if len(net_positions) >= 4:
                recent_avg = sum(net_positions[:4]) / 4
                prior_avg = sum(net_positions[4:8]) / max(len(net_positions[4:8]), 1) if len(net_positions) > 4 else recent_avg
                trend = "building" if abs(recent_avg) > abs(prior_avg) * 1.1 else "unwinding" if abs(recent_avg) < abs(prior_avg) * 0.9 else "stable"
            else:
                trend = "unknown"

            return {
                "net_position": current_net,
                "percentile": percentile,
                "signal": signal,
                "trend": trend,
                "weeks_of_data": len(net_positions),
            }

        except Exception as exc:
            logger.warning("CFTC COT fetch failed: %s", exc)
            return {"signal": "unavailable"}

    # ------------------------------------------------------------------ #
    #  Finnhub Macro Surprise Index (beat/miss tracker)
    # ------------------------------------------------------------------ #

    def get_macro_surprise(self) -> dict:
        """Compute a rolling macro surprise score from recent economic releases.

        Uses Finnhub's economic calendar to track whether recent US macro
        data (NFP, CPI, GDP, etc.) is beating or missing consensus estimates.

        A positive surprise score means the economy is outperforming
        expectations → supports momentum and low-VIX regimes.
        A negative score means macro data is disappointing → risk-off
        pressure, VIX likely to rise, momentum may not persist over 5 days.

        Returns
        -------
        dict
            surprise_score (-1 to +1), recent_beats, recent_misses,
            signal (beating/inline/missing).
        """
        if not config.has_finnhub():
            return {"signal": "unavailable", "reason": "no finnhub key"}

        try:
            # Look back 30 days for recent releases
            from scanners.sentiment import _cached_finnhub_get
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            data = _cached_finnhub_get(
                "https://finnhub.io/api/v1/calendar/economic",
                {
                    "from": start_date,
                    "to": end_date,
                    "token": config.finnhub_api_key,
                },
            )
            events = data.get("economicCalendar", [])

            if not events:
                return {"signal": "insufficient_data"}

            # Filter for US events with actual and estimate values
            # Focus on high-impact indicators
            high_impact_keywords = {
                "nonfarm", "payroll", "cpi", "ppi", "gdp", "retail sales",
                "pce", "unemployment", "jobless", "ism", "consumer confidence",
                "housing starts", "industrial production", "durable goods",
            }

            beats = 0
            misses = 0
            total_scored = 0

            for event in events:
                if event.get("country") != "US":
                    continue

                actual = event.get("actual")
                estimate = event.get("estimate")

                if actual is None or estimate is None:
                    continue

                # Check if this is a high-impact event
                event_name = (event.get("event", "") or "").lower()
                is_high_impact = any(kw in event_name for kw in high_impact_keywords)
                weight = 2.0 if is_high_impact else 1.0

                try:
                    actual_val = float(actual)
                    estimate_val = float(estimate)
                except (ValueError, TypeError):
                    continue

                if actual_val > estimate_val:
                    beats += weight
                elif actual_val < estimate_val:
                    misses += weight
                total_scored += weight

            if total_scored == 0:
                return {"signal": "insufficient_data"}

            # Surprise score: -1 (all missing) to +1 (all beating)
            surprise_score = (beats - misses) / total_scored

            # Classify
            if surprise_score > 0.3:
                signal = "beating"      # economy outperforming → momentum-friendly
            elif surprise_score < -0.3:
                signal = "missing"      # economy disappointing → risk-off pressure
            else:
                signal = "inline"

            return {
                "surprise_score": round(surprise_score, 3),
                "beats": round(beats, 1),
                "misses": round(misses, 1),
                "total_events": round(total_scored, 1),
                "signal": signal,
            }

        except Exception as exc:
            logger.warning("Macro surprise calculation failed: %s", exc)
            return {"signal": "unavailable"}

    # ------------------------------------------------------------------ #
    #  Cross-Asset Leading Indicators
    # ------------------------------------------------------------------ #

    def get_cross_asset_signals(self) -> dict:
        """Fetch leading indicators from bonds, dollar, and oil.

        These assets often lead equity moves by 1-2 days:
        - TLT (20yr bonds): inverse to equities, rate-sensitive
        - UUP (US dollar): strong dollar hurts exporters, emerging markets
        - USO (crude oil): energy sector proxy, inflation signal

        Returns
        -------
        dict
            Per-asset 5d return, direction, and composite equity signal.
        """
        assets = {
            "TLT": {"name": "bonds_20y", "equity_inverse": True},
            "UUP": {"name": "us_dollar", "equity_inverse": True},
            "USO": {"name": "crude_oil", "equity_inverse": False},
        }
        signals = {}
        equity_headwinds = 0
        equity_tailwinds = 0

        for ticker, meta in assets.items():
            try:
                data = yf.Ticker(ticker).history(period="10d")
                if data.empty or len(data) < 5:
                    signals[meta["name"]] = {"return_5d": None, "signal": "no_data"}
                    continue

                ret_5d = (data["Close"].iloc[-1] / data["Close"].iloc[-5] - 1) * 100
                ret_1d = (data["Close"].iloc[-1] / data["Close"].iloc[-2] - 1) * 100

                # Determine equity impact
                if meta["equity_inverse"]:
                    # TLT/UUP up = equity headwind, down = tailwind
                    if ret_5d > 1.0:
                        equity_headwinds += 1
                    elif ret_5d < -1.0:
                        equity_tailwinds += 1
                else:
                    # USO up = mixed (energy bullish, inflation concern)
                    if ret_5d > 3.0:
                        equity_headwinds += 1  # Oil spike = inflation fear
                    elif ret_5d < -3.0:
                        equity_headwinds += 1  # Oil crash = demand fear

                signals[meta["name"]] = {
                    "return_5d": round(ret_5d, 2),
                    "return_1d": round(ret_1d, 2),
                    "signal": "headwind" if (meta["equity_inverse"] and ret_5d > 1.0)
                             else "tailwind" if (meta["equity_inverse"] and ret_5d < -1.0)
                             else "neutral",
                }
            except Exception as exc:
                logger.debug("Cross-asset fetch failed for %s: %s", ticker, exc)
                signals[meta["name"]] = {"return_5d": None, "signal": "no_data"}

        # Composite signal
        if equity_headwinds >= 2:
            composite = "risk_off"
        elif equity_tailwinds >= 2:
            composite = "risk_on"
        else:
            composite = "mixed"

        signals["composite"] = composite
        signals["headwinds"] = equity_headwinds
        signals["tailwinds"] = equity_tailwinds

        logger.info(
            "Cross-asset signals: %s (headwinds=%d, tailwinds=%d)",
            composite, equity_headwinds, equity_tailwinds,
        )
        return signals

    # ------------------------------------------------------------------ #
    #  Sector News
    # ------------------------------------------------------------------ #

    # Keyword map: sector name → terms to match in headlines
    SECTOR_NEWS_KEYWORDS = {
        "Technology": ["tech", "software", "semiconductor", "chip", "ai ", "artificial intelligence",
                       "cloud", "cyber", "saas", "apple", "microsoft", "google", "nvidia", "meta"],
        "Financials": ["bank", "banking", "financial", "lending", "credit card", "mortgage",
                       "interest rate", "fed ", "federal reserve", "insurance", "fintech"],
        "Energy": ["oil", "crude", "natural gas", "energy", "opec", "drilling", "renewable",
                   "solar", "wind power", "pipeline", "refin"],
        "Health Care": ["pharma", "biotech", "fda", "drug", "clinical trial", "health",
                        "hospital", "medical", "vaccine", "therapeut"],
        "Industrials": ["industrial", "manufacturing", "defense", "aerospace", "transport",
                        "infrastructure", "construction", "supply chain", "logistics"],
        "Consumer Discretionary": ["retail", "consumer", "auto", "ev ", "electric vehicle",
                                   "housing", "travel", "leisure", "restaurant", "apparel"],
        "Consumer Staples": ["grocery", "food", "beverage", "tobacco", "household",
                             "personal care", "staple"],
        "Communication Services": ["media", "streaming", "social media", "telecom",
                                   "advertising", "content", "broadcast"],
        "Materials": ["mining", "steel", "chemical", "fertilizer", "commodity", "gold",
                      "copper", "lithium", "rare earth"],
        "Utilities": ["utility", "power grid", "electric utility", "water utility", "nuclear"],
        "Real Estate": ["real estate", "reit", "property", "housing market", "commercial real estate"],
    }

    def get_sector_news(self) -> dict:
        """Fetch broad market news and categorize by sector.

        Uses Finnhub general news endpoint (single API call) and classifies
        each headline into relevant sectors using keyword matching.

        Returns
        -------
        dict
            Keyed by sector name. Each value is a list of dicts with
            headline, source, and VADER sentiment. Also includes a 'macro'
            key for non-sector-specific market-moving headlines.
        """
        try:
            finnhub_key = config.finnhub_api_key
            if not finnhub_key:
                return {}

            from scanners.sentiment import _cached_finnhub_get
            articles = _cached_finnhub_get(
                "https://finnhub.io/api/v1/news",
                {"category": "general", "token": finnhub_key},
            )

            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()

            # Noise filter
            skip_keywords = {"crypto", "bitcoin", "ethereum", "nft", "meme coin",
                             "celebrity", "kardashian"}

            sector_news = {sector: [] for sector in self.SECTOR_NEWS_KEYWORDS}
            sector_news["macro"] = []  # Non-sector-specific macro headlines

            for article in articles[:80]:  # Cap to avoid excessive processing
                headline = (article.get("headline") or "").strip()
                if not headline:
                    continue
                hl_lower = headline.lower()
                if any(kw in hl_lower for kw in skip_keywords):
                    continue

                source = article.get("source", "")
                sentiment = vader.polarity_scores(headline)["compound"]

                entry = {
                    "headline": headline,
                    "source": source,
                    "sentiment": round(sentiment, 3),
                }

                # Classify into sectors (a headline can match multiple sectors)
                matched_sectors = []
                for sector, keywords in self.SECTOR_NEWS_KEYWORDS.items():
                    if any(kw in hl_lower for kw in keywords):
                        matched_sectors.append(sector)
                        if len(sector_news[sector]) < 5:  # Cap per sector
                            sector_news[sector].append(entry)

                # If no sector matched, check for macro relevance
                macro_keywords = ["tariff", "trade war", "sanctions", "geopolitical",
                                  "fed ", "fomc", "rate cut", "rate hike", "inflation",
                                  "recession", "gdp", "jobs", "unemployment", "treasury",
                                  "fiscal", "stimulus", "shutdown", "debt ceiling",
                                  "war", "conflict", "ceasefire", "election"]
                if not matched_sectors and any(kw in hl_lower for kw in macro_keywords):
                    if len(sector_news["macro"]) < 8:
                        sector_news["macro"].append(entry)

            # Remove empty sectors
            sector_news = {k: v for k, v in sector_news.items() if v}

            total = sum(len(v) for v in sector_news.values())
            logger.info("Sector news: %d headlines across %d sectors", total, len(sector_news))
            return sector_news

        except Exception as exc:
            logger.warning("Sector news fetch failed: %s", exc)
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
        holding_window = self.get_holding_window_events()
        regime_persistence = self.assess_regime_persistence()
        skew = self.get_cboe_skew()
        market_pc = self.get_market_put_call_ratio()
        vix_term = self.get_vix_term_structure()
        cot = self.get_cot_positioning()
        macro_surprise = self.get_macro_surprise()
        cross_asset = self.get_cross_asset_signals()
        sector_news = self.get_sector_news()

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
            "holding_window": holding_window,
            "regime_persistence": regime_persistence,
            "skew": skew,
            "market_put_call": market_pc,
            "vix_term_structure": vix_term,
            "cot_positioning": cot,
            "macro_surprise": macro_surprise,
            "cross_asset": cross_asset,
            "sector_news": sector_news,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            "Market summary complete — VIX %.2f (%s), breadth: %s",
            vix_data.get("current", 0),
            vix_regime.get("regime", "unknown"),
            breadth.get("breadth_signal", "unknown"),
        )

        return summary
