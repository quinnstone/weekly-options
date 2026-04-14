"""
Options-chain scanner for weekly options trade selection.

Analyzes options chains to surface put/call ratios, IV rank, max pain,
strike-level open interest, and optimal entry strikes for directional
weekly options plays (Monday entry, Friday expiry, 5-day hold).
"""

import logging
import time
from datetime import datetime, timedelta
from math import erf as _erf, sqrt as _msqrt, log as _mlog, exp as _mexp

import numpy as np
import pandas as pd
import yfinance as yf

from config import Config

logger = logging.getLogger(__name__)
config = Config()

# ------------------------------------------------------------------ #
#  Black-Scholes-Merton helpers (no scipy dependency)
# ------------------------------------------------------------------ #

_RISK_FREE_RATE = 0.045  # ~current short-term Treasury yield


def _norm_cdf(x):
    """Standard normal CDF via math.erf — no scipy needed."""
    return 0.5 * (1 + _erf(x / _msqrt(2)))


def _bsm_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes-Merton European option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    sqrt_T = _msqrt(T)
    d1 = (_mlog(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return S * _norm_cdf(d1) - K * _mexp(-r * T) * _norm_cdf(d2)
    return K * _mexp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _solve_iv(market_price, S, K, T, r, option_type="call",
              tol=0.001, max_iter=100):
    """Implied volatility via bisection — no scipy dependency.

    Returns annualised IV as a decimal (e.g. 0.35 = 35%) or None on failure.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic * 0.99:
        return None  # below intrinsic — bad data

    low, high = 0.01, 10.0  # 1 % – 1000 % annualised
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = _bsm_price(S, K, T, r, mid, option_type)

        if abs(price - market_price) < tol:
            return mid

        if price > market_price:
            high = mid
        else:
            low = mid

    result = (low + high) / 2
    if result <= 0.02 or result >= 9.9:
        return None  # hit solver bounds — no convergence
    return result


def _bsm_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute BSM Greeks for a European option.

    Returns dict with:
        delta, gamma, theta (per day), vega (per 1% IV move),
        charm (delta decay per day), vanna (delta sensitivity to IV).

    NOTE on vanna/charm usage (2026-04-09 design decision):
        Charm is used as a scoring input via the theta_cost sub-score — it
        directly tells us how much our delta erodes overnight even if the stock
        doesn't move, which is critical for weekly hold decisions.

        Vanna is DISPLAY-ONLY for now. It quantifies how delta shifts when IV
        changes (e.g., post-earnings IV crush), but we lack sufficient live data
        to calibrate its scoring weight. After 4-6 weeks of scorecard data, if
        IV-crush-adjusted delta predictions outperform raw delta, promote vanna
        to a scoring input. Until then, it's informational on pick output only.
        This follows the test-and-revert discipline from ATLAS lessons.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0,
                "charm": 0, "vanna": 0}

    sqrt_T = _msqrt(T)
    d1 = (_mlog(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Standard normal PDF
    n_d1 = _mexp(-0.5 * d1 ** 2) / _msqrt(2 * 3.14159265359)

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta_annual = (
            -(S * n_d1 * sigma) / (2 * sqrt_T)
            - r * K * _mexp(-r * T) * _norm_cdf(d2)
        )
    else:
        delta = _norm_cdf(d1) - 1
        theta_annual = (
            -(S * n_d1 * sigma) / (2 * sqrt_T)
            + r * K * _mexp(-r * T) * _norm_cdf(-d2)
        )

    gamma = n_d1 / (S * sigma * sqrt_T)
    vega = S * n_d1 * sqrt_T / 100  # per 1% IV move
    theta_daily = theta_annual / 365  # per calendar day

    # Charm (delta bleed): d(delta)/d(time) — how much delta decays per day
    # Charm = -n(d1) * [2*r*T - d2*sigma*sqrt(T)] / (2*T*sigma*sqrt(T))
    # For calls; puts have opposite sign adjustment
    charm_annual = -n_d1 * (
        2 * r * T - d2 * sigma * sqrt_T
    ) / (2 * T * sigma * sqrt_T)
    if option_type == "put":
        charm_annual += r * _mexp(-r * T) * _norm_cdf(-d1)
    else:
        charm_annual -= r * _mexp(-r * T) * _norm_cdf(-d1)
    charm_daily = charm_annual / 365

    # Vanna: d(delta)/d(sigma) = d(vega)/d(S) — sensitivity of delta to IV change
    # Vanna = -n(d1) * d2 / sigma  (per 100% IV; we scale to per 1% IV)
    vanna = -n_d1 * d2 / sigma / 100  # per 1% IV change

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_daily, 4),
        "vega": round(vega, 4),
        "charm": round(charm_daily, 6),
        "vanna": round(vanna, 6),
    }


def _project_greeks_over_hold(S, K, T_start, r, sigma, option_type="call",
                               hold_days=5):
    """Project how Greeks evolve over the holding period (Mon→Fri).

    Returns list of dicts (one per trading day) showing delta, gamma,
    theta, vega, charm, vanna, plus derived metrics:
    - delta_change: how much delta shifted from day 1 (charm effect)
    - cumulative_theta: total theta paid through this day
    """
    projections = []
    day1_delta = None
    cumulative_theta = 0.0

    for day in range(hold_days):
        T_remaining = max(T_start - (day / 252), 1 / 252)  # min 1 trading day
        greeks = _bsm_greeks(S, K, T_remaining, r, sigma, option_type)
        greeks["day"] = day + 1
        greeks["days_to_expiry"] = round(T_remaining * 252, 1)

        if day1_delta is None:
            day1_delta = greeks["delta"]
        greeks["delta_change"] = round(greeks["delta"] - day1_delta, 4)

        cumulative_theta += greeks["theta"]
        greeks["cumulative_theta"] = round(cumulative_theta, 4)

        projections.append(greeks)
    return projections


class OptionsScanner:
    """Fetch and analyze options chains with a focus on weekly expiries.

    Optionally uses Tradier for real-time quotes and market-maker Greeks
    when configured. Falls back to yfinance (adequate for our pre-market
    cron schedule). Tradier/Polygon can be added later for intraday use.
    """

    def __init__(self):
        self._tradier = None
        try:
            from scanners.tradier import TradierClient
            client = TradierClient()
            if client.enabled:
                self._tradier = client
                logger.info("OptionsScanner using Tradier as primary data source")
            else:
                logger.debug("Tradier not configured — using yfinance (adequate for pre-market pipeline)")
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    #  Expiry helpers
    # ------------------------------------------------------------------ #

    def get_weekly_expiry(self, ticker: str, target_days: int = 5) -> "str | None":
        """Find the expiry closest to target_days from today.

        For Monday entry: target_days=5 finds this Friday.
        For Wednesday scan: target_days=9 finds next Friday.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        target_days : int
            Number of calendar days ahead to target (default 5).

        Returns
        -------
        str or None
            Expiration date string as returned by yfinance, or *None* on failure.
        """
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options

            if not expirations:
                logger.warning("No option expirations found for %s", ticker)
                return None

            today = datetime.now().date()
            target_date = today + timedelta(days=target_days)
            min_date = today + timedelta(days=3)  # never buy options expiring in < 3 days

            # Collect Friday expiries that are >= min_date
            friday_candidates = []
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= min_date and exp_date.weekday() == 4:
                    friday_candidates.append((exp_str, exp_date))

            if friday_candidates:
                # Pick the Friday closest to target_date, preferring the soonest
                # that is >= min_date
                best = min(
                    friday_candidates,
                    key=lambda x: abs((x[1] - target_date).days),
                )
                return best[0]

            # Fallback: nearest expiry >= min_date regardless of weekday
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= min_date:
                    logger.info(
                        "No Friday expiry for %s; using nearest: %s",
                        ticker,
                        exp_str,
                    )
                    return exp_str

            logger.warning("No valid expirations for %s (all < 3 days out)", ticker)
            return None

        except Exception as exc:
            logger.error("Failed to get weekly expiry for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------ #
    #  Chain retrieval
    # ------------------------------------------------------------------ #

    def get_options_chain(self, ticker: str, expiry_date: str) -> dict:
        """Fetch calls and puts DataFrames for a given expiry.

        Returns
        -------
        dict
            Keys ``'calls'`` and ``'puts'``, each a :class:`pd.DataFrame`,
            or empty dict on failure.
        """
        try:
            tk = yf.Ticker(ticker)
            chain = tk.option_chain(expiry_date)
            return {"calls": chain.calls, "puts": chain.puts}
        except Exception as exc:
            logger.error(
                "Failed to fetch options chain for %s exp %s: %s",
                ticker,
                expiry_date,
                exc,
            )
            return {}

    # ------------------------------------------------------------------ #
    #  Chain analysis
    # ------------------------------------------------------------------ #

    def analyze_chain(self, ticker: str) -> dict:
        """Compute key options metrics for the weekly expiry.

        Metrics
        -------
        - Put/Call ratio (by volume and OI)
        - IV rank (current ATM IV vs realized vol ratio)
        - Max pain strike
        - Highest OI strikes (calls and puts)
        - Highest volume strikes (calls and puts)
        - IV skew (OTM puts vs OTM calls)
        - ATM implied volatility
        - Realized vol (5-day and 20-day)
        - Weekly expected move
        - Theta per day as % of premium

        Returns
        -------
        dict
            All computed metrics, or empty dict on failure.
        """
        try:
            expiry = self.get_weekly_expiry(ticker)
            if not expiry:
                return {}

            chain = self.get_options_chain(ticker, expiry)
            if not chain:
                return {}

            calls = chain["calls"].copy()
            puts = chain["puts"].copy()

            # Current price + price history for realized vol
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1mo")
            if hist.empty:
                return {}
            current_price = float(hist["Close"].iloc[-1])

            # Days to expiry for BSM IV calculation
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
            days_to_exp = max((expiry_dt - datetime.now().date()).days, 1)
            T = days_to_exp / 365.0

            # ---- Put/Call ratio ----
            total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
            total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
            total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0

            pc_ratio_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else None
            pc_ratio_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

            # ---- ATM implied volatility (BSM-validated) ----
            atm_iv = self._compute_validated_atm_iv(
                calls, puts, current_price, T,
            )

            # ---- Realized volatility (20-day and 5-day) & IV / RV ratio ----
            realized_vol = self._compute_realized_vol(hist)
            realized_vol_5d = self._compute_realized_vol_5d(hist)
            iv_rv_ratio = None
            if atm_iv and realized_vol and realized_vol > 0:
                iv_rv_ratio = round(atm_iv / realized_vol, 3)
            iv_rank = self._iv_rank_from_rv_ratio(iv_rv_ratio)

            # ---- Max pain ----
            max_pain = self._compute_max_pain(calls, puts)

            # ---- Highest OI strikes ----
            highest_oi_call = self._top_strike(calls, "openInterest")
            highest_oi_put = self._top_strike(puts, "openInterest")

            # ---- Highest volume strikes ----
            highest_vol_call = self._top_strike(calls, "volume")
            highest_vol_put = self._top_strike(puts, "volume")

            # ---- IV skew ----
            iv_skew = self._compute_iv_skew(calls, puts, current_price)

            # ---- ATM premium data (mid-price, bid-ask spread) ----
            atm_premium = self._get_atm_premium(calls, puts, current_price)

            # ---- Straddle-based implied move (more reliable than IV for near-expiry) ----
            _call_mid = atm_premium.get("call_mid") or 0
            _put_mid = atm_premium.get("put_mid") or 0
            straddle_mid = round(_call_mid + _put_mid, 4) if (_call_mid > 0 and _put_mid > 0) else None
            implied_move_pct = (
                round((straddle_mid / current_price) * 100, 2)
                if straddle_mid and current_price > 0 else None
            )

            # ---- Weekly expected move (ATR-based) ----
            atr_daily = float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1])
            weekly_expected_move_pct = round(
                atr_daily * _msqrt(5) / current_price * 100, 2
            ) if current_price > 0 else None

            # ---- Theta per day as % of premium ----
            theta_per_day_pct = None
            if straddle_mid and straddle_mid > 0 and days_to_exp > 0:
                daily_theta = straddle_mid / days_to_exp
                theta_per_day_pct = round(daily_theta / straddle_mid * 100, 2)

            # ---- Theta decay curve over holding period (BSM-based) ----
            theta_decay_curve = None
            if atm_iv and atm_iv > 0:
                atm_strike = atm_premium.get("call_strike") or current_price
                theta_decay_curve = []
                for day_offset in range(min(days_to_exp, 5)):
                    T_remain = max((days_to_exp - day_offset) / 365.0, 1 / 365)
                    g = _bsm_greeks(current_price, atm_strike, T_remain,
                                    _RISK_FREE_RATE, atm_iv, "call")
                    theta_decay_curve.append({
                        "day": day_offset + 1,
                        "theta_per_day": g["theta"],
                        "charm_per_day": g["charm"],
                        "theta_pct_of_premium": round(
                            abs(g["theta"]) / straddle_mid * 100, 2
                        ) if straddle_mid and straddle_mid > 0 else None,
                    })

            # ---- Stock-level weekly vs monthly IV term structure ----
            iv_term_structure = self._compute_iv_term_structure(
                ticker, expiry, atm_iv, current_price,
            )

            return {
                "ticker": ticker,
                "expiry": expiry,
                "days_to_expiry": days_to_exp,
                "current_price": round(current_price, 2),
                "pc_ratio_volume": round(pc_ratio_vol, 3) if pc_ratio_vol is not None else None,
                "pc_ratio_oi": round(pc_ratio_oi, 3) if pc_ratio_oi is not None else None,
                "atm_iv": round(atm_iv, 4) if atm_iv is not None else None,
                "iv_rank": round(iv_rank, 1) if iv_rank is not None else None,
                "max_pain": max_pain,
                "highest_oi_call_strike": highest_oi_call,
                "highest_oi_put_strike": highest_oi_put,
                "highest_vol_call_strike": highest_vol_call,
                "highest_vol_put_strike": highest_vol_put,
                "iv_skew": round(iv_skew, 4) if iv_skew is not None else None,
                "total_call_volume": int(total_call_vol),
                "total_put_volume": int(total_put_vol),
                "total_call_oi": int(total_call_oi),
                "total_put_oi": int(total_put_oi),
                "atm_call_mid": atm_premium.get("call_mid"),
                "atm_put_mid": atm_premium.get("put_mid"),
                "atm_call_spread_pct": atm_premium.get("call_spread_pct"),
                "atm_put_spread_pct": atm_premium.get("put_spread_pct"),
                "atm_call_strike": atm_premium.get("call_strike"),
                "atm_put_strike": atm_premium.get("put_strike"),
                "realized_vol_20d": round(realized_vol, 4) if realized_vol is not None else None,
                "iv_rv_ratio": iv_rv_ratio,
                "implied_move_pct": implied_move_pct,
                "atm_straddle_mid": straddle_mid,
                # --- New weekly fields ---
                "realized_vol_5d": round(realized_vol_5d, 4) if realized_vol_5d is not None else None,
                "weekly_expected_move_pct": weekly_expected_move_pct,
                "theta_per_day_pct": theta_per_day_pct,
                "theta_decay_curve": theta_decay_curve,
                "iv_term_structure": iv_term_structure,
            }

        except Exception as exc:
            logger.error("Options chain analysis failed for %s: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Optimal strike selection
    # ------------------------------------------------------------------ #

    def find_optimal_strikes(
        self,
        ticker: str,
        direction: str,
        budget_range: tuple[float, float] = (0.10, 10.00),
    ) -> dict:
        """Find the best strike for a directional weekly options trade.

        Uses Tradier real-time Greeks when available (market-maker computed
        delta, gamma, theta, vega), falls back to yfinance + BSM estimates.

        Strike selection is expected-move-aware: prefers strikes within
        the stock's weekly ATR-based range, targeting 0.30-0.40 delta.
        Weekly options use wider stops and time-decay-aware profit targets
        compared to 0DTE plays.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        direction : str
            ``'bullish'`` (buy calls) or ``'bearish'`` (buy puts).
        budget_range : tuple
            Acceptable premium range ``(min, max)`` per contract.

        Returns
        -------
        dict
            strike, premium, iv, estimated_delta, volume, oi, option_type,
            plus entry/exit guidance fields.
        """
        # --- Tradier fast-path: real-time Greeks + quotes ---
        if self._tradier:
            tradier_result = self._find_strikes_tradier(ticker, direction)
            if tradier_result:
                return tradier_result
            logger.info("Tradier strike search failed for %s — falling back to yfinance", ticker)

        try:
            expiry = self.get_weekly_expiry(ticker)
            if not expiry:
                return {}

            chain = self.get_options_chain(ticker, expiry)
            if not chain:
                return {}

            tk = yf.Ticker(ticker)
            hist = tk.history(period="1mo")
            if hist.empty:
                return {}
            current_price = float(hist["Close"].iloc[-1])

            # Days to expiry
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
            days_to_exp = max((expiry_dt - datetime.now().date()).days, 1)

            # Compute ATR for expected move sizing
            atr = float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1])
            atr_pct = atr / current_price

            # Weekly expected move: ATR * sqrt(5) for 5-day hold
            weekly_atr = atr * _msqrt(5)
            weekly_atr_pct = weekly_atr / current_price
            weekly_exp_move = round(weekly_atr_pct * 100, 2)

            # Scale budget range for weekly options
            min_prem = atr_pct * _msqrt(5) * 0.3 * current_price
            max_prem = atr_pct * _msqrt(5) * 2.5 * current_price
            # Respect caller's bounds as outer limits
            min_prem = max(min_prem, budget_range[0])
            max_prem = min(max_prem, budget_range[1])

            if direction == "bullish":
                options = chain["calls"].copy()
                option_type = "call"
            elif direction == "bearish":
                options = chain["puts"].copy()
                option_type = "put"
            else:
                logger.error("Invalid direction '%s'; use 'bullish' or 'bearish'", direction)
                return {}

            if options.empty:
                return {}

            # Midpoint price as premium estimate
            options = options.copy()
            options["mid"] = (options["bid"] + options["ask"]) / 2

            # Filter: bid > 0, within budget, some volume or OI
            mask = (
                (options["bid"] > 0.05)
                & (options["mid"] >= min_prem)
                & (options["mid"] <= max_prem)
                & ((options["volume"] > 0) | (options["openInterest"] > 10))
            )
            filtered = options.loc[mask].copy()

            if filtered.empty:
                logger.info(
                    "No strikes pass filters for %s %s (budget %.2f-%.2f)",
                    ticker, direction, min_prem, max_prem,
                )
                return {}

            # Strike distance from current price as % of weekly ATR
            # Target delta: 0.30-0.40 (slightly more OTM than 0DTE to reduce theta cost)
            max_otm_distance = weekly_atr_pct * 0.5  # penalize > 0.5x weekly ATR

            if option_type == "call":
                filtered["strike_dist_pct"] = (filtered["strike"] - current_price) / current_price
                filtered["moneyness"] = current_price / filtered["strike"]
                filtered["est_delta"] = np.clip(0.5 * filtered["moneyness"], 0.05, 0.95)
                target_delta = 0.35
            else:
                filtered["strike_dist_pct"] = (current_price - filtered["strike"]) / current_price
                filtered["moneyness"] = filtered["strike"] / current_price
                filtered["est_delta"] = -np.clip(0.5 * filtered["moneyness"], 0.05, 0.95)
                target_delta = -0.35

            # Penalise strikes that are too far OTM
            filtered["otm_penalty"] = np.where(
                filtered["strike_dist_pct"] > max_otm_distance,
                (filtered["strike_dist_pct"] - max_otm_distance) * 10,
                0,
            )

            # Score: delta closeness + volume + tight spread - OTM penalty
            filtered["spread_pct"] = (
                (filtered["ask"] - filtered["bid"]) / filtered["mid"]
            ).clip(0, 5)
            filtered["delta_dist"] = (filtered["est_delta"] - target_delta).abs()
            vol_max = max(filtered["volume"].max(), 1)
            filtered["score"] = (
                (1 - filtered["delta_dist"]) * 3       # delta closeness
                + (filtered["volume"] / vol_max) * 2   # volume
                + (1 - filtered["spread_pct"].clip(0, 1)) * 2  # tight spread
                - filtered["otm_penalty"]               # penalise deep OTM
            )

            best = filtered.sort_values("score", ascending=False).iloc[0]
            iv_value = float(best.get("impliedVolatility", 0))
            premium = round(float(best["mid"]), 2)

            # Compute breakeven and risk/reward metrics
            strike = float(best["strike"])
            if option_type == "call":
                breakeven = strike + premium
                breakeven_move_pct = ((breakeven - current_price) / current_price) * 100
            else:
                breakeven = strike - premium
                breakeven_move_pct = ((current_price - breakeven) / current_price) * 100

            # Expected move based on weekly ATR
            expected_move_dollar = round(weekly_atr, 2)
            expected_move_pct = weekly_exp_move

            # Time-decay-aware profit targets
            day_targets = "40% day-1 / 35% day-2 / 25% day-3 / 15% day-4"

            # BSM Greeks at entry
            T = days_to_exp / 365.0
            iv_for_greeks = iv_value if iv_value > 0.05 else 0.30
            greeks = _bsm_greeks(current_price, strike, T, _RISK_FREE_RATE,
                                 iv_for_greeks, option_type)
            greeks_projection = _project_greeks_over_hold(
                current_price, strike, T, _RISK_FREE_RATE,
                iv_for_greeks, option_type, hold_days=min(days_to_exp, 5),
            )

            # Earnings IV crush risk
            earnings_iv_crush = self._estimate_earnings_iv_crush(
                ticker, expiry_dt, iv_for_greeks,
            )

            return {
                "ticker": ticker,
                "option_type": option_type,
                "direction": direction,
                "expiry": expiry,
                "strike": strike,
                "premium": premium,
                "bid": float(best["bid"]),
                "ask": float(best["ask"]),
                "iv": round(iv_value, 4),
                "estimated_delta": round(float(best["est_delta"]), 3),
                "volume": int(best["volume"]),
                "oi": int(best.get("openInterest", 0)),
                "score": round(float(best["score"]), 2),
                # BSM Greeks
                "greeks": greeks,
                "greeks_projection": greeks_projection,
                # Earnings risk
                "earnings_iv_crush": earnings_iv_crush,
                # Execution guidance
                "current_price": round(current_price, 2),
                "breakeven": round(breakeven, 2),
                "breakeven_move_pct": round(breakeven_move_pct, 2),
                "expected_daily_move": round(atr, 2),
                "expected_daily_move_pct": round(atr_pct * 100, 2),
                "expected_weekly_move": expected_move_dollar,
                "expected_weekly_move_pct": expected_move_pct,
                "days_to_expiry": days_to_exp,
                "entry": {
                    "method": "limit_monday_open",
                    "wait_minutes": 20,
                    "instruction": (
                        f"Enter Monday morning. Wait 15-20 min after open for spread to tighten. "
                        f"Limit order at mid ({premium:.2f}) or better. Skip if gap > 2%."
                    ),
                },
                "exit": {
                    "profit_targets": {
                        "day_1": 40,  # Take 40% profit if hit by end of Monday
                        "day_2": 35,  # 35% by Tuesday
                        "day_3": 25,  # 25% by Wednesday
                        "day_4": 15,  # 15% by Thursday (theta eating gains)
                    },
                    "stop_loss_pct": 50,  # Wider stop for weekly (more time for thesis to play out)
                    "delta_stop": 0.10,   # Exit if delta drops below 0.10 (nearly worthless)
                    "instruction": (
                        f"Time-decay-aware exits: take {day_targets} profit. "
                        f"Hard stop at 50% loss. Exit if delta < 0.10. "
                        f"Breakeven requires {breakeven_move_pct:.1f}% move to ${breakeven:.2f} "
                        f"(weekly expected move: {weekly_exp_move:.1f}%)."
                    ),
                },
            }

        except Exception as exc:
            logger.error("Optimal strike search failed for %s: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Tradier-powered strike selection
    # ------------------------------------------------------------------ #

    def _find_strikes_tradier(self, ticker: str, direction: str) -> dict:
        """Find optimal strikes using Tradier real-time data.

        Returns the same dict format as find_optimal_strikes() so callers
        don't need to know which data source was used.
        """
        try:
            result = self._tradier.find_best_strike(ticker, direction)
            if not result:
                return {}

            # Enrich with ATR-based expected move (still need yfinance for history)
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1mo")
            if not hist.empty:
                atr = float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1])
                current_price = result["current_price"]
                atr_pct = atr / current_price if current_price > 0 else 0
                weekly_atr = atr * _msqrt(5)

                result["expected_daily_move"] = round(atr, 2)
                result["expected_daily_move_pct"] = round(atr_pct * 100, 2)
                result["expected_weekly_move"] = round(weekly_atr, 2)
                result["expected_weekly_move_pct"] = round(atr_pct * _msqrt(5) * 100, 2)

            # Add Greeks projection using Tradier's real IV
            iv = result.get("iv") or 0.30
            strike = result["strike"]
            current_price = result["current_price"]
            expiry_dt = datetime.strptime(result["expiry"], "%Y-%m-%d").date()
            days_to_exp = max((expiry_dt - datetime.now().date()).days, 1)
            T = days_to_exp / 365.0
            option_type = result["option_type"]

            result["greeks_projection"] = _project_greeks_over_hold(
                current_price, strike, T, _RISK_FREE_RATE,
                iv, option_type, hold_days=min(days_to_exp, 5),
            )

            # Earnings IV crush check
            result["earnings_iv_crush"] = self._estimate_earnings_iv_crush(
                ticker, expiry_dt, iv,
            )

            result["days_to_expiry"] = days_to_exp

            # Entry/exit guidance
            premium = result["premium"]
            breakeven_move_pct = result["breakeven_move_pct"]
            breakeven = result["breakeven"]
            weekly_exp_move = result.get("expected_weekly_move_pct", 5.0)
            day_targets = "40% day-1 / 35% day-2 / 25% day-3 / 15% day-4"

            result["entry"] = {
                "method": "limit_monday_open",
                "wait_minutes": 20,
                "instruction": (
                    f"Enter Monday morning. Wait 15-20 min after open for spread to tighten. "
                    f"Limit order at mid ({premium:.2f}) or better. Skip if gap > 2%."
                ),
            }
            result["exit"] = {
                "profit_targets": {"day_1": 40, "day_2": 35, "day_3": 25, "day_4": 15},
                "stop_loss_pct": 50,
                "delta_stop": 0.10,
                "instruction": (
                    f"Time-decay-aware exits: take {day_targets} profit. "
                    f"Hard stop at 50% loss. Exit if delta < 0.10. "
                    f"Breakeven requires {breakeven_move_pct:.1f}% move to ${breakeven:.2f} "
                    f"(weekly expected move: {weekly_exp_move:.1f}%)."
                ),
            }

            # Opening range data for Monday entry
            opening = self._tradier.get_opening_range(ticker)
            if opening:
                result["opening_range"] = opening
                # Skip signal: gap > 2%
                if abs(opening.get("gap_pct", 0)) > 2.0:
                    result["entry_warning"] = (
                        f"Gap {opening['gap_pct']:+.1f}% — consider skipping "
                        f"(move may already be priced in)"
                    )

            logger.info(
                "Tradier strike for %s: %s %s @ $%.2f (delta %.3f, spread %.1f%%)",
                ticker, direction, strike, premium,
                abs(result.get("estimated_delta", 0)),
                result.get("spread_pct", 0),
            )
            return result

        except Exception as exc:
            logger.warning("Tradier strike selection failed for %s: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Batch
    # ------------------------------------------------------------------ #

    def scan_batch(self, tickers: list) -> dict:
        """Run :meth:`analyze_chain` on multiple tickers.

        Returns
        -------
        dict
            Keyed by ticker, values are analysis dicts.
        """
        results = {}
        for ticker in tickers:
            logger.info("Analyzing options chain for %s ...", ticker)
            analysis = self.analyze_chain(ticker)
            if analysis:
                results[ticker] = analysis
            time.sleep(0.3)  # Rate-limit

        return results

    # ------------------------------------------------------------------ #
    #  Stock-level IV term structure (weekly vs monthly)
    # ------------------------------------------------------------------ #

    def _compute_iv_term_structure(
        self,
        ticker: str,
        weekly_expiry: str,
        weekly_iv: "float | None",
        current_price: float,
    ) -> "dict | None":
        """Compare weekly IV to monthly IV for the same stock.

        When weekly IV < monthly IV (contango), buying the weekly is
        relatively cheap — the market prices more risk in the longer term.
        When weekly IV > monthly IV (backwardation), the weekly is expensive,
        often due to a near-term event (earnings, FDA, etc.).

        Returns
        -------
        dict or None
            weekly_iv, monthly_iv, ratio, structure (contango/backwardation/flat),
            mispricing_signal (cheap/fair/expensive).
        """
        if weekly_iv is None or weekly_iv <= 0:
            return None

        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options
            if not expirations or len(expirations) < 2:
                return None

            weekly_dt = datetime.strptime(weekly_expiry, "%Y-%m-%d").date()

            # Find the nearest monthly expiry: 25-45 days out from today
            today = datetime.now().date()
            monthly_expiry = None
            for exp_str in expirations:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                days_out = (exp_dt - today).days
                if 25 <= days_out <= 45 and exp_dt != weekly_dt:
                    monthly_expiry = exp_str
                    break

            if not monthly_expiry:
                # Fallback: any expiry 20-60 days out that isn't the weekly
                for exp_str in expirations:
                    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    days_out = (exp_dt - today).days
                    if 20 <= days_out <= 60 and exp_dt != weekly_dt:
                        monthly_expiry = exp_str
                        break

            if not monthly_expiry:
                return None

            # Fetch monthly chain and compute ATM IV
            monthly_chain = self.get_options_chain(ticker, monthly_expiry)
            if not monthly_chain:
                return None

            monthly_dt = datetime.strptime(monthly_expiry, "%Y-%m-%d").date()
            T_monthly = max((monthly_dt - today).days, 1) / 365.0

            monthly_iv = self._compute_validated_atm_iv(
                monthly_chain["calls"], monthly_chain["puts"],
                current_price, T_monthly,
            )

            if monthly_iv is None or monthly_iv <= 0:
                return None

            ratio = weekly_iv / monthly_iv

            # Classify structure
            if ratio < 0.90:
                structure = "contango"       # weekly cheap vs monthly
                mispricing = "cheap"         # favorable for buying weeklies
            elif ratio > 1.10:
                structure = "backwardation"  # weekly expensive (event premium)
                mispricing = "expensive"
            else:
                structure = "flat"
                mispricing = "fair"

            return {
                "weekly_iv": round(weekly_iv, 4),
                "monthly_iv": round(monthly_iv, 4),
                "weekly_expiry": weekly_expiry,
                "monthly_expiry": monthly_expiry,
                "ratio": round(ratio, 3),
                "structure": structure,
                "mispricing_signal": mispricing,
            }

        except Exception as exc:
            logger.debug("IV term structure failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------ #
    #  Earnings IV crush estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_earnings_iv_crush(
        ticker: str,
        expiry_date,
        current_iv: float,
    ) -> dict:
        """Estimate earnings-related IV crush risk within the holding window.

        Checks if an earnings date falls between now and expiry.
        If so, estimates the post-earnings IV crush (typically 30-60% of
        pre-earnings IV for weekly options) and the premium at risk.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        expiry_date : date
            Option expiry date.
        current_iv : float
            Current implied volatility (annualized decimal).

        Returns
        -------
        dict
            earnings_in_window (bool), earnings_date (str or None),
            estimated_iv_crush_pct (float), premium_at_risk_pct (float).
        """
        result = {
            "earnings_in_window": False,
            "earnings_date": None,
            "estimated_iv_crush_pct": 0,
            "premium_at_risk_pct": 0,
        }

        try:
            tk = yf.Ticker(ticker)
            cal = tk.calendar
            if cal is None:
                return result

            # yfinance returns calendar as a dict or DataFrame
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        earnings_date = pd.Timestamp(ed[0]).date()
                    else:
                        earnings_date = pd.Timestamp(ed).date()
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.columns:
                    vals = cal["Earnings Date"].dropna()
                    if not vals.empty:
                        earnings_date = pd.Timestamp(vals.iloc[0]).date()

            if earnings_date is None:
                return result

            today = datetime.now().date()
            if today <= earnings_date <= expiry_date:
                result["earnings_in_window"] = True
                result["earnings_date"] = str(earnings_date)

                # IV crush model: weeklies typically lose 35-50% of IV
                # post-earnings. Use 40% as base, scale by current IV level.
                base_crush = 0.40
                # Higher IV → more crush (convex relationship)
                if current_iv > 0.60:
                    crush_factor = 0.50
                elif current_iv > 0.40:
                    crush_factor = 0.45
                else:
                    crush_factor = base_crush

                result["estimated_iv_crush_pct"] = round(crush_factor * 100, 1)
                # Vega-based premium impact: rough estimate
                # Premium at risk ≈ crush_factor × current_iv × vega_sensitivity
                # For ATM weekly: vega ≈ 0.04-0.08 per 1% IV per $100 underlying
                # Simplified: premium drops ~crush_factor × (IV_portion_of_premium)
                # For weeklies, IV is ~60-80% of the premium (rest is intrinsic)
                iv_portion = 0.70  # typical for near-ATM weekly
                result["premium_at_risk_pct"] = round(
                    crush_factor * iv_portion * 100, 1
                )

        except Exception as exc:
            logger.debug("Earnings IV crush check failed for %s: %s", ticker, exc)

        return result

    # ================================================================== #
    #  Private helpers
    # ================================================================== #

    @staticmethod
    def _get_atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, price: float) -> "float | None":
        """Return ATM implied volatility (average of nearest call & put IV)."""
        try:
            call_atm = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]
            put_atm = puts.iloc[(puts["strike"] - price).abs().argsort()[:1]]

            ivs = []
            if not call_atm.empty and "impliedVolatility" in call_atm.columns:
                ivs.append(float(call_atm["impliedVolatility"].iloc[0]))
            if not put_atm.empty and "impliedVolatility" in put_atm.columns:
                ivs.append(float(put_atm["impliedVolatility"].iloc[0]))

            return np.mean(ivs) if ivs else None
        except Exception:
            return None

    @staticmethod
    def _compute_validated_atm_iv(
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        price: float,
        T: float,
        r: float = _RISK_FREE_RATE,
    ) -> "float | None":
        """Compute ATM IV using BSM from option mid-prices, yfinance fallback.

        For each of the nearest ATM call and put:
        1. Compute mid-price from bid/ask.
        2. Solve BSM for IV (bisection).
        3. Fall back to yfinance ``impliedVolatility`` only when BSM fails.
        4. Average available IVs.
        """
        ivs = []

        for opts, opt_type in [(calls, "call"), (puts, "put")]:
            if opts.empty:
                continue

            atm = opts.iloc[(opts["strike"] - price).abs().argsort()[:1]]
            if atm.empty:
                continue

            strike = float(atm["strike"].iloc[0])
            bid = float(atm["bid"].iloc[0]) if "bid" in atm.columns else 0
            ask = float(atm["ask"].iloc[0]) if "ask" in atm.columns else 0

            # Compute mid-price
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
            elif ask > 0:
                mid = ask * 0.85  # discount ask-only (illiquid)
            else:
                mid = 0

            # BSM solve if we have a mid
            if mid > 0:
                iv = _solve_iv(mid, price, strike, T, r, opt_type)
                if iv is not None and 0.05 < iv < 5.0:
                    ivs.append(iv)
                    continue

            # Fallback: yfinance impliedVolatility (sanity-checked)
            yf_iv = 0.0
            if "impliedVolatility" in atm.columns:
                yf_iv = float(atm["impliedVolatility"].iloc[0] or 0)
            if yf_iv > 0.05:
                ivs.append(yf_iv)

        return float(np.mean(ivs)) if ivs else None

    @staticmethod
    def _compute_realized_vol(hist, window: int = 20) -> "float | None":
        """Annualised realized volatility from daily close-to-close returns."""
        if hist is None or len(hist) < 6:
            return None

        closes = hist["Close"].tail(min(window + 1, len(hist)))
        returns = closes.pct_change().dropna()

        if len(returns) < 5:
            return None

        daily_std = float(returns.std())
        return daily_std * np.sqrt(252)

    @staticmethod
    def _compute_realized_vol_5d(hist, window: int = 5) -> "float | None":
        """5-day annualized realized vol for weekly comparison."""
        if hist is None or len(hist) < 6:
            return None

        closes = hist["Close"].tail(min(window + 1, len(hist)))
        returns = closes.pct_change().dropna()

        if len(returns) < 3:
            return None

        daily_std = float(returns.std())
        return daily_std * np.sqrt(252)

    @staticmethod
    def _iv_rank_from_rv_ratio(iv_rv_ratio: "float | None") -> "float | None":
        """Map IV / RV ratio to a 0-100 rank (options cheapness / richness).

        NOTE: this is NOT industry-standard IV rank (which uses 52-week IV
        percentile). This is a lightweight proxy derived from IV/RV ratio.
        For weekly options IV/RV is structurally elevated, so the distribution
        skews toward the "expensive" end vs. true IV rank. Do not treat this
        as a broker-equivalent IV rank — see memory note iv_rank_caveats.

        Semantics (inverted for legacy compatibility):
        - iv_rv < 0.8  -> high opportunity rank (70-100)
        - iv_rv ~ 1.0  -> moderate rank (40-60)
        - iv_rv > 1.5  -> low rank (0-30)
        Higher rank = better buying opportunity.
        """
        if iv_rv_ratio is None:
            return None
        rank = float(np.clip((2.0 - iv_rv_ratio) / 2.0 * 100, 0, 100))
        return rank

    @staticmethod
    def _compute_iv_rank(calls: pd.DataFrame, puts: pd.DataFrame, atm_iv: "float | None") -> "float | None":
        """Approximate IV rank using the range of IVs across strikes.

        A proper IV rank requires historical IV data. This uses the chain's
        own IV distribution as a rough proxy.
        """
        if atm_iv is None:
            return None
        try:
            all_ivs = pd.concat([
                calls["impliedVolatility"].dropna(),
                puts["impliedVolatility"].dropna(),
            ])
            if all_ivs.empty:
                return None

            iv_min = float(all_ivs.min())
            iv_max = float(all_ivs.max())
            if iv_max == iv_min:
                return 50.0

            rank = ((atm_iv - iv_min) / (iv_max - iv_min)) * 100
            return float(np.clip(rank, 0, 100))
        except Exception:
            return None

    @staticmethod
    def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> "float | None":
        """Calculate the max-pain strike (where total option-holder losses are greatest).

        At each candidate strike, compute the aggregate intrinsic value paid
        out to all call and put holders.  The strike with the *minimum* total
        payout is max pain.
        """
        try:
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            if not strikes:
                return None

            min_pain = float("inf")
            max_pain_strike = None

            for strike in strikes:
                # Call holders lose when price <= strike
                call_pain = 0.0
                for _, row in calls.iterrows():
                    if strike > row["strike"]:
                        call_pain += (strike - row["strike"]) * row.get("openInterest", 0)

                # Put holders lose when price >= strike
                put_pain = 0.0
                for _, row in puts.iterrows():
                    if strike < row["strike"]:
                        put_pain += (row["strike"] - strike) * row.get("openInterest", 0)

                total = call_pain + put_pain
                if total < min_pain:
                    min_pain = total
                    max_pain_strike = strike

            return float(max_pain_strike) if max_pain_strike is not None else None
        except Exception:
            return None

    @staticmethod
    def _top_strike(options: pd.DataFrame, column: str) -> "float | None":
        """Return the strike with the highest value in *column*."""
        try:
            if options.empty or column not in options.columns:
                return None
            idx = options[column].idxmax()
            return float(options.loc[idx, "strike"])
        except Exception:
            return None

    @staticmethod
    def _get_atm_premium(
        calls: pd.DataFrame, puts: pd.DataFrame, price: float
    ) -> dict:
        """Extract ATM option mid-prices and bid-ask spread percentages.

        Returns dict with call_mid, put_mid, call_spread_pct, put_spread_pct,
        call_strike, put_strike. Values are None if data is unavailable.
        """
        result = {
            "call_mid": None, "put_mid": None,
            "call_spread_pct": None, "put_spread_pct": None,
            "call_strike": None, "put_strike": None,
        }
        try:
            # Nearest call strike to current price
            if not calls.empty:
                call_atm = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]
                if not call_atm.empty:
                    bid = float(call_atm["bid"].iloc[0]) if "bid" in call_atm.columns else 0
                    ask = float(call_atm["ask"].iloc[0]) if "ask" in call_atm.columns else 0
                    if bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                        result["call_mid"] = round(mid, 4)
                        result["call_spread_pct"] = round((ask - bid) / mid * 100, 2) if mid > 0 else None
                    elif ask > 0:
                        result["call_mid"] = round(ask, 4)
                    result["call_strike"] = float(call_atm["strike"].iloc[0])

            # Nearest put strike to current price
            if not puts.empty:
                put_atm = puts.iloc[(puts["strike"] - price).abs().argsort()[:1]]
                if not put_atm.empty:
                    bid = float(put_atm["bid"].iloc[0]) if "bid" in put_atm.columns else 0
                    ask = float(put_atm["ask"].iloc[0]) if "ask" in put_atm.columns else 0
                    if bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                        result["put_mid"] = round(mid, 4)
                        result["put_spread_pct"] = round((ask - bid) / mid * 100, 2) if mid > 0 else None
                    elif ask > 0:
                        result["put_mid"] = round(ask, 4)
                    result["put_strike"] = float(put_atm["strike"].iloc[0])

        except Exception as exc:
            logger.debug("ATM premium extraction failed: %s", exc)

        return result

    @staticmethod
    def _compute_iv_skew(
        calls: pd.DataFrame, puts: pd.DataFrame, price: float
    ) -> "float | None":
        """Compute IV skew: average OTM put IV minus average OTM call IV.

        Positive skew means puts are more expensive (typical for equities).
        """
        try:
            otm_puts = puts.loc[puts["strike"] < price]
            otm_calls = calls.loc[calls["strike"] > price]

            if otm_puts.empty or otm_calls.empty:
                return None

            avg_put_iv = float(otm_puts["impliedVolatility"].mean())
            avg_call_iv = float(otm_calls["impliedVolatility"].mean())

            return avg_put_iv - avg_call_iv
        except Exception:
            return None
