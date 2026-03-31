"""
Options-chain scanner for zero-DTE trade selection.

Analyzes options chains to surface put/call ratios, IV rank, max pain,
strike-level open interest, and optimal entry strikes for directional
zero-DTE plays.
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


class OptionsScanner:
    """Fetch and analyze options chains with a focus on near-term expiries."""

    # ------------------------------------------------------------------ #
    #  Expiry helpers
    # ------------------------------------------------------------------ #

    def get_friday_expiry(self, ticker: str) -> "str | None":
        """Return the nearest Friday (or end-of-week) expiration string.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.

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

            # Walk through expirations and pick the nearest Friday (weekday 4)
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= today and exp_date.weekday() == 4:
                    return exp_str

            # Fallback: nearest expiry regardless of weekday
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= today:
                    logger.info(
                        "No Friday expiry for %s; using nearest: %s",
                        ticker,
                        exp_str,
                    )
                    return exp_str

            logger.warning("All expirations for %s are in the past", ticker)
            return None

        except Exception as exc:
            logger.error("Failed to get Friday expiry for %s: %s", ticker, exc)
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
        """Compute key options metrics for the nearest Friday expiry.

        Metrics
        -------
        - Put/Call ratio (by volume and OI)
        - IV rank (current ATM IV vs 30-day range)
        - Max pain strike
        - Highest OI strikes (calls and puts)
        - Highest volume strikes (calls and puts)
        - IV skew (OTM puts vs OTM calls)
        - ATM implied volatility

        Returns
        -------
        dict
            All computed metrics, or empty dict on failure.
        """
        try:
            expiry = self.get_friday_expiry(ticker)
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

            # ---- Realized volatility & IV / RV ratio ----
            realized_vol = self._compute_realized_vol(hist)
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
        """Find the best strike for a directional zero-DTE trade.

        Strike selection is expected-move-aware: prefers strikes within
        the stock's ATR-based daily range, targeting 0.35-0.50 delta.
        Previous version used a flat $0.10-$5.00 filter that produced
        strikes too far OTM on volatile stocks.

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
        try:
            expiry = self.get_friday_expiry(ticker)
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

            # Compute ATR for expected move sizing
            atr = float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1])
            atr_pct = atr / current_price

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
            min_prem, max_prem = budget_range
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

            # Strike distance from current price as % of ATR
            if option_type == "call":
                filtered["strike_dist_pct"] = (filtered["strike"] - current_price) / current_price
                filtered["moneyness"] = current_price / filtered["strike"]
                filtered["est_delta"] = np.clip(0.5 * filtered["moneyness"], 0.05, 0.95)
                # Target: slightly OTM, within ~0.5 ATR of current price
                # This keeps us close enough that a normal daily move makes us profitable
                target_delta = 0.42
                max_otm_distance = atr_pct * 0.7  # don't go further than 70% of daily ATR
            else:
                filtered["strike_dist_pct"] = (current_price - filtered["strike"]) / current_price
                filtered["moneyness"] = filtered["strike"] / current_price
                filtered["est_delta"] = -np.clip(0.5 * filtered["moneyness"], 0.05, 0.95)
                target_delta = -0.42
                max_otm_distance = atr_pct * 0.7

            # Penalise strikes that are too far OTM (key fix: was missing before)
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

            # Expected move based on ATR
            expected_move_dollar = round(atr, 2)
            expected_move_pct = round(atr_pct * 100, 2)

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
                # Execution guidance
                "current_price": round(current_price, 2),
                "breakeven": round(breakeven, 2),
                "breakeven_move_pct": round(breakeven_move_pct, 2),
                "expected_daily_move": expected_move_dollar,
                "expected_daily_move_pct": expected_move_pct,
                "entry": {
                    "method": "limit_after_open",
                    "wait_minutes": 15,
                    "instruction": (
                        f"Wait 15-30 min after open for range to establish. "
                        f"Enter on a limit order at the mid ({premium:.2f}) or better. "
                        f"Do NOT chase — if the move has already started without you, skip."
                    ),
                },
                "exit": {
                    "profit_target_pct": 50,
                    "stop_loss_pct": 40,
                    "time_stop": "12:00 ET",
                    "instruction": (
                        f"Take profit at 50% gain (premium {premium:.2f} → {premium * 1.5:.2f}). "
                        f"Cut loss at 40% loss (premium → {premium * 0.6:.2f}). "
                        f"If neither hit by 12:00 ET, close — theta accelerates after noon on 0DTE. "
                        f"Breakeven requires a {breakeven_move_pct:.1f}% move to ${breakeven:.2f}."
                    ),
                },
            }

        except Exception as exc:
            logger.error("Optimal strike search failed for %s: %s", ticker, exc)
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
    def _iv_rank_from_rv_ratio(iv_rv_ratio: "float | None") -> "float | None":
        """Map IV / RV ratio to a 0-100 rank (options cheapness / richness).

        For *buying* 0-DTE:
        - iv_rv < 0.8  → cheap options → high opportunity rank (70-100)
        - iv_rv ~ 1.0  → fair              → moderate rank (40-60)
        - iv_rv > 1.5  → expensive          → low rank (0-30)

        The ranking intentionally inverts so that *higher rank = better
        buying opportunity* (consistent with how downstream scoring uses
        the field).
        """
        if iv_rv_ratio is None:
            return None
        # Invert: lower ratio = better buying opportunity = higher rank
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
