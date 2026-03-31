"""
Directional backtest for the Zero-DTE Options Trading Analysis System.

Combines historical price data with macro context (FRED) and event
calendars (Finnhub) to answer the most important question: does our
direction prediction actually work, and *why*?

For each Wednesday in the lookback period, computes technical indicators,
enriches with macro regime and event context, runs direction prediction,
and checks whether the stock moved in the predicted direction by Friday
close.  Reports per-signal accuracy, regime breakdowns, contextual
analysis (earnings gaps, macro events, yield curve), and optimal weight
derivations.

Signals tested (available from price data):
    RSI, MACD histogram, price vs SMA20, gap %, volume ratio,
    Bollinger Band position

Contextual features (FRED + Finnhub):
    Earnings proximity (gap classification), high-impact macro events,
    yield curve state, credit spread level, financial conditions

Signals NOT tested (require live options/sentiment data):
    P/C ratio, max pain, IV, sentiment, flow, finviz, insider
"""

import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()


class DirectionalBacktester:
    """Backtest direction prediction accuracy on historical Wed->Fri moves."""

    LOOKBACK_WEEKS = 52

    # Signals we can test from price history alone.
    TESTABLE_SIGNALS = ["rsi", "macd", "sma20", "gap", "volume", "bb"]

    # Contextual features from external data sources.
    CONTEXT_FEATURES = [
        "earnings_gap", "macro_event", "yield_curve", "credit_spread",
        "financial_conditions",
    ]

    # Signals we CANNOT test historically (documented for transparency).
    UNTESTABLE_SIGNALS = [
        "pc_ratio", "max_pain", "iv_data", "sentiment",
        "flow", "finviz_pre_market", "analyst_rating", "insider",
    ]

    # FRED series IDs for macro context
    FRED_SERIES = {
        "T10Y2Y": "10Y-2Y Treasury Spread (daily)",
        "BAMLH0A0HYM2": "High Yield OAS (credit spread)",
        "NFCI": "Chicago Fed National Financial Conditions Index",
    }

    def __init__(self, tickers=None, lookback_weeks=None):
        if tickers is None:
            from universe.robinhood import get_full_universe
            self.tickers = get_full_universe()
        else:
            self.tickers = list(tickers)

        self.lookback_weeks = lookback_weeks or self.LOOKBACK_WEEKS

        # Populated by _fetch_* methods in run()
        self._earnings_dates = {}   # {ticker: set of date strings}
        self._economic_events = {}  # {date_str: [event_dicts]}
        self._fred_data = {}        # {series_id: pd.Series}
        self._has_external_data = False

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run full directional backtest across the universe.

        Returns
        -------
        dict
            Comprehensive results: overall accuracy, per-signal stats,
            regime breakdown, contextual analysis, confidence calibration,
            signal correlations, and derived optimal weights.
        """
        logger.info(
            "Starting directional backtest: %d tickers, %d-week lookback",
            len(self.tickers), self.lookback_weeks,
        )

        # Fetch external data ONCE before the ticker loop
        vix_hist = self._fetch_vix_history()
        self._fetch_fred_macro_history()
        self._fetch_earnings_calendar()
        self._fetch_economic_events()
        self._has_external_data = bool(self._fred_data or self._earnings_dates or self._economic_events)

        all_results = []
        failed = 0

        for i, ticker in enumerate(self.tickers, 1):
            try:
                ticker_results = self._backtest_ticker(ticker, vix_hist)
                all_results.extend(ticker_results)
            except Exception as exc:
                logger.debug("Backtest failed for %s: %s", ticker, exc)
                failed += 1

            if i % 25 == 0:
                logger.info("Backtest progress: %d / %d tickers", i, len(self.tickers))
            time.sleep(0.15)  # light rate limit

        logger.info(
            "Backtest complete: %d predictions from %d tickers (%d failed)",
            len(all_results), len(self.tickers) - failed, failed,
        )

        compiled = self._compile_results(all_results)
        compiled["meta"] = {
            "tickers_scanned": len(self.tickers),
            "tickers_failed": failed,
            "lookback_weeks": self.lookback_weeks,
            "run_date": datetime.now().isoformat(),
            "testable_signals": self.TESTABLE_SIGNALS,
            "context_features": self.CONTEXT_FEATURES,
            "untestable_signals": self.UNTESTABLE_SIGNALS,
            "external_data": {
                "fred_series": list(self._fred_data.keys()),
                "earnings_tickers": len(self._earnings_dates),
                "economic_event_dates": len(self._economic_events),
            },
        }
        return compiled

    def save_results(self, results: dict, date_str: str = None) -> Path:
        """Persist backtest results to data/performance/."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        filepath = config.performance_dir / f"backtest_{date_str}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as fh:
            json.dump(results, fh, indent=2, default=str)

        logger.info("Backtest results saved to %s", filepath)
        return filepath

    def print_report(self, results: dict) -> str:
        """Print a human-readable backtest report and return as string."""
        lines = []
        _p = lines.append

        _p("=" * 64)
        _p("  DIRECTIONAL BACKTEST REPORT")
        _p("=" * 64)

        meta = results.get("meta", {})
        _p(f"  Universe: {meta.get('tickers_scanned', '?')} tickers "
           f"({meta.get('tickers_failed', 0)} failed)")
        _p(f"  Lookback: {meta.get('lookback_weeks', '?')} weeks")
        _p(f"  Run date: {meta.get('run_date', '?')[:10]}")
        _p("")

        # -- Overall accuracy --
        acc = results.get("overall_accuracy", 0)
        total = results.get("total_predictions", 0)
        correct = results.get("correct_predictions", 0)
        _p(f"  OVERALL ACCURACY:  {acc:.1%}  ({correct}/{total})")
        edge = acc - 0.5
        _p(f"  Edge over coin flip: {edge:+.1%}")
        _p("")

        # -- Per-signal accuracy --
        _p("-" * 64)
        _p("  PER-SIGNAL ACCURACY  (when signal was active)")
        _p("-" * 64)
        _p(f"  {'Signal':<12} {'Accuracy':>9} {'Active':>8} {'Bullish':>9} {'Bearish':>9} {'Corr':>7}")

        per_signal = results.get("per_signal", {})
        correlations = results.get("signal_correlations", {})
        for sig in self.TESTABLE_SIGNALS:
            s = per_signal.get(sig, {})
            acc_s = s.get("accuracy")
            acc_str = f"{acc_s:.1%}" if acc_s is not None else "N/A"
            bull_acc = s.get("bullish_accuracy")
            bear_acc = s.get("bearish_accuracy")
            bull_str = f"{bull_acc:.1%}" if bull_acc is not None else "N/A"
            bear_str = f"{bear_acc:.1%}" if bear_acc is not None else "N/A"
            corr = correlations.get(sig)
            corr_str = f"{corr:+.3f}" if corr is not None else "N/A"
            _p(f"  {sig:<12} {acc_str:>9} {s.get('times_active', 0):>8} "
               f"{bull_str:>9} {bear_str:>9} {corr_str:>7}")
        _p("")

        # -- By regime --
        _p("-" * 64)
        _p("  ACCURACY BY VIX REGIME")
        _p("-" * 64)
        by_regime = results.get("by_regime", {})
        for regime in ["low", "normal", "elevated", "high", "extreme"]:
            r = by_regime.get(regime, {})
            if r:
                _p(f"  {regime:<12} {r['accuracy']:.1%}  (n={r['count']})")
        _p("")

        # -- By confidence --
        _p("-" * 64)
        _p("  ACCURACY BY CONFIDENCE BUCKET")
        _p("-" * 64)
        _p(f"  {'Bucket':<12} {'Accuracy':>9} {'Count':>8}")
        by_conf = results.get("by_confidence", {})
        for bucket in ["very_low", "low", "medium", "high", "very_high"]:
            b = by_conf.get(bucket, {})
            if b:
                _p(f"  {bucket:<12} {b['accuracy']:.1%} {b['count']:>8}")
        _p("")

        # -- Signal correlations & optimal weights --
        _p("-" * 64)
        _p("  SIGNAL CORRELATIONS & OPTIMAL WEIGHTS")
        _p("-" * 64)
        optimal = results.get("optimal_weights", {})
        _p(f"  {'Signal':<12} {'Correlation':>12} {'Optimal Wt':>12} {'Verdict':>12}")
        for sig in self.TESTABLE_SIGNALS:
            corr = correlations.get(sig)
            wt = optimal.get(sig)
            if corr is not None:
                if corr > 0.02:
                    verdict = "KEEP"
                elif corr < -0.02:
                    verdict = "INVERT?"
                else:
                    verdict = "WEAK"
                _p(f"  {sig:<12} {corr:>+12.4f} {wt:>12.4f} {verdict:>12}")
        _p("")

        # -- Contextual analysis --
        ctx = results.get("contextual", {})
        if ctx:
            # Gap classification
            gap_cls = ctx.get("gap_classification", {})
            if any(v for v in gap_cls.values() if v):
                _p("-" * 64)
                _p("  GAP CLASSIFICATION (Finnhub earnings calendar)")
                _p("-" * 64)
                _p(f"  {'Type':<22} {'Accuracy':>9} {'Count':>8}")
                for label, display in [("earnings_gap", "Earnings gap"),
                                       ("non_earnings_gap", "Non-earnings gap"),
                                       ("no_significant_gap", "No gap (<0.5%)")]:
                    g = gap_cls.get(label)
                    if g:
                        _p(f"  {display:<22} {g['accuracy']:.1%} {g['count']:>8}")
                gap_corr = ctx.get("gap_correlation_by_type", {})
                if gap_corr:
                    _p("")
                    _p("  Gap signal correlation by type:")
                    for gtype, corr_val in gap_corr.items():
                        _p(f"    {gtype}: r={corr_val:+.4f}")
                _p("")

            # Macro event impact
            macro = ctx.get("macro_event", {})
            if macro.get("with_event") or macro.get("without_event"):
                _p("-" * 64)
                _p("  MACRO EVENT IMPACT (Finnhub economic calendar)")
                _p("-" * 64)
                we = macro.get("with_event")
                wo = macro.get("without_event")
                if we:
                    _p(f"  During high-impact events:  {we['accuracy']:.1%}  (n={we['count']})")
                if wo:
                    _p(f"  Calm days (no event):       {wo['accuracy']:.1%}  (n={wo['count']})")
                avg_e = macro.get("avg_abs_return_event")
                avg_c = macro.get("avg_abs_return_calm")
                if avg_e is not None and avg_c is not None:
                    _p(f"  Avg |move| with event: {avg_e:.3f}%  vs calm: {avg_c:.3f}%")
                _p("")

            # Yield curve
            by_yield = ctx.get("by_yield_curve", {})
            if by_yield:
                _p("-" * 64)
                _p("  ACCURACY BY YIELD CURVE STATE (FRED)")
                _p("-" * 64)
                for state in ["inverted", "flat", "slightly_positive", "normal", "steep"]:
                    y = by_yield.get(state)
                    if y:
                        _p(f"  {state:<12} {y['accuracy']:.1%}  (n={y['count']})")
                _p("")

            # Credit spread
            by_credit = ctx.get("by_credit_spread", {})
            if by_credit:
                _p("-" * 64)
                _p("  ACCURACY BY CREDIT SPREAD (FRED HY OAS)")
                _p("-" * 64)
                for state in ["tight", "normal", "wide", "stressed"]:
                    cs = by_credit.get(state)
                    if cs:
                        _p(f"  {state:<12} {cs['accuracy']:.1%}  (n={cs['count']})")
                _p("")

            # Financial conditions
            by_nfci = ctx.get("by_financial_conditions", {})
            if by_nfci:
                _p("-" * 64)
                _p("  ACCURACY BY FINANCIAL CONDITIONS (FRED NFCI)")
                _p("-" * 64)
                for state in ["loose", "normal", "tightening", "tight"]:
                    fc = by_nfci.get(state)
                    if fc:
                        _p(f"  {state:<12} {fc['accuracy']:.1%}  (n={fc['count']})")
                _p("")

            # Signal-macro interaction
            smi = ctx.get("signal_macro_interaction", {})
            if smi:
                _p("-" * 64)
                _p("  SIGNAL ACCURACY: MACRO EVENTS vs CALM DAYS")
                _p("-" * 64)
                _p(f"  {'Signal':<12} {'Event':>9} {'Calm':>9} {'Delta':>9}")
                for sig in self.TESTABLE_SIGNALS:
                    s = smi.get(sig)
                    if s:
                        _p(f"  {sig:<12} {s['macro_event_accuracy']:.1%}"
                           f" {s['calm_accuracy']:.1%}"
                           f" {s['delta']:>+8.1%}")
                _p("")

        # -- Actionable takeaways --
        _p("-" * 64)
        _p("  ACTIONABLE TAKEAWAYS")
        _p("-" * 64)

        harmful = [s for s in self.TESTABLE_SIGNALS
                   if correlations.get(s, 0) < -0.01]
        helpful = sorted(
            [(s, correlations.get(s, 0)) for s in self.TESTABLE_SIGNALS
             if correlations.get(s, 0) > 0.01],
            key=lambda x: -x[1],
        )

        if helpful:
            best = helpful[0]
            _p(f"  Best signal: {best[0]} (r={best[1]:+.4f})")
        if harmful:
            _p(f"  Harmful signals (negative correlation): {', '.join(harmful)}")
            _p("  -> Consider reducing weight or inverting logic")
        if acc < 0.52:
            _p("  WARNING: Overall accuracy near coin-flip — model has no edge")
        elif acc >= 0.55:
            _p("  Model shows meaningful directional edge")

        # Contextual takeaways
        if ctx:
            gap_corr = ctx.get("gap_correlation_by_type", {})
            if gap_corr:
                eg = gap_corr.get("earnings")
                neg = gap_corr.get("non_earnings")
                if eg is not None and neg is not None:
                    if (eg > 0) != (neg > 0):
                        _p(f"  GAP INSIGHT: Earnings gaps (r={eg:+.4f}) and "
                           f"non-earnings gaps (r={neg:+.4f}) have OPPOSITE behavior")
                        _p("  -> Gap signal should be context-aware (earnings vs liquidity)")

            macro = ctx.get("macro_event", {})
            we = macro.get("with_event")
            wo = macro.get("without_event")
            if we and wo:
                delta = we["accuracy"] - wo["accuracy"]
                if abs(delta) > 0.03:
                    direction = "better" if delta > 0 else "worse"
                    _p(f"  MACRO INSIGHT: Model is {abs(delta):.1%} {direction} during "
                       f"high-impact events")

        _p("")
        _p("  Note: Technical signals + macro context tested. Options-derived")
        _p("  signals (P/C ratio, IV, flow) cannot be backtested without")
        _p("  historical option chain data.")
        _p("=" * 64)

        report = "\n".join(lines)
        print(report)
        return report

    # ------------------------------------------------------------------
    #  Core backtest logic
    # ------------------------------------------------------------------

    def _fetch_vix_history(self) -> pd.DataFrame:
        """Fetch VIX history for regime classification."""
        months = max(self.lookback_weeks // 4 + 3, 15)
        vix = yf.Ticker("^VIX")
        hist = vix.history(period=f"{months}mo")
        logger.info("Fetched %d days of VIX history", len(hist))
        return hist

    def _backtest_ticker(self, ticker: str, vix_hist: pd.DataFrame) -> list:
        """Backtest a single ticker across all historical Wednesdays."""
        months = max(self.lookback_weeks // 4 + 3, 15)
        tk = yf.Ticker(ticker)
        hist = tk.history(period=f"{months}mo")

        if len(hist) < 60:
            return []

        indicators = self._compute_indicator_series(hist)
        results = []

        cutoff = datetime.now() - timedelta(weeks=self.lookback_weeks)

        for i in range(50, len(hist)):
            dt = hist.index[i]

            # Only Wednesdays within lookback window
            if dt.weekday() != 2:
                continue
            if dt.tz_localize(None) < cutoff:
                continue

            # Find the corresponding Friday (typically index i+2)
            fri_idx = None
            for offset in range(1, 5):
                check = i + offset
                if check < len(hist) and hist.index[check].weekday() == 4:
                    fri_idx = check
                    break

            if fri_idx is None or fri_idx >= len(hist):
                continue

            signals = self._extract_signals(indicators, i)
            if signals is None:
                continue

            regime = self._classify_regime(vix_hist, dt)
            prediction = self._predict_direction(signals, regime)

            wed_close = float(hist["Close"].iloc[i])
            fri_close = float(hist["Close"].iloc[fri_idx])
            actual_return = (fri_close - wed_close) / wed_close
            actual_direction = "call" if actual_return >= 0 else "put"

            date_str = dt.strftime("%Y-%m-%d")
            context = self._get_context(ticker, date_str, signals)

            results.append({
                "ticker": ticker,
                "date": date_str,
                "wed_close": round(wed_close, 2),
                "fri_close": round(fri_close, 2),
                "actual_return_pct": round(actual_return * 100, 3),
                "actual_direction": actual_direction,
                "predicted_direction": prediction["direction"],
                "confidence": prediction["confidence"],
                "correct": prediction["direction"] == actual_direction,
                "regime": regime,
                "signal_votes": prediction["signal_votes"],
                "context": context,
            })

        return results

    # ------------------------------------------------------------------
    #  Indicator computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_indicator_series(hist: pd.DataFrame) -> dict:
        """Compute all technical indicators as full time series."""
        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]
        open_price = hist["Open"]

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # MACD(12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # Bollinger Bands(20, 2)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_range = bb_upper - bb_lower
        bb_pct = (close - bb_lower) / bb_range.replace(0, np.nan)

        # ATR(14) — for filtering, not direction
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = atr / close * 100

        # Volume ratio
        vol_sma20 = volume.rolling(20).mean()
        vol_ratio = volume / vol_sma20.replace(0, np.nan)

        # Price vs SMA20
        price_vs_sma20 = (close - sma20) / sma20.replace(0, np.nan) * 100

        # Gap %  (open vs previous close)
        gap_pct = (open_price - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100

        return {
            "close": close,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "bb_pct": bb_pct,
            "atr_pct": atr_pct,
            "vol_ratio": vol_ratio,
            "price_vs_sma20": price_vs_sma20,
            "gap_pct": gap_pct,
        }

    @staticmethod
    def _extract_signals(indicators: dict, idx: int) -> "dict | None":
        """Extract indicator values at index *idx*, return None if NaN."""
        vals = {}
        for key in ["rsi", "macd_hist", "bb_pct", "atr_pct",
                     "vol_ratio", "price_vs_sma20", "gap_pct"]:
            v = indicators[key].iloc[idx]
            if pd.isna(v):
                return None
            vals[key] = float(v)
        return vals

    # ------------------------------------------------------------------
    #  Direction prediction (technical signals only)
    # ------------------------------------------------------------------

    # Gap weight by VIX regime — mirrors scoring.py GAP_WEIGHT_BY_REGIME
    GAP_WEIGHT_BY_REGIME = {
        "low": 0.3,
        "normal": 0.5,
        "elevated": 1.0,
        "high": 1.2,
        "extreme": 1.2,
    }

    # Confidence calibration — mirrors scoring.py CONFIDENCE_CALIBRATION
    CONFIDENCE_CALIBRATION = {
        (0.0, 0.2): 0.35,
        (0.2, 0.4): 0.55,
        (0.4, 0.6): 0.55,
        (0.6, 0.8): 0.60,
        (0.8, 1.01): 0.40,
    }

    @classmethod
    def _predict_direction(cls, signals: dict, regime: str = "normal") -> dict:
        """Mirror determine_direction() using only price-derived signals.

        Each signal casts a weighted vote.  Gap weight is regime-aware
        and confidence is empirically calibrated — matching scoring.py.
        """
        bullish = 0.0
        bearish = 0.0
        votes = {}

        # RSI — same thresholds as scoring.py
        rsi = signals["rsi"]
        if rsi < 40:
            bearish += 1
            votes["rsi"] = -1
        elif rsi > 60:
            bullish += 1
            votes["rsi"] = 1
        else:
            votes["rsi"] = 0

        # MACD histogram direction
        hist = signals["macd_hist"]
        if hist > 0:
            bullish += 1
            votes["macd"] = 1
        elif hist < 0:
            bearish += 1
            votes["macd"] = -1
        else:
            votes["macd"] = 0

        # Price vs SMA20
        sma_pct = signals["price_vs_sma20"]
        if sma_pct < 0:
            bearish += 1
            votes["sma20"] = -1
        elif sma_pct > 0:
            bullish += 1
            votes["sma20"] = 1
        else:
            votes["sma20"] = 0

        # Gap % — context-aware weight based on VIX regime
        gap_weight = cls.GAP_WEIGHT_BY_REGIME.get(regime, 0.5)
        gap = signals["gap_pct"]
        if gap >= 0.5:
            bullish += gap_weight
            votes["gap"] = 1
        elif gap <= -0.5:
            bearish += gap_weight
            votes["gap"] = -1
        else:
            votes["gap"] = 0

        # Volume confirms prevailing direction
        vol_ratio = signals["vol_ratio"]
        if vol_ratio > 1.5:
            if sma_pct > 0:
                bullish += 0.5
                votes["volume"] = 1
            elif sma_pct < 0:
                bearish += 0.5
                votes["volume"] = -1
            else:
                votes["volume"] = 0
        else:
            votes["volume"] = 0

        # BB extremes — mean reversion bias
        bb = signals["bb_pct"]
        if bb < 0:
            bullish += 0.5  # below lower band -> oversold bounce
            votes["bb"] = 1
        elif bb > 1:
            bearish += 0.5  # above upper band -> overbought reversal
            votes["bb"] = -1
        else:
            votes["bb"] = 0

        total = bullish + bearish
        if total == 0:
            return {"direction": "call", "confidence": 0.0, "signal_votes": votes}

        raw_confidence = abs(bullish - bearish) / total
        direction = "call" if bullish >= bearish else "put"

        # Apply empirical confidence calibration
        calibrated = raw_confidence
        for (lo, hi), cal_val in cls.CONFIDENCE_CALIBRATION.items():
            if lo <= raw_confidence < hi:
                calibrated = cal_val
                break

        return {
            "direction": direction,
            "confidence": round(calibrated, 3),
            "raw_confidence": round(raw_confidence, 3),
            "signal_votes": votes,
        }

    # ------------------------------------------------------------------
    #  VIX regime classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_regime(vix_hist: pd.DataFrame, date) -> str:
        """Return the VIX regime for the given date."""
        try:
            # Find nearest date on or before
            mask = vix_hist.index <= date
            if not mask.any():
                return "normal"
            vix = float(vix_hist.loc[mask, "Close"].iloc[-1])
            if vix < 15:
                return "low"
            if vix < 20:
                return "normal"
            if vix < 25:
                return "elevated"
            if vix < 35:
                return "high"
            return "extreme"
        except Exception:
            return "normal"

    # ------------------------------------------------------------------
    #  External data fetching (FRED + Finnhub)
    # ------------------------------------------------------------------

    def _fetch_fred_macro_history(self):
        """Fetch historical FRED macro series for the full lookback period."""
        if not config.has_fred():
            logger.info("FRED API key not configured; skipping macro data")
            return

        try:
            from fredapi import Fred
            fred = Fred(api_key=config.fred_api_key)
        except Exception as exc:
            logger.warning("Failed to initialize FRED client: %s", exc)
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.lookback_weeks + 4)

        for series_id, description in self.FRED_SERIES.items():
            try:
                data = fred.get_series(series_id, start_date, end_date)
                if data is not None and not data.empty:
                    self._fred_data[series_id] = data.dropna()
                    logger.info("FRED %s: %d observations loaded (%s)",
                                series_id, len(self._fred_data[series_id]), description)
                else:
                    logger.warning("FRED %s returned empty", series_id)
            except Exception as exc:
                logger.warning("Failed to fetch FRED %s: %s", series_id, exc)
            time.sleep(0.2)

    def _fetch_earnings_calendar(self):
        """Fetch Finnhub earnings calendar for all tickers across lookback period.

        Stores {ticker: set of earnings date strings} for gap classification.
        """
        if not config.has_finnhub():
            logger.info("Finnhub API key not configured; skipping earnings calendar")
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.lookback_weeks + 1)

        try:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "token": config.finnhub_api_key,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            earnings = data.get("earningsCalendar", [])
            for item in earnings:
                sym = item.get("symbol", "")
                date = item.get("date", "")
                if sym and date:
                    if sym not in self._earnings_dates:
                        self._earnings_dates[sym] = set()
                    self._earnings_dates[sym].add(date)

            logger.info("Finnhub earnings calendar: %d events across %d tickers",
                        len(earnings), len(self._earnings_dates))

        except Exception as exc:
            logger.warning("Failed to fetch Finnhub earnings calendar: %s", exc)

    def _fetch_economic_events(self):
        """Fetch Finnhub economic calendar for the lookback period.

        Stores {date_str: [event_dicts]} for high-impact event flagging.
        """
        if not config.has_finnhub():
            logger.info("Finnhub API key not configured; skipping economic calendar")
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.lookback_weeks + 1)

        # Finnhub economic calendar has a max range, so chunk by quarter
        current = start_date
        all_events = []
        while current < end_date:
            chunk_end = min(current + timedelta(days=90), end_date)
            try:
                url = "https://finnhub.io/api/v1/calendar/economic"
                params = {
                    "from": current.strftime("%Y-%m-%d"),
                    "to": chunk_end.strftime("%Y-%m-%d"),
                    "token": config.finnhub_api_key,
                }
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                events = data.get("economicCalendar", [])
                all_events.extend(events)
            except Exception as exc:
                logger.warning("Failed to fetch economic events for %s to %s: %s",
                               current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), exc)
            current = chunk_end + timedelta(days=1)
            time.sleep(0.3)

        # Known high-impact event keywords
        high_impact_keywords = [
            "fomc", "cpi", "ppi", "nonfarm", "non-farm", "nfp", "pce",
            "gdp", "jolts", "retail sales", "jobless claims",
        ]

        for item in all_events:
            event_name = item.get("event", "")
            date = item.get("date", "")
            impact = item.get("impact", 1)
            if isinstance(impact, (int, float)):
                is_high = impact >= 3
            else:
                is_high = str(impact).lower() == "high"

            # Override impact for known high-impact events
            name_lower = event_name.lower()
            for kw in high_impact_keywords:
                if kw in name_lower:
                    is_high = True
                    break

            if date and is_high:
                if date not in self._economic_events:
                    self._economic_events[date] = []
                self._economic_events[date].append({
                    "event": event_name,
                    "impact": "high",
                })

        logger.info("Economic calendar: %d high-impact event dates loaded",
                    len(self._economic_events))

    # ------------------------------------------------------------------
    #  Context enrichment
    # ------------------------------------------------------------------

    def _get_context(self, ticker: str, date_str: str, signals: dict) -> dict:
        """Build contextual features for a single Wednesday prediction."""
        context = {}

        # --- Earnings gap classification ---
        # Check if ticker had earnings within 2 days before this Wednesday
        is_earnings_gap = False
        if ticker in self._earnings_dates:
            try:
                wed = datetime.strptime(date_str, "%Y-%m-%d")
                for offset in range(0, 3):  # Same day, day before, 2 days before
                    check = (wed - timedelta(days=offset)).strftime("%Y-%m-%d")
                    if check in self._earnings_dates[ticker]:
                        is_earnings_gap = True
                        break
            except ValueError:
                pass
        context["earnings_gap"] = is_earnings_gap

        # Classify gap type using earnings context
        gap_pct = signals.get("gap_pct", 0)
        if abs(gap_pct) >= 0.5:
            context["gap_type"] = "earnings" if is_earnings_gap else "non_earnings"
        else:
            context["gap_type"] = "none"

        # --- Macro event in Wed-Fri window ---
        has_macro_event = False
        try:
            wed = datetime.strptime(date_str, "%Y-%m-%d")
            for offset in range(0, 3):  # Wed, Thu, Fri
                check = (wed + timedelta(days=offset)).strftime("%Y-%m-%d")
                if check in self._economic_events:
                    has_macro_event = True
                    break
        except ValueError:
            pass
        context["macro_event"] = has_macro_event

        # --- Yield curve state ---
        context["yield_curve"] = self._get_yield_curve_state(date_str)

        # --- Credit spread level ---
        context["credit_spread"] = self._get_credit_spread_state(date_str)

        # --- Financial conditions ---
        context["financial_conditions"] = self._get_nfci_state(date_str)

        return context

    def _get_fred_value_at(self, series_id: str, date_str: str) -> "float | None":
        """Get the most recent FRED observation on or before the given date."""
        series = self._fred_data.get(series_id)
        if series is None or series.empty:
            return None
        try:
            target = pd.Timestamp(date_str)
            mask = series.index <= target
            if not mask.any():
                return None
            return float(series.loc[mask].iloc[-1])
        except Exception:
            return None

    def _get_yield_curve_state(self, date_str: str) -> str:
        """Classify yield curve state from 10Y-2Y spread.

        Uses finer-grained thresholds that distinguish meaningful
        shifts within the normal range (which is where the curve
        has spent most of 2025-2026).
        """
        spread = self._get_fred_value_at("T10Y2Y", date_str)
        if spread is None:
            return "unknown"
        if spread < -0.1:
            return "inverted"
        if spread < 0.15:
            return "flat"
        if spread < 0.40:
            return "slightly_positive"
        if spread < 0.65:
            return "normal"
        return "steep"

    def _get_credit_spread_state(self, date_str: str) -> str:
        """Classify high-yield credit spread as tight/normal/wide/stressed."""
        hy = self._get_fred_value_at("BAMLH0A0HYM2", date_str)
        if hy is None:
            return "unknown"
        if hy < 3.0:
            return "tight"
        if hy < 4.5:
            return "normal"
        if hy < 6.0:
            return "wide"
        return "stressed"

    def _get_nfci_state(self, date_str: str) -> str:
        """Classify Chicago Fed financial conditions as loose/normal/tight."""
        nfci = self._get_fred_value_at("NFCI", date_str)
        if nfci is None:
            return "unknown"
        # NFCI: negative = loose, positive = tight, 0 = average
        if nfci < -0.5:
            return "loose"
        if nfci < 0.0:
            return "normal"
        if nfci < 0.5:
            return "tightening"
        return "tight"

    # ------------------------------------------------------------------
    #  Results compilation
    # ------------------------------------------------------------------

    def _compile_results(self, all_results: list) -> dict:
        """Aggregate individual predictions into summary statistics."""
        if not all_results:
            return {"error": "No backtest results generated"}

        total = len(all_results)
        correct = sum(1 for r in all_results if r["correct"])
        overall_accuracy = correct / total

        # -- Per-signal accuracy --
        per_signal = {}
        for sig in self.TESTABLE_SIGNALS:
            voted_bull = [r for r in all_results if r["signal_votes"].get(sig) == 1]
            voted_bear = [r for r in all_results if r["signal_votes"].get(sig) == -1]
            neutral = total - len(voted_bull) - len(voted_bear)

            bull_correct = sum(1 for r in voted_bull if r["actual_direction"] == "call")
            bear_correct = sum(1 for r in voted_bear if r["actual_direction"] == "put")
            active = len(voted_bull) + len(voted_bear)
            sig_correct = bull_correct + bear_correct

            per_signal[sig] = {
                "times_active": active,
                "times_neutral": neutral,
                "accuracy": round(sig_correct / active, 4) if active > 0 else None,
                "bullish_calls": len(voted_bull),
                "bullish_accuracy": round(bull_correct / len(voted_bull), 4) if voted_bull else None,
                "bearish_calls": len(voted_bear),
                "bearish_accuracy": round(bear_correct / len(voted_bear), 4) if voted_bear else None,
            }

        # -- By VIX regime --
        by_regime = {}
        for regime in ["low", "normal", "elevated", "high", "extreme"]:
            subset = [r for r in all_results if r["regime"] == regime]
            if subset:
                rc = sum(1 for r in subset if r["correct"])
                by_regime[regime] = {
                    "count": len(subset),
                    "accuracy": round(rc / len(subset), 4),
                    "avg_return_pct": round(
                        np.mean([r["actual_return_pct"] for r in subset]), 3
                    ),
                }

        # -- By confidence bucket --
        by_confidence = {}
        buckets = [
            (0.0, 0.2, "very_low"),
            (0.2, 0.4, "low"),
            (0.4, 0.6, "medium"),
            (0.6, 0.8, "high"),
            (0.8, 1.01, "very_high"),
        ]
        for lo, hi, label in buckets:
            subset = [r for r in all_results if lo <= r["confidence"] < hi]
            if subset:
                bc = sum(1 for r in subset if r["correct"])
                by_confidence[label] = {
                    "count": len(subset),
                    "accuracy": round(bc / len(subset), 4),
                }

        # -- Signal correlations (Pearson r of vote vs actual return) --
        signal_correlations = {}
        for sig in self.TESTABLE_SIGNALS:
            votes = np.array([r["signal_votes"].get(sig, 0) for r in all_results])
            returns = np.array([r["actual_return_pct"] for r in all_results])
            # Only correlate where the signal was active
            active_mask = votes != 0
            if active_mask.sum() > 30:
                r_val = float(np.corrcoef(votes[active_mask], returns[active_mask])[0, 1])
                signal_correlations[sig] = round(r_val, 4)

        # -- Derive optimal weights --
        optimal_weights = self._derive_optimal_weights(signal_correlations)

        # -- Weekly breakdown --
        weekly = {}
        for r in all_results:
            wk = r["date"][:10]
            if wk not in weekly:
                weekly[wk] = {"total": 0, "correct": 0}
            weekly[wk]["total"] += 1
            if r["correct"]:
                weekly[wk]["correct"] += 1

        weekly_summary = []
        for date_str in sorted(weekly.keys()):
            w = weekly[date_str]
            weekly_summary.append({
                "week": date_str,
                "predictions": w["total"],
                "correct": w["correct"],
                "accuracy": round(w["correct"] / w["total"], 4) if w["total"] else 0,
            })

        # -- Contextual analysis (FRED + Finnhub enrichment) --
        contextual = self._compile_contextual_analysis(all_results)

        return {
            "overall_accuracy": round(overall_accuracy, 4),
            "total_predictions": total,
            "correct_predictions": correct,
            "per_signal": per_signal,
            "by_regime": by_regime,
            "by_confidence": by_confidence,
            "signal_correlations": signal_correlations,
            "optimal_weights": optimal_weights,
            "weekly_summary": weekly_summary,
            "contextual": contextual,
        }

    def _compile_contextual_analysis(self, all_results: list) -> dict:
        """Break down accuracy by contextual features from FRED/Finnhub data."""
        contextual = {}

        def _accuracy_of(subset):
            if not subset:
                return None
            c = sum(1 for r in subset if r["correct"])
            return {"count": len(subset), "accuracy": round(c / len(subset), 4)}

        # -- Gap classification: earnings vs non-earnings --
        earnings_gaps = [r for r in all_results
                         if r.get("context", {}).get("gap_type") == "earnings"]
        non_earnings_gaps = [r for r in all_results
                             if r.get("context", {}).get("gap_type") == "non_earnings"]
        no_gap = [r for r in all_results
                  if r.get("context", {}).get("gap_type") == "none"]

        contextual["gap_classification"] = {
            "earnings_gap": _accuracy_of(earnings_gaps),
            "non_earnings_gap": _accuracy_of(non_earnings_gaps),
            "no_significant_gap": _accuracy_of(no_gap),
        }

        # Gap signal correlation split by type
        gap_corr = {}
        for label, subset in [("earnings", earnings_gaps), ("non_earnings", non_earnings_gaps)]:
            if len(subset) > 20:
                votes = np.array([r["signal_votes"].get("gap", 0) for r in subset])
                returns = np.array([r["actual_return_pct"] for r in subset])
                active = votes != 0
                if active.sum() > 15:
                    r_val = float(np.corrcoef(votes[active], returns[active])[0, 1])
                    gap_corr[label] = round(r_val, 4)
        contextual["gap_correlation_by_type"] = gap_corr

        # -- Macro event impact --
        macro_event = [r for r in all_results if r.get("context", {}).get("macro_event")]
        no_macro = [r for r in all_results if not r.get("context", {}).get("macro_event")]
        contextual["macro_event"] = {
            "with_event": _accuracy_of(macro_event),
            "without_event": _accuracy_of(no_macro),
        }
        # Average absolute return during macro events vs calm days
        if macro_event:
            contextual["macro_event"]["avg_abs_return_event"] = round(
                np.mean([abs(r["actual_return_pct"]) for r in macro_event]), 3)
        if no_macro:
            contextual["macro_event"]["avg_abs_return_calm"] = round(
                np.mean([abs(r["actual_return_pct"]) for r in no_macro]), 3)

        # -- By yield curve state --
        by_yield = {}
        for state in ["inverted", "flat", "normal", "steep"]:
            subset = [r for r in all_results
                      if r.get("context", {}).get("yield_curve") == state]
            result = _accuracy_of(subset)
            if result:
                by_yield[state] = result
        contextual["by_yield_curve"] = by_yield

        # -- By credit spread --
        by_credit = {}
        for state in ["tight", "normal", "wide", "stressed"]:
            subset = [r for r in all_results
                      if r.get("context", {}).get("credit_spread") == state]
            result = _accuracy_of(subset)
            if result:
                by_credit[state] = result
        contextual["by_credit_spread"] = by_credit

        # -- By financial conditions --
        by_nfci = {}
        for state in ["loose", "normal", "tightening", "tight"]:
            subset = [r for r in all_results
                      if r.get("context", {}).get("financial_conditions") == state]
            result = _accuracy_of(subset)
            if result:
                by_nfci[state] = result
        contextual["by_financial_conditions"] = by_nfci

        # -- Cross-analysis: signal accuracy during macro events vs calm --
        signal_macro_interaction = {}
        for sig in self.TESTABLE_SIGNALS:
            if macro_event and no_macro:
                active_event = [r for r in macro_event if r["signal_votes"].get(sig, 0) != 0]
                active_calm = [r for r in no_macro if r["signal_votes"].get(sig, 0) != 0]
                event_acc = _accuracy_of(active_event)
                calm_acc = _accuracy_of(active_calm)
                if event_acc and calm_acc:
                    signal_macro_interaction[sig] = {
                        "macro_event_accuracy": event_acc["accuracy"],
                        "calm_accuracy": calm_acc["accuracy"],
                        "delta": round(event_acc["accuracy"] - calm_acc["accuracy"], 4),
                    }
        contextual["signal_macro_interaction"] = signal_macro_interaction

        return contextual

    @staticmethod
    def _derive_optimal_weights(correlations: dict) -> dict:
        """Derive signal weights proportional to |correlation|.

        Signals with negative correlation are flagged — their weight is
        still derived from |r| so they appear in the output, but the
        sign indicates they should be *inverted* not just down-weighted.
        """
        if not correlations:
            return {}

        abs_corrs = {k: abs(v) for k, v in correlations.items()}
        total = sum(abs_corrs.values())
        if total <= 0:
            return {}

        weights = {}
        for k in correlations:
            raw_weight = abs_corrs[k] / total
            # Prefix negative-correlation signals so it's obvious
            weights[k] = round(raw_weight, 4)

        return weights
