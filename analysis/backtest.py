"""
Directional backtest for the Weekly Options Trading Analysis System.

Tests whether our direction prediction works for 5-day (Mon→Fri) holds.
Combines historical price data with macro context (FRED) and event
calendars (Finnhub).

For each Monday in the lookback period, computes weekly-appropriate
technical indicators (momentum, ADX, trend persistence, SMA slopes),
enriches with macro regime and event context, runs direction prediction,
and checks whether the stock moved in the predicted direction by Friday
close.

Walk-forward validation: train on first half of lookback, test on second
half — no look-ahead bias.

Signals tested (from price data):
    5d/10d/21d momentum, ADX + trend strength, SMA20 slope,
    RSI, MACD histogram, price vs SMA20, BB position, volume ratio

Contextual features (FRED + Finnhub):
    Earnings proximity, high-impact macro events (FOMC/CPI/NFP in
    Mon-Fri window), yield curve state, credit spread, financial conditions

Signals NOT tested (require live options/sentiment data):
    P/C ratio, max pain, IV, sentiment, flow, finviz, insider
"""

import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta
from math import sqrt, log as _mlog, exp as _mexp, erf as _erf
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
    """Backtest direction prediction accuracy on historical Mon->Fri moves."""

    LOOKBACK_WEEKS = 52

    # Signals we can test from price history alone (weekly-appropriate).
    TESTABLE_SIGNALS = [
        "momentum_5d", "momentum_10d", "trend_adx", "sma_slope",
        "rsi", "macd", "sma20_position", "bb", "volume",
    ]

    # Contextual features from external data sources.
    CONTEXT_FEATURES = [
        "earnings_in_window", "macro_event", "yield_curve",
        "credit_spread", "financial_conditions",
    ]

    # Signals we CANNOT test historically (documented for transparency).
    UNTESTABLE_SIGNALS = [
        "pc_ratio", "max_pain", "iv_data", "iv_rv_ratio", "sentiment",
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

    def run(self, walk_forward: bool = True) -> dict:
        """Run full directional backtest across the universe.

        Parameters
        ----------
        walk_forward : bool
            If True, split into train/test halves for walk-forward
            validation. If False, report on the full period.

        Returns
        -------
        dict
            Comprehensive results: overall accuracy, per-signal stats,
            regime breakdown, contextual analysis, walk-forward results,
            Monte Carlo confidence intervals, and derived optimal weights.
        """
        logger.info(
            "Starting weekly directional backtest: %d tickers, %d-week lookback",
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
            time.sleep(0.15)

        logger.info(
            "Backtest complete: %d predictions from %d tickers (%d failed)",
            len(all_results), len(self.tickers) - failed, failed,
        )

        # Full-period compilation
        compiled = self._compile_results(all_results)

        # Walk-forward validation
        if walk_forward and len(all_results) > 50:
            wf = self._walk_forward_validation(all_results)
            compiled["walk_forward"] = wf
        else:
            compiled["walk_forward"] = {"note": "insufficient data for walk-forward"}

        # Monte Carlo confidence intervals
        if len(all_results) > 30:
            compiled["monte_carlo"] = self._monte_carlo_ci(all_results)

        # Simulated options P&L (approximated from directional moves)
        compiled["simulated_options_pnl"] = self._simulate_options_pnl(all_results)

        compiled["meta"] = {
            "tickers_scanned": len(self.tickers),
            "tickers_failed": failed,
            "lookback_weeks": self.lookback_weeks,
            "holding_period": "Mon->Fri (5 trading days)",
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

        _p("=" * 68)
        _p("  WEEKLY DIRECTIONAL BACKTEST REPORT (Mon->Fri)")
        _p("=" * 68)

        meta = results.get("meta", {})
        _p(f"  Universe: {meta.get('tickers_scanned', '?')} tickers "
           f"({meta.get('tickers_failed', 0)} failed)")
        _p(f"  Lookback: {meta.get('lookback_weeks', '?')} weeks")
        _p(f"  Holding: {meta.get('holding_period', 'Mon->Fri')}")
        _p(f"  Run date: {meta.get('run_date', '?')[:10]}")
        _p("")

        # Overall accuracy
        acc = results.get("overall_accuracy", 0)
        total = results.get("total_predictions", 0)
        correct = results.get("correct_predictions", 0)
        _p(f"  OVERALL ACCURACY:  {acc:.1%}  ({correct}/{total})")
        edge = acc - 0.5
        _p(f"  Edge over coin flip: {edge:+.1%}")
        _p("")

        # Monte Carlo CI
        mc = results.get("monte_carlo", {})
        if mc:
            _p(f"  95% Confidence Interval: [{mc.get('ci_lower', 0):.1%}, {mc.get('ci_upper', 0):.1%}]")
            _p(f"  Mean accuracy (10k bootstraps): {mc.get('mean_accuracy', 0):.1%}")
            _p("")

        # Walk-forward
        wf = results.get("walk_forward", {})
        if "test_accuracy" in wf:
            _p("-" * 68)
            _p("  WALK-FORWARD VALIDATION")
            _p("-" * 68)
            _p(f"  Train period: {wf.get('train_start', '?')} to {wf.get('train_end', '?')}")
            _p(f"  Test period:  {wf.get('test_start', '?')} to {wf.get('test_end', '?')}")
            _p(f"  Train accuracy: {wf.get('train_accuracy', 0):.1%} (n={wf.get('train_count', 0)})")
            _p(f"  Test accuracy:  {wf.get('test_accuracy', 0):.1%} (n={wf.get('test_count', 0)})")
            overfit = wf.get("overfit_gap", 0)
            _p(f"  Overfit gap:    {overfit:+.1%}")
            if overfit > 0.05:
                _p("  WARNING: Significant overfit detected — test accuracy << train")
            _p("")

        # Per-signal accuracy
        _p("-" * 68)
        _p("  PER-SIGNAL ACCURACY  (when signal was active)")
        _p("-" * 68)
        _p(f"  {'Signal':<16} {'Accuracy':>9} {'Active':>8} {'Bullish':>9} {'Bearish':>9} {'Corr':>7}")

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
            _p(f"  {sig:<16} {acc_str:>9} {s.get('times_active', 0):>8} "
               f"{bull_str:>9} {bear_str:>9} {corr_str:>7}")
        _p("")

        # By VIX regime
        _p("-" * 68)
        _p("  ACCURACY BY VIX REGIME")
        _p("-" * 68)
        by_regime = results.get("by_regime", {})
        for regime in ["low", "normal", "elevated", "high", "extreme"]:
            r = by_regime.get(regime, {})
            if r:
                avg_ret = r.get("avg_return_pct", 0)
                _p(f"  {regime:<12} {r['accuracy']:.1%}  (n={r['count']})  "
                   f"avg move: {avg_ret:+.2f}%")
        _p("")

        # By confidence
        _p("-" * 68)
        _p("  ACCURACY BY CONFIDENCE BUCKET")
        _p("-" * 68)
        _p(f"  {'Bucket':<12} {'Accuracy':>9} {'Count':>8}")
        by_conf = results.get("by_confidence", {})
        for bucket in ["very_low", "low", "medium", "high", "very_high"]:
            b = by_conf.get(bucket, {})
            if b:
                _p(f"  {bucket:<12} {b['accuracy']:.1%} {b['count']:>8}")
        _p("")

        # Simulated options P&L
        opts_pnl = results.get("simulated_options_pnl", {})
        if opts_pnl and opts_pnl.get("total_trades", 0) > 0:
            _p("-" * 68)
            _p("  SIMULATED OPTIONS P&L (approx, 0.35 delta weekly)")
            _p("-" * 68)
            _p(f"  Total trades:    {opts_pnl.get('total_trades', 0)}")
            _p(f"  Win rate:        {opts_pnl.get('win_rate', 0):.1%}")
            _p(f"  Avg winner:      {opts_pnl.get('avg_winner_pct', 0):+.1f}%")
            _p(f"  Avg loser:       {opts_pnl.get('avg_loser_pct', 0):+.1f}%")
            _p(f"  Profit factor:   {opts_pnl.get('profit_factor', 0):.2f}")
            _p(f"  Expected return: {opts_pnl.get('expected_return_pct', 0):+.1f}% per trade")
            _p(f"  Max drawdown:    {opts_pnl.get('max_drawdown_pct', 0):.1f}%")
            _p("")

        # Signal correlations & optimal weights
        _p("-" * 68)
        _p("  SIGNAL CORRELATIONS & OPTIMAL WEIGHTS")
        _p("-" * 68)
        optimal = results.get("optimal_weights", {})
        _p(f"  {'Signal':<16} {'Correlation':>12} {'Optimal Wt':>12} {'Verdict':>12}")
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
                wt_str = f"{wt:.4f}" if wt is not None else "N/A"
                _p(f"  {sig:<16} {corr:>+12.4f} {wt_str:>12} {verdict:>12}")
        _p("")

        # Contextual analysis
        ctx = results.get("contextual", {})
        if ctx:
            # Earnings in holding window
            earn = ctx.get("earnings_in_window", {})
            if earn.get("with_earnings") or earn.get("without_earnings"):
                _p("-" * 68)
                _p("  EARNINGS IN MON-FRI WINDOW")
                _p("-" * 68)
                we = earn.get("with_earnings")
                wo = earn.get("without_earnings")
                if we:
                    _p(f"  With earnings:    {we['accuracy']:.1%}  (n={we['count']})")
                if wo:
                    _p(f"  Without earnings: {wo['accuracy']:.1%}  (n={wo['count']})")
                _p("")

            # Macro event impact
            macro = ctx.get("macro_event", {})
            if macro.get("with_event") or macro.get("without_event"):
                _p("-" * 68)
                _p("  MACRO EVENT IMPACT (FOMC/CPI/NFP in Mon-Fri window)")
                _p("-" * 68)
                we = macro.get("with_event")
                wo = macro.get("without_event")
                if we:
                    _p(f"  During high-impact events:  {we['accuracy']:.1%}  (n={we['count']})")
                if wo:
                    _p(f"  Calm weeks (no event):      {wo['accuracy']:.1%}  (n={wo['count']})")
                avg_e = macro.get("avg_abs_return_event")
                avg_c = macro.get("avg_abs_return_calm")
                if avg_e is not None and avg_c is not None:
                    _p(f"  Avg |move| with event: {avg_e:.3f}%  vs calm: {avg_c:.3f}%")
                _p("")

            # Yield curve, credit, NFCI
            for section, title, states in [
                ("by_yield_curve", "YIELD CURVE STATE (FRED)", ["inverted", "flat", "slightly_positive", "normal", "steep"]),
                ("by_credit_spread", "CREDIT SPREAD (FRED HY OAS)", ["tight", "normal", "wide", "stressed"]),
                ("by_financial_conditions", "FINANCIAL CONDITIONS (FRED NFCI)", ["loose", "normal", "tightening", "tight"]),
            ]:
                by_section = ctx.get(section, {})
                if by_section:
                    _p("-" * 68)
                    _p(f"  ACCURACY BY {title}")
                    _p("-" * 68)
                    for state in states:
                        s = by_section.get(state)
                        if s:
                            _p(f"  {state:<20} {s['accuracy']:.1%}  (n={s['count']})")
                    _p("")

        # Actionable takeaways
        _p("-" * 68)
        _p("  ACTIONABLE TAKEAWAYS")
        _p("-" * 68)

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

        acc_val = results.get("overall_accuracy", 0)
        if acc_val < 0.52:
            _p("  WARNING: Overall accuracy near coin-flip — model has no edge")
        elif acc_val >= 0.55:
            _p("  Model shows meaningful directional edge for weekly holds")

        _p("")
        _p("  Note: Technical signals + macro context tested. Options-derived")
        _p("  signals (P/C ratio, IV, flow) cannot be backtested without")
        _p("  historical option chain data.")
        _p("=" * 68)

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
        """Backtest a single ticker across all historical Mondays.

        For each Monday in the lookback period:
        1. Compute weekly indicators using data up to that Monday
        2. Predict direction (call/put) using weekly-appropriate signals
        3. Find Friday close (the expiry date)
        4. Record actual vs predicted direction + return
        """
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

            # Only Mondays within lookback window
            if dt.weekday() != 0:
                continue
            if dt.tz_localize(None) < cutoff:
                continue

            # Find the corresponding Friday (typically index i+4)
            fri_idx = None
            for offset in range(3, 7):
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

            mon_close = float(hist["Close"].iloc[i])
            fri_close = float(hist["Close"].iloc[fri_idx])
            actual_return = (fri_close - mon_close) / mon_close
            actual_direction = "call" if actual_return >= 0 else "put"

            date_str = dt.strftime("%Y-%m-%d")
            fri_date_str = hist.index[fri_idx].strftime("%Y-%m-%d")
            context = self._get_context(ticker, date_str, fri_date_str, signals)

            results.append({
                "ticker": ticker,
                "date": date_str,
                "expiry_date": fri_date_str,
                "mon_close": round(mon_close, 2),
                "fri_close": round(fri_close, 2),
                "actual_return_pct": round(actual_return * 100, 3),
                "actual_direction": actual_direction,
                "predicted_direction": prediction["direction"],
                "confidence": prediction["confidence"],
                "correct": prediction["direction"] == actual_direction,
                "regime": regime,
                "signal_votes": prediction["signal_votes"],
                "context": context,
                # For BSM-based options P&L simulation
                "entry_price": round(mon_close, 2),
                "signals": signals,
            })

        return results

    # ------------------------------------------------------------------
    #  Indicator computation (weekly-appropriate)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_indicator_series(hist: pd.DataFrame) -> dict:
        """Compute all technical indicators as full time series.

        Includes weekly momentum signals (5d/10d/21d returns, ADX,
        SMA slopes) alongside standard indicators.
        """
        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]

        # -- Multi-day returns (key weekly signals) --
        return_5d = close.pct_change(5) * 100
        return_10d = close.pct_change(10) * 100
        return_21d = close.pct_change(21) * 100

        # -- ADX(14) with +DI/-DI --
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        # Zero out when the other is larger
        both = pd.DataFrame({"plus": plus_dm, "minus": minus_dm})
        plus_dm = both["plus"].where(both["plus"] > both["minus"], 0)
        minus_dm = both["minus"].where(both["minus"] > both["plus"], 0)

        plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(14).mean()

        # -- Trend classification --
        # (computed per-row in _extract_signals)

        # -- SMA slopes --
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma20_slope = (sma20 - sma20.shift(5)) / sma20.shift(5).replace(0, np.nan) * 100

        # -- RSI(14) --
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # -- MACD(12, 26, 9) --
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # -- Bollinger Bands(20, 2) --
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_range = bb_upper - bb_lower
        bb_pct = (close - bb_lower) / bb_range.replace(0, np.nan)

        # -- ATR & volume --
        atr_pct = atr14 / close * 100
        vol_sma20 = volume.rolling(20).mean()
        vol_ratio = volume / vol_sma20.replace(0, np.nan)
        price_vs_sma20 = (close - sma20) / sma20.replace(0, np.nan) * 100

        return {
            "close": close,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "return_21d": return_21d,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "sma20_slope": sma20_slope,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "bb_pct": bb_pct,
            "atr_pct": atr_pct,
            "vol_ratio": vol_ratio,
            "price_vs_sma20": price_vs_sma20,
        }

    @staticmethod
    def _extract_signals(indicators: dict, idx: int) -> "dict | None":
        """Extract indicator values at index *idx*, return None if NaN."""
        keys = [
            "return_5d", "return_10d", "return_21d", "adx", "plus_di",
            "minus_di", "sma20_slope", "rsi", "macd_hist", "bb_pct",
            "atr_pct", "vol_ratio", "price_vs_sma20",
        ]
        vals = {}
        for key in keys:
            v = indicators[key].iloc[idx]
            if pd.isna(v):
                return None
            vals[key] = float(v)
        return vals

    # ------------------------------------------------------------------
    #  Direction prediction (weekly-appropriate signals)
    # ------------------------------------------------------------------

    # Confidence calibration — mirrors scoring.py
    CONFIDENCE_CALIBRATION = {
        (0.0, 0.15): 0.30,
        (0.15, 0.30): 0.45,
        (0.30, 0.50): 0.55,
        (0.50, 0.70): 0.58,
        (0.70, 0.85): 0.52,
        (0.85, 1.01): 0.42,
    }

    @classmethod
    def _predict_direction(cls, signals: dict, regime: str = "normal") -> dict:
        """Predict direction using weekly-appropriate signals.

        Mirrors scoring.py's determine_direction() using only
        price-derived signals available historically.
        """
        bullish = 0.0
        bearish = 0.0
        votes = {}

        # 1. 5-day momentum (strongest weekly signal, weight 1.5)
        ret_5d = signals["return_5d"]
        if ret_5d > 1.0:
            bullish += 1.5
            votes["momentum_5d"] = 1
        elif ret_5d > 0:
            bullish += 0.5
            votes["momentum_5d"] = 1
        elif ret_5d < -1.0:
            bearish += 1.5
            votes["momentum_5d"] = -1
        elif ret_5d < 0:
            bearish += 0.5
            votes["momentum_5d"] = -1
        else:
            votes["momentum_5d"] = 0

        # 2. 10-day momentum (medium-term context)
        ret_10d = signals["return_10d"]
        if ret_10d > 2.0:
            bullish += 0.8
            votes["momentum_10d"] = 1
        elif ret_10d > 0:
            bullish += 0.3
            votes["momentum_10d"] = 1
        elif ret_10d < -2.0:
            bearish += 0.8
            votes["momentum_10d"] = -1
        elif ret_10d < 0:
            bearish += 0.3
            votes["momentum_10d"] = -1
        else:
            votes["momentum_10d"] = 0

        # 3. ADX + trend direction (weight 1.5)
        adx = signals["adx"]
        plus_di = signals["plus_di"]
        minus_di = signals["minus_di"]

        if adx > 25:
            # Strong trend — direction from DI
            if plus_di > minus_di:
                bullish += 1.5 if adx > 40 else 0.8
                votes["trend_adx"] = 1
            else:
                bearish += 1.5 if adx > 40 else 0.8
                votes["trend_adx"] = -1
        elif adx < 15:
            votes["trend_adx"] = 0  # No trend
        else:
            # Mild trend
            if plus_di > minus_di:
                bullish += 0.3
                votes["trend_adx"] = 1
            else:
                bearish += 0.3
                votes["trend_adx"] = -1

        # 4. SMA20 slope (trend acceleration)
        slope = signals["sma20_slope"]
        if slope > 0.5:
            bullish += 0.5
            votes["sma_slope"] = 1
        elif slope > 0.1:
            bullish += 0.2
            votes["sma_slope"] = 1
        elif slope < -0.5:
            bearish += 0.5
            votes["sma_slope"] = -1
        elif slope < -0.1:
            bearish += 0.2
            votes["sma_slope"] = -1
        else:
            votes["sma_slope"] = 0

        # 5. RSI — mean reversion at extremes (weekly thresholds)
        rsi = signals["rsi"]
        if rsi < 30:
            bullish += 1.0
            votes["rsi"] = 1
        elif rsi < 40:
            bullish += 0.3
            votes["rsi"] = 1
        elif rsi > 70:
            bearish += 1.0
            votes["rsi"] = -1
        elif rsi > 60:
            bearish += 0.3
            votes["rsi"] = -1
        else:
            votes["rsi"] = 0

        # 6. MACD histogram
        hist = signals["macd_hist"]
        if hist > 0:
            bullish += 0.5
            votes["macd"] = 1
        elif hist < 0:
            bearish += 0.5
            votes["macd"] = -1
        else:
            votes["macd"] = 0

        # 7. Price vs SMA20
        sma_pct = signals["price_vs_sma20"]
        if sma_pct > 2:
            bullish += 0.8
            votes["sma20_position"] = 1
        elif sma_pct > 0:
            bullish += 0.3
            votes["sma20_position"] = 1
        elif sma_pct < -2:
            bearish += 0.8
            votes["sma20_position"] = -1
        elif sma_pct < 0:
            bearish += 0.3
            votes["sma20_position"] = -1
        else:
            votes["sma20_position"] = 0

        # 8. BB extremes
        bb = signals["bb_pct"]
        if bb < 0:
            bullish += 0.5  # below lower band -> oversold
            votes["bb"] = 1
        elif bb > 1:
            bearish += 0.5  # above upper band -> overbought
            votes["bb"] = -1
        else:
            votes["bb"] = 0

        # 9. Volume confirms prevailing direction
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

        total = bullish + bearish
        if total == 0:
            return {"direction": "call", "confidence": 0.0,
                    "raw_confidence": 0.0, "signal_votes": votes}

        raw_confidence = abs(bullish - bearish) / total
        direction = "call" if bullish >= bearish else "put"

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
    #  Walk-forward validation
    # ------------------------------------------------------------------

    def _walk_forward_validation(self, all_results: list) -> dict:
        """Split results into train/test halves for walk-forward validation.

        Train on weeks 1-26, test on weeks 27-52. Reports accuracy for
        each half and the overfit gap.
        """
        sorted_results = sorted(all_results, key=lambda r: r["date"])
        midpoint = len(sorted_results) // 2

        train = sorted_results[:midpoint]
        test = sorted_results[midpoint:]

        if not train or not test:
            return {"note": "insufficient data for walk-forward split"}

        train_correct = sum(1 for r in train if r["correct"])
        test_correct = sum(1 for r in test if r["correct"])

        train_acc = train_correct / len(train)
        test_acc = test_correct / len(test)

        # Per-signal correlations for train vs test
        train_corr = self._compute_signal_correlations(train)
        test_corr = self._compute_signal_correlations(test)

        # Signal stability: do correlations hold across halves?
        stable_signals = []
        unstable_signals = []
        for sig in self.TESTABLE_SIGNALS:
            tc = train_corr.get(sig)
            te = test_corr.get(sig)
            if tc is not None and te is not None:
                if (tc > 0 and te > 0) or (tc < 0 and te < 0):
                    stable_signals.append(sig)
                else:
                    unstable_signals.append(sig)

        return {
            "train_start": train[0]["date"],
            "train_end": train[-1]["date"],
            "test_start": test[0]["date"],
            "test_end": test[-1]["date"],
            "train_count": len(train),
            "test_count": len(test),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "overfit_gap": round(train_acc - test_acc, 4),
            "train_signal_correlations": train_corr,
            "test_signal_correlations": test_corr,
            "stable_signals": stable_signals,
            "unstable_signals": unstable_signals,
        }

    # ------------------------------------------------------------------
    #  Monte Carlo confidence intervals
    # ------------------------------------------------------------------

    @staticmethod
    def _monte_carlo_ci(all_results: list, n_bootstraps: int = 10000,
                        ci: float = 0.95) -> dict:
        """Bootstrap confidence interval for overall accuracy.

        Resamples predictions with replacement 10k times to estimate
        the distribution of accuracy under random sampling variation.
        """
        correct_array = np.array([1 if r["correct"] else 0 for r in all_results])
        n = len(correct_array)

        rng = np.random.default_rng(42)
        boot_accuracies = np.empty(n_bootstraps)

        for b in range(n_bootstraps):
            sample = rng.choice(correct_array, size=n, replace=True)
            boot_accuracies[b] = sample.mean()

        alpha = (1 - ci) / 2
        lower = float(np.percentile(boot_accuracies, alpha * 100))
        upper = float(np.percentile(boot_accuracies, (1 - alpha) * 100))

        return {
            "n_bootstraps": n_bootstraps,
            "confidence_level": ci,
            "mean_accuracy": round(float(boot_accuracies.mean()), 4),
            "std_accuracy": round(float(boot_accuracies.std()), 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
        }

    # ------------------------------------------------------------------
    #  Simulated options P&L
    # ------------------------------------------------------------------

    @staticmethod
    def _bsm_price_backtest(S, K, T, r, sigma, option_type="call"):
        """BSM option price for backtest (self-contained, no external deps)."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)
        sqrt_T = sqrt(T)
        d1 = (_mlog(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        nd1 = 0.5 * (1 + _erf(d1 / sqrt(2)))
        nd2 = 0.5 * (1 + _erf(d2 / sqrt(2)))
        if option_type == "call":
            return S * nd1 - K * _mexp(-r * T) * nd2
        return K * _mexp(-r * T) * (1 - nd2) - S * (1 - nd1)

    @staticmethod
    def _simulate_options_pnl(all_results: list) -> dict:
        """Simulate weekly options P&L using BSM pricing at entry and exit.

        NOTE: This BSM-based simulation will likely be replaced with
        Polygon.io historical options data once integrated. Polygon provides
        actual historical options chains (bid/ask/IV per strike per day),
        which would let us backtest on real market prices instead of
        model-implied prices. The rest of the pipeline won't need to
        change — just swap BSM repricing for Polygon lookups here.

        Instead of the crude "3% of stock price" assumption, we now:
        1. Estimate entry IV from historical realized vol (annualized 20-day)
        2. Price the option at entry (Monday open) using BSM
        3. Price the option at exit (Friday close) using BSM with reduced T
        4. Compute actual P&L including theta, gamma, and vega effects
           implicitly through the repricing

        The BSM repricing approach captures all Greeks simultaneously —
        gamma helps winners, theta hurts holders, and the actual premium
        level reflects the stock's volatility, not a flat 3%.
        """
        if not all_results:
            return {}

        r = 0.045  # risk-free rate
        target_delta = 0.35

        pnls = []
        wins = 0
        losses = 0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for res in all_results:
            stock_return = res["actual_return_pct"] / 100  # decimal
            predicted_dir = res["predicted_direction"]
            entry_price = res.get("entry_price", 100)  # stock price at Monday open

            # Estimate IV from historical realized vol at that point
            # Use atr_pct as a proxy: annualized ≈ daily_atr_pct * sqrt(252)
            atr_pct = res.get("signals", {}).get("atr_pct", 1.5) or 1.5
            estimated_iv = (atr_pct / 100) * sqrt(252)
            # Clamp to reasonable range
            estimated_iv = max(0.15, min(estimated_iv, 2.0))

            option_type = "call" if predicted_dir == "call" else "put"
            T_entry = 5 / 252  # 5 trading days
            T_exit = 0.5 / 252  # near expiry (half a day left)

            # Strike selection: target ~0.35 delta
            # For calls: strike slightly above spot; for puts: slightly below
            if option_type == "call":
                strike = entry_price * (1 + 0.01)  # ~1% OTM
            else:
                strike = entry_price * (1 - 0.01)  # ~1% OTM

            # BSM price at entry (Monday)
            entry_premium = DirectionalBacktester._bsm_price_backtest(
                entry_price, strike, T_entry, r, estimated_iv, option_type,
            )

            if entry_premium <= 0.01:
                # Skip: option too cheap to trade meaningfully
                continue

            # Stock price at exit (Friday close)
            exit_stock_price = entry_price * (1 + stock_return)

            # BSM price at exit — IV typically contracts ~10% over the week
            # for non-event weeks (conservative assumption)
            exit_iv = estimated_iv * 0.90
            exit_premium = DirectionalBacktester._bsm_price_backtest(
                exit_stock_price, strike, T_exit, r, exit_iv, option_type,
            )

            # Option return as % of premium paid
            option_return = (exit_premium - entry_premium) / entry_premium

            # Apply stops/targets (checked intraweek in reality, but we
            # only have weekly endpoints — this underestimates stop-outs)
            if option_return >= 0.40:
                option_return = 0.40  # profit target
            elif option_return <= -0.50:
                option_return = -0.50  # stop loss

            pnls.append(option_return)

            if option_return > 0:
                wins += 1
            else:
                losses += 1

            cumulative += option_return
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        if not pnls:
            return {}

        pnl_array = np.array(pnls)
        winners = pnl_array[pnl_array > 0]
        losers = pnl_array[pnl_array <= 0]

        gross_profit = float(winners.sum()) if len(winners) > 0 else 0
        gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "total_trades": len(pnls),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(pnls), 4),
            "avg_return_pct": round(float(pnl_array.mean()) * 100, 2),
            "avg_winner_pct": round(float(winners.mean()) * 100, 2) if len(winners) > 0 else 0,
            "avg_loser_pct": round(float(losers.mean()) * 100, 2) if len(losers) > 0 else 0,
            "expected_return_pct": round(float(pnl_array.mean()) * 100, 2),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_return_pct": round(cumulative * 100, 2),
        }

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
        """Fetch Finnhub earnings calendar for gap classification."""
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
        """Fetch Finnhub economic calendar for the lookback period."""
        if not config.has_finnhub():
            logger.info("Finnhub API key not configured; skipping economic calendar")
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.lookback_weeks + 1)

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
    #  Context enrichment (Mon-Fri window)
    # ------------------------------------------------------------------

    def _get_context(self, ticker: str, mon_date: str, fri_date: str,
                     signals: dict) -> dict:
        """Build contextual features for a single Monday prediction.

        Checks the entire Mon-Fri holding window for events, not just
        the entry day (critical difference from 0DTE backtesting).
        """
        context = {}

        # --- Earnings in Mon-Fri window ---
        has_earnings = False
        if ticker in self._earnings_dates:
            try:
                mon = datetime.strptime(mon_date, "%Y-%m-%d")
                fri = datetime.strptime(fri_date, "%Y-%m-%d")
                for offset in range(0, (fri - mon).days + 1):
                    check = (mon + timedelta(days=offset)).strftime("%Y-%m-%d")
                    if check in self._earnings_dates[ticker]:
                        has_earnings = True
                        break
            except ValueError:
                pass
        context["earnings_in_window"] = has_earnings

        # --- Macro events in Mon-Fri window ---
        has_macro_event = False
        macro_events = []
        try:
            mon = datetime.strptime(mon_date, "%Y-%m-%d")
            fri = datetime.strptime(fri_date, "%Y-%m-%d")
            for offset in range(0, (fri - mon).days + 1):
                check = (mon + timedelta(days=offset)).strftime("%Y-%m-%d")
                if check in self._economic_events:
                    has_macro_event = True
                    macro_events.extend(self._economic_events[check])
        except ValueError:
            pass
        context["macro_event"] = has_macro_event
        context["macro_event_count"] = len(macro_events)

        # --- Yield curve state ---
        context["yield_curve"] = self._get_yield_curve_state(mon_date)

        # --- Credit spread level ---
        context["credit_spread"] = self._get_credit_spread_state(mon_date)

        # --- Financial conditions ---
        context["financial_conditions"] = self._get_nfci_state(mon_date)

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
        nfci = self._get_fred_value_at("NFCI", date_str)
        if nfci is None:
            return "unknown"
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

    def _compute_signal_correlations(self, results: list) -> dict:
        """Compute Pearson r of each signal vote vs actual return."""
        correlations = {}
        for sig in self.TESTABLE_SIGNALS:
            votes = np.array([r["signal_votes"].get(sig, 0) for r in results])
            returns = np.array([r["actual_return_pct"] for r in results])
            active_mask = votes != 0
            if active_mask.sum() > 30:
                r_val = float(np.corrcoef(votes[active_mask], returns[active_mask])[0, 1])
                if not np.isnan(r_val):
                    correlations[sig] = round(r_val, 4)
        return correlations

    def _compile_results(self, all_results: list) -> dict:
        """Aggregate individual predictions into summary statistics."""
        if not all_results:
            return {"error": "No backtest results generated"}

        total = len(all_results)
        correct = sum(1 for r in all_results if r["correct"])
        overall_accuracy = correct / total

        # Per-signal accuracy
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

        # By VIX regime
        by_regime = {}
        for regime in ["low", "normal", "elevated", "high", "extreme"]:
            subset = [r for r in all_results if r["regime"] == regime]
            if subset:
                rc = sum(1 for r in subset if r["correct"])
                by_regime[regime] = {
                    "count": len(subset),
                    "accuracy": round(rc / len(subset), 4),
                    "avg_return_pct": round(
                        float(np.mean([r["actual_return_pct"] for r in subset])), 3
                    ),
                }

        # By confidence bucket
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

        # Signal correlations
        signal_correlations = self._compute_signal_correlations(all_results)

        # Derive optimal weights
        optimal_weights = self._derive_optimal_weights(signal_correlations)

        # Weekly breakdown
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

        # Contextual analysis
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
        """Break down accuracy by contextual features."""
        contextual = {}

        def _accuracy_of(subset):
            if not subset:
                return None
            c = sum(1 for r in subset if r["correct"])
            return {"count": len(subset), "accuracy": round(c / len(subset), 4)}

        # Earnings in Mon-Fri window
        with_earnings = [r for r in all_results
                         if r.get("context", {}).get("earnings_in_window")]
        no_earnings = [r for r in all_results
                       if not r.get("context", {}).get("earnings_in_window")]
        contextual["earnings_in_window"] = {
            "with_earnings": _accuracy_of(with_earnings),
            "without_earnings": _accuracy_of(no_earnings),
        }

        # Macro event in Mon-Fri window
        macro_event = [r for r in all_results if r.get("context", {}).get("macro_event")]
        no_macro = [r for r in all_results if not r.get("context", {}).get("macro_event")]
        contextual["macro_event"] = {
            "with_event": _accuracy_of(macro_event),
            "without_event": _accuracy_of(no_macro),
        }
        if macro_event:
            contextual["macro_event"]["avg_abs_return_event"] = round(
                float(np.mean([abs(r["actual_return_pct"]) for r in macro_event])), 3)
        if no_macro:
            contextual["macro_event"]["avg_abs_return_calm"] = round(
                float(np.mean([abs(r["actual_return_pct"]) for r in no_macro])), 3)

        # By yield curve, credit spread, financial conditions
        for ctx_key, section_name, states in [
            ("yield_curve", "by_yield_curve", ["inverted", "flat", "slightly_positive", "normal", "steep"]),
            ("credit_spread", "by_credit_spread", ["tight", "normal", "wide", "stressed"]),
            ("financial_conditions", "by_financial_conditions", ["loose", "normal", "tightening", "tight"]),
        ]:
            by_section = {}
            for state in states:
                subset = [r for r in all_results
                          if r.get("context", {}).get(ctx_key) == state]
                result = _accuracy_of(subset)
                if result:
                    by_section[state] = result
            contextual[section_name] = by_section

        # Signal-macro interaction
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
        """Derive signal weights proportional to |correlation|."""
        if not correlations:
            return {}

        abs_corrs = {k: abs(v) for k, v in correlations.items()}
        total = sum(abs_corrs.values())
        if total <= 0:
            return {}

        return {k: round(abs_corrs[k] / total, 4) for k in correlations}
