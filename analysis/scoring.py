"""
Candidate scoring engine for the Zero-DTE Options Trading Analysis System.

Combines technical, options, sentiment, flow, market-regime, and expected-move
signals into a single composite score (0-100) for each candidate ticker.
Weights are persisted to disk and evolve through weekly reflections.
"""

import sys
import os
import json
import logging
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()

WEIGHTS_FILE = config.performance_dir / "weights.json"


class CandidateScorer:
    """Score and rank zero-DTE candidates using a weighted multi-signal model."""

    DEFAULT_WEIGHTS = {
        "technical": 0.18,
        "options": 0.18,
        "sentiment": 0.12,
        "flow": 0.08,
        "market_regime": 0.12,
        "expected_move": 0.18,
        "finviz": 0.08,
        "insider": 0.06,
    }

    # Empirical confidence calibration from 52-week backtest (2026-03-27).
    # Raw confidence buckets mapped to observed accuracy:
    #   very_high (0.8-1.0) → 52.4%  (worst — overconfident)
    #   high      (0.6-0.8) → 55.6%
    #   medium    (0.4-0.6) → 55.0%
    #   low       (0.2-0.4) → 55.0%
    #   very_low  (0.0-0.2) → 53.4%
    # Calibration shrinks extreme raw confidence toward the empirical mean.
    CONFIDENCE_CALIBRATION = {
        # (raw_lo, raw_hi) → calibrated_value
        (0.0, 0.2): 0.35,
        (0.2, 0.4): 0.55,
        (0.4, 0.6): 0.55,
        (0.6, 0.8): 0.60,
        (0.8, 1.01): 0.40,  # penalise overconfidence
    }

    # Gap signal weight by VIX regime — backtest shows gap is unreliable
    # in calm markets (r=-0.080 at 12wk) but useful in stress (r=+0.088 at 52wk).
    GAP_WEIGHT_BY_REGIME = {
        "low": 0.3,       # barely trust it — calm markets, gaps are noise
        "normal": 0.5,    # reduced trust
        "elevated": 1.0,  # full weight — momentum markets
        "high": 1.2,      # gap continuation is strong in fear
        "extreme": 1.2,
    }

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def __init__(self, weights=None):
        """Load scoring weights from disk or use supplied / default values.

        Parameters
        ----------
        weights : dict or None
            If provided, these weights override any saved or default values.
        """
        if weights is not None:
            self.weights = dict(weights)
        else:
            self.weights = self._load_weights()

        self.market_summary = None

    def set_market_summary(self, summary: dict) -> None:
        """Attach a market summary so the scorer can use economic event data.

        Parameters
        ----------
        summary : dict
            The dict returned by ``MarketScanner.get_market_summary()``.
        """
        self.market_summary = summary

    # Regime-adaptive multipliers applied on top of base weights.
    # Each regime shifts emphasis toward signals that perform best
    # in that environment, then re-normalises to sum to 1.0.
    REGIME_MULTIPLIERS = {
        "low": {
            # Calm markets: technicals are cleaner, sentiment drives moves,
            # options premiums are cheap (expected move matters less)
            "technical": 1.3,
            "options": 0.9,
            "sentiment": 1.3,
            "flow": 1.0,
            "market_regime": 0.8,
            "expected_move": 0.8,
            "finviz": 1.1,
            "insider": 1.1,
        },
        "normal": {
            # Balanced — use base weights as-is
            "technical": 1.0,
            "options": 1.0,
            "sentiment": 1.0,
            "flow": 1.0,
            "market_regime": 1.0,
            "expected_move": 1.0,
            "finviz": 1.0,
            "insider": 1.0,
        },
        "elevated": {
            # Rising vol: options and flow become more informative,
            # technicals start to break down
            "technical": 0.9,
            "options": 1.2,
            "sentiment": 0.9,
            "flow": 1.3,
            "market_regime": 1.1,
            "expected_move": 1.2,
            "finviz": 1.0,
            "insider": 0.8,
        },
        "high": {
            # Fear regime: options pricing and expected move dominate,
            # technicals unreliable, sentiment is noise
            "technical": 0.7,
            "options": 1.4,
            "sentiment": 0.7,
            "flow": 1.3,
            "market_regime": 1.2,
            "expected_move": 1.4,
            "finviz": 0.9,
            "insider": 0.6,
        },
        "extreme": {
            # Panic: options and expected move are everything,
            # other signals largely useless
            "technical": 0.5,
            "options": 1.5,
            "sentiment": 0.5,
            "flow": 1.4,
            "market_regime": 1.3,
            "expected_move": 1.5,
            "finviz": 0.8,
            "insider": 0.5,
        },
    }

    def _load_weights(self) -> dict:
        """Attempt to read weights from *WEIGHTS_FILE*; fall back to defaults."""
        if WEIGHTS_FILE.exists():
            try:
                with open(WEIGHTS_FILE, "r") as fh:
                    data = json.load(fh)
                logger.info("Loaded scoring weights from %s", WEIGHTS_FILE)
                return data
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not parse weights file: %s — using defaults", exc)
        return dict(self.DEFAULT_WEIGHTS)

    def _regime_adjusted_weights(self, regime: str) -> dict:
        """Return weights adjusted for the current VIX regime.

        Applies multipliers from ``REGIME_MULTIPLIERS`` to the base weights
        and re-normalises so they sum to 1.0.
        """
        multipliers = self.REGIME_MULTIPLIERS.get(regime, self.REGIME_MULTIPLIERS["normal"])
        raw = {
            cat: self.weights.get(cat, 0) * multipliers.get(cat, 1.0)
            for cat in self.weights
        }
        total = sum(raw.values())
        if total > 0:
            return {k: v / total for k, v in raw.items()}
        return dict(self.weights)

    # ------------------------------------------------------------------
    #  Sub-scores
    # ------------------------------------------------------------------

    def _technical_score(self, data: dict) -> float:
        """Derive a 0-100 technical sub-score.

        Considers RSI extremes, ATR, volume ratio, Bollinger Band position,
        MACD crossover signals, and intraday 0DTE signals (gap, VWAP, intraday ATR).
        """
        tech = data.get("technical", {})
        score = 0.0

        # RSI extremity
        rsi = tech.get("rsi", 50) or 50
        if rsi < 30 or rsi > 70:
            score += min(100, abs(rsi - 50) * 3)
        else:
            score += abs(rsi - 50)

        # ATR contribution — prefer intraday ATR if available
        intraday_atr = tech.get("intraday_atr_pct")
        atr_pct = intraday_atr if intraday_atr else (tech.get("atr_pct", 0) or 0)
        score += min(30, atr_pct * 15)

        # Volume contribution
        volume_ratio = tech.get("volume_ratio", 0) or 0
        score += min(20, volume_ratio * 10)

        # Bollinger contribution
        bb_pct = tech.get("bb_pct", 0.5)
        if bb_pct is None:
            bb_pct = 0.5
        if bb_pct < 0 or bb_pct > 1:
            score += 20
        else:
            score += abs(bb_pct - 0.5) * 20

        # MACD crossover bonus
        hist = tech.get("macd_histogram", 0) or 0
        hist_prev = tech.get("macd_histogram_prev", 0) or 0
        if hist * hist_prev < 0:
            score += 15

        # --- Intraday 0DTE signals ---

        # Pre-market gap: large gaps = strong intraday signal
        gap_pct = tech.get("gap_pct", 0) or 0
        if abs(gap_pct) >= 1.0:
            score += min(15, abs(gap_pct) * 5)

        # Price vs previous day's VWAP: confirms buyer/seller control
        price_vs_vwap = tech.get("price_vs_prev_vwap")
        if price_vs_vwap is not None:
            score += min(10, abs(price_vs_vwap) * 3)

        return float(np.clip(score, 0, 100))

    def _options_score(self, data: dict) -> float:
        """Derive a 0-100 options sub-score.

        Evaluates bid-ask spread quality, put/call ratio extremity,
        total options volume, and max pain divergence.
        """
        opts = data.get("options", {})
        score = 0.0

        # Bid-ask spread quality (critical for 0DTE execution cost)
        call_spread = opts.get("atm_call_spread_pct")
        put_spread = opts.get("atm_put_spread_pct")
        avg_spread = None
        if call_spread is not None and put_spread is not None:
            avg_spread = (call_spread + put_spread) / 2
        elif call_spread is not None:
            avg_spread = call_spread
        elif put_spread is not None:
            avg_spread = put_spread

        if avg_spread is not None:
            if avg_spread < 5:
                score += 20      # tight spread — excellent liquidity
            elif avg_spread < 15:
                score += 12
            elif avg_spread < 30:
                score += 5
            # else: wide spread — 0 points
        else:
            score += 8  # neutral when no data

        # P/C ratio extremity
        pc_ratio = opts.get("pc_ratio_volume", 1.0) or 1.0
        score += min(30, abs(pc_ratio - 1.0) * 30)

        # Total options volume (liquidity)
        total_call_vol = opts.get("total_call_volume", 0) or 0
        total_put_vol = opts.get("total_put_volume", 0) or 0
        total_vol = total_call_vol + total_put_vol
        score += min(20, total_vol / 100000)

        # Max pain divergence
        current_price = opts.get("current_price", 0) or 0
        max_pain = opts.get("max_pain", 0) or 0
        if current_price > 0 and max_pain > 0:
            divergence = abs(current_price - max_pain) / current_price * 100
            score += min(30, divergence)

        return float(np.clip(score, 0, 100))

    def _sentiment_score(self, data: dict) -> float:
        """Derive a 0-100 sentiment sub-score.

        Maps composite_score from (-1, 1) to (0, 100).
        """
        sent = data.get("sentiment", {})
        if not sent:
            return 50.0  # neutral default

        # composite_score from sentiment scanner is -1 to +1
        composite = sent.get("composite_score", 0.0)
        if composite is None:
            composite = 0.0
        score = (composite + 1) * 50  # map to 0-100

        return float(np.clip(score, 0, 100))

    def _flow_score(self, data: dict) -> float:
        """Derive a 0-100 flow sub-score.

        Looks for unusual flow signals and directional indicators.
        """
        flow = data.get("flow", {})
        if not flow:
            return 30.0  # default when no flow data

        score = 30.0  # baseline

        # Unusual flow flag
        if flow.get("unusual", False) or flow.get("unusual_volume", False):
            score += 30

        # Directional indicator
        if flow.get("direction"):
            score += 20

        # Volume / OI ratio — high ratio means fresh positioning
        vol_oi = flow.get("volume_oi_ratio", 0) or 0
        if vol_oi >= 2.0:
            score += 15
        elif vol_oi >= 1.0:
            score += 10
        elif vol_oi >= 0.5:
            score += 5

        return float(np.clip(score, 0, 100))

    def _market_regime_score(self, data: dict) -> float:
        """Derive a 0-100 market regime sub-score.

        High VIX favours puts (boost put candidates, penalise calls).
        Low VIX environments favour calls.
        """
        regime = data.get("market_regime", {})
        vix_regime = regime.get("regime", "normal")
        direction = data.get("direction_hint", "neutral")

        score = 50.0  # neutral baseline

        if vix_regime in ("high", "extreme"):
            if direction == "put":
                score += 10
            elif direction == "call":
                score -= 5
        elif vix_regime == "low":
            if direction == "call":
                score += 10
        elif vix_regime == "elevated":
            if direction == "put":
                score += 5

        return float(np.clip(score, 0, 100))

    def _finviz_score(self, data: dict) -> float:
        """Derive a 0-100 Finviz sub-score.

        Considers pre-market movement, analyst consensus, relative volume,
        and short float (squeeze potential).
        """
        fv = data.get("finviz", {})
        if not fv:
            return 50.0  # neutral when no data

        score = 40.0  # baseline

        # Pre-market change: large moves = strong intraday opportunity
        pre_change = fv.get("pre_market_change_pct")
        if pre_change is not None:
            score += min(20, abs(pre_change) * 5)

        # Analyst rating
        rating = fv.get("analyst_rating")
        if rating == "Strong Buy":
            score += 15
        elif rating == "Buy":
            score += 10
        elif rating == "Hold":
            score += 0
        elif rating == "Sell":
            score -= 5
        elif rating == "Strong Sell":
            score -= 10

        # Relative volume (>1.5 = unusual activity, good for 0DTE)
        rel_vol = fv.get("rel_volume")
        if rel_vol is not None:
            if rel_vol >= 2.0:
                score += 15
            elif rel_vol >= 1.5:
                score += 10
            elif rel_vol >= 1.0:
                score += 5

        # Short float — high short interest = squeeze potential
        short_float = fv.get("short_float_pct")
        if short_float is not None and short_float > 10:
            score += min(15, (short_float - 10) * 1.5)

        return float(np.clip(score, 0, 100))

    def _insider_score(self, data: dict) -> float:
        """Derive a 0-100 insider-trading sub-score from SEC EDGAR data.

        Insider buying is a stronger signal than selling (insiders sell for
        many reasons, but buy for one: they think the stock will go up).
        """
        insider = data.get("insider", {})
        if not insider:
            return 50.0  # neutral when no data

        score = 50.0  # neutral baseline

        signal = insider.get("insider_signal", "neutral")
        if signal == "bullish":
            score += 20
        elif signal == "bearish":
            score -= 10  # selling is weaker signal

        # Net value magnitude — large insider buys are very bullish
        net_value = insider.get("net_value", 0)
        if net_value > 0:
            # Scale: $100k+ net buy is meaningful
            score += min(15, net_value / 100_000 * 5)
        elif net_value < 0:
            score -= min(10, abs(net_value) / 500_000 * 5)

        # Number of recent buys — cluster buying is a stronger signal
        total_buys = insider.get("total_buys", 0)
        if total_buys >= 3:
            score += 10
        elif total_buys >= 1:
            score += 5

        return float(np.clip(score, 0, 100))

    def expected_move_score(self, ticker_data: dict) -> float:
        """Score 0-100 based on whether option premium makes sense vs typical move.

        Compares the stock's realised daily move (ATR-based) against the
        market-implied move.  Prefers the straddle-based implied move (direct
        market price of expected movement) when available; falls back to
        ATM IV / sqrt(252).

        When realised > implied the options are cheap relative to actual
        movement — good for buying 0-DTE.

        Parameters
        ----------
        ticker_data : dict
            Must contain 'technical' and 'options' sub-dicts.

        Returns
        -------
        float
            Score between 0 and 100.
        """
        tech = ticker_data.get("technical", {})
        opts = ticker_data.get("options", {})
        atr = tech.get("atr", 0)
        price = tech.get("price", 1)

        # Guard against missing / zero data
        if not atr or not price or price <= 0:
            return 50.0

        expected_pct_move = atr / price

        # Prefer straddle-based implied move (more accurate for near-expiry)
        implied_move_pct = opts.get("implied_move_pct")
        if implied_move_pct is not None and implied_move_pct > 0:
            implied_pct_move = implied_move_pct / 100.0
        else:
            # Fall back to IV-derived daily move
            atm_iv = opts.get("atm_iv", 0) or 0
            if atm_iv <= 0:
                return 50.0
            implied_pct_move = atm_iv / sqrt(252)

        if expected_pct_move > implied_pct_move:
            # Options are cheap relative to movement -- GOOD for buying
            ratio = expected_pct_move / implied_pct_move if implied_pct_move > 0 else 2.0
            score = 70 + min(30, (ratio - 1.0) * 30)
        elif expected_pct_move < implied_pct_move * 0.5:
            # Options are expensive relative to movement -- BAD
            ratio = expected_pct_move / implied_pct_move if implied_pct_move > 0 else 0.5
            score = max(0, ratio * 60)
        else:
            # Neutral zone
            ratio = expected_pct_move / implied_pct_move if implied_pct_move > 0 else 1.0
            score = 40 + (ratio - 0.5) * 40

        return float(np.clip(score, 0, 100))

    def _economic_event_adjustment(self) -> dict:
        """Calculate score and confidence adjustments based on economic events.

        High-impact events (FOMC, CPI, NFP, etc.) create large intraday moves
        that are good for 0DTE profitability, so the overall score is boosted.
        However, the direction of the move is unpredictable, so confidence is
        reduced slightly.

        Returns
        -------
        dict
            Keys: score_bonus (float), confidence_penalty (float).
        """
        if not self.market_summary:
            return {"score_bonus": 0.0, "confidence_penalty": 0.0}

        has_high_impact = self.market_summary.get("has_high_impact_event", False)

        if has_high_impact:
            return {"score_bonus": 10.0, "confidence_penalty": 0.05}

        return {"score_bonus": 0.0, "confidence_penalty": 0.0}

    # ------------------------------------------------------------------
    #  Public scoring API
    # ------------------------------------------------------------------

    def score_candidate(self, ticker_data: dict) -> dict:
        """Produce a composite score (0-100) for a single candidate.

        Parameters
        ----------
        ticker_data : dict
            Must contain sub-dicts keyed by 'technical', 'options',
            'sentiment', 'flow', 'market_regime'.  Missing keys are
            tolerated (neutral baseline used).

        Returns
        -------
        dict
            Original data augmented with 'scores' (per-category) and
            'composite_score'.
        """
        sub_scores = {
            "technical": self._technical_score(ticker_data),
            "options": self._options_score(ticker_data),
            "sentiment": self._sentiment_score(ticker_data),
            "flow": self._flow_score(ticker_data),
            "market_regime": self._market_regime_score(ticker_data),
            "expected_move": self.expected_move_score(ticker_data),
            "finviz": self._finviz_score(ticker_data),
            "insider": self._insider_score(ticker_data),
        }

        # Regime-adaptive weighted combination
        regime = ticker_data.get("market_regime", {}).get("regime", "normal")
        active_weights = self._regime_adjusted_weights(regime)

        composite = sum(
            sub_scores[cat] * active_weights.get(cat, 0)
            for cat in sub_scores
        )

        # Economic event adjustment — boost score on high-impact event days
        econ_adj = self._economic_event_adjustment()
        composite += econ_adj["score_bonus"]

        result = dict(ticker_data)
        result["scores"] = sub_scores
        result["composite_score"] = round(float(np.clip(composite, 0, 100)), 2)
        result["economic_event_bonus"] = econ_adj["score_bonus"]
        result["active_regime"] = regime
        result["active_weights"] = {k: round(v, 4) for k, v in active_weights.items()}
        return result

    def rank_candidates(self, candidates_list: list, top_n: int = 10) -> list:
        """Score every candidate and return the top *top_n* sorted by composite.

        Parameters
        ----------
        candidates_list : list[dict]
            Each element is a ticker_data dict suitable for ``score_candidate``.
        top_n : int
            Number of top candidates to return.

        Returns
        -------
        list[dict]
            Sorted (highest first), each with 'composite_score' attached.
        """
        scored = [self.score_candidate(c) for c in candidates_list]
        scored.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        logger.info(
            "Ranked %d candidates — top score: %.1f, bottom score: %.1f",
            len(scored),
            scored[0]["composite_score"] if scored else 0,
            scored[-1]["composite_score"] if scored else 0,
        )
        return scored[:top_n]

    # ------------------------------------------------------------------
    #  Direction determination
    # ------------------------------------------------------------------

    def determine_direction(self, ticker_data: dict) -> dict:
        """Decide call vs put based on aggregate signals.

        Returns
        -------
        dict
            Keys: direction ('call' or 'put'), confidence (0.0 - 1.0).
        """
        bullish_count = 0
        bearish_count = 0

        tech = ticker_data.get("technical", {})
        opts = ticker_data.get("options", {})

        # RSI signal
        rsi = tech.get("rsi", 50) or 50
        if rsi < 40:
            bearish_count += 1
        elif rsi > 60:
            bullish_count += 1

        # MACD histogram direction
        hist = tech.get("macd_histogram", 0) or 0
        if hist > 0:
            bullish_count += 1
        elif hist < 0:
            bearish_count += 1

        # Put/call ratio
        pc_ratio = opts.get("pc_ratio_volume", 1.0) or 1.0
        if pc_ratio > 1.2:
            bearish_count += 1  # more puts = bearish bias
        elif pc_ratio < 0.8:
            bullish_count += 1  # more calls = bullish bias

        # Price vs max pain (price tends to pull toward max pain)
        current_price = opts.get("current_price", 0) or 0
        max_pain = opts.get("max_pain", 0) or 0
        if current_price > 0 and max_pain > 0:
            if current_price < max_pain:
                bullish_count += 1  # below max pain = bullish pull
            elif current_price > max_pain:
                bearish_count += 1  # above max pain = bearish pull

        # Price vs SMA20
        price_vs_sma20 = tech.get("price_vs_sma20_pct", 0)
        if price_vs_sma20 is not None:
            if price_vs_sma20 < 0:
                bearish_count += 1  # below SMA20 = bearish trend
            elif price_vs_sma20 > 0:
                bullish_count += 1  # above SMA20 = bullish trend

        # --- Intraday 0DTE signals ---

        # Pre-market gap direction — context-aware weight
        # Backtest shows gap is unreliable in calm markets but useful in stress
        regime_info = ticker_data.get("market_regime", {})
        vix_regime_name = regime_info.get("regime", "normal")
        gap_weight = self.GAP_WEIGHT_BY_REGIME.get(vix_regime_name, 0.5)
        gap_pct = tech.get("gap_pct", 0) or 0
        if gap_pct >= 0.5:
            bullish_count += gap_weight
        elif gap_pct <= -0.5:
            bearish_count += gap_weight

        # Price vs previous day VWAP (confirms buyer/seller control)
        price_vs_vwap = tech.get("price_vs_prev_vwap")
        if price_vs_vwap is not None:
            if price_vs_vwap > 0.3:
                bullish_count += 1
            elif price_vs_vwap < -0.3:
                bearish_count += 1

        # --- Finviz signals ---
        fv = ticker_data.get("finviz", {})

        # Finviz pre-market change (fresh Friday morning data)
        fv_pre_change = fv.get("pre_market_change_pct")
        if fv_pre_change is not None:
            if fv_pre_change >= 0.5:
                bullish_count += 1
            elif fv_pre_change <= -0.5:
                bearish_count += 1

        # Analyst consensus
        rating = fv.get("analyst_rating")
        if rating in ("Strong Buy", "Buy"):
            bullish_count += 0.5
        elif rating in ("Strong Sell", "Sell"):
            bearish_count += 0.5

        # --- SEC EDGAR insider signal ---
        insider = ticker_data.get("insider", {})
        insider_signal = insider.get("insider_signal", "neutral")
        if insider_signal == "bullish":
            bullish_count += 1
        elif insider_signal == "bearish":
            bearish_count += 0.5  # selling is weaker directional signal

        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            return {"direction": "call", "confidence": 0.0}

        # VIX bias: if high, nudge toward puts
        regime = ticker_data.get("market_regime", {})
        vix_current = regime.get("current", 0) or 0
        if vix_current > 25:
            bearish_count += 0.5
            total_signals += 0.5

        raw_confidence = abs(bullish_count - bearish_count) / total_signals
        direction = "call" if bullish_count >= bearish_count else "put"

        # Calibrate confidence using empirical accuracy curve
        confidence = self._calibrate_confidence(raw_confidence)

        # Reduce confidence slightly on high-impact economic event days
        econ_adj = self._economic_event_adjustment()
        confidence -= econ_adj["confidence_penalty"]

        return {
            "direction": direction,
            "confidence": round(float(np.clip(confidence, 0, 1)), 3),
            "raw_confidence": round(float(raw_confidence), 3),
            "gap_weight_used": gap_weight,
        }

    # ------------------------------------------------------------------
    #  Confidence calibration
    # ------------------------------------------------------------------

    def _calibrate_confidence(self, raw: float) -> float:
        """Map raw vote-ratio confidence to empirically calibrated confidence.

        The backtest showed that very high raw confidence (>0.8) corresponds
        to LOWER accuracy (52.4%) than medium confidence (55.0%).  This was
        caused by the gap signal's 1.5x weight pushing most predictions into
        the very_high bucket with unstable accuracy.

        The calibration curve is derived from 52-week backtest accuracy by
        bucket, rescaled so the highest-accuracy bucket gets the highest
        calibrated value.
        """
        for (lo, hi), calibrated in self.CONFIDENCE_CALIBRATION.items():
            if lo <= raw < hi:
                return calibrated
        return raw  # fallback if somehow outside all buckets

    # ------------------------------------------------------------------
    #  Macro regime gate
    # ------------------------------------------------------------------

    def assess_macro_edge(self, market_summary: dict = None) -> dict:
        """Evaluate whether the current macro environment gives the model an edge.

        Backtest findings (52-week, 5150 predictions):
            - Credit spread tight (<3% HY OAS) + NFCI loose (<-0.5) → ~50% accuracy
            - Credit spread normal + NFCI normal → ~59% accuracy
            - VIX normal (15-20) → 50.5%, VIX elevated+ → 57-72%

        Returns
        -------
        dict
            Keys: has_edge (bool), confidence_multiplier (0.0-1.0),
            reasons (list of strings explaining the assessment).
        """
        summary = market_summary or self.market_summary
        if not summary:
            return {"has_edge": True, "confidence_multiplier": 1.0, "reasons": ["no macro data"]}

        reasons = []
        penalty = 0.0

        # VIX regime
        vix_regime = summary.get("vix_regime", {})
        regime_name = vix_regime.get("regime", "normal")
        vix_level = vix_regime.get("vix_level")

        if regime_name in ("low", "normal"):
            penalty += 0.20
            reasons.append(f"VIX {regime_name} ({vix_level:.1f}) — model has ~50% accuracy here")
        elif regime_name in ("high", "extreme"):
            penalty -= 0.15  # boost
            reasons.append(f"VIX {regime_name} ({vix_level:.1f}) — model's strongest regime")
        elif regime_name == "elevated":
            penalty -= 0.05
            reasons.append(f"VIX elevated ({vix_level:.1f}) — model has 57% accuracy")

        # Credit spread (from FRED via market summary)
        credit = summary.get("credit_spread", {})
        credit_state = credit.get("credit_state")
        hy_oas = credit.get("hy_oas")
        if credit_state == "tight":
            penalty += 0.15
            reasons.append(f"Credit spreads tight (HY OAS {hy_oas}%) — complacent market, 50.4% accuracy")
        elif credit_state in ("wide", "stressed"):
            penalty -= 0.10
            reasons.append(f"Credit spreads {credit_state} (HY OAS {hy_oas}%) — stress = better signal clarity")
        elif credit_state == "normal" and hy_oas:
            reasons.append(f"Credit spreads normal (HY OAS {hy_oas}%) — model's best environment (57%)")

        # Financial conditions (from FRED via market summary)
        conditions = summary.get("financial_conditions", {})
        cond_state = conditions.get("conditions_state")
        nfci = conditions.get("nfci")
        if cond_state == "loose":
            penalty += 0.15
            reasons.append(f"Financial conditions loose (NFCI {nfci}) — 50.8% accuracy")
        elif cond_state == "normal" and nfci is not None:
            reasons.append(f"Financial conditions normal (NFCI {nfci}) — 59.4% accuracy")
        elif cond_state in ("tightening", "tight"):
            penalty -= 0.05
            reasons.append(f"Financial conditions {cond_state} (NFCI {nfci}) — stress = better signals")

        # Treasury yields (from market summary)
        yields = summary.get("yields", {})
        curve_signal = yields.get("curve_signal")
        if curve_signal == "steep":
            penalty += 0.05
            reasons.append("Yield curve steep — model accuracy drops ~3%")

        # Check for high-impact economic events
        has_event = summary.get("has_high_impact_event", False)
        if has_event:
            reasons.append("High-impact economic event this week — larger moves but less predictable direction")

        confidence_multiplier = max(0.0, min(1.0, 1.0 - penalty))
        has_edge = confidence_multiplier >= 0.55

        if not has_edge:
            reasons.append("REGIME GATE: complacent macro environment — reduce position sizing or skip")

        return {
            "has_edge": has_edge,
            "confidence_multiplier": round(confidence_multiplier, 3),
            "regime": regime_name,
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    #  Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self) -> None:
        """Persist current weights to disk."""
        WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WEIGHTS_FILE, "w") as fh:
            json.dump(self.weights, fh, indent=2)
        logger.info("Saved scoring weights to %s", WEIGHTS_FILE)

    def update_weights(self, performance_data: dict) -> None:
        """Adjust weights toward categories that correlated with wins.

        Uses a simple exponential moving average approach:
            new_weight = old_weight * (1 - alpha) + target * alpha

        Parameters
        ----------
        performance_data : dict
            Expected keys: signal_correlations — a dict mapping each category
            name to its correlation with positive outcomes (-1 to +1).
        """
        correlations = performance_data.get("signal_correlations", {})
        if not correlations:
            logger.info("No signal correlations provided; weights unchanged")
            return

        alpha = 0.15  # learning rate
        raw = {}
        for category, current_w in self.weights.items():
            corr = correlations.get(category, 0.0)
            # Target weight: boost categories with positive correlation
            target = current_w * (1 + corr)
            raw[category] = current_w * (1 - alpha) + target * alpha

        # Re-normalise so weights sum to 1.0
        total = sum(raw.values())
        if total > 0:
            self.weights = {k: round(v / total, 4) for k, v in raw.items()}
        else:
            self.weights = dict(self.DEFAULT_WEIGHTS)

        self.save_weights()
        logger.info("Updated scoring weights: %s", self.weights)
