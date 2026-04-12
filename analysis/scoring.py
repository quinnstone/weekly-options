"""
Candidate scoring engine for the Weekly Options Trading Analysis System.

Three-tier scoring architecture optimized for 5-day directional holds:
    Tier 1 — DIRECTION (60%): momentum, mean-reversion, regime bias
    Tier 2 — EDGE QUALITY (25%): IV/RV mispricing, flow conviction, event risk
    Tier 3 — EXECUTION QUALITY (15%): liquidity, strike efficiency, theta cost

Direction determination uses weekly-appropriate signals (multi-day momentum,
trend persistence, sector rotation) rather than intraday noise.

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
from analysis.patterns import PatternLibrary

logger = logging.getLogger(__name__)
config = Config()

WEIGHTS_FILE = config.performance_dir / "weights.json"


class CandidateScorer:
    """Score and rank weekly options candidates using a 3-tier weighted model."""

    # ======================================================================
    #  Tier weights and sub-weights
    # ======================================================================

    DEFAULT_WEIGHTS = {
        # Tier 1: Direction (60%)
        "momentum": 0.20,
        "mean_reversion": 0.15,
        "regime_bias": 0.10,
        "trend_persistence": 0.15,
        # Tier 2: Edge Quality (25%)
        "iv_mispricing": 0.10,
        "flow_conviction": 0.08,
        "event_risk": 0.07,
        # Tier 3: Execution Quality (15%)
        "liquidity": 0.05,
        "strike_efficiency": 0.05,
        "theta_cost": 0.05,
    }

    # Regime-adaptive multipliers: shift emphasis based on VIX environment.
    REGIME_MULTIPLIERS = {
        "low": {
            "momentum": 1.3, "mean_reversion": 0.8, "regime_bias": 0.7,
            "trend_persistence": 1.3, "iv_mispricing": 0.8, "flow_conviction": 1.0,
            "event_risk": 0.9, "liquidity": 1.0, "strike_efficiency": 1.0,
            "theta_cost": 1.2,
        },
        "normal": {
            "momentum": 1.0, "mean_reversion": 1.0, "regime_bias": 1.0,
            "trend_persistence": 1.0, "iv_mispricing": 1.0, "flow_conviction": 1.0,
            "event_risk": 1.0, "liquidity": 1.0, "strike_efficiency": 1.0,
            "theta_cost": 1.0,
        },
        "elevated": {
            "momentum": 0.9, "mean_reversion": 1.2, "regime_bias": 1.1,
            "trend_persistence": 0.9, "iv_mispricing": 1.3, "flow_conviction": 1.2,
            "event_risk": 1.1, "liquidity": 1.0, "strike_efficiency": 1.0,
            "theta_cost": 0.9,
        },
        "high": {
            "momentum": 0.7, "mean_reversion": 1.4, "regime_bias": 1.3,
            "trend_persistence": 0.7, "iv_mispricing": 1.4, "flow_conviction": 1.3,
            "event_risk": 1.2, "liquidity": 1.1, "strike_efficiency": 1.0,
            "theta_cost": 0.8,
        },
        "extreme": {
            "momentum": 0.5, "mean_reversion": 1.5, "regime_bias": 1.4,
            "trend_persistence": 0.5, "iv_mispricing": 1.5, "flow_conviction": 1.4,
            "event_risk": 1.3, "liquidity": 1.2, "strike_efficiency": 1.0,
            "theta_cost": 0.7,
        },
    }

    # Confidence calibration — placeholder until we have weekly backtest data.
    # Conservative: assume ~52-55% directional accuracy.
    CONFIDENCE_CALIBRATION = {
        (0.0, 0.15): 0.30,
        (0.15, 0.30): 0.45,
        (0.30, 0.50): 0.55,
        (0.50, 0.70): 0.58,
        (0.70, 0.85): 0.52,
        (0.85, 1.01): 0.42,  # penalize extreme unanimity
    }

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def __init__(self, weights=None):
        if weights is not None:
            self.weights = dict(weights)
        else:
            self.weights = self._load_weights()
        self.market_summary = None
        self.pattern_library = PatternLibrary()

    def set_market_summary(self, summary: dict) -> None:
        self.market_summary = summary

    def _load_weights(self) -> dict:
        if WEIGHTS_FILE.exists():
            try:
                with open(WEIGHTS_FILE, "r") as fh:
                    data = json.load(fh)
                # Migrate old 8-factor weights to new 10-factor
                if "technical" in data and "momentum" not in data:
                    logger.info("Migrating old weights to new 10-factor model")
                    return dict(self.DEFAULT_WEIGHTS)
                logger.info("Loaded scoring weights from %s", WEIGHTS_FILE)
                return data
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not parse weights file: %s — using defaults", exc)
        return dict(self.DEFAULT_WEIGHTS)

    def _regime_adjusted_weights(self, regime: str) -> dict:
        multipliers = self.REGIME_MULTIPLIERS.get(regime, self.REGIME_MULTIPLIERS["normal"])
        raw = {
            cat: self.weights.get(cat, 0) * multipliers.get(cat, 1.0)
            for cat in self.weights
        }
        total = sum(raw.values())
        if total > 0:
            return {k: v / total for k, v in raw.items()}
        return dict(self.weights)

    # ======================================================================
    #  TIER 1: DIRECTION SUB-SCORES (60% of composite)
    # ======================================================================

    def _momentum_score(self, data: dict) -> float:
        """Score 0-100 based on multi-day price momentum.

        Uses 5-day, 10-day, and 21-day returns. Strong recent momentum
        in a trending stock is a good weekly signal.
        """
        tech = data.get("technical", {})
        score = 50.0  # neutral baseline

        ret_5d = tech.get("return_5d") or 0
        ret_10d = tech.get("return_10d") or 0
        ret_21d = tech.get("return_21d") or 0

        # Direction hint for context
        direction = data.get("direction_hint", "neutral")
        is_bullish = direction == "call"
        is_bearish = direction == "put"

        # Short-term momentum (5-day): most important for weekly trades
        if is_bullish:
            score += min(25, max(-15, ret_5d * 5))
        elif is_bearish:
            score += min(25, max(-15, -ret_5d * 5))
        else:
            score += min(15, abs(ret_5d) * 3)

        # Medium-term momentum (10-day)
        if is_bullish:
            score += min(15, max(-10, ret_10d * 2))
        elif is_bearish:
            score += min(15, max(-10, -ret_10d * 2))
        else:
            score += min(10, abs(ret_10d) * 1.5)

        # Longer-term context (21-day): momentum should be aligned
        alignment_bonus = 0
        if is_bullish and ret_5d > 0 and ret_21d > 0:
            alignment_bonus = 10  # all timeframes agree
        elif is_bearish and ret_5d < 0 and ret_21d < 0:
            alignment_bonus = 10
        score += alignment_bonus

        return float(np.clip(score, 0, 100))

    def _mean_reversion_score(self, data: dict) -> float:
        """Score 0-100 for mean-reversion setups.

        RSI extremes, distance from 52-week high/low, and Bollinger Band
        position identify oversold/overbought conditions.
        """
        tech = data.get("technical", {})
        score = 50.0

        rsi = tech.get("rsi", 50) or 50
        direction = data.get("direction_hint", "neutral")

        # RSI extremity — for weeklies, use wider thresholds (30/70)
        if direction == "call" and rsi < 30:
            score += min(30, (30 - rsi) * 3)  # deep oversold = strong call signal
        elif direction == "put" and rsi > 70:
            score += min(30, (rsi - 70) * 3)  # deep overbought = strong put signal
        elif rsi < 35 or rsi > 65:
            score += min(15, abs(rsi - 50) * 0.8)

        # Bollinger Band position
        bb_pct = tech.get("bb_pct", 0.5) or 0.5
        if direction == "call" and bb_pct < 0.1:
            score += 15  # near lower band = potential bounce
        elif direction == "put" and bb_pct > 0.9:
            score += 15  # near upper band = potential pullback
        elif bb_pct < 0 or bb_pct > 1:
            score += 10  # outside bands = strong setup

        # 52-week context
        pct_from_high = tech.get("pct_from_52w_high")
        pct_from_low = tech.get("pct_from_52w_low")
        if pct_from_high is not None and direction == "call" and pct_from_high < -20:
            score += 10  # >20% below 52w high = value play for calls
        if pct_from_low is not None and direction == "put" and pct_from_low > 50:
            score += 10  # >50% above 52w low = stretched for puts

        return float(np.clip(score, 0, 100))

    def _regime_bias_score(self, data: dict) -> float:
        """Score 0-100 based on VIX regime alignment with direction.

        High VIX favors puts; low VIX favors calls. Elevated regimes
        are the model's strongest environment.
        """
        regime = data.get("market_regime", {})
        vix_regime = regime.get("regime", "normal")
        direction = data.get("direction_hint", "neutral")
        score = 50.0

        regime_direction_map = {
            ("high", "put"): 20, ("extreme", "put"): 25,
            ("high", "call"): -10, ("extreme", "call"): -15,
            ("low", "call"): 15, ("low", "put"): -5,
            ("elevated", "put"): 10, ("elevated", "call"): 0,
        }
        score += regime_direction_map.get((vix_regime, direction), 0)

        # Regime persistence bonus
        persistence = data.get("regime_persistence", {})
        if persistence.get("is_stable"):
            score += 5  # stable regime = more reliable signals
        elif persistence.get("transition_direction") == "escalating" and direction == "put":
            score += 10  # VIX rising = puts stronger
        elif persistence.get("transition_direction") == "calming" and direction == "call":
            score += 5  # VIX falling = calls recover

        # Cross-asset leading indicators
        # Bonds/dollar moves lead equities by 1-2 days
        cross_asset = data.get("cross_asset", {})
        composite = cross_asset.get("composite", "mixed")
        if composite == "risk_off" and direction == "put":
            score += 10  # Bonds/dollar confirm bearish thesis
        elif composite == "risk_off" and direction == "call":
            score -= 8  # Cross-asset headwinds against bullish thesis
        elif composite == "risk_on" and direction == "call":
            score += 8  # Cross-asset tailwinds support bullish thesis
        elif composite == "risk_on" and direction == "put":
            score -= 5  # Cross-asset tailwinds weaken bearish thesis

        return float(np.clip(score, 0, 100))

    def _trend_persistence_score(self, data: dict) -> float:
        """Score 0-100 for trend strength and persistence.

        ADX, SMA slopes, and trend classification. Strong trends
        persist over weekly timeframes.
        """
        tech = data.get("technical", {})
        score = 50.0

        adx = tech.get("adx") or 0
        trend = tech.get("trend_strength", "flat")
        direction = data.get("direction_hint", "neutral")

        # ADX strength: >25 = trending, >40 = very strong
        if adx > 40:
            score += 20
        elif adx > 25:
            score += 10
        elif adx < 15:
            score -= 10  # choppy, directionless

        # Trend alignment with direction
        trend_bullish = trend in ("strong_up", "up")
        trend_bearish = trend in ("strong_down", "down")

        if direction == "call" and trend_bullish:
            score += 15
            if trend == "strong_up":
                score += 5
        elif direction == "put" and trend_bearish:
            score += 15
            if trend == "strong_down":
                score += 5
        elif direction == "call" and trend_bearish:
            score -= 10  # fighting the trend
        elif direction == "put" and trend_bullish:
            score -= 10

        # SMA slope confirmation
        sma20_slope = tech.get("sma20_slope") or 0
        if direction == "call" and sma20_slope > 0.1:
            score += 5
        elif direction == "put" and sma20_slope < -0.1:
            score += 5

        return float(np.clip(score, 0, 100))

    # ======================================================================
    #  TIER 2: EDGE QUALITY SUB-SCORES (25% of composite)
    # ======================================================================

    def _iv_mispricing_score(self, data: dict) -> float:
        """Score 0-100 for IV vs realized vol mispricing.

        When options are cheap (IV < RV), buying is favorable.
        Also incorporates stock-level IV term structure: when the weekly
        IV is cheap relative to the monthly, we're buying at a discount.
        """
        opts = data.get("options", {})
        tech = data.get("technical", {})
        score = 50.0

        # Primary: IV/RV ratio
        iv_rv = opts.get("iv_rv_ratio")
        if iv_rv is not None:
            if iv_rv < 0.8:
                score = min(100, 80 + (0.8 - iv_rv) * 50)
            elif iv_rv < 1.0:
                score = 65 + (1.0 - iv_rv) * 50
            elif iv_rv < 1.3:
                score = 45
            elif iv_rv < 1.5:
                score = 30
            else:
                score = max(5, 30 - (iv_rv - 1.5) * 20)
        else:
            # Fallback: compare ATR expected move to straddle implied move
            atr_pct = tech.get("atr_pct", 0) or 0
            implied_move = opts.get("implied_move_pct") or 0
            if atr_pct > 0 and implied_move > 0:
                ratio = (atr_pct * sqrt(5)) / implied_move
                if ratio > 1.2:
                    score = 75
                elif ratio > 0.8:
                    score = 50
                else:
                    score = 25

        # Stock-level IV term structure bonus/penalty
        iv_ts = opts.get("iv_term_structure")
        if iv_ts and isinstance(iv_ts, dict):
            ratio = iv_ts.get("ratio", 1.0)
            if ratio < 0.85:
                score += 10  # weekly significantly cheap vs monthly
            elif ratio < 0.95:
                score += 5   # modestly cheap
            elif ratio > 1.15:
                score -= 10  # weekly expensive (event premium baked in)
            elif ratio > 1.05:
                score -= 5

        return float(np.clip(score, 0, 100))

    def _flow_conviction_score(self, data: dict) -> float:
        """Score 0-100 for options flow conviction.

        Unusual flow + directional alignment = institutional positioning.
        """
        flow = data.get("flow", {})
        opts = data.get("options", {})
        if not flow and not opts:
            return 40.0

        score = 40.0  # baseline

        # Unusual flow detection
        if flow.get("unusual") or flow.get("unusual_volume"):
            score += 25
            # Direction alignment bonus
            flow_dir = flow.get("direction", "")
            pick_dir = data.get("direction_hint", "")
            if (flow_dir == "bullish" and pick_dir == "call") or \
               (flow_dir == "bearish" and pick_dir == "put"):
                score += 15  # flow agrees with our direction

        # P/C ratio conviction
        pc_ratio = opts.get("pc_ratio_volume", 1.0) or 1.0
        direction = data.get("direction_hint", "")
        if direction == "put" and pc_ratio > 1.3:
            score += 10  # heavy put buying confirms bearish
        elif direction == "call" and pc_ratio < 0.7:
            score += 10  # heavy call buying confirms bullish

        # Volume/OI ratio
        vol_oi = flow.get("volume_oi_ratio", 0) or 0
        if vol_oi >= 2.0:
            score += 10
        elif vol_oi >= 1.0:
            score += 5

        return float(np.clip(score, 0, 100))

    def _event_risk_score(self, data: dict) -> float:
        """Score 0-100 for event risk in the holding window.

        High-impact events create uncertainty but also opportunity.
        Score reflects whether events help or hurt the trade.
        """
        score = 60.0  # baseline (no events = slightly favorable)

        # Macro events in holding window
        holding = data.get("holding_window", {})
        if not holding and self.market_summary:
            holding = self.market_summary.get("holding_window", {})

        if holding:
            risk_level = holding.get("risk_level", "low")
            if risk_level == "high":
                score -= 20  # FOMC or multiple events = uncertainty
            elif risk_level == "medium":
                score -= 5   # one event = minor uncertainty

            if holding.get("has_fomc"):
                score -= 10  # FOMC is uniquely unpredictable

        # Sentiment as proxy for event positioning
        sent = data.get("sentiment", {})
        composite = sent.get("composite_score", 0) or 0
        direction = data.get("direction_hint", "")
        if direction == "call" and composite > 0.3:
            score += 10
        elif direction == "put" and composite < -0.3:
            score += 10

        # Social intelligence signals (catalysts, flow, risks)
        social = sent.get("social", {})
        if social:
            # Catalyst bonus: specific catalysts being discussed = informed positioning
            catalysts = social.get("catalysts", [])
            if catalysts:
                score += min(len(catalysts) * 4, 12)

            # Flow alignment: institutional flow (UW sweeps) confirming direction
            flow_consensus = social.get("flow_consensus", "neutral")
            flow_conviction = social.get("flow_conviction", 0)
            if flow_consensus != "neutral" and flow_conviction > 0.3:
                if (direction == "call" and flow_consensus == "bullish") or \
                   (direction == "put" and flow_consensus == "bearish"):
                    score += 8  # Flow confirms thesis
                elif (direction == "call" and flow_consensus == "bearish") or \
                     (direction == "put" and flow_consensus == "bullish"):
                    score -= 6  # Flow contradicts thesis — caution

            # DD-quality posts: substantive analysis adds conviction
            post_types = social.get("post_types", {})
            dd_count = post_types.get("dd", 0)
            if dd_count >= 2:
                score += 6  # Multiple DD posts = informed community attention

            # Risk flags from social: crowd-sourced risk detection
            risks = social.get("risks", [])
            if risks:
                score -= min(len(risks) * 3, 10)

        # Analyst consensus (Finviz)
        fv = data.get("finviz", {})
        rating = fv.get("analyst_rating")
        if rating in ("Strong Buy", "Buy") and direction == "call":
            score += 10
        elif rating in ("Strong Sell", "Sell") and direction == "put":
            score += 10

        # Insider signal
        insider = data.get("insider", {})
        insider_signal = insider.get("insider_signal", "neutral")
        if insider_signal == "bullish" and direction == "call":
            score += 8
        elif insider_signal == "bearish" and direction == "put":
            score += 5

        # Earnings during Mon-Fri holding window — hard penalty.
        # Earnings create gamma spikes and IV crush that invalidate
        # the delta/theta assumptions the rest of the model relies on.
        earnings_date = (
            data.get("earnings_date")
            or fv.get("earnings_date")
            or data.get("finviz", {}).get("earnings_date")
        )
        if earnings_date:
            # If we have a date string, check if it's this week
            try:
                from datetime import datetime as _dt, timedelta
                if isinstance(earnings_date, str):
                    # Try common formats
                    for fmt in ("%Y-%m-%d", "%b %d", "%m/%d/%Y"):
                        try:
                            ed = _dt.strptime(earnings_date, fmt)
                            if fmt == "%b %d":
                                ed = ed.replace(year=_dt.now().year)
                            today = _dt.now()
                            # Monday of this week to Friday
                            mon = today - timedelta(days=today.weekday())
                            fri = mon + timedelta(days=4)
                            if mon.date() <= ed.date() <= fri.date():
                                score -= 25  # Heavy penalty: earnings in hold window
                                data["_earnings_in_window"] = True
                            break
                        except ValueError:
                            continue
            except Exception:
                pass
        # Also flag the boolean from narrowing/finviz
        if data.get("earnings_warning"):
            score -= 15  # Already flagged by upstream — additional penalty

        return float(np.clip(score, 0, 100))

    # ======================================================================
    #  TIER 3: EXECUTION QUALITY SUB-SCORES (15% of composite)
    # ======================================================================

    def _liquidity_score(self, data: dict) -> float:
        """Score 0-100 for options liquidity (bid-ask, volume, OI)."""
        opts = data.get("options", {})
        score = 50.0

        # Bid-ask spread quality
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
                score += 25
            elif avg_spread < 10:
                score += 15
            elif avg_spread < 20:
                score += 5
            elif avg_spread > 30:
                score -= 15

        # Total options volume
        total_vol = (opts.get("total_call_volume", 0) or 0) + (opts.get("total_put_volume", 0) or 0)
        score += min(20, total_vol / 50000)

        return float(np.clip(score, 0, 100))

    def _strike_efficiency_score(self, data: dict) -> float:
        """Score 0-100 for how well the optimal strike matches the setup.

        Max pain divergence + volume concentration at nearby strikes.
        """
        opts = data.get("options", {})
        score = 50.0

        # Max pain divergence: price should be near max pain for pin potential,
        # or far from it if we're betting on a breakout
        current_price = opts.get("current_price", 0) or 0
        max_pain = opts.get("max_pain", 0) or 0
        if current_price > 0 and max_pain > 0:
            divergence_pct = abs(current_price - max_pain) / current_price * 100
            if divergence_pct > 3:
                score += 15  # enough room for max-pain pull
            elif divergence_pct > 1:
                score += 5

        # IV skew: negative skew (calls expensive) favors puts and vice versa
        iv_skew = opts.get("iv_skew")
        direction = data.get("direction_hint", "")
        if iv_skew is not None:
            if iv_skew > 0.05 and direction == "call":
                score += 10  # puts expensive, calls relatively cheap
            elif iv_skew < -0.05 and direction == "put":
                score += 10

        return float(np.clip(score, 0, 100))

    def _theta_cost_score(self, data: dict) -> float:
        """Score 0-100 for theta efficiency and charm (delta decay).

        Lower daily theta as % of premium = better for weekly holds.
        Charm measures how much delta erodes per day even without stock
        movement — high charm means your position is silently dying.
        """
        opts = data.get("options", {})
        score = 50.0

        theta_pct = opts.get("theta_per_day_pct")
        if theta_pct is not None:
            if theta_pct < 10:
                score += 25  # excellent theta efficiency
            elif theta_pct < 15:
                score += 15
            elif theta_pct < 20:
                score += 5
            elif theta_pct > 25:
                score -= 15  # expensive theta

        # Days to expiry bonus (more days = better for weeklies)
        dte = opts.get("days_to_expiry", 5)
        if dte >= 5:
            score += 10
        elif dte >= 3:
            score += 5
        elif dte < 2:
            score -= 20  # too close to expiry

        # Charm penalty: if delta decays rapidly, the position loses
        # directional exposure even on flat days. Use theta_decay_curve
        # or greeks from strike data.
        theta_curve = opts.get("theta_decay_curve")
        if theta_curve and len(theta_curve) >= 3:
            # Compare charm on day 1 vs day 3 — accelerating charm is bad
            charm_d1 = abs(theta_curve[0].get("charm_per_day", 0) or 0)
            charm_d3 = abs(theta_curve[2].get("charm_per_day", 0) or 0)
            if charm_d3 > 0 and charm_d1 > 0:
                charm_accel = charm_d3 / charm_d1
                if charm_accel > 2.0:
                    score -= 10  # charm accelerating fast (delta dying)
                elif charm_accel > 1.5:
                    score -= 5

        # Charm-adjusted delta: estimate what delta will be by Wednesday.
        # If charm eats delta below 0.15 by mid-week, the position is
        # effectively dead before the thesis can play out.
        greeks = opts.get("greeks", {})
        entry_delta = abs(greeks.get("delta") or opts.get("estimated_delta") or 0)
        charm = abs(greeks.get("charm") or 0)
        if entry_delta > 0 and charm > 0:
            # Project delta at day 3 (Wednesday): delta - 3 * charm
            wed_delta = entry_delta - (3 * charm)
            if wed_delta < 0.10:
                score -= 20  # Position will be nearly worthless by mid-week
            elif wed_delta < 0.15:
                score -= 10  # Marginal delta by mid-week — risky
            # Also penalize if more than half of delta lost to charm by Wed
            if charm * 3 > entry_delta * 0.5:
                score -= 5  # Charm eating >50% of delta in 3 days

        # Stock-level IV term structure: cheap weekly IV = better entry
        iv_ts = opts.get("iv_term_structure")
        if iv_ts and isinstance(iv_ts, dict):
            mispricing = iv_ts.get("mispricing_signal")
            if mispricing == "cheap":
                score += 8  # weekly IV below monthly = buying at discount
            elif mispricing == "expensive":
                score -= 8  # weekly IV elevated (event premium)

        return float(np.clip(score, 0, 100))

    # ======================================================================
    #  Public scoring API
    # ======================================================================

    def score_candidate(self, ticker_data: dict) -> dict:
        """Produce a composite score (0-100) for a single candidate.

        Uses ensemble of 3 independent scoring models for consensus:
        1. Linear factor model (primary 10-factor weighted)
        2. Momentum-only model (trend + momentum signals)
        3. Mean-reversion model (RSI extremes + IV mispricing)

        Final score is a consensus-weighted average. Agreement between
        models increases confidence; disagreement reduces it.
        """
        sub_scores = {
            # Tier 1: Direction
            "momentum": self._momentum_score(ticker_data),
            "mean_reversion": self._mean_reversion_score(ticker_data),
            "regime_bias": self._regime_bias_score(ticker_data),
            "trend_persistence": self._trend_persistence_score(ticker_data),
            # Tier 2: Edge Quality
            "iv_mispricing": self._iv_mispricing_score(ticker_data),
            "flow_conviction": self._flow_conviction_score(ticker_data),
            "event_risk": self._event_risk_score(ticker_data),
            # Tier 3: Execution
            "liquidity": self._liquidity_score(ticker_data),
            "strike_efficiency": self._strike_efficiency_score(ticker_data),
            "theta_cost": self._theta_cost_score(ticker_data),
        }

        # --- New market signal adjustments ---
        market_adj = self._market_signal_adjustments(ticker_data)

        regime = ticker_data.get("market_regime", {}).get("regime", "normal")
        active_weights = self._regime_adjusted_weights(regime)

        # Model 1: Primary linear factor model (weight: 0.50)
        linear_score = sum(
            sub_scores[cat] * active_weights.get(cat, 0)
            for cat in sub_scores
        ) + market_adj

        # Model 2: Momentum-only model (weight: 0.25)
        momentum_score = (
            sub_scores["momentum"] * 0.40
            + sub_scores["trend_persistence"] * 0.35
            + sub_scores["regime_bias"] * 0.15
            + sub_scores["flow_conviction"] * 0.10
        )

        # Model 3: Mean-reversion + value model (weight: 0.25)
        reversion_score = (
            sub_scores["mean_reversion"] * 0.35
            + sub_scores["iv_mispricing"] * 0.30
            + sub_scores["event_risk"] * 0.15
            + sub_scores["theta_cost"] * 0.10
            + sub_scores["liquidity"] * 0.10
        )

        # Ensemble consensus
        models = [linear_score, momentum_score, reversion_score]
        model_weights = [0.50, 0.25, 0.25]
        composite = sum(m * w for m, w in zip(models, model_weights))

        # Model agreement bonus/penalty
        model_std = float(np.std(models))
        if model_std < 5:
            composite *= 1.05  # strong consensus boost
        elif model_std > 15:
            composite *= 0.92  # disagreement penalty

        # Holding window event adjustment
        holding = ticker_data.get("holding_window", {})
        if not holding and self.market_summary:
            holding = self.market_summary.get("holding_window", {})
        if holding and holding.get("risk_level") == "high":
            composite *= 0.95

        # Pattern library adjustment
        pattern_info = self.pattern_library.get_pattern_adjustment(ticker_data)
        composite += pattern_info["adjustment"]

        result = dict(ticker_data)
        result["scores"] = sub_scores
        result["ensemble"] = {
            "linear": round(float(linear_score), 2),
            "momentum": round(float(momentum_score), 2),
            "reversion": round(float(reversion_score), 2),
            "model_std": round(model_std, 2),
            "consensus": "strong" if model_std < 5 else ("weak" if model_std > 15 else "moderate"),
        }
        result["pattern"] = pattern_info
        result["composite_score"] = round(float(np.clip(composite, 0, 100)), 2)
        result["active_regime"] = regime
        result["active_weights"] = {k: round(v, 4) for k, v in active_weights.items()}
        return result

    def _market_signal_adjustments(self, ticker_data: dict) -> float:
        """Score adjustment from market-wide signals.

        Integrates: CBOE skew, market P/C, VIX term structure,
        CFTC COT speculator positioning, macro surprise index.
        """
        adj = 0.0
        market = ticker_data.get("market_regime", {})
        if not market and self.market_summary:
            market = self.market_summary or {}

        direction = ticker_data.get("direction_hint", "neutral")

        # CBOE Skew Index — tail risk sentiment
        skew = market.get("skew", {})
        skew_class = skew.get("classification", "normal")
        if skew_class == "extreme_fear" and direction == "put":
            adj += 3.0  # tail risk confirms put direction
        elif skew_class == "extreme_fear" and direction == "call":
            adj -= 2.0  # going against tail fear
        elif skew_class == "complacent" and direction == "call":
            adj += 1.5  # low fear environment favors calls

        # Market-wide P/C ratio (contrarian)
        mkt_pc = market.get("market_put_call", {})
        mkt_ratio = mkt_pc.get("ratio")
        if mkt_ratio is not None:
            if mkt_ratio > 1.2 and direction == "call":
                adj += 2.0  # extreme put buying = contrarian bullish
            elif mkt_ratio < 0.7 and direction == "put":
                adj += 2.0  # extreme call buying = contrarian bearish

        # VIX term structure (contango vs backwardation)
        vix_term = market.get("vix_term_structure", {})
        structure = vix_term.get("structure")
        if structure == "backwardation" and direction == "put":
            adj += 2.5  # near-term fear elevated = put confirmation
        elif structure == "backwardation" and direction == "call":
            adj -= 1.5  # near-term fear not ideal for calls
        elif structure == "steep_contango" and direction == "call":
            adj += 1.5  # calm near-term, risk further out = call-friendly

        # CFTC COT — speculator positioning (contrarian weekly signal)
        cot = market.get("cot_positioning", {})
        cot_signal = cot.get("signal", "neutral")
        if cot_signal == "extreme_long" and direction == "put":
            adj += 2.5  # speculators max long → vulnerable to pullback
        elif cot_signal == "extreme_long" and direction == "call":
            adj -= 1.5  # crowded long = risky for new longs
        elif cot_signal == "extreme_short" and direction == "call":
            adj += 2.5  # speculators max short → contrarian bullish
        elif cot_signal == "extreme_short" and direction == "put":
            adj -= 1.5  # short squeeze risk for puts
        elif cot_signal == "long" and direction == "put":
            adj += 1.0
        elif cot_signal == "short" and direction == "call":
            adj += 1.0

        # Macro surprise — regime persistence signal
        macro = market.get("macro_surprise", {})
        macro_signal = macro.get("signal", "inline")
        if macro_signal == "beating" and direction == "call":
            adj += 2.0  # economy outperforming → momentum persists, bullish
        elif macro_signal == "beating" and direction == "put":
            adj -= 1.0  # harder to be bearish when macro beats
        elif macro_signal == "missing" and direction == "put":
            adj += 2.0  # economy disappointing → risk-off, bearish
        elif macro_signal == "missing" and direction == "call":
            adj -= 1.0  # uphill for calls when macro misses

        return adj

    def rank_candidates(self, candidates_list: list, top_n: int = 10) -> list:
        scored = [self.score_candidate(c) for c in candidates_list]
        scored.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        logger.info(
            "Ranked %d candidates — top: %.1f, bottom: %.1f",
            len(scored),
            scored[0]["composite_score"] if scored else 0,
            scored[-1]["composite_score"] if scored else 0,
        )
        return scored[:top_n]

    # ======================================================================
    #  Direction determination — weekly-appropriate signals
    # ======================================================================

    def determine_direction(self, ticker_data: dict) -> dict:
        """Decide call vs put using weekly-horizon signals.

        Replaces the 0DTE vote system with signals that predict
        5-day directional moves.
        """
        bullish = 0.0
        bearish = 0.0

        tech = ticker_data.get("technical", {})
        opts = ticker_data.get("options", {})

        # 1. Multi-day momentum (strongest weekly signal)
        ret_5d = tech.get("return_5d", 0) or 0
        ret_10d = tech.get("return_10d", 0) or 0
        if ret_5d > 1.0:
            bullish += 1.5
        elif ret_5d > 0:
            bullish += 0.5
        elif ret_5d < -1.0:
            bearish += 1.5
        elif ret_5d < 0:
            bearish += 0.5

        # 2. Trend strength (ADX + direction)
        trend = tech.get("trend_strength", "flat")
        if trend == "strong_up":
            bullish += 1.5
        elif trend == "up":
            bullish += 0.8
        elif trend == "strong_down":
            bearish += 1.5
        elif trend == "down":
            bearish += 0.8

        # 3. RSI — mean reversion at extremes (weekly thresholds)
        rsi = tech.get("rsi", 50) or 50
        if rsi < 30:
            bullish += 1.0  # deeply oversold → bounce
        elif rsi < 40:
            bullish += 0.3
        elif rsi > 70:
            bearish += 1.0  # deeply overbought → pullback
        elif rsi > 60:
            bearish += 0.3

        # 4. Price vs SMA20 (trend bias)
        price_vs_sma20 = tech.get("price_vs_sma20_pct", 0)
        if price_vs_sma20 is not None:
            if price_vs_sma20 > 2:
                bullish += 0.8
            elif price_vs_sma20 > 0:
                bullish += 0.3
            elif price_vs_sma20 < -2:
                bearish += 0.8
            elif price_vs_sma20 < 0:
                bearish += 0.3

        # 5. MACD histogram direction (momentum confirmation)
        hist = tech.get("macd_histogram", 0) or 0
        hist_prev = tech.get("macd_histogram_prev", 0) or 0
        if hist > 0 and hist > hist_prev:
            bullish += 0.5  # expanding bullish
        elif hist < 0 and hist < hist_prev:
            bearish += 0.5  # expanding bearish
        elif hist * hist_prev < 0:
            # crossover
            if hist > 0:
                bullish += 0.8
            else:
                bearish += 0.8

        # 6. Put/call ratio (contrarian for weeklies)
        pc_ratio = opts.get("pc_ratio_volume", 1.0) or 1.0
        if pc_ratio > 1.3:
            bearish += 0.5
        elif pc_ratio < 0.7:
            bullish += 0.5

        # 7. Max pain pull (weekly effect is stronger than daily)
        current_price = opts.get("current_price", 0) or 0
        max_pain = opts.get("max_pain", 0) or 0
        if current_price > 0 and max_pain > 0:
            mp_dist = (max_pain - current_price) / current_price * 100
            if mp_dist > 1:
                bullish += 0.5  # price pulled up toward max pain
            elif mp_dist < -1:
                bearish += 0.5

        # 8. Analyst consensus (Finviz)
        fv = ticker_data.get("finviz", {})
        rating = fv.get("analyst_rating")
        if rating in ("Strong Buy", "Buy"):
            bullish += 0.4
        elif rating in ("Strong Sell", "Sell"):
            bearish += 0.4

        # 9. Insider signal
        insider = ticker_data.get("insider", {})
        insider_signal = insider.get("insider_signal", "neutral")
        if insider_signal == "bullish":
            bullish += 0.6
        elif insider_signal == "bearish":
            bearish += 0.3

        # 10. VIX regime bias
        regime = ticker_data.get("market_regime", {})
        vix_current = regime.get("current", 0) or 0
        if vix_current > 25:
            bearish += 0.4
        elif vix_current < 14:
            bullish += 0.2

        total_signals = bullish + bearish
        if total_signals == 0:
            return {"direction": "call", "confidence": 0.0, "raw_confidence": 0.0}

        raw_confidence = abs(bullish - bearish) / total_signals
        direction = "call" if bullish >= bearish else "put"
        # Use pattern library's live calibration if available, else static table
        calibrated = self.pattern_library.get_calibrated_confidence(raw_confidence)
        confidence = self._calibrate_confidence(raw_confidence)
        # Blend: 60% live calibration, 40% static (if live is available)
        if calibrated != raw_confidence:
            confidence = 0.6 * calibrated + 0.4 * confidence

        return {
            "direction": direction,
            "confidence": round(float(np.clip(confidence, 0, 1)), 3),
            "raw_confidence": round(float(raw_confidence), 3),
            "bullish_score": round(float(bullish), 2),
            "bearish_score": round(float(bearish), 2),
        }

    def _calibrate_confidence(self, raw: float) -> float:
        for (lo, hi), calibrated in self.CONFIDENCE_CALIBRATION.items():
            if lo <= raw < hi:
                return calibrated
        return raw

    # ======================================================================
    #  Macro regime gate
    # ======================================================================

    def assess_macro_edge(self, market_summary: dict = None) -> dict:
        """Evaluate whether the macro environment gives the model an edge.

        For weekly options, regime persistence matters more than for 0DTE.
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
            penalty += 0.15
            reasons.append(f"VIX {regime_name} ({vix_level:.1f}) — lower directional edge")
        elif regime_name in ("high", "extreme"):
            penalty -= 0.15
            reasons.append(f"VIX {regime_name} ({vix_level:.1f}) — higher directional clarity")
        elif regime_name == "elevated":
            penalty -= 0.05
            reasons.append(f"VIX elevated ({vix_level:.1f}) — moderate edge")

        # Regime persistence (new for weekly)
        persistence = summary.get("regime_persistence", {})
        if not persistence.get("is_stable", True):
            penalty += 0.10
            trans = persistence.get("transition_direction", "")
            reasons.append(f"VIX regime transitioning ({trans}) — signals may not hold 5 days")
        elif persistence.get("persistence_score", 0.5) > 0.7:
            penalty -= 0.05
            reasons.append("VIX regime stable — good for weekly holds")

        # Credit spread
        credit = summary.get("credit_spread", {})
        credit_state = credit.get("credit_state")
        hy_oas = credit.get("hy_oas")
        if credit_state == "tight":
            penalty += 0.10
            reasons.append(f"Credit spreads tight (HY OAS {hy_oas}%) — complacent")
        elif credit_state in ("wide", "stressed"):
            penalty -= 0.10
            reasons.append(f"Credit spreads {credit_state} — better signal clarity")

        # Financial conditions
        conditions = summary.get("financial_conditions", {})
        cond_state = conditions.get("conditions_state")
        nfci = conditions.get("nfci")
        if cond_state == "loose":
            penalty += 0.10
            reasons.append(f"Financial conditions loose (NFCI {nfci})")
        elif cond_state in ("tightening", "tight"):
            penalty -= 0.05
            reasons.append(f"Financial conditions {cond_state} — better signals")

        # Holding window events
        holding = summary.get("holding_window", {})
        if holding.get("risk_level") == "high":
            penalty += 0.10
            reasons.append(f"High-impact events in holding window: {holding.get('event_days', [])}")

        # COT positioning — extreme crowding reduces edge
        cot = summary.get("cot_positioning", {})
        cot_signal = cot.get("signal", "neutral")
        if cot_signal in ("extreme_long", "extreme_short"):
            penalty -= 0.05  # extreme positioning = clearer contrarian signal
            reasons.append(f"COT {cot_signal} — strong contrarian signal available")

        # Macro surprise — inline macro = less directional clarity
        macro = summary.get("macro_surprise", {})
        macro_signal = macro.get("signal", "inline")
        if macro_signal == "inline":
            penalty += 0.05
            reasons.append("Macro data inline with expectations — less directional clarity")
        elif macro_signal in ("beating", "missing"):
            penalty -= 0.05
            score_val = macro.get("surprise_score", 0)
            reasons.append(f"Macro {macro_signal} (score {score_val:+.2f}) — directional clarity")

        confidence_multiplier = max(0.0, min(1.0, 1.0 - penalty))
        has_edge = confidence_multiplier >= 0.50

        if not has_edge:
            reasons.append("REGIME GATE: low-edge environment — reduce sizing or skip")

        return {
            "has_edge": has_edge,
            "confidence_multiplier": round(confidence_multiplier, 3),
            "regime": regime_name,
            "reasons": reasons,
        }

    # ======================================================================
    #  Weight persistence
    # ======================================================================

    def save_weights(self) -> None:
        WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WEIGHTS_FILE, "w") as fh:
            json.dump(self.weights, fh, indent=2)
        logger.info("Saved scoring weights to %s", WEIGHTS_FILE)

    def update_weights(self, performance_data: dict) -> None:
        """Adjust weights based on signal correlations with outcomes.

        Uses a conservative Bayesian-inspired update (alpha=0.08) with
        minimum sample requirements.
        """
        correlations = performance_data.get("signal_correlations", {})
        min_samples = performance_data.get("sample_size", 0)

        if not correlations:
            logger.info("No signal correlations — weights unchanged")
            return

        # Require minimum 15 picks before adjusting weights
        if min_samples < 15:
            logger.info("Only %d samples — need 15+ before weight adjustment", min_samples)
            return

        alpha = 0.08  # conservative learning rate (was 0.15 in 0DTE)
        raw = {}
        for category, current_w in self.weights.items():
            corr = correlations.get(category, 0.0)
            target = current_w * (1 + corr)
            raw[category] = current_w * (1 - alpha) + target * alpha

        total = sum(raw.values())
        if total > 0:
            self.weights = {k: round(v / total, 4) for k, v in raw.items()}
        else:
            self.weights = dict(self.DEFAULT_WEIGHTS)

        self.save_weights()
        logger.info("Updated scoring weights: %s", self.weights)
