"""
Narrowing pipeline for the Weekly Options Trading Analysis System.

Filters candidates through three stages on a Wed/Fri/Mon cadence:
    Wednesday scan      - Broad universe down to 25 candidates + 10 bench
    Friday refresh      - Delta-aware re-ranking of 35 (25 + bench) to best 20
    Monday picks        - 20 candidates down to 3 high-conviction picks with diversity
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()


class NarrowingPipeline:
    """Filter candidates through the Wednesday/Friday/Monday pipeline."""

    STAGE_TARGETS = {
        "wednesday_scan": 25,
        "friday_refresh": 20,
        "monday_picks": 3,
    }

    BENCH_SIZE = 10  # extra candidates saved as alternates

    STAGE_ORDER = list(STAGE_TARGETS.keys())

    def __init__(self):
        self.stage_names = list(self.STAGE_TARGETS.keys())

    # ------------------------------------------------------------------
    #  Stage-specific filtering
    # ------------------------------------------------------------------

    def narrow(self, candidates: list, stage: str, scanner_results: dict = None) -> list:
        """Apply stage-specific filtering and return narrowed list.

        Parameters
        ----------
        candidates : list[dict]
            Each element is a candidate dict (must contain 'ticker' key).
        stage : str
            One of the stage names in ``STAGE_TARGETS``.
        scanner_results : dict or None
            Additional scanner output used by certain stages.

        Returns
        -------
        list[dict]
            Filtered and (possibly) enriched candidate list.
        """
        scanner_results = scanner_results or {}
        target = self.STAGE_TARGETS.get(stage, len(candidates))

        if stage == "wednesday_scan":
            filtered = self._wednesday_scan_filter(candidates, scanner_results)
        elif stage == "friday_refresh":
            filtered = self._friday_refresh_filter(candidates, scanner_results)
        elif stage == "monday_picks":
            filtered = self._monday_picks_filter(candidates, scanner_results)
        else:
            logger.warning("Unknown stage '%s' — returning candidates unchanged", stage)
            return candidates

        result = filtered[:target]
        logger.info(
            "Stage '%s': %d -> %d candidates (target %d)",
            stage, len(candidates), len(result), target,
        )
        return result

    def narrow_with_bench(self, candidates: list, stage: str,
                          bench_size: int = None,
                          scanner_results: dict = None) -> tuple:
        """Like narrow() but also returns the next *bench_size* candidates.

        Parameters
        ----------
        candidates : list[dict]
        stage : str
        bench_size : int or None
            Defaults to ``BENCH_SIZE``.
        scanner_results : dict or None
            Additional context passed to the stage filter.

        Returns
        -------
        tuple[list, list]
            (narrowed top candidates, bench candidates)
        """
        if bench_size is None:
            bench_size = self.BENCH_SIZE
        scanner_results = scanner_results or {}

        target = self.STAGE_TARGETS.get(stage, len(candidates))

        if stage == "wednesday_scan":
            filtered = self._wednesday_scan_filter(candidates, scanner_results)
        elif stage == "friday_refresh":
            filtered = self._friday_refresh_filter(candidates, scanner_results)
        else:
            logger.warning("narrow_with_bench not supported for stage '%s'", stage)
            return candidates[:target], []

        top = filtered[:target]
        bench = filtered[target:target + bench_size]

        logger.info(
            "Stage '%s': %d -> %d candidates + %d bench",
            stage, len(candidates), len(top), len(bench),
        )
        return top, bench

    # ------------------------------------------------------------------
    #  Individual stage filters
    # ------------------------------------------------------------------

    def _wednesday_scan_filter(self, candidates: list, results: dict) -> list:
        """Filter and rank for Wednesday broad scan — target 25.

        Uses weekly-appropriate signals: multi-day momentum, ADX/trend
        strength, weekly ATR, and options liquidity. Replaces 0DTE's
        ATR-extremity/RSI-extremity focus with directional persistence.

        Hard filters:
        - Weekly ATR% > 1.5 (stock moves enough for weekly option profit)
        - Options volume > 500 (enough liquidity for entry/exit)
        - Price > $10 (avoid penny stock noise)

        Ranking by weekly_scan_rank composite:
        - Momentum quality (30%): 5d/10d returns, direction consistency
        - Trend strength (25%): ADX, trend classification, SMA slopes
        - Volatility profile (20%): weekly ATR, relative volume
        - Options quality (15%): volume, spread quality, IV rank
        - Setup extremity (10%): RSI/BB for mean-reversion setups
        """
        scored = []
        for c in candidates:
            if isinstance(c, str):
                c = {"ticker": c}

            tech = c.get("technical", {})
            opts = c.get("options", {})
            ticker = c.get("ticker", "?")

            # --- Hard filters ---
            # Weekly ATR: use weekly_atr_pct if available, else scale daily
            weekly_atr = tech.get("weekly_atr_pct") or 0
            if weekly_atr == 0:
                daily_atr = tech.get("atr_pct", 0) or 0
                weekly_atr = daily_atr * 2.236  # sqrt(5) scaling
            if weekly_atr < 1.5:
                logger.debug("Removing %s — weekly ATR%% %.2f < 1.5", ticker, weekly_atr)
                continue

            # Price floor
            price = tech.get("price") or opts.get("current_price") or 0
            if price < 10:
                logger.debug("Removing %s — price $%.2f < $10", ticker, price)
                continue

            # Options data must exist with minimum volume
            if not opts:
                logger.debug("Removing %s — no options data", ticker)
                continue
            total_vol = (opts.get("total_call_volume", 0) or 0) + (opts.get("total_put_volume", 0) or 0)
            if total_vol < 500:
                logger.debug("Removing %s — options volume %d < 500", ticker, total_vol)
                continue

            # --- Momentum quality (30%) ---
            ret_5d = tech.get("return_5d", 0) or 0
            ret_10d = tech.get("return_10d", 0) or 0
            ret_21d = tech.get("return_21d", 0) or 0

            # Magnitude of recent move (higher = more interesting)
            mag_5d = min(abs(ret_5d) / 5.0, 1.0)  # 5% move maxes out
            mag_10d = min(abs(ret_10d) / 8.0, 1.0)

            # Direction consistency bonus: 5d and 10d agree
            if ret_5d != 0 and ret_10d != 0:
                direction_match = 1.0 if (ret_5d * ret_10d > 0) else 0.3
            else:
                direction_match = 0.5

            # Multi-timeframe alignment: 5d, 10d, 21d all agree
            signs = [ret_5d > 0, ret_10d > 0, ret_21d > 0]
            alignment = sum(signs) / 3.0 if all(r != 0 for r in [ret_5d, ret_10d, ret_21d]) else 0.5
            # Reward strong agreement (all bullish or all bearish)
            alignment_score = abs(alignment - 0.5) * 2  # 0 to 1

            momentum_score = (
                mag_5d * 0.40
                + mag_10d * 0.25
                + direction_match * 0.20
                + alignment_score * 0.15
            )

            # --- Trend strength (25%) ---
            adx = tech.get("adx") or 0
            trend = tech.get("trend_strength", "flat")
            sma20_slope = tech.get("sma20_slope") or 0
            sma50_slope = tech.get("sma50_slope") or 0

            # ADX: >25 trending, >40 very strong
            adx_sub = min(adx / 50.0, 1.0) if adx > 15 else adx / 30.0

            # Trend classification
            trend_map = {
                "strong_up": 1.0, "strong_down": 1.0,
                "up": 0.7, "down": 0.7,
                "flat": 0.2,
            }
            trend_sub = trend_map.get(trend, 0.3)

            # SMA slope confirmation (both slopes agree = strong trend)
            if sma20_slope != 0 and sma50_slope != 0:
                slopes_agree = 1.0 if (sma20_slope * sma50_slope > 0) else 0.3
            else:
                slopes_agree = 0.5

            trend_score = adx_sub * 0.40 + trend_sub * 0.35 + slopes_agree * 0.25

            # --- Volatility profile (20%) ---
            # Weekly ATR normalized (3-8% is sweet spot for weeklies)
            if weekly_atr >= 3 and weekly_atr <= 8:
                atr_sub = 1.0  # ideal range
            elif weekly_atr > 8:
                atr_sub = max(0.4, 1.0 - (weekly_atr - 8) / 10.0)  # too volatile
            else:
                atr_sub = min(weekly_atr / 3.0, 1.0)  # below sweet spot

            # Relative volume
            vol_ratio = tech.get("volume_ratio", 1.0) or 1.0
            vol_sub = min(vol_ratio / 2.0, 1.0)

            vol_score = atr_sub * 0.60 + vol_sub * 0.40

            # --- Options quality (15%) ---
            # Spread quality
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
                    spread_sub = 1.0
                elif avg_spread < 15:
                    spread_sub = 0.7
                elif avg_spread < 30:
                    spread_sub = 0.4
                else:
                    spread_sub = 0.15
            else:
                spread_sub = 0.5

            # Options volume quality
            vol_quality = min(total_vol / 10000, 1.0)

            # IV rank (higher = more premium but also more opportunity)
            iv_rank = opts.get("iv_rank", 0) or 0
            # Sweet spot: 30-70 IV rank (not too cheap, not too expensive)
            if 30 <= iv_rank <= 70:
                iv_sub = 0.8 + (1.0 - abs(iv_rank - 50) / 20) * 0.2
            elif iv_rank > 70:
                iv_sub = 0.5  # expensive premium
            else:
                iv_sub = 0.6  # cheap but may lack catalyst

            opts_score = spread_sub * 0.40 + vol_quality * 0.30 + iv_sub * 0.30

            # --- Setup extremity (10%) ---
            rsi = tech.get("rsi", 50) or 50
            rsi_extremity = min(abs(rsi - 50) / 30.0, 1.0)  # peaks at RSI 20 or 80

            bb_pct = tech.get("bb_pct", 0.5)
            if bb_pct is None:
                bb_pct = 0.5
            bb_extremity = min(abs(bb_pct - 0.5) / 0.8, 1.0)

            extremity_score = rsi_extremity * 0.60 + bb_extremity * 0.40

            # --- Composite scan rank ---
            scan_rank = (
                momentum_score * 0.30
                + trend_score * 0.25
                + vol_score * 0.20
                + opts_score * 0.15
                + extremity_score * 0.10
            )

            c = dict(c)
            c["scan_rank"] = round(scan_rank, 4)
            c["scan_components"] = {
                "momentum": round(momentum_score, 4),
                "trend": round(trend_score, 4),
                "volatility": round(vol_score, 4),
                "options": round(opts_score, 4),
                "extremity": round(extremity_score, 4),
            }
            scored.append(c)

        scored.sort(key=lambda x: x.get("scan_rank", 0), reverse=True)
        return scored

    def _friday_refresh_filter(self, candidates: list, results: dict) -> list:
        """Delta-aware re-ranking for Friday — target 20.

        Evaluates how each candidate's setup is *evolving* over the
        Wednesday-to-Friday window. Compares Wednesday snapshot against
        Friday's fresh data to detect setups that are strengthening,
        weakening, or catalyzed by new information.

        Hard filters (relaxed from Wednesday):
        - Weekly ATR% >= 1.0 (was 1.5 — allow slight compression if setup evolving)
        - Options volume >= 300 (was 500)
        - wednesday_snapshot must exist (need baseline for deltas)

        Scoring components (each 0-1, weighted):
        - Setup momentum (0.25): RSI trajectory, MACD change, BB evolution, volume sustaining
        - Trend evolution (0.20): ADX change, trend persistence, SMA slope acceleration
        - IV & premium trajectory (0.20): IV expanding vs collapsing, premium resilience
        - Options positioning (0.15): P/C ratio shift, max pain convergence
        - Base quality floor (0.15): Recalculated scan_rank on fresh data
        - Sentiment evolution (0.05): Sentiment delta + article count change
        """
        scored = []

        for c in candidates:
            c = dict(c)
            ticker = c.get("ticker", "?")
            tech = c.get("technical", {})
            opts = c.get("options", {})
            snap = c.get("wednesday_snapshot", {})
            snap_tech = snap.get("technical", {})
            snap_opts = snap.get("options", {})

            # --- Hard filters ---
            weekly_atr = tech.get("weekly_atr_pct") or 0
            if weekly_atr == 0:
                daily_atr = tech.get("atr_pct", 0) or 0
                weekly_atr = daily_atr * 2.236
            if weekly_atr < 1.0:
                logger.debug("Fri filter: removing %s — weekly ATR%% %.2f < 1.0", ticker, weekly_atr)
                continue

            if not opts:
                logger.debug("Fri filter: removing %s — no options data", ticker)
                continue
            total_vol = (opts.get("total_call_volume", 0) or 0) + (opts.get("total_put_volume", 0) or 0)
            if total_vol < 300:
                logger.debug("Fri filter: removing %s — options volume %d < 300", ticker, total_vol)
                continue

            has_snapshot = bool(snap and snap_tech)

            # ============================================================
            # Component 1: Setup Momentum (weight 0.25)
            # ============================================================
            if has_snapshot:
                # RSI trajectory
                rsi_fri = (tech.get("rsi") or 50)
                rsi_wed = (snap_tech.get("rsi") or 50)
                rsi_delta = rsi_fri - rsi_wed

                if rsi_wed < 35 and rsi_delta > 0:
                    rsi_sub = min(abs(rsi_delta) / 10.0, 1.0)
                elif rsi_wed > 65 and rsi_delta < 0:
                    rsi_sub = min(abs(rsi_delta) / 10.0, 1.0)
                elif abs(rsi_fri - 50) > abs(rsi_wed - 50):
                    rsi_sub = 0.8  # moving further from neutral
                elif abs(rsi_fri - 50) < abs(rsi_wed - 50):
                    rsi_sub = 0.3  # reverting — setup weakening
                else:
                    rsi_sub = 0.5

                # MACD histogram momentum
                hist_fri = tech.get("macd_histogram", 0) or 0
                hist_wed = snap_tech.get("macd_histogram", 0) or 0

                if hist_fri != 0 and hist_wed != 0 and (hist_fri * hist_wed < 0):
                    macd_sub = 1.0  # fresh crossover
                elif abs(hist_fri) > abs(hist_wed) and hist_fri * hist_wed >= 0:
                    macd_sub = 0.8  # expanding same direction
                elif abs(hist_fri) <= abs(hist_wed) and hist_fri * hist_wed >= 0:
                    macd_sub = 0.4  # contracting same sign
                else:
                    macd_sub = 0.3

                # BB position evolution
                bb_fri = tech.get("bb_pct", 0.5) if tech.get("bb_pct") is not None else 0.5
                bb_wed = snap_tech.get("bb_pct", 0.5) if snap_tech.get("bb_pct") is not None else 0.5

                now_outside = bb_fri < 0 or bb_fri > 1
                was_inside = 0 <= bb_wed <= 1
                was_outside = bb_wed < 0 or bb_wed > 1

                if was_inside and now_outside:
                    bb_sub = 1.0  # fresh breakout
                elif was_outside and now_outside:
                    bb_sub = 0.9 if abs(bb_fri - 0.5) > abs(bb_wed - 0.5) else 0.6
                elif was_outside and not now_outside:
                    bb_sub = 0.2  # failed breakout
                else:
                    bb_sub = min(abs(bb_fri - 0.5) / 0.5, 1.0) * 0.7

                # Volume sustaining
                vol_fri = tech.get("volume_ratio", 0) or 0
                vol_wed = snap_tech.get("volume_ratio", 0) or 0

                if vol_fri >= 1.0 and (vol_wed == 0 or vol_fri >= vol_wed * 0.7):
                    vol_sub = 1.0
                elif vol_fri >= 0.8 and (vol_wed == 0 or vol_fri >= vol_wed * 0.5):
                    vol_sub = 0.6
                else:
                    vol_sub = 0.2

                setup_score = rsi_sub * 0.35 + macd_sub * 0.25 + bb_sub * 0.20 + vol_sub * 0.20
            else:
                setup_score = 0.5

            # ============================================================
            # Component 2: Trend Evolution (weight 0.20)
            # ============================================================
            if has_snapshot:
                # ADX change — is trend strengthening?
                adx_fri = tech.get("adx") or 0
                adx_wed = snap_tech.get("adx") or 0

                if adx_fri > adx_wed and adx_fri > 20:
                    adx_sub = min((adx_fri - adx_wed) / 10.0 + 0.7, 1.0)
                elif adx_fri > 25:
                    adx_sub = 0.7  # still strong even if slightly declining
                elif adx_fri < 15:
                    adx_sub = 0.2  # lost trend
                else:
                    adx_sub = 0.5

                # Trend classification persistence
                trend_fri = tech.get("trend_strength", "flat")
                trend_wed = snap_tech.get("trend_strength", "flat")
                strong_trends = {"strong_up", "strong_down"}
                trends = {"up", "down", "strong_up", "strong_down"}

                if trend_fri in strong_trends and trend_wed in trends:
                    trend_sub = 1.0  # strengthening trend
                elif trend_fri in trends and trend_wed in trends and trend_fri == trend_wed:
                    trend_sub = 0.8  # persisting
                elif trend_fri in trends and trend_wed == "flat":
                    trend_sub = 0.9  # new trend emerging
                elif trend_fri == "flat" and trend_wed in trends:
                    trend_sub = 0.2  # trend died
                else:
                    trend_sub = 0.5

                # SMA slope acceleration
                slope20_fri = tech.get("sma20_slope", 0) or 0
                slope20_wed = snap_tech.get("sma20_slope", 0) or 0
                if abs(slope20_fri) > abs(slope20_wed) and slope20_fri * slope20_wed >= 0:
                    slope_sub = 0.8  # accelerating
                elif slope20_fri * slope20_wed < 0:
                    slope_sub = 0.3  # reversed
                else:
                    slope_sub = 0.5

                trend_evo_score = adx_sub * 0.40 + trend_sub * 0.35 + slope_sub * 0.25
            else:
                trend_evo_score = 0.5

            # ============================================================
            # Component 3: IV & Premium Trajectory (weight 0.20)
            # ============================================================
            if has_snapshot:
                iv_fri = opts.get("atm_iv", 0) or 0
                iv_wed = snap_opts.get("atm_iv", 0) or 0

                if iv_wed > 0 and iv_fri > 0:
                    iv_change = (iv_fri - iv_wed) / iv_wed
                    if iv_change > 0.05:
                        iv_sub = 0.9  # IV expanding — market expects bigger move
                    elif iv_change > -0.05:
                        iv_sub = 0.6  # IV stable
                    elif iv_change > -0.10:
                        iv_sub = 0.35  # mild compression
                    else:
                        iv_sub = 0.15  # IV collapsing
                else:
                    iv_sub = 0.5

                iv_rank_fri = opts.get("iv_rank", 0) or 0
                if iv_rank_fri > 0.5:
                    iv_sub = min(iv_sub + 0.1, 1.0)

                # Premium resilience (has premium held up against 2-day theta?)
                call_mid_fri = opts.get("atm_call_mid") or 0
                call_mid_wed = snap_opts.get("atm_call_mid") or 0
                put_mid_fri = opts.get("atm_put_mid") or 0
                put_mid_wed = snap_opts.get("atm_put_mid") or 0

                mid_fri = max(call_mid_fri, put_mid_fri)
                mid_wed = max(call_mid_wed, put_mid_wed)
                premium_sub = 0.5
                if mid_wed > 0 and mid_fri > 0:
                    prem_change = (mid_fri - mid_wed) / mid_wed
                    if prem_change > 0.05:
                        premium_sub = 0.85  # premium growing despite theta
                    elif prem_change > -0.15:
                        premium_sub = 0.55  # normal theta decay
                    else:
                        premium_sub = 0.2   # collapsing beyond theta

                # Spread quality
                spread_fri = opts.get("atm_call_spread_pct") or opts.get("atm_put_spread_pct")
                spread_sub = 0.5
                if spread_fri is not None:
                    if spread_fri < 10:
                        spread_sub = 0.8
                    elif spread_fri < 25:
                        spread_sub = 0.5
                    else:
                        spread_sub = 0.2

                iv_score = iv_sub * 0.40 + premium_sub * 0.35 + spread_sub * 0.25
            else:
                iv_score = 0.5

            # ============================================================
            # Component 4: Options Positioning Shift (weight 0.15)
            # ============================================================
            if has_snapshot:
                pc_fri = opts.get("pc_ratio_volume", 1.0) or 1.0
                pc_wed = snap_opts.get("pc_ratio_volume", 1.0) or 1.0

                if abs(pc_fri - 1.0) > abs(pc_wed - 1.0):
                    pc_sub = 0.8  # conviction building
                elif abs(pc_fri - 1.0) > 0.5:
                    pc_sub = 0.7  # still extreme
                elif abs(pc_fri - 1.0) < abs(pc_wed - 1.0):
                    pc_sub = 0.3  # conviction fading
                else:
                    pc_sub = 0.5

                # Max pain convergence
                price_fri = (tech.get("price") or opts.get("current_price") or 0)
                max_pain_fri = opts.get("max_pain", 0) or 0
                price_wed = (snap_tech.get("price") or snap_opts.get("current_price") or 0)
                max_pain_wed = snap_opts.get("max_pain", 0) or 0

                if price_fri > 0 and max_pain_fri > 0:
                    div_fri = abs(price_fri - max_pain_fri) / price_fri
                    div_wed = (abs(price_wed - max_pain_wed) / price_wed
                               if price_wed > 0 and max_pain_wed > 0 else div_fri)
                    if div_fri < div_wed:
                        mp_sub = 0.7  # converging toward max pain
                    else:
                        mp_sub = 0.4  # diverging
                else:
                    mp_sub = 0.5

                opts_shift_score = pc_sub * 0.6 + mp_sub * 0.4
            else:
                opts_shift_score = 0.5

            # ============================================================
            # Component 5: Base Quality Floor (weight 0.15)
            # ============================================================
            # Recalculate scan_rank components on Friday's data
            ret_5d = tech.get("return_5d", 0) or 0
            adx_now = tech.get("adx") or 0
            vol_ratio = tech.get("volume_ratio", 1.0) or 1.0
            rsi = tech.get("rsi", 50) or 50

            base_quality = (
                min(abs(ret_5d) / 5.0, 1.0) * 0.30
                + min(adx_now / 50.0, 1.0) * 0.30
                + min(vol_ratio / 2.0, 1.0) * 0.20
                + min(abs(rsi - 50) / 30.0, 1.0) * 0.20
            )

            # ============================================================
            # Component 6: Sentiment Evolution (weight 0.05)
            # ============================================================
            if has_snapshot:
                sent_fri = c.get("sentiment", {}).get("composite_score", 0) or 0
                sent_wed = snap.get("sentiment", {}).get("composite_score", 0) or 0

                if abs(sent_fri) > abs(sent_wed) and sent_fri * sent_wed >= 0:
                    sent_score = 0.8  # strengthening
                elif sent_fri * sent_wed < 0:
                    sent_score = 0.4  # flipped sign
                else:
                    sent_score = 0.5

                art_fri = c.get("sentiment", {}).get("article_count", 0) or 0
                art_wed = snap.get("sentiment", {}).get("article_count", 0) or 0
                if art_fri > art_wed:
                    sent_score = min(sent_score + 0.2, 1.0)
            else:
                sent_score = 0.5

            # ============================================================
            # Composite Friday score
            # ============================================================
            friday_score = (
                setup_score * 0.25
                + trend_evo_score * 0.20
                + iv_score * 0.20
                + opts_shift_score * 0.15
                + base_quality * 0.15
                + sent_score * 0.05
            )

            c["friday_score"] = round(friday_score, 4)
            c["friday_components"] = {
                "setup_momentum": round(setup_score, 4),
                "trend_evolution": round(trend_evo_score, 4),
                "iv_premium_trajectory": round(iv_score, 4),
                "options_positioning": round(opts_shift_score, 4),
                "base_quality": round(base_quality, 4),
                "sentiment_evolution": round(sent_score, 4),
            }
            scored.append(c)

        scored.sort(key=lambda x: x.get("friday_score", 0), reverse=True)
        return scored

    def _monday_picks_filter(self, candidates: list, results: dict) -> list:
        """Select top 3 high-conviction picks with diversity enforcement.

        Candidates are already scored with composite_score by the scoring
        engine. This filter enforces:

        1. Minimum confidence threshold: drop picks below 0.25 confidence
        2. Sector diversity: max 2 from same sector, but only if both rank
           in the top 5 by composite score AND their 20-day return
           correlation is < 0.60 (stricter than the general 0.75 dedup).
           This lets genuinely independent same-sector plays through while
           still protecting against single-headline blowups.
        3. Direction balance: if all 3 same direction AND average confidence
           < 0.75, swap lowest-scored pick for best contrarian
        4. Earnings-week guard: flag (but don't remove) picks with earnings
           within the Mon-Fri holding window
        5. Correlation-based diversity: avoid highly correlated tickers
           (e.g., AAPL + MSFT both in tech with 0.8+ correlation)
        """
        # Sort by composite score
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("composite_score", 0),
            reverse=True,
        )

        # Filter out low confidence — METHODOLOGY.md §Confidence Calibration
        # defines this as a HARD gate. Do not relax it even if fewer than 3
        # picks pass; an all-low-confidence week is a signal the system is
        # saying "no edge here" and should be respected.
        confident = [
            c for c in sorted_candidates
            if c.get("direction_confidence", c.get("confidence", 0)) >= 0.25
        ]
        if len(confident) < 3:
            logger.warning(
                "Only %d candidates cleared 0.25 confidence gate; "
                "proceeding with best available rather than diluting with "
                "low-confidence picks",
                len(confident),
            )

        # Build sector map
        sector_map = {}
        for c in confident:
            ticker = c.get("ticker", "")
            sector = c.get("sector", "unknown")
            sector_map[ticker] = sector

        # Sector diversity: allow up to 2 from same sector (was 1).
        # Same-sector pairs face tighter correlation dedup (0.60 vs 0.75)
        # and a top-5 rank gate below.
        diverse = enforce_diversity(confident, sector_map, max_per_sector=2)

        # Correlation-based dedup: avoid highly correlated pairs
        diverse = self._correlation_dedup(diverse)

        # Take top 3 — prefer fewer high-conviction picks over more mediocre ones
        from config import PORTFOLIO_SIZE
        top5 = diverse[:PORTFOLIO_SIZE]

        # Same-sector top-5 gate: if 2 of the 3 picks share a sector,
        # both must have ranked in the top 5 of the original composite-scored
        # list. This prevents a mediocre same-sector pick from riding the
        # coattails of a strong one. If the gate fails, swap the lower one
        # for the next best cross-sector candidate.
        if len(top5) >= 2:
            top5_tickers = {c.get("ticker") for c in sorted_candidates[:5]}
            sectors_in_picks = {}
            for pick in top5:
                s = pick.get("sector", "unknown")
                sectors_in_picks.setdefault(s, []).append(pick)
            for sector, picks_in_sector in sectors_in_picks.items():
                if len(picks_in_sector) >= 2:
                    # Check both were top-5 in the pre-diversity ranking
                    for pick in picks_in_sector:
                        if pick.get("ticker") not in top5_tickers:
                            # This pick wasn't top-5 — demote it
                            logger.info(
                                "Same-sector gate: %s (%s) not in top-5 composite; "
                                "replacing with next cross-sector candidate",
                                pick.get("ticker"), sector,
                            )
                            top5.remove(pick)
                            # Find replacement from diverse pool
                            existing_tickers = {c.get("ticker") for c in top5}
                            for replacement in diverse:
                                if (replacement.get("ticker") not in existing_tickers
                                        and replacement.get("sector") != sector):
                                    top5.append(replacement)
                                    break
                            break  # Only fix one per sector per pass

        # Flag earnings-week picks
        for pick in top5:
            earnings_date = pick.get("earnings_date") or pick.get("finviz", {}).get("earnings_date")
            if earnings_date:
                pick["earnings_warning"] = True
                logger.info(
                    "Pick %s has earnings on %s within holding window",
                    pick.get("ticker"), earnings_date,
                )

        # Direction balance check — with 3 picks, all-same is more likely
        if len(top5) >= 3:
            directions = [c.get("direction", "call") for c in top5]
            confidences = [c.get("direction_confidence", c.get("confidence", 0)) for c in top5]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            all_same = len(set(directions)) == 1
            if all_same and avg_confidence < 0.75:
                dominant = directions[0]
                contrarian = "put" if dominant == "call" else "call"

                top5_tickers = {c.get("ticker") for c in top5}
                contrarian_pool = [
                    c for c in diverse[PORTFOLIO_SIZE:]
                    if c.get("direction") == contrarian
                    and c.get("ticker") not in top5_tickers
                ]

                if contrarian_pool:
                    best_contrarian = contrarian_pool[0]
                    top5[-1] = best_contrarian
                    logger.info(
                        "Direction balance: swapped %s for contrarian %s (%s)",
                        dominant, best_contrarian.get("ticker"), contrarian,
                    )

        # Kelly criterion position sizing
        top5 = self._apply_kelly_sizing(top5)

        return top5

    def _apply_kelly_sizing(self, picks: list) -> list:
        """Apply Kelly criterion to determine confidence-weighted position sizes.

        Kelly fraction f* = (bp - q) / b, where:
            p = estimated win probability (from confidence calibration)
            q = 1 - p
            b = reward/risk ratio (from breakeven move vs expected move)

        We use half-Kelly (f*/2) for safety, capped at 3% per trade.
        """
        for pick in picks:
            confidence = pick.get("direction_confidence", pick.get("confidence", 0.5))
            # Win probability estimate: calibrated confidence
            p = max(0.30, min(0.70, confidence))
            q = 1 - p

            # Reward/risk ratio: weekly expected move / breakeven move needed
            be_move = abs(pick.get("breakeven_move_pct", 5.0) or 5.0)
            exp_move = pick.get("expected_weekly_move_pct", 5.0) or 5.0
            b = max(0.5, exp_move / be_move) if be_move > 0 else 1.0

            # Full Kelly
            kelly_full = (b * p - q) / b if b > 0 else 0
            # Half-Kelly for safety, capped at 3% of portfolio
            kelly_half = max(0, kelly_full / 2)
            position_pct = min(0.03, kelly_half)

            # Regime gate reduction
            macro_edge = pick.get("macro_edge", {})
            if macro_edge and not macro_edge.get("has_edge", True):
                position_pct *= 0.5  # half size in low-edge regimes

            pick["kelly"] = {
                "win_prob": round(p, 3),
                "reward_risk": round(b, 3),
                "full_kelly": round(kelly_full, 4),
                "half_kelly": round(kelly_half, 4),
                "position_pct": round(position_pct, 4),
                "position_pct_display": f"{position_pct * 100:.1f}%",
            }

        return picks

    def _correlation_dedup(self, candidates: list, corr_threshold: float = 0.75,
                           same_sector_threshold: float = 0.60) -> list:
        """Remove highly correlated tickers using rolling 20-day return correlations.

        Fetches 1-month price history for each candidate, computes pairwise
        correlation, and drops lower-scored duplicates above *corr_threshold*.
        Same-sector pairs use a tighter *same_sector_threshold* (0.60) to
        ensure genuine independence when allowing 2 picks from one sector.
        Falls back to sector/sub-industry grouping if price data is unavailable.
        """
        if len(candidates) <= 1:
            return candidates

        tickers = [c.get("ticker", "") for c in candidates]
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # Fetch return data for pairwise correlation
        returns_map = {}
        try:
            for ticker in tickers:
                hist = yf.Ticker(ticker).history(period="1mo")
                if hist is not None and len(hist) >= 10:
                    returns_map[ticker] = hist["Close"].pct_change().dropna().values
        except Exception as exc:
            logger.debug("Correlation fetch failed: %s — using sector proxy", exc)

        # Build correlation matrix
        excluded = set()
        if len(returns_map) >= 2:
            checked = set()
            for i, c1 in enumerate(candidates):
                t1 = c1.get("ticker", "")
                if t1 in excluded or t1 not in returns_map:
                    continue
                for j, c2 in enumerate(candidates):
                    if j <= i:
                        continue
                    t2 = c2.get("ticker", "")
                    if t2 in excluded or t2 not in returns_map:
                        continue
                    pair = (t1, t2)
                    if pair in checked:
                        continue
                    checked.add(pair)

                    r1, r2 = returns_map[t1], returns_map[t2]
                    min_len = min(len(r1), len(r2))
                    if min_len < 5:
                        continue
                    corr = float(np.corrcoef(r1[:min_len], r2[:min_len])[0, 1])
                    # Same-sector pairs use tighter threshold (0.60) to ensure
                    # genuine independence when 2 picks from one sector are allowed
                    s1_sector = c1.get("sector", "unknown")
                    s2_sector = c2.get("sector", "unknown")
                    threshold = same_sector_threshold if s1_sector == s2_sector else corr_threshold
                    if abs(corr) >= threshold:
                        # Drop the lower-scored one
                        s1 = c1.get("composite_score", 0)
                        s2 = c2.get("composite_score", 0)
                        drop = t2 if s1 >= s2 else t1
                        excluded.add(drop)
                        logger.info(
                            "Correlation dedup: dropping %s (corr %.2f with %s, threshold %.2f%s)",
                            drop, corr, t1 if drop == t2 else t2, threshold,
                            " [same-sector]" if s1_sector == s2_sector else "",
                        )
        else:
            # Fallback: sector + sub-industry grouping
            group_counts = {}
            for c in candidates:
                sector = c.get("sector", "unknown")
                industry = c.get("industry", c.get("sub_industry", "unknown"))
                group_key = (sector, industry)
                count = group_counts.get(group_key, 0)
                if count >= 2:
                    excluded.add(c.get("ticker", ""))
                    logger.debug(
                        "Correlation dedup (sector proxy): skipping %s from %s/%s",
                        c.get("ticker"), sector, industry,
                    )
                else:
                    group_counts[group_key] = count + 1

        return [c for c in candidates if c.get("ticker", "") not in excluded]

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def save_stage_results(self, stage: str, candidates: list, date_str: str) -> Path:
        """Save stage output to data/candidates/{date}/{stage}.json."""
        stage_dir = config.candidates_dir / date_str
        stage_dir.mkdir(parents=True, exist_ok=True)
        filepath = stage_dir / f"{stage}.json"

        serialisable = _make_serialisable(candidates)

        with open(filepath, "w") as fh:
            json.dump(serialisable, fh, indent=2, default=str)

        logger.info("Saved %d candidates for stage '%s' -> %s", len(candidates), stage, filepath)
        return filepath

    def load_stage_results(self, stage: str, date_str: str) -> list:
        """Load previously saved stage results."""
        filepath = config.candidates_dir / date_str / f"{stage}.json"
        if not filepath.exists():
            logger.warning("No saved results for stage '%s' on %s", stage, date_str)
            return []
        try:
            with open(filepath, "r") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, IOError) as exc:
            logger.error("Failed to load stage results from %s: %s", filepath, exc)
            return []


# ------------------------------------------------------------------
#  Public diversity filter
# ------------------------------------------------------------------

def enforce_diversity(candidates: list, sector_map: dict, max_per_sector: int = 2) -> list:
    """Filter candidates so no sector is over-represented.

    Parameters
    ----------
    candidates : list[dict]
        Already sorted by score (highest first).
    sector_map : dict
        Mapping of ticker -> sector name.
    max_per_sector : int
        Maximum picks allowed from any single sector.

    Returns
    -------
    list[dict]
        Filtered list preserving score order.
    """
    sector_counts = {}
    filtered = []

    for c in candidates:
        ticker = c.get("ticker", "")
        sector = sector_map.get(ticker, c.get("sector", "unknown"))

        current_count = sector_counts.get(sector, 0)
        if current_count >= max_per_sector:
            logger.debug(
                "Skipping %s — sector '%s' already has %d picks",
                ticker, sector, current_count,
            )
            continue

        sector_counts[sector] = current_count + 1
        filtered.append(c)

    return filtered


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _make_serialisable(obj):
    """Recursively convert non-JSON-serialisable types."""
    import numpy as _np
    import pandas as _pd

    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    if isinstance(obj, _pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, _pd.Series):
        return obj.tolist()
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj
