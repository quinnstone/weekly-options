"""
Narrowing pipeline for the Zero-DTE Options Trading Analysis System.

Filters candidates through three stages:
    Wednesday scan      - Broad universe down to 20 candidates + 10 bench
    Thursday re-eval    - Delta-aware re-ranking of 30 (20 + bench) to best 20
    Friday picks        - 20 candidates down to 3 high-conviction picks with diversity
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()


class NarrowingPipeline:
    """Filter candidates through the Wednesday/Friday two-pass pipeline."""

    # Stage name -> target count coming *out* of that stage
    STAGE_TARGETS = {
        "wednesday_scan": 20,
        "thursday_reeval": 20,
        "friday_picks": 3,
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
        elif stage == "thursday_reeval":
            filtered = self._thursday_reeval_filter(candidates, scanner_results)
        elif stage == "friday_picks":
            filtered = self._friday_picks_filter(candidates, scanner_results)
        else:
            logger.warning("Unknown stage '%s' — returning candidates unchanged", stage)
            return candidates

        # Truncate to target if we still have too many
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
        elif stage == "thursday_reeval":
            filtered = self._thursday_reeval_filter(candidates, scanner_results)
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
        """Filter and rank for Wednesday broad scan -- target 20.

        Requirements:
        - ATR% > 0.8 (stock actually moves)
        - Options data exists and total options volume > 1000
        Ranking by composite tech_rank score.
        """
        scored = []
        for c in candidates:
            # Accept raw ticker strings or dicts
            if isinstance(c, str):
                c = {"ticker": c}

            tech = c.get("technical", {})
            opts = c.get("options", {})

            # Require ATR% > 0.8
            atr_pct = tech.get("atr_pct", 0) or 0
            if atr_pct <= 0.8:
                logger.debug("Removing %s — ATR%% %.2f below 0.8", c.get("ticker"), atr_pct)
                continue

            # Require options data exists and total volume > 1000
            if not opts:
                logger.debug("Removing %s — no options data", c.get("ticker"))
                continue
            total_vol = (opts.get("total_call_volume", 0) or 0) + (opts.get("total_put_volume", 0) or 0)
            if total_vol <= 1000:
                logger.debug(
                    "Removing %s — options volume %d below 1000",
                    c.get("ticker"), total_vol,
                )
                continue

            # Compute tech_rank for ranking
            if not tech:
                c = dict(c)
                c["tech_rank"] = 0.1
                scored.append(c)
                continue

            # ATR pct — higher = more movement potential (weight 30%)
            atr_component = min(atr_pct / 5.0, 1.0) * 0.30

            # RSI extremity — farther from 50 = more interesting (weight 25%)
            rsi = tech.get("rsi", 50) or 50
            rsi_extremity = abs(rsi - 50) / 50.0  # normalised 0-1
            rsi_component = rsi_extremity * 0.25

            # Volume ratio — higher = more active (weight 20%)
            volume_ratio = tech.get("volume_ratio", 0) or 0
            vol_component = min(volume_ratio / 3.0, 1.0) * 0.20

            # Bollinger band position extremity (weight 15%)
            bb_pct = tech.get("bb_pct", 0.5)
            if bb_pct is None:
                bb_pct = 0.5
            bb_extremity = abs(bb_pct - 0.5)
            bb_component = min(bb_extremity / 1.0, 1.0) * 0.15

            # MACD histogram sign flip (weight 10%)
            hist = tech.get("macd_histogram", 0) or 0
            hist_prev = tech.get("macd_histogram_prev", 0) or 0
            macd_flip = 1.0 if (hist * hist_prev < 0) else 0.0
            macd_component = macd_flip * 0.10

            tech_rank = atr_component + rsi_component + vol_component + bb_component + macd_component

            c = dict(c)
            c["tech_rank"] = round(tech_rank, 4)
            scored.append(c)

        # Sort by tech_rank descending
        scored.sort(key=lambda x: x.get("tech_rank", 0), reverse=True)
        return scored[:20]

    def _thursday_reeval_filter(self, candidates: list, results: dict) -> list:
        """Delta-aware re-ranking for Thursday — target 20.

        Unlike Wednesday's screening filter, this evaluates how each
        candidate's setup is *evolving* toward Friday by comparing
        Wednesday's snapshot against Thursday's fresh data.

        Hard filters (relaxed from Wednesday):
        - ATR% >= 0.5 (was 0.8 — allow slight compression if setup evolving)
        - Options volume >= 500 (was 1000)
        - wednesday_snapshot must exist (need baseline for deltas)

        Scoring components (each 0-1, weighted):
        - Setup momentum (0.30): RSI trajectory, MACD change, BB evolution, volume sustaining
        - IV trajectory (0.20): IV expanding vs collapsing heading into Friday
        - Options positioning (0.15): P/C ratio shift, max pain convergence
        - Fresh catalysts (0.15): Thursday gap, pre-market move, new headlines
        - Base quality floor (0.15): Recalculated tech_rank (absolute quality)
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
            atr_pct = tech.get("atr_pct", 0) or 0
            if atr_pct < 0.5:
                logger.debug("Thu filter: removing %s — ATR%% %.2f < 0.5", ticker, atr_pct)
                continue

            if not opts:
                logger.debug("Thu filter: removing %s — no options data", ticker)
                continue
            total_vol = (opts.get("total_call_volume", 0) or 0) + (opts.get("total_put_volume", 0) or 0)
            if total_vol < 500:
                logger.debug("Thu filter: removing %s — options volume %d < 500", ticker, total_vol)
                continue

            has_snapshot = bool(snap and snap_tech)

            # ============================================================
            # Component 1: Setup Momentum (weight 0.30)
            # ============================================================
            if has_snapshot:
                # 1a. RSI trajectory (40% of component)
                rsi_thu = (tech.get("rsi") or 50)
                rsi_wed = (snap_tech.get("rsi") or 50)
                rsi_delta = rsi_thu - rsi_wed

                # Evaluate whether RSI is moving in a "useful" direction
                if rsi_wed < 35 and rsi_delta > 0:
                    # Bouncing from oversold — ideal call setup
                    rsi_sub = min(abs(rsi_delta) / 10.0, 1.0)
                elif rsi_wed > 65 and rsi_delta < 0:
                    # Pulling back from overbought — ideal put setup
                    rsi_sub = min(abs(rsi_delta) / 10.0, 1.0)
                elif abs(rsi_thu - 50) > abs(rsi_wed - 50):
                    # Moving further from 50 — momentum continuing
                    rsi_sub = 0.8
                elif abs(rsi_thu - 50) < abs(rsi_wed - 50):
                    # Reverting toward 50 — setup weakening
                    rsi_sub = 0.3
                else:
                    rsi_sub = 0.5

                # 1b. MACD histogram momentum (25% of component)
                hist_thu = tech.get("macd_histogram", 0) or 0
                hist_wed = snap_tech.get("macd_histogram", 0) or 0

                if hist_thu != 0 and hist_wed != 0 and (hist_thu * hist_wed < 0):
                    # Fresh crossover on Thursday
                    macd_sub = 1.0
                elif abs(hist_thu) > abs(hist_wed) and hist_thu * hist_wed >= 0:
                    # Expanding in same direction
                    macd_sub = 0.8
                elif abs(hist_thu) <= abs(hist_wed) and hist_thu * hist_wed >= 0:
                    # Contracting but same sign
                    macd_sub = 0.4
                else:
                    macd_sub = 0.3

                # 1c. BB position evolution (20% of component)
                bb_thu = tech.get("bb_pct", 0.5) if tech.get("bb_pct") is not None else 0.5
                bb_wed = snap_tech.get("bb_pct", 0.5) if snap_tech.get("bb_pct") is not None else 0.5

                was_inside = 0 <= bb_wed <= 1
                now_outside = bb_thu < 0 or bb_thu > 1
                was_outside = bb_wed < 0 or bb_wed > 1
                now_inside = 0 <= bb_thu <= 1

                if was_inside and now_outside:
                    bb_sub = 1.0  # fresh breakout
                elif was_outside and now_outside:
                    # Still outside — check if extending
                    if abs(bb_thu - 0.5) > abs(bb_wed - 0.5):
                        bb_sub = 0.9  # extending breakout
                    else:
                        bb_sub = 0.6  # holding outside
                elif was_outside and now_inside:
                    bb_sub = 0.2  # failed breakout
                else:
                    # Inside both days — reward movement toward edge
                    bb_sub = min(abs(bb_thu - 0.5) / 0.5, 1.0) * 0.7

                # 1d. Volume sustaining (15% of component)
                vol_thu = tech.get("volume_ratio", 0) or 0
                vol_wed = snap_tech.get("volume_ratio", 0) or 0

                if vol_thu >= 1.0 and (vol_wed == 0 or vol_thu >= vol_wed * 0.7):
                    vol_sub = 1.0  # sustained or growing interest
                elif vol_thu >= 0.8 and (vol_wed == 0 or vol_thu >= vol_wed * 0.5):
                    vol_sub = 0.6  # acceptable decay
                else:
                    vol_sub = 0.2  # Wednesday was a blip

                setup_score = rsi_sub * 0.40 + macd_sub * 0.25 + bb_sub * 0.20 + vol_sub * 0.15
            else:
                # No snapshot — fall back to neutral
                setup_score = 0.5

            # ============================================================
            # Component 2: IV & Premium Trajectory (weight 0.20)
            # ============================================================
            if has_snapshot:
                iv_thu = opts.get("atm_iv", 0) or 0
                iv_wed = snap_opts.get("atm_iv", 0) or 0
                iv_rank_thu = opts.get("iv_rank", 0) or 0

                # IV direction
                if iv_wed > 0 and iv_thu > 0:
                    iv_change = (iv_thu - iv_wed) / iv_wed
                    if iv_change > 0.05:
                        iv_sub = 0.9  # IV expanding — market expects bigger Friday move
                    elif iv_change > -0.05:
                        iv_sub = 0.6  # IV stable
                    elif iv_change > -0.10:
                        iv_sub = 0.35  # mild compression
                    else:
                        iv_sub = 0.15  # IV collapsing — catalyst may have passed
                else:
                    iv_sub = 0.5  # no baseline

                # IV rank bonus
                if iv_rank_thu > 0.5:
                    iv_sub = min(iv_sub + 0.1, 1.0)

                # Actual premium delta (more direct than IV proxy)
                call_mid_thu = opts.get("atm_call_mid") or 0
                call_mid_wed = snap_opts.get("atm_call_mid") or 0
                put_mid_thu = opts.get("atm_put_mid") or 0
                put_mid_wed = snap_opts.get("atm_put_mid") or 0

                premium_sub = 0.5  # neutral default
                # Use whichever side has data; prefer the higher-premium side
                mid_thu = max(call_mid_thu, put_mid_thu)
                mid_wed = max(call_mid_wed, put_mid_wed)
                if mid_wed > 0 and mid_thu > 0:
                    prem_change = (mid_thu - mid_wed) / mid_wed
                    if prem_change > 0.05:
                        premium_sub = 0.85  # premium growing despite theta — strong signal
                    elif prem_change > -0.15:
                        premium_sub = 0.55  # normal theta decay range
                    else:
                        premium_sub = 0.2   # premium collapsing beyond theta

                # Bid-ask spread quality (liquidity check)
                spread_thu = opts.get("atm_call_spread_pct") or opts.get("atm_put_spread_pct")
                spread_wed = snap_opts.get("atm_call_spread_pct") or snap_opts.get("atm_put_spread_pct")
                spread_sub = 0.5
                if spread_thu is not None:
                    if spread_thu < 10:
                        spread_sub = 0.8  # tight spread — good liquidity
                    elif spread_thu < 25:
                        spread_sub = 0.5  # acceptable
                    else:
                        spread_sub = 0.2  # wide spread — slippage risk
                    # Penalize if spread widened significantly from Wednesday
                    if spread_wed is not None and spread_wed > 0:
                        if spread_thu > spread_wed * 1.5:
                            spread_sub = max(spread_sub - 0.2, 0.0)

                # Combine: IV direction (40%), premium delta (35%), spread quality (25%)
                iv_score = iv_sub * 0.40 + premium_sub * 0.35 + spread_sub * 0.25
            else:
                iv_score = 0.5

            # ============================================================
            # Component 3: Options Positioning Shift (weight 0.15)
            # ============================================================
            if has_snapshot:
                pc_thu = opts.get("pc_ratio_volume", 1.0) or 1.0
                pc_wed = snap_opts.get("pc_ratio_volume", 1.0) or 1.0

                # Is conviction building (P/C moving further from 1.0)?
                if abs(pc_thu - 1.0) > abs(pc_wed - 1.0):
                    pc_sub = 0.8  # conviction building
                elif abs(pc_thu - 1.0) > 0.5:
                    pc_sub = 0.7  # still extreme even if slightly fading
                elif abs(pc_thu - 1.0) < abs(pc_wed - 1.0):
                    pc_sub = 0.3  # conviction fading
                else:
                    pc_sub = 0.5

                # Max pain convergence
                price_thu = (tech.get("price") or opts.get("current_price") or 0)
                max_pain_thu = opts.get("max_pain", 0) or 0
                price_wed = (snap_tech.get("price") or snap_opts.get("current_price") or 0)
                max_pain_wed = snap_opts.get("max_pain", 0) or 0

                if price_thu > 0 and max_pain_thu > 0:
                    div_thu = abs(price_thu - max_pain_thu) / price_thu
                    div_wed = abs(price_wed - max_pain_wed) / price_wed if price_wed > 0 and max_pain_wed > 0 else div_thu
                    if div_thu < div_wed:
                        mp_sub = 0.7  # converging toward max pain — Friday pin risk info
                    else:
                        mp_sub = 0.4  # diverging — potential breakout
                else:
                    mp_sub = 0.5

                opts_shift_score = pc_sub * 0.6 + mp_sub * 0.4
            else:
                opts_shift_score = 0.5

            # ============================================================
            # Component 4: Fresh Catalysts (weight 0.15)
            # ============================================================
            gap_pct = abs(tech.get("gap_pct", 0) or 0)
            finviz = c.get("finviz", {})
            pre_mkt = abs(finviz.get("pre_market_change_pct", 0) or 0)

            if gap_pct >= 1.5:
                catalyst_score = 0.9
            elif gap_pct >= 0.5:
                catalyst_score = 0.6
            else:
                catalyst_score = 0.3

            # Pre-market move bonus
            if pre_mkt >= 1.0:
                catalyst_score = min(catalyst_score + 0.15, 1.0)

            # New headlines bonus
            if has_snapshot:
                thu_headlines = set(c.get("sentiment", {}).get("headlines", []) or [])
                wed_headlines = set(snap.get("sentiment", {}).get("headlines", []) or [])
                new_headlines = len(thu_headlines - wed_headlines)
                if new_headlines >= 2:
                    catalyst_score = min(catalyst_score + 0.15, 1.0)

            # ============================================================
            # Component 5: Base Quality Floor (weight 0.15)
            # ============================================================
            # Reuse Wednesday's tech_rank formula on Thursday's fresh data
            # to ensure we don't promote a weak stock purely on deltas
            rsi = tech.get("rsi", 50) or 50
            volume_ratio = tech.get("volume_ratio", 0) or 0
            bb_pct_raw = tech.get("bb_pct", 0.5)
            if bb_pct_raw is None:
                bb_pct_raw = 0.5

            base_quality = (
                min(atr_pct / 5.0, 1.0) * 0.35
                + (abs(rsi - 50) / 50.0) * 0.30
                + min(volume_ratio / 3.0, 1.0) * 0.20
                + min(abs(bb_pct_raw - 0.5), 1.0) * 0.15
            )

            # ============================================================
            # Component 6: Sentiment Evolution (weight 0.05)
            # ============================================================
            if has_snapshot:
                sent_thu = c.get("sentiment", {}).get("composite_score", 0) or 0
                sent_wed = snap.get("sentiment", {}).get("composite_score", 0) or 0

                # Strengthening in same direction?
                if abs(sent_thu) > abs(sent_wed) and sent_thu * sent_wed >= 0:
                    sent_score = 0.8
                elif sent_thu * sent_wed < 0:
                    sent_score = 0.4  # flipped sign — confused signal
                else:
                    sent_score = 0.5

                # More articles = increasing coverage
                art_thu = c.get("sentiment", {}).get("article_count", 0) or 0
                art_wed = snap.get("sentiment", {}).get("article_count", 0) or 0
                if art_thu > art_wed:
                    sent_score = min(sent_score + 0.2, 1.0)
            else:
                sent_score = 0.5

            # ============================================================
            # Composite Thursday score
            # ============================================================
            thursday_score = (
                setup_score * 0.30
                + iv_score * 0.20
                + opts_shift_score * 0.15
                + catalyst_score * 0.15
                + base_quality * 0.15
                + sent_score * 0.05
            )

            c["thursday_score"] = round(thursday_score, 4)
            c["thursday_components"] = {
                "setup_momentum": round(setup_score, 4),
                "iv_premium_trajectory": round(iv_score, 4),
                "options_positioning": round(opts_shift_score, 4),
                "fresh_catalysts": round(catalyst_score, 4),
                "base_quality": round(base_quality, 4),
                "sentiment_evolution": round(sent_score, 4),
            }
            scored.append(c)

        # Sort by thursday_score descending
        scored.sort(key=lambda x: x.get("thursday_score", 0), reverse=True)
        return scored

    def _friday_picks_filter(self, candidates: list, results: dict) -> list:
        """Select top 3 high-conviction candidates with diversity enforcement.

        Candidates are already scored with composite_score by this point.
        - No more than 1 from same sector (3 picks = 3 different sectors)
        - If all 3 are same direction (all calls or all puts) AND average
          confidence < 0.8, swap the lowest-scored pick for the best
          contrarian candidate from the remaining pool
        - Minimum confidence threshold: drop picks below 0.2 confidence
        """
        # Sort by composite score
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("composite_score", 0),
            reverse=True,
        )

        # Filter out very low confidence picks
        confident = [
            c for c in sorted_candidates
            if c.get("direction_confidence", c.get("confidence", 0)) >= 0.2
        ]
        # Fall back to unfiltered if too aggressive
        if len(confident) < 3:
            confident = sorted_candidates

        # Build sector map from candidates
        sector_map = {}
        for c in confident:
            ticker = c.get("ticker", "")
            sector = c.get("sector", "unknown")
            sector_map[ticker] = sector

        # Apply sector diversity: max 1 per sector for 3 picks
        diverse = enforce_diversity(confident, sector_map, max_per_sector=1)

        # Take top 3
        top3 = diverse[:3]

        # Direction mix check
        if len(top3) == 3:
            directions = [c.get("direction", "call") for c in top3]
            confidences = [c.get("direction_confidence", c.get("confidence", 0)) for c in top3]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            all_same_direction = len(set(directions)) == 1
            if all_same_direction and avg_confidence < 0.8:
                # Find best contrarian candidate from remaining pool
                dominant_direction = directions[0]
                contrarian_direction = "put" if dominant_direction == "call" else "call"

                # Look through remaining candidates (not in top3) for contrarian
                top3_tickers = {c.get("ticker") for c in top3}
                contrarian_candidates = [
                    c for c in diverse[3:]
                    if c.get("direction") == contrarian_direction
                    and c.get("ticker") not in top3_tickers
                ]

                if contrarian_candidates:
                    # Swap lowest-scored pick for best contrarian
                    best_contrarian = contrarian_candidates[0]
                    top3[-1] = best_contrarian
                    logger.info(
                        "Direction mix: swapped %s for contrarian %s (%s)",
                        directions[0], best_contrarian.get("ticker"),
                        contrarian_direction,
                    )

        return top3

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def save_stage_results(self, stage: str, candidates: list, date_str: str) -> Path:
        """Save stage output to data/candidates/{date}/{stage}.json.

        Returns the file path written.
        """
        stage_dir = config.candidates_dir / date_str
        stage_dir.mkdir(parents=True, exist_ok=True)
        filepath = stage_dir / f"{stage}.json"

        # Make data JSON-serialisable (drop DataFrames, numpy types)
        serialisable = _make_serialisable(candidates)

        with open(filepath, "w") as fh:
            json.dump(serialisable, fh, indent=2, default=str)

        logger.info("Saved %d candidates for stage '%s' -> %s", len(candidates), stage, filepath)
        return filepath

    def load_stage_results(self, stage: str, date_str: str) -> list:
        """Load previously saved stage results.

        Returns
        -------
        list[dict]
            The candidate list, or an empty list if the file does not exist.
        """
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
