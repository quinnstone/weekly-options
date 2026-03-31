"""
Performance tracking for the Zero-DTE Options Trading Analysis System.

Records picks, outcomes, and computes aggregate statistics to enable
the weekly reflection / weight-adjustment loop.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()


class PerformanceTracker:
    """Record and query pick / outcome history."""

    def __init__(self):
        self.perf_dir = config.performance_dir
        self.perf_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  Recording
    # ------------------------------------------------------------------

    def record_picks(self, date_str: str, picks: list) -> Path:
        """Save Friday picks to data/performance/{date}_picks.json.

        Parameters
        ----------
        date_str : str
            Date string (YYYY-MM-DD).
        picks : list[dict]
            The final picks for the day.

        Returns
        -------
        Path
            The file path written.
        """
        filepath = self.perf_dir / f"{date_str}_picks.json"

        serialisable = []
        for p in picks:
            serialisable.append({
                "ticker": p.get("ticker"),
                "direction": p.get("direction"),
                "selected_strike": p.get("selected_strike"),
                "selected_expiry": p.get("selected_expiry"),
                "estimated_premium": p.get("estimated_premium"),
                "composite_score": p.get("composite_score"),
                "direction_confidence": p.get("direction_confidence"),
                "scores": p.get("scores"),
            })

        with open(filepath, "w") as fh:
            json.dump({
                "date": date_str,
                "picks": serialisable,
                "recorded_at": datetime.now().isoformat(),
            }, fh, indent=2, default=str)

        logger.info("Recorded %d picks for %s -> %s", len(picks), date_str, filepath)
        return filepath

    def record_outcomes(self, date_str: str, outcomes: list) -> Path:
        """Record actual market outcomes for a day's picks.

        Parameters
        ----------
        date_str : str
            Date string of the original picks (YYYY-MM-DD).
        outcomes : list[dict]
            Each dict should contain:
                - ticker (str)
                - reached_strike (bool): did price reach the strike?
                - peak_value (float): highest option value during the day
                - entry_premium (float): what the option cost at entry
                - close_value (float): value at market close
                - price_at_various (dict): option price at timestamps

        Returns
        -------
        Path
            The file path written.
        """
        filepath = self.perf_dir / f"{date_str}_outcomes.json"

        processed = []
        for o in outcomes:
            entry = o.get("entry_premium", 0)
            peak = o.get("peak_value", 0)

            # Win = option was profitable at any point during the day
            win = peak > entry if entry > 0 else False
            peak_return = ((peak - entry) / entry) if entry > 0 else 0.0
            close_value = o.get("close_value", 0)
            close_return = ((close_value - entry) / entry) if entry > 0 else 0.0

            processed.append({
                "ticker": o.get("ticker"),
                "reached_strike": o.get("reached_strike", False),
                "entry_premium": entry,
                "peak_value": peak,
                "close_value": close_value,
                "peak_return": round(peak_return, 4),
                "close_return": round(close_return, 4),
                "win": win,
                "price_at_various": o.get("price_at_various", {}),
            })

        with open(filepath, "w") as fh:
            json.dump({
                "date": date_str,
                "outcomes": processed,
                "recorded_at": datetime.now().isoformat(),
            }, fh, indent=2, default=str)

        logger.info("Recorded %d outcomes for %s -> %s", len(outcomes), date_str, filepath)
        return filepath

    # ------------------------------------------------------------------
    #  History queries
    # ------------------------------------------------------------------

    def get_history(self, n_weeks: int = 10) -> list:
        """Load the last *n_weeks* of picks and outcomes.

        Returns
        -------
        list[dict]
            Each element has keys: date, picks (list), outcomes (list or None).
            Sorted newest first.
        """
        history = []

        # Scan performance directory for pick files
        pick_files = sorted(self.perf_dir.glob("*_picks.json"), reverse=True)

        for pick_file in pick_files:
            date_str = pick_file.stem.replace("_picks", "")
            try:
                with open(pick_file, "r") as fh:
                    pick_data = json.load(fh)
            except (json.JSONDecodeError, IOError):
                continue

            # Try to load matching outcomes
            outcome_file = self.perf_dir / f"{date_str}_outcomes.json"
            outcome_data = None
            if outcome_file.exists():
                try:
                    with open(outcome_file, "r") as fh:
                        outcome_data = json.load(fh)
                except (json.JSONDecodeError, IOError):
                    pass

            history.append({
                "date": date_str,
                "picks": pick_data.get("picks", []),
                "outcomes": outcome_data.get("outcomes", []) if outcome_data else None,
            })

            if len(history) >= n_weeks:
                break

        return history

    def get_stats(self) -> dict:
        """Calculate aggregate statistics across all recorded outcomes.

        Returns
        -------
        dict
            Keys: total_picks, total_with_outcomes, wins, losses,
            win_rate, avg_peak_return, avg_close_return,
            best_pick, worst_pick, accuracy_by_direction.
        """
        history = self.get_history(n_weeks=52)

        total_picks = 0
        total_with_outcomes = 0
        wins = 0
        losses = 0
        peak_returns = []
        close_returns = []
        best_pick = None
        worst_pick = None
        best_return = float("-inf")
        worst_return = float("inf")
        direction_stats = {"call": {"wins": 0, "total": 0},
                           "put": {"wins": 0, "total": 0}}

        for week in history:
            picks = week["picks"]
            outcomes = week["outcomes"]
            total_picks += len(picks)

            if outcomes is None:
                continue

            # Build lookup by ticker
            outcome_map = {o["ticker"]: o for o in outcomes}

            for pick in picks:
                ticker = pick.get("ticker")
                outcome = outcome_map.get(ticker)
                if outcome is None:
                    continue

                total_with_outcomes += 1
                peak_ret = outcome.get("peak_return", 0)
                close_ret = outcome.get("close_return", 0)
                is_win = outcome.get("win", False)

                peak_returns.append(peak_ret)
                close_returns.append(close_ret)

                if is_win:
                    wins += 1
                else:
                    losses += 1

                # Track best / worst
                if peak_ret > best_return:
                    best_return = peak_ret
                    best_pick = {
                        "ticker": ticker,
                        "date": week["date"],
                        "return": peak_ret,
                    }
                if close_ret < worst_return:
                    worst_return = close_ret
                    worst_pick = {
                        "ticker": ticker,
                        "date": week["date"],
                        "return": close_ret,
                    }

                # Direction accuracy
                direction = pick.get("direction", "unknown")
                if direction in direction_stats:
                    direction_stats[direction]["total"] += 1
                    if is_win:
                        direction_stats[direction]["wins"] += 1

        # Compute aggregates
        win_rate = (wins / total_with_outcomes) if total_with_outcomes > 0 else 0.0
        avg_peak = sum(peak_returns) / len(peak_returns) if peak_returns else 0.0
        avg_close = sum(close_returns) / len(close_returns) if close_returns else 0.0

        accuracy_by_direction = {}
        for d, stats in direction_stats.items():
            if stats["total"] > 0:
                accuracy_by_direction[d] = {
                    "wins": stats["wins"],
                    "total": stats["total"],
                    "accuracy": round(stats["wins"] / stats["total"], 4),
                }
            else:
                accuracy_by_direction[d] = {
                    "wins": 0, "total": 0, "accuracy": 0.0,
                }

        return {
            "total_picks": total_picks,
            "total_with_outcomes": total_with_outcomes,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "avg_peak_return": round(avg_peak, 4),
            "avg_close_return": round(avg_close, 4),
            "best_pick": best_pick,
            "worst_pick": worst_pick,
            "accuracy_by_direction": accuracy_by_direction,
        }
