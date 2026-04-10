"""
Pattern Library for the Weekly Options Trading Analysis System.

Learns conditional patterns from graded trades to improve future predictions.
Tracks which signal combinations lead to wins/losses under different regimes,
and feeds pattern-based adjustments back into the scoring engine.

The pattern library is the system's long-term memory: it captures non-obvious
relationships between market conditions and trade outcomes that pure weight
adjustment cannot express (e.g., "momentum + low VIX + earnings week → loss").
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import Config

logger = logging.getLogger(__name__)
config = Config()

PATTERNS_FILE = config.performance_dir / "patterns.json"
CALIBRATION_FILE = config.performance_dir / "confidence_calibration.json"


class PatternLibrary:
    """Learn and apply conditional patterns from historical trade outcomes."""

    # Minimum observations before a pattern is considered reliable
    MIN_PATTERN_OBS = 5

    def __init__(self):
        self.patterns = self._load_patterns()
        self.calibration = self._load_calibration()

    # ------------------------------------------------------------------
    #  Pattern recording
    # ------------------------------------------------------------------

    def record_trade(self, pick: dict, outcome: dict) -> None:
        """Record a completed trade for pattern extraction.

        Parameters
        ----------
        pick : dict
            The original pick data (scores, regime, direction, signals).
        outcome : dict
            Trade result: 'result' (WIN/LOSS/PARTIAL), 'pnl', 'return_pct'.
        """
        pattern_key = self._extract_pattern_key(pick)
        result = outcome.get("result", "LOSS")
        pnl = outcome.get("return_pct", 0)

        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = {
                "wins": 0, "losses": 0, "partials": 0,
                "total_return": 0, "observations": 0,
                "examples": [],
            }

        entry = self.patterns[pattern_key]
        entry["observations"] += 1
        entry["total_return"] += pnl

        if result == "WIN":
            entry["wins"] += 1
        elif result == "PARTIAL":
            entry["partials"] += 1
        else:
            entry["losses"] += 1

        # Keep last 10 examples for debugging
        entry["examples"] = entry.get("examples", [])[-9:] + [{
            "ticker": pick.get("ticker"),
            "date": pick.get("date", ""),
            "result": result,
            "pnl": round(pnl, 4),
        }]

        self._save_patterns()

        # Update confidence calibration with this observation
        raw_conf = pick.get("raw_confidence", pick.get("confidence", 0.5))
        self._update_calibration(raw_conf, result == "WIN")

    def _extract_pattern_key(self, pick: dict) -> str:
        """Extract a pattern key from pick signals and conditions.

        Pattern keys capture the most discriminative signal combinations:
        - Regime (low/normal/elevated/high/extreme)
        - Direction (call/put)
        - Dominant signal (highest-scoring factor)
        - RSI zone (oversold/neutral/overbought)
        - Trend state (trending/choppy)
        - IV state (cheap/fair/expensive)
        """
        regime = pick.get("active_regime", pick.get("market_regime", {}).get("regime", "normal"))
        direction = pick.get("direction", "call")

        # Dominant signal from scores
        scores = pick.get("scores", {})
        dominant = "unknown"
        if scores:
            tier1 = {k: scores[k] for k in ("momentum", "mean_reversion", "trend_persistence") if k in scores}
            if tier1:
                dominant = max(tier1, key=tier1.get)

        # RSI zone
        tech = pick.get("technical", {})
        rsi = tech.get("rsi", 50) or 50
        if rsi < 35:
            rsi_zone = "oversold"
        elif rsi > 65:
            rsi_zone = "overbought"
        else:
            rsi_zone = "neutral"

        # Trend state
        adx = tech.get("adx", 0) or 0
        trend_state = "trending" if adx > 25 else "choppy"

        # IV state
        opts = pick.get("options", {})
        iv_rv = opts.get("iv_rv_ratio")
        if iv_rv is not None:
            if iv_rv < 0.9:
                iv_state = "cheap"
            elif iv_rv > 1.3:
                iv_state = "expensive"
            else:
                iv_state = "fair"
        else:
            iv_state = "unknown"

        return f"{regime}|{direction}|{dominant}|{rsi_zone}|{trend_state}|{iv_state}"

    # ------------------------------------------------------------------
    #  Pattern-based score adjustments
    # ------------------------------------------------------------------

    def get_pattern_adjustment(self, pick: dict) -> dict:
        """Look up the pattern for a candidate and return a score adjustment.

        Returns
        -------
        dict
            'adjustment' (float, -10 to +10), 'pattern_win_rate' (float),
            'pattern_observations' (int), 'pattern_key' (str).
        """
        key = self._extract_pattern_key(pick)
        entry = self.patterns.get(key)

        if entry is None or entry["observations"] < self.MIN_PATTERN_OBS:
            return {
                "adjustment": 0,
                "pattern_win_rate": None,
                "pattern_observations": 0,
                "pattern_key": key,
            }

        obs = entry["observations"]
        win_rate = (entry["wins"] + 0.5 * entry["partials"]) / obs
        avg_return = entry["total_return"] / obs

        # Adjustment: scale by deviation from baseline (50% win rate)
        # Positive patterns boost score, negative patterns penalize
        deviation = win_rate - 0.50
        adjustment = deviation * 20  # ±10 points max at 100%/0% win rate

        # Confidence in the pattern scales with observations
        confidence_factor = min(1.0, obs / 20)
        adjustment *= confidence_factor

        return {
            "adjustment": round(float(np.clip(adjustment, -10, 10)), 2),
            "pattern_win_rate": round(win_rate, 3),
            "pattern_avg_return": round(avg_return, 4),
            "pattern_observations": obs,
            "pattern_key": key,
        }

    # ------------------------------------------------------------------
    #  Confidence calibration auto-update
    # ------------------------------------------------------------------

    def _update_calibration(self, raw_confidence: float, was_win: bool) -> None:
        """Update confidence calibration buckets with a new observation.

        Tracks actual win rate for each raw confidence bucket so the
        system learns its own accuracy over time.
        """
        # Bucket raw confidence into bins of 0.10
        bucket = str(round(raw_confidence * 10) / 10)

        if bucket not in self.calibration:
            self.calibration[bucket] = {"wins": 0, "total": 0}

        self.calibration[bucket]["total"] += 1
        if was_win:
            self.calibration[bucket]["wins"] += 1

        self._save_calibration()

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """Return calibrated confidence based on historical accuracy.

        If we have enough observations for this confidence bucket,
        return the actual observed win rate. Otherwise return the
        raw confidence unchanged.
        """
        bucket = str(round(raw_confidence * 10) / 10)
        entry = self.calibration.get(bucket)

        if entry and entry["total"] >= 10:
            return entry["wins"] / entry["total"]

        return raw_confidence

    def get_calibration_table(self) -> dict:
        """Return the full calibration table for inspection."""
        table = {}
        for bucket, data in sorted(self.calibration.items()):
            total = data["total"]
            win_rate = data["wins"] / total if total > 0 else 0
            table[bucket] = {
                "total": total,
                "wins": data["wins"],
                "win_rate": round(win_rate, 3),
                "reliable": total >= 10,
            }
        return table

    # ------------------------------------------------------------------
    #  Backtest validation gate
    # ------------------------------------------------------------------

    def validate_weights(self, proposed_weights: dict, backtest_results: dict) -> dict:
        """Gate check: only deploy new weights if they improve backtest metrics.

        Compares proposed weights against the current weights using
        backtest accuracy and Sharpe ratio.

        Parameters
        ----------
        proposed_weights : dict
            New weight vector from reflector/weight updater.
        backtest_results : dict
            Must contain 'current_accuracy', 'proposed_accuracy',
            'current_sharpe', 'proposed_sharpe'.

        Returns
        -------
        dict
            'approved' (bool), 'reason' (str), metrics.
        """
        curr_acc = backtest_results.get("current_accuracy", 0)
        prop_acc = backtest_results.get("proposed_accuracy", 0)
        curr_sharpe = backtest_results.get("current_sharpe", 0)
        prop_sharpe = backtest_results.get("proposed_sharpe", 0)

        # Gate criteria:
        # 1. Proposed accuracy must not drop more than 2%
        # 2. Proposed Sharpe must not drop more than 0.1
        # 3. At least one metric must improve
        acc_delta = prop_acc - curr_acc
        sharpe_delta = prop_sharpe - curr_sharpe

        if acc_delta < -0.02:
            return {
                "approved": False,
                "reason": f"Accuracy dropped {acc_delta:.1%} (threshold: -2%)",
                "accuracy_delta": round(acc_delta, 4),
                "sharpe_delta": round(sharpe_delta, 4),
            }

        if sharpe_delta < -0.1:
            return {
                "approved": False,
                "reason": f"Sharpe dropped {sharpe_delta:.2f} (threshold: -0.1)",
                "accuracy_delta": round(acc_delta, 4),
                "sharpe_delta": round(sharpe_delta, 4),
            }

        if acc_delta <= 0 and sharpe_delta <= 0:
            return {
                "approved": False,
                "reason": "No improvement in either metric — keeping current weights",
                "accuracy_delta": round(acc_delta, 4),
                "sharpe_delta": round(sharpe_delta, 4),
            }

        return {
            "approved": True,
            "reason": f"Approved: accuracy {acc_delta:+.1%}, Sharpe {sharpe_delta:+.2f}",
            "accuracy_delta": round(acc_delta, 4),
            "sharpe_delta": round(sharpe_delta, 4),
        }

    # ------------------------------------------------------------------
    #  Summary / reporting
    # ------------------------------------------------------------------

    def get_top_patterns(self, n: int = 10) -> list:
        """Return the top-N most reliable patterns (by observation count)."""
        reliable = [
            (key, data) for key, data in self.patterns.items()
            if data["observations"] >= self.MIN_PATTERN_OBS
        ]
        reliable.sort(key=lambda x: x[1]["observations"], reverse=True)

        results = []
        for key, data in reliable[:n]:
            obs = data["observations"]
            win_rate = (data["wins"] + 0.5 * data["partials"]) / obs
            results.append({
                "pattern": key,
                "win_rate": round(win_rate, 3),
                "avg_return": round(data["total_return"] / obs, 4),
                "observations": obs,
            })
        return results

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def _load_patterns(self) -> dict:
        if PATTERNS_FILE.exists():
            try:
                with open(PATTERNS_FILE, "r") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not parse patterns file — starting fresh")
        return {}

    def _save_patterns(self) -> None:
        PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PATTERNS_FILE, "w") as fh:
            json.dump(self.patterns, fh, indent=2)

    def _load_calibration(self) -> dict:
        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE, "r") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not parse calibration file — starting fresh")
        return {}

    def _save_calibration(self) -> None:
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION_FILE, "w") as fh:
            json.dump(self.calibration, fh, indent=2)
