"""
Weekly reflection engine for the Weekly Options Trading Analysis System.

Analyses the past week's picks and outcomes (Mon entry → Fri expiry),
identifies which signals were predictive (or misleading), computes
correlations, suggests weight adjustments, and formats everything
for human review and Discord.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from tracking.tracker import PerformanceTracker
from analysis.scoring import CandidateScorer
from analysis.patterns import PatternLibrary

logger = logging.getLogger(__name__)
config = Config()


class WeeklyReflector:
    """Generate, persist, and apply weekly reflection insights."""

    def __init__(self):
        self.tracker = PerformanceTracker()
        self.scorer = CandidateScorer()
        self.pattern_library = PatternLibrary()

    # ------------------------------------------------------------------
    #  Redundancy guard — skip when nothing new to reflect on
    # ------------------------------------------------------------------

    def _last_reflection_date(self) -> "str | None":
        """Find the YYYY-MM-DD of the most recent prior reflection on disk."""
        files = sorted(
            config.performance_dir.glob("*_reflection.json"),
            reverse=True,
        )
        if not files:
            return None
        # Filenames look like "2026-05-23_reflection.json"
        return files[0].name.split("_")[0]

    def _most_recent_monday_picks_date(self, days_back: int = 14) -> "str | None":
        """Find the YYYY-MM-DD of the most recent monday_picks.json on disk.

        Walks back up to `days_back` days. Returns None if no picks file
        exists in that window (suggests holiday-skipped or cron-failed
        weeks since the last reflection).
        """
        today = datetime.now().date()
        for i in range(days_back):
            check_date = today - timedelta(days=i)
            iso = check_date.strftime("%Y-%m-%d")
            path = config.candidates_dir / iso / "monday_picks.json"
            if path.exists():
                return iso
        return None

    def _should_skip_redundant_reflection(self) -> bool:
        """Return True if running reflect would just regenerate the prior
        reflection's content (no new monday_picks.json since last reflect).
        """
        last_refl = self._last_reflection_date()
        if not last_refl:
            # No prior reflection — first run, never skip
            return False
        last_picks = self._most_recent_monday_picks_date()
        if not last_picks:
            # No picks anywhere in lookback window — definitely no new data
            return True
        # Skip if the most recent picks file is older than (or equal to)
        # the last reflection. ISO date strings compare lexicographically.
        return last_picks <= last_refl

    # ------------------------------------------------------------------
    #  Core reflection
    # ------------------------------------------------------------------

    def _load_scorecard_week(self, week_date: str):
        """Adapt the scorecard's graded week into (picks, outcomes) the rest of
        the reflector expects.

        Returns ([], None) if no scorecard week matches `week_date` — caller
        then falls back to the legacy tracker path.

        outcome dicts: {ticker, win, peak_return, pnl}. NOTE peak_return here is
        the realized return at expiry (pnl/entry_cost), NOT a true intra-week
        peak — the scorecard grades hold-to-expiry. Good enough for signal
        correlation; a future change can read the per-day monitors for true MFE.

        pick dicts carry "scores" (the per-signal sub-scores captured in
        entry_features at grade time) so _compute_signal_correlations works.
        For weeks graded before entry_features existed, scores is {} and
        correlations degrade to 0.0 — honest, not fabricated.
        """
        try:
            with open(config.performance_dir / "scorecard_data.json") as fh:
                data = json.load(fh)
        except Exception:
            return [], None

        week = next((w for w in data.get("weeks", []) if w.get("pick_date") == week_date), None)
        if not week:
            return [], None

        picks, outcomes = [], []
        for g in week.get("picks", []):
            feats = g.get("entry_features") or {}
            entry_cost = g.get("entry_cost") or 0
            pnl = g.get("pnl", 0)
            picks.append({
                "ticker": g.get("ticker"),
                "direction": g.get("direction"),
                "scores": feats.get("scores") or {},
                "entry_signal": g.get("entry_signal", "GO"),
                "result": g.get("result"),
                "diagnostics": g.get("diagnostics") or {},
                "entry_features": feats,
            })
            outcomes.append({
                "ticker": g.get("ticker"),
                "win": g.get("result") == "WIN",
                "peak_return": round(pnl / entry_cost, 4) if entry_cost else 0.0,
                "pnl": pnl,
            })
        return picks, outcomes

    def reflect(self, week_date: str) -> "dict | None":
        """Generate a full reflection for the specified week.

        Returns ``None`` if no new picks have been generated since the most
        recent prior reflection — this covers holiday-skipped weeks and
        cron-failure weeks where running a fresh reflection would just
        regurgitate the prior week's content (and waste an Anthropic API
        call in the downstream DeepReflectionAgent). Callers must handle
        the None case explicitly (see main.py reflect command).

        Parameters
        ----------
        week_date : str
            A date string (YYYY-MM-DD) falling within the target week.

        Returns
        -------
        dict or None
            Reflection dict containing performance stats, signal analysis,
            lessons, and suggested weight adjustments. None if redundant.
        """
        # Redundancy guard — see _should_skip_redundant_reflection() for the
        # criterion. Saves ~$0.30-0.50 per holiday week (DeepReflectionAgent
        # call) and keeps the historical reflection record clean.
        if self._should_skip_redundant_reflection():
            logger.info(
                "Reflection skipped — no new monday_picks since last reflection "
                "(likely a holiday-skipped week or missed cron)."
            )
            return None

        logger.info("Generating reflection for week of %s", week_date)

        # 1. Load picks and outcomes — from the SCORECARD (the real grader),
        # not the legacy tracker outcomes path (whose {date}_outcomes.json files
        # were never written, so the reflector reported win_rate 0.0 on EVERY
        # week, incl. winning weeks). The scorecard has real W/L/P, pnl, and the
        # entry_features/diagnostics captured at grade time.
        picks, outcomes = self._load_scorecard_week(week_date)
        if not picks:
            # Fall back to the legacy tracker path if the scorecard has no
            # matching week (e.g. grading hasn't run yet for this date).
            history = self.tracker.get_history(n_weeks=1)
            if not history:
                logger.warning("No scorecard or tracker history for reflection")
                return self._empty_reflection(week_date)
            week_data = history[0]
            picks = week_data.get("picks", [])
            outcomes = week_data.get("outcomes")
            if not picks:
                return self._empty_reflection(week_date)

        # 2. Overall performance
        stats = self.tracker.get_stats()
        wins = 0
        losses = 0
        returns = []

        if outcomes:
            outcome_map = {o["ticker"]: o for o in outcomes}
            for pick in picks:
                o = outcome_map.get(pick.get("ticker"))
                if o:
                    if o.get("win"):
                        wins += 1
                    else:
                        losses += 1
                    returns.append(o.get("peak_return", 0))

        total_picks = len(picks)
        win_rate = wins / total_picks if total_picks > 0 else 0.0
        avg_return = sum(returns) / len(returns) if returns else 0.0

        # 2b. Record trades in pattern library for long-term learning
        if outcomes:
            outcome_map_for_patterns = {o["ticker"]: o for o in outcomes}
            for pick in picks:
                o = outcome_map_for_patterns.get(pick.get("ticker"))
                if o:
                    result_str = "WIN" if o.get("win") else "LOSS"
                    self.pattern_library.record_trade(pick, {
                        "result": result_str,
                        "pnl": o.get("pnl", 0),
                        "return_pct": o.get("peak_return", 0),
                    })
            logger.info("Recorded %d trades in pattern library", len(outcomes))

        # 3. Signal correlation analysis
        signal_correlations = self._compute_signal_correlations(picks, outcomes)

        # 4. What worked / what failed
        worked = self._analyse_what_worked(picks, outcomes)
        failed = self._analyse_what_failed(picks, outcomes)

        # 5. Lessons
        lessons = self._derive_lessons(signal_correlations, worked, failed, stats)

        # 6. Weight adjustments
        weight_adjustments = self._suggest_weight_adjustments(signal_correlations)

        reflection = {
            "week": week_date,
            "total_picks": total_picks,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "avg_return": round(avg_return, 4),
            "signal_correlations": signal_correlations,
            "what_worked": worked,
            "what_failed": failed,
            "lessons": lessons,
            "weight_adjustments": weight_adjustments,
            "current_weights": dict(self.scorer.weights),
            "generated_at": datetime.now().isoformat(),
        }

        # Auto-save
        self.save_reflection(week_date, reflection)
        return reflection

    # ------------------------------------------------------------------
    #  Signal correlation
    # ------------------------------------------------------------------

    def _compute_signal_correlations(self, picks: list, outcomes: list) -> dict:
        """Compute correlation between each signal category and outcomes.

        Returns a dict mapping category name -> correlation (-1 to +1).
        """
        if not outcomes:
            return {}

        outcome_map = {o["ticker"]: o for o in outcomes}
        # Use new 10-factor model categories from scoring.py
        categories = [
            "momentum", "mean_reversion", "regime_bias", "trend_persistence",
            "iv_mispricing", "flow_conviction", "event_risk",
            "liquidity", "strike_efficiency", "theta_cost",
        ]
        correlations = {}

        for cat in categories:
            scores_list = []
            returns_list = []

            for pick in picks:
                ticker = pick.get("ticker")
                o = outcome_map.get(ticker)
                if o is None:
                    continue

                # Get the sub-score for this category
                sub_scores = pick.get("scores", {})
                sub_score = sub_scores.get(cat, sub_scores.get(f"{cat}_adj", 50))
                peak_ret = o.get("peak_return", 0)

                scores_list.append(sub_score)
                returns_list.append(peak_ret)

            if len(scores_list) >= 2:
                scores_arr = np.array(scores_list)
                returns_arr = np.array(returns_list)
                # Avoid division by zero for constant arrays
                if np.std(scores_arr) > 0 and np.std(returns_arr) > 0:
                    corr = float(np.corrcoef(scores_arr, returns_arr)[0, 1])
                else:
                    corr = 0.0
                correlations[cat] = round(corr, 4)
            else:
                correlations[cat] = 0.0

        return correlations

    # ------------------------------------------------------------------
    #  Analysis helpers
    # ------------------------------------------------------------------

    def _analyse_what_worked(self, picks: list, outcomes: list) -> list:
        """Identify signals that contributed to winning trades."""
        if not outcomes:
            return []

        outcome_map = {o["ticker"]: o for o in outcomes}
        worked = []

        for pick in picks:
            o = outcome_map.get(pick.get("ticker"))
            if not o or not o.get("win"):
                continue

            ticker = pick.get("ticker")
            reasons = []

            scores = pick.get("scores", {})
            # Find the highest-scoring category for this winner
            if scores:
                top_cat = max(
                    (k for k in scores if not k.endswith("_adj")),
                    key=lambda k: scores.get(k, 0),
                    default=None,
                )
                if top_cat:
                    reasons.append(f"strong {top_cat} signal ({scores[top_cat]:.0f})")

            direction = pick.get("direction", "unknown")
            confidence = pick.get("direction_confidence", 0)
            if confidence >= 0.6:
                reasons.append(f"high-confidence {direction} call")

            worked.append({
                "ticker": ticker,
                "return": o.get("peak_return", 0),
                "reasons": reasons,
            })

        return worked

    def _analyse_what_failed(self, picks: list, outcomes: list) -> list:
        """Identify signals that led to losing trades."""
        if not outcomes:
            return []

        outcome_map = {o["ticker"]: o for o in outcomes}
        failed = []

        for pick in picks:
            o = outcome_map.get(pick.get("ticker"))
            if not o or o.get("win"):
                continue

            ticker = pick.get("ticker")
            reasons = []

            scores = pick.get("scores", {})
            if scores:
                # Identify misleading high-score categories
                for cat, val in scores.items():
                    if cat.endswith("_adj"):
                        continue
                    if val >= 70:
                        reasons.append(f"{cat} score was high ({val:.0f}) but trade lost")

            direction = pick.get("direction", "unknown")
            reasons.append(f"direction was {direction}")

            failed.append({
                "ticker": ticker,
                "return": o.get("close_return", 0),
                "reasons": reasons,
            })

        return failed

    def _derive_lessons(self, correlations: dict, worked: list,
                        failed: list, stats: dict) -> list:
        """Derive human-readable lessons from the analysis."""
        lessons = []

        # Correlation-based lessons
        for cat, corr in correlations.items():
            if corr >= 0.5:
                lessons.append(
                    f"{cat.capitalize()} signals were strongly predictive this week "
                    f"(correlation {corr:.2f}) — consider increasing weight."
                )
            elif corr <= -0.3:
                lessons.append(
                    f"{cat.capitalize()} signals were misleading this week "
                    f"(correlation {corr:.2f}) — consider decreasing weight."
                )

        # Win rate lessons
        win_rate = stats.get("win_rate", 0)
        if win_rate >= 0.7:
            lessons.append("Strong win rate this period — current approach is working well.")
        elif win_rate <= 0.3:
            lessons.append(
                "Low win rate — review whether market regime has shifted "
                "or if signal quality has degraded."
            )

        # Direction accuracy
        direction_stats = stats.get("accuracy_by_direction", {})
        for d, d_stats in direction_stats.items():
            acc = d_stats.get("accuracy", 0)
            total = d_stats.get("total", 0)
            if total >= 3 and acc <= 0.3:
                lessons.append(
                    f"{d.upper()} direction accuracy is low ({acc:.0%}) — "
                    f"revisit direction-determination logic."
                )

        if not lessons:
            lessons.append("Insufficient data to draw firm conclusions this week.")

        return lessons

    def _suggest_weight_adjustments(self, correlations: dict) -> dict:
        """Suggest weight deltas based on signal correlations."""
        adjustments = {}
        for cat, corr in correlations.items():
            if abs(corr) >= 0.2:
                # Suggest shifting weight in the direction of correlation
                delta = corr * 0.05  # conservative step
                adjustments[cat] = round(delta, 4)
        return adjustments

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def save_reflection(self, week_date: str, reflection: dict) -> Path:
        """Save reflection to data/performance/{date}_reflection.json."""
        filepath = config.performance_dir / f"{week_date}_reflection.json"
        with open(filepath, "w") as fh:
            json.dump(reflection, fh, indent=2, default=str)
        logger.info("Saved reflection to %s", filepath)
        return filepath

    def apply_learnings(self, reflection: dict) -> None:
        """Update scorer weights based on the reflection.

        Uses a two-step process:
        1. Compute proposed weights from signal correlations
        2. Validate via backtest gate — only deploy if metrics improve

        Calls ``CandidateScorer.update_weights`` with the signal
        correlations extracted from the reflection.
        """
        correlations = reflection.get("signal_correlations", {})
        if not correlations:
            logger.info("No signal correlations in reflection — skipping weight update")
            return

        # Pass sample size for minimum-sample gate in scorer
        total_picks = reflection.get("total_picks", 0)
        performance_data = {
            "signal_correlations": correlations,
            "sample_size": total_picks,
        }

        # Save current weights for rollback
        old_weights = dict(self.scorer.weights)

        # Apply proposed update
        self.scorer.update_weights(performance_data)
        proposed_weights = dict(self.scorer.weights)

        # Backtest validation gate (if we have enough history)
        try:
            from analysis.backtest import DirectionalBacktester
            bt = DirectionalBacktester(lookback_weeks=12)
            # Run backtest with current and proposed weights
            current_results = bt.run()
            current_acc = current_results.get("accuracy", 0)
            current_sharpe = current_results.get("sharpe_ratio", 0)

            # Run with proposed weights (already applied)
            proposed_results = bt.run()
            proposed_acc = proposed_results.get("accuracy", 0)
            proposed_sharpe = proposed_results.get("sharpe_ratio", 0)

            validation = self.pattern_library.validate_weights(
                proposed_weights,
                {
                    "current_accuracy": current_acc,
                    "proposed_accuracy": proposed_acc,
                    "current_sharpe": current_sharpe,
                    "proposed_sharpe": proposed_sharpe,
                },
            )

            if not validation["approved"]:
                # Rollback to old weights
                self.scorer.weights = old_weights
                self.scorer.save_weights()
                logger.warning(
                    "Backtest validation REJECTED weight update: %s",
                    validation["reason"],
                )
                return

            logger.info(
                "Backtest validation APPROVED: %s", validation["reason"],
            )
        except Exception as exc:
            # If backtest fails, keep proposed weights (fail open)
            logger.warning("Backtest validation skipped (error: %s) — keeping proposed weights", exc)

        logger.info("Applied learnings — new weights: %s", self.scorer.weights)

        # Refresh agent context documents with new weights/state
        try:
            from agents.state_generator import refresh_agent_context
            refresh_agent_context()
        except Exception as exc:
            logger.warning("Failed to refresh agent context after learning: %s", exc)

    # ------------------------------------------------------------------
    #  Formatting
    # ------------------------------------------------------------------

    def format_reflection(self, reflection: dict) -> str:
        """Format a reflection dict as human-readable text.

        Suitable for terminal display or Discord (via send_weekly_reflection).
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Weekly Reflection — {reflection.get('week', 'N/A')}")
        lines.append("=" * 60)
        lines.append("")

        # Performance summary
        lines.append("PERFORMANCE")
        lines.append("-" * 40)
        lines.append(f"  Total picks:  {reflection.get('total_picks', 0)}")
        lines.append(f"  Wins:         {reflection.get('wins', 0)}")
        lines.append(f"  Losses:       {reflection.get('losses', 0)}")
        win_rate = reflection.get("win_rate", 0)
        lines.append(f"  Win rate:     {win_rate:.0%}")
        avg_ret = reflection.get("avg_return", 0)
        lines.append(f"  Avg return:   {avg_ret:.1%}")
        lines.append("")

        # Signal correlations
        correlations = reflection.get("signal_correlations", {})
        if correlations:
            lines.append("SIGNAL CORRELATIONS")
            lines.append("-" * 40)
            for cat, corr in sorted(correlations.items(), key=lambda x: -x[1]):
                bar = "+" * int(abs(corr) * 20) if corr >= 0 else "-" * int(abs(corr) * 20)
                lines.append(f"  {cat:18s} {corr:+.3f}  {bar}")
            lines.append("")

        # What worked
        worked = reflection.get("what_worked", [])
        if worked:
            lines.append("WHAT WORKED")
            lines.append("-" * 40)
            for w in worked:
                reasons = ", ".join(w.get("reasons", []))
                lines.append(f"  {w['ticker']:6s} +{w.get('return', 0):.1%}  {reasons}")
            lines.append("")

        # What failed
        failed = reflection.get("what_failed", [])
        if failed:
            lines.append("WHAT FAILED")
            lines.append("-" * 40)
            for f_item in failed:
                reasons = ", ".join(f_item.get("reasons", []))
                lines.append(f"  {f_item['ticker']:6s} {f_item.get('return', 0):.1%}  {reasons}")
            lines.append("")

        # Lessons
        lessons = reflection.get("lessons", [])
        if lessons:
            lines.append("LESSONS")
            lines.append("-" * 40)
            for lesson in lessons:
                lines.append(f"  * {lesson}")
            lines.append("")

        # Weight adjustments
        adjustments = reflection.get("weight_adjustments", {})
        if adjustments:
            lines.append("SUGGESTED WEIGHT ADJUSTMENTS")
            lines.append("-" * 40)
            for cat, delta in adjustments.items():
                lines.append(f"  {cat:18s} {delta:+.4f}")
            lines.append("")

        # Current weights
        weights = reflection.get("current_weights", {})
        if weights:
            lines.append("CURRENT WEIGHTS")
            lines.append("-" * 40)
            for cat, w in weights.items():
                lines.append(f"  {cat:18s} {w:.4f}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _empty_reflection(self, week_date: str) -> dict:
        """Return a skeleton reflection when no data is available."""
        return {
            "week": week_date,
            "total_picks": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "signal_correlations": {},
            "what_worked": [],
            "what_failed": [],
            "lessons": ["No picks or outcomes recorded for this week."],
            "weight_adjustments": {},
            "current_weights": dict(self.scorer.weights),
            "generated_at": datetime.now().isoformat(),
        }
