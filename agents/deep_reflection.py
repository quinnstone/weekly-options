"""
Deep Reflection Agent — runs Saturday morning.

Instead of the mechanical reflector, uses Claude to analyze the week's
trades with nuance: why specific picks failed, what regime shifts occurred,
and what tactical adjustments to make for next week.

This is the "CIO weekly review" that quant desks do on Saturdays.
"""

import json
import logging
from pathlib import Path

from agents.base import BaseAgent
from config import Config

logger = logging.getLogger(__name__)
config = Config()

SYSTEM_PROMPT = """You are the Chief Investment Officer reviewing this week's weekly options
trades. You have full access to the methodology reference above.

Analyze each trade through the lens of our compound methodology:

**For each trade, diagnose the failure/success mode:**
- **Direction wrong:** Which of the 10 direction signals misled us? Was the
  ensemble in consensus (std < 5) or disagreement (std > 15)?
- **Direction right, theta killed it:** Was charm decay faster than expected?
  Did we hold too long past the day-specific target (40%→35%→25%→15%→10%)?
- **Regime shift:** Did VIX move between regimes mid-week, invalidating the
  weight multipliers that were set at entry? (e.g., entered in "normal" but
  VIX spiked to "elevated" on Wednesday)
- **IV mispriced:** Was IV/RV ratio accurate at entry? Did IV term structure
  (weekly vs monthly) correctly signal cheap/expensive?
- **Event dominated:** Did FOMC/CPI/earnings overwhelm the directional thesis?
  Was the event_risk score adequate (-20 for high-impact)?
- **Pattern library insight:** If this trade matched a known pattern, did the
  historical win rate predict the outcome?

**Propose 1-3 SPECIFIC tactical adjustments.** Frame them as testable hypotheses
that must pass the validation gate (accuracy drop ≤2%, Sharpe drop ≤0.1):
- Weight changes: "Reduce momentum from 0.20 to 0.17 in elevated+ regimes"
- Threshold changes: "Widen stops to 60% when VIX > 25"
- Process changes: "For FOMC weeks, enter Tuesday after announcement instead of Monday"

**Note:** These proposals will be tested against the backtest validation gate
before deployment. Be specific enough to implement and test.

End with a one-sentence market outlook for next week, referencing the current
VIX regime, credit conditions, and any macro events in the coming holding window."""


class DeepReflectionAgent(BaseAgent):
    """Generate deep qualitative analysis of the week's trades."""

    AGENT_NAME = "deep_reflection"
    MAX_TOKENS = 3000

    def reflect(self, reflection: dict, scorecard: dict = None,
                market_summary: dict = None) -> dict:
        """Generate a deep reflection on the week's performance.

        Parameters
        ----------
        reflection : dict
            Mechanical reflection from WeeklyReflector.
        scorecard : dict or None
            Graded results from Scorecard.
        market_summary : dict or None
            This week's market context.

        Returns
        -------
        dict
            'analysis' (str), 'tactical_adjustments' (list),
            'market_outlook' (str).
        """
        if not self.enabled:
            return {"analysis": "", "tactical_adjustments": [], "market_outlook": ""}

        # Build context from reflection
        week = reflection.get("week", "unknown")
        win_rate = reflection.get("win_rate", 0)
        total = reflection.get("total_picks", 0)
        wins = reflection.get("wins", 0)
        losses = reflection.get("losses", 0)
        avg_ret = reflection.get("avg_return", 0)

        worked = reflection.get("what_worked", [])
        failed = reflection.get("what_failed", [])
        correlations = reflection.get("signal_correlations", {})
        lessons = reflection.get("lessons", [])
        weights = reflection.get("current_weights", {})

        worked_text = "\n".join(
            f"  {w['ticker']}: +{w.get('return',0):.1%} — {', '.join(w.get('reasons',[]))}"
            for w in worked
        ) or "  None"

        failed_text = "\n".join(
            f"  {f['ticker']}: {f.get('return',0):.1%} — {', '.join(f.get('reasons',[]))}"
            for f in failed
        ) or "  None"

        corr_text = "\n".join(
            f"  {cat}: {corr:+.3f}" for cat, corr in
            sorted(correlations.items(), key=lambda x: -x[1])
        ) or "  No data"

        # Scorecard details
        sc_text = ""
        if scorecard:
            for p in scorecard.get("picks", []):
                sc_text += (
                    f"\n  {p.get('ticker')}: {p.get('direction','?').upper()} "
                    f"${p.get('strike','?')} — {p.get('result','?')} "
                    f"(entry ${p.get('entry_premium','?')}, close ${p.get('closing_price','?')}, "
                    f"P&L ${p.get('pnl', 0):+,.2f})"
                )

        market_text = self._format_market_context(market_summary) if market_summary else "No market data."

        user_msg = (
            f"WEEK OF {week}\n"
            f"Win rate: {win_rate:.0%} ({wins}W / {losses}L / {total} total)\n"
            f"Average return: {avg_ret:.1%}\n\n"
            f"MARKET CONTEXT:\n{market_text}\n\n"
            f"WHAT WORKED:\n{worked_text}\n\n"
            f"WHAT FAILED:\n{failed_text}\n\n"
            f"SIGNAL CORRELATIONS:\n{corr_text}\n\n"
            f"SCORECARD DETAIL:{sc_text or ' No scorecard data.'}\n\n"
            f"CURRENT WEIGHTS:\n{json.dumps(weights, indent=2)}\n\n"
            f"Give your CIO analysis, tactical adjustments, and one-sentence market outlook."
        )

        response = self._call(SYSTEM_PROMPT, user_msg)

        if not response:
            return {"analysis": "", "tactical_adjustments": [], "market_outlook": ""}

        # Save the full analysis to disk
        analysis_path = config.performance_dir / f"{week}_deep_reflection.md"
        with open(analysis_path, "w") as fh:
            fh.write(f"# Deep Reflection — Week of {week}\n\n")
            fh.write(response)

        # Append methodology changes since the prior reflection. Documentation
        # only — does NOT get fed back into the agent prompt (avoiding any
        # influence on its analysis). Surfaces what code interventions shipped
        # between weeks so future readers can correlate outcomes with changes.
        try:
            changes_section = self._methodology_changes_section(week)
            if changes_section:
                with open(analysis_path, "a") as fh:
                    fh.write(changes_section)
                logger.info("Appended methodology changes section to %s", analysis_path.name)
        except Exception as exc:
            logger.warning("Methodology changes section failed (non-fatal): %s", exc)

        logger.info("Deep reflection saved to %s", analysis_path)

        return {
            "analysis": response,
            "tactical_adjustments": [],  # Could parse from response if needed
            "market_outlook": "",
        }

    def _methodology_changes_section(self, current_week: str) -> str:
        """Build a markdown section listing code commits since the prior reflection.

        Pure documentation — never injected into agent prompts. Filters out
        bot data commits ('pipeline: ...') so only methodology changes appear.
        Returns empty string if no prior reflection found or no changes detected.
        """
        import subprocess

        # Find the most recent prior deep_reflection.md (excluding the current week)
        reflection_files = sorted(
            config.performance_dir.glob("*_deep_reflection.md"),
            reverse=True,
        )
        prior_files = [f for f in reflection_files if not f.name.startswith(current_week)]
        if not prior_files:
            return ""

        prior_date = prior_files[0].name.split("_")[0]  # e.g., "2026-04-25"

        try:
            result = subprocess.run(
                ["git", "log", "--no-merges", "--pretty=format:%h %s",
                 f"--since={prior_date}"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as exc:
            logger.debug("git log failed for changes section: %s", exc)
            return ""

        if result.returncode != 0:
            return ""

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        # Filter out bot data commits — only keep methodology changes
        meaningful = [l for l in lines if "pipeline:" not in l.lower()]

        section = f"\n\n---\n\n## Methodology Changes Since {prior_date}\n\n"
        section += (
            "_Documentation only — these commits shipped between the prior reflection "
            "and this one. The CIO analysis above does not incorporate this list "
            "(avoiding any anchoring on recent interventions). Use this section to "
            "correlate outcomes with code changes when reviewing weeks side-by-side._\n\n"
        )

        if not meaningful:
            section += "_No methodology changes since the prior reflection — picks were generated by the same code._\n"
        else:
            for line in meaningful:
                section += f"- `{line}`\n"

        return section
