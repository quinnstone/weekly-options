"""
Post-Mortem Agent — runs during Friday scorecard grading.

Analyzes ALL graded picks in a SINGLE batched call. For each WIN/LOSS/PARTIAL,
explains WHY it won or lost — identifying which signals were right/wrong,
whether the thesis held, and what event or regime shift drove the outcome.

Batched architecture: one Opus call covers all 3 picks for cost efficiency
and more coherent cross-pick pattern recognition.
"""

import logging

from agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the trade post-mortem analyst. You have full access to the
methodology reference above.

You will receive ALL graded picks for the week in one batch. For EACH pick,
write a 2-3 sentence post-mortem explaining WHY it won or lost.

Focus on MECHANISM, not outcome:
- **WINS:** Which signal(s) drove the correct thesis? Was the win expected
  (strong consensus) or surprising (weak consensus)? Did it hit the day-specific
  target or expire ITM through luck?
- **LOSSES:** What invalidated the thesis? Event risk? Regime shift? Gap against?
  Was the loss predictable (model_std > 15, event_risk penalized)?
- **PARTIALS:** Why did the stock move right but not enough to cover premium + theta?

After individual picks, add a PATTERNS section:
- Any common thread across wins or losses this week?
- Did the VIX regime hold stable or shift mid-week?
- Were direction signals generally reliable or noisy?

Reference specific numbers. Don't say "the stock went down" — say "AAPL dropped
3.2% Thursday after earnings miss, invalidating the momentum thesis (5d return
was +2.1% at entry)."

Format:
[TICKER]: [WIN/LOSS/PARTIAL] — [2-3 sentence post-mortem]
...
PATTERNS: [2-3 sentences on cross-pick observations]"""


class PostMortemAgent(BaseAgent):
    """Generate batched post-mortems for all graded picks."""

    AGENT_NAME = "post_mortem"
    MAX_TOKENS = 2000

    def analyze_week(self, picks: list, outcomes: list,
                     market_summary: dict = None) -> list:
        """Generate post-mortems for all picks in a single batched call.

        Parameters
        ----------
        picks : list
            Original pick dicts.
        outcomes : list
            Grading results from scorecard.
        market_summary : dict or None
            Market context from the pick's week.

        Returns
        -------
        list[dict]
            Each with 'ticker', 'result', 'pnl', 'post_mortem'.
        """
        if not self.enabled or not picks or not outcomes:
            return []

        outcome_map = {o.get("ticker"): o for o in outcomes}

        # Build all picks into a single prompt
        picks_text = []
        for pick in picks:
            ticker = pick.get("ticker", "?")
            outcome = outcome_map.get(ticker, {})
            if not outcome:
                continue

            pick_text = self._format_pick_data(pick)
            result = outcome.get("result", "unknown")
            pnl = outcome.get("pnl", 0)
            entry = outcome.get("entry_premium", "?")
            close = outcome.get("closing_price", "?")

            picks_text.append(
                f"### {ticker}: {result.upper()} (${pnl:+,.2f})\n"
                f"{pick_text}\n"
                f"Entry premium: ${entry}, Closing price: ${close}"
            )

        if not picks_text:
            return []

        market_ctx = ""
        if market_summary:
            market_ctx = f"MARKET CONTEXT:\n{self._format_market_context(market_summary)}\n\n"

        user_msg = (
            f"{market_ctx}"
            f"GRADED PICKS:\n\n"
            + "\n\n".join(picks_text) +
            f"\n\nWrite post-mortems for each pick, then identify cross-pick patterns."
        )

        response = self._call(SYSTEM_PROMPT, user_msg)
        if not response:
            return []

        # Parse per-ticker post-mortems from batched response
        results = []
        for pick in picks:
            ticker = pick.get("ticker", "?")
            outcome = outcome_map.get(ticker, {})
            if not outcome:
                continue

            # Find this ticker's section
            lines = response.split("\n")
            pm_lines = []
            capturing = False
            for line in lines:
                if ticker.upper() in line.upper() and any(s in line.upper() for s in ["WIN", "LOSS", "PARTIAL"]):
                    capturing = True
                    # Get text after the marker
                    for marker in ["—", "-", ":"]:
                        parts = line.split(marker, 1)
                        if len(parts) > 1 and len(parts[1].strip()) > 10:
                            pm_lines.append(parts[1].strip())
                            break
                    continue
                if capturing:
                    if line.strip() == "" or "PATTERN" in line.upper():
                        break
                    if any(p.get("ticker", "").upper() in line.upper()
                           for p in picks if p.get("ticker") != ticker):
                        break
                    pm_lines.append(line.strip())

            post_mortem = " ".join(pm_lines).strip()

            results.append({
                "ticker": ticker,
                "result": outcome.get("result", "unknown"),
                "pnl": outcome.get("pnl", 0),
                "post_mortem": post_mortem or f"No specific analysis parsed for {ticker}.",
            })
            logger.info("Post-mortem for %s (%s): %s",
                         ticker, outcome.get("result", "?"),
                         post_mortem[:80] if post_mortem else "N/A")

        return results
