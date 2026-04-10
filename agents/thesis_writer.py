"""
Thesis Writer Agent — runs after Monday picks are finalized.

For each of the 5 final picks, generates a 2-3 sentence trading thesis
that explains the reasoning in plain English. These theses are embedded
in the Discord picks notification.

This is the "analyst note" attached to each trade recommendation — not
a recap of the data, but a synthesis of WHY this trade should work.
"""

import logging

from agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the thesis writer for the weekly options trading desk. You have
full access to the methodology reference above.

For each pick, write a 2-3 sentence trading thesis. The thesis must:

1. **Name the dominant signal** — What's driving this pick? Momentum
   continuation, mean-reversion snap, IV mispricing, flow confirmation?
   Reference the actual score (e.g., "momentum scored 78/100").

2. **State the catalyst** — What makes THIS WEEK the right week? Is
   there an event, a technical breakout, a sector rotation underway?

3. **Name the kill condition** — What would invalidate the thesis?
   "If VIX breaks above 30, momentum multiplier drops to 0.5x and this
   trade loses its edge." Or "If earnings miss expectations, IV crush
   wipes out any directional gain."

Don't repeat data. Don't say "the composite score is 74.2" — the trader
can see that. Explain what the numbers MEAN for this specific trade
this specific week.

Format: TICKER — thesis text. One thesis per pick."""


class ThesisWriter(BaseAgent):
    """Generate trading theses for Discord pick notifications."""

    AGENT_NAME = "thesis_writer"
    MAX_TOKENS = 1500

    def write_theses(self, picks: list, market_summary: dict = None,
                     market_narrative: str = None) -> list:
        """Generate theses for all Monday picks.

        Parameters
        ----------
        picks : list
            Final picks with full data.
        market_summary : dict or None
            Current market context.
        market_narrative : str or None
            Market narrative from MarketNarrative agent.

        Returns
        -------
        list[dict]
            Each with 'ticker', 'thesis'.
        """
        if not self.enabled or not picks:
            return []

        market_ctx = ""
        if market_summary:
            market_ctx = self._format_market_context(market_summary)
        if market_narrative:
            market_ctx += f"\n\nMARKET NARRATIVE:\n{market_narrative}"

        # Build all picks in one call (cheaper and more coherent)
        picks_text = []
        for i, pick in enumerate(picks, 1):
            pick_text = self._format_pick_data(pick)
            sector = pick.get("sector", "unknown")
            earnings = "EARNINGS THIS WEEK" if pick.get("earnings_warning") else ""
            picks_text.append(f"### Pick #{i}\n{pick_text}\nSector: {sector} {earnings}")

        user_msg = (
            f"MARKET CONTEXT:\n{market_ctx}\n\n"
            f"PICKS:\n\n" + "\n\n".join(picks_text) +
            f"\n\nWrite a 2-3 sentence thesis for each pick."
        )

        response = self._call(SYSTEM_PROMPT, user_msg)
        if not response:
            return []

        # Parse theses — look for ticker mentions
        results = []
        for pick in picks:
            ticker = pick.get("ticker", "?")
            # Find the section for this ticker
            lines = response.split("\n")
            thesis_lines = []
            capturing = False
            for line in lines:
                if ticker.upper() in line.upper() and ("—" in line or "-" in line or ":" in line):
                    capturing = True
                    # Include the line after the ticker header
                    after_ticker = line.split("—", 1)[-1].strip() if "—" in line else line.split("-", 1)[-1].strip() if "-" in line else ""
                    if after_ticker:
                        thesis_lines.append(after_ticker)
                    continue
                if capturing:
                    if line.strip() == "" or (any(t.get("ticker", "").upper() in line.upper() for t in picks if t.get("ticker") != ticker)):
                        break
                    thesis_lines.append(line.strip())

            thesis = " ".join(thesis_lines).strip()
            if not thesis:
                # Fallback: just attribute the full response
                thesis = ""

            results.append({
                "ticker": ticker,
                "thesis": thesis,
            })
            if thesis:
                logger.info("Thesis for %s: %s", ticker, thesis[:80])

        return results
