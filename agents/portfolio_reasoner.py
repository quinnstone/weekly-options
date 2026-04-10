"""
Portfolio Reasoner Agent — runs during Monday picks (final 5 selection).

Combined agent: selects the optimal 3-pick portfolio AND writes a trading
thesis for each. One Opus call replaces two separate agents.

Reviews concentration risk, macro exposure, correlation structure, and
regime fit. Then generates per-pick theses for Discord delivery.
"""

import json
import logging
from datetime import datetime

from agents.base import BaseAgent
from config import Config

logger = logging.getLogger(__name__)
_config = Config()

SYSTEM_PROMPT = """You are the portfolio construction specialist AND thesis writer for the
weekly options trading desk. You have full access to the methodology
reference above.

You receive 5-6 scored candidates. Do TWO things in one response:

## PART 1: Select the Best 3

We run 3 high-conviction picks per week — quality over quantity. Consider:
1. **Concentration risk:** With only 3 picks, each position matters more.
   Avoid macro correlation (e.g., 2 tech stocks sensitive to rate moves).
2. **Regime fit:** Which picks align with this VIX regime's weight
   multipliers? In elevated+ VIX, mean-reversion gets 1.2-1.5x.
3. **Direction balance:** If all 3 are calls and FOMC is this week,
   the entire book has the same event risk. Consider adding a hedge.
4. **Event exposure:** With 3 picks, even 1 earnings surprise can blow
   up a third of the book. Avoid stacking event risk.
5. **Quality bar:** Every pick must be high-conviction. If candidate #3
   has model_std > 15 or weak consensus, run 2 picks instead of 3.

## PART 2: Write Theses

For each selected pick, write a 2-3 sentence trading thesis:
- Name the dominant signal driving the pick
- State what makes THIS WEEK the right entry
- Name the kill condition that would invalidate the thesis

## Response Format

SELECTED:
1. TICKER — [1-line portfolio justification]
   THESIS: [2-3 sentence trading thesis]
2. TICKER — [1-line portfolio justification]
   THESIS: [2-3 sentence trading thesis]
...
DROPPED: TICKER — [reason], TICKER — [reason]
PORTFOLIO NOTE: [1-2 sentences on overall book risk/construction]"""


class PortfolioReasoner(BaseAgent):
    """Select optimal portfolio and write theses in a single call."""

    AGENT_NAME = "portfolio_reasoner"
    MAX_TOKENS = 2500

    def select(self, candidates: list, market_summary: dict = None,
               macro_edge: dict = None) -> dict:
        """Review candidates and propose optimal 3-pick portfolio with theses.

        Parameters
        ----------
        candidates : list
            Top 5-6 scored candidates (post-diversity filter).
        market_summary : dict or None
            Current market context.
        macro_edge : dict or None
            Macro edge assessment from scorer.

        Returns
        -------
        dict
            'selected' (list of 5 tickers), 'dropped' (list),
            'theses' (dict: ticker -> thesis), 'reasoning' (str).
        """
        if not self.enabled:
            from config import PORTFOLIO_SIZE
            return {
                "selected": [c.get("ticker") for c in candidates[:PORTFOLIO_SIZE]],
                "dropped": [c.get("ticker") for c in candidates[PORTFOLIO_SIZE:]],
                "theses": {},
                "reasoning": "",
            }

        # Format candidates
        candidate_text = []
        for i, c in enumerate(candidates, 1):
            pick_text = self._format_pick_data(c)
            sector = c.get("sector", "unknown")
            kelly = c.get("kelly_pct", "N/A")
            earnings = " [EARNINGS THIS WEEK]" if c.get("earnings_warning") else ""
            candidate_text.append(
                f"### Rank #{i}\n{pick_text}\n"
                f"Sector: {sector} | Kelly size: {kelly}{earnings}"
            )

        market_ctx = ""
        if market_summary:
            market_ctx = self._format_market_context(market_summary)

            # Load narrative if available
            date_str = datetime.now().strftime("%Y-%m-%d")
            narrative_path = _config.candidates_dir / date_str / "market_narrative.md"
            try:
                if narrative_path.exists():
                    market_ctx += f"\n\nMARKET NARRATIVE:\n{narrative_path.read_text()}"
            except Exception:
                pass

        edge_text = ""
        if macro_edge:
            edge_text = (
                f"\nMacro edge: {'YES' if macro_edge.get('has_edge') else 'NO'} "
                f"(multiplier: {macro_edge.get('confidence_multiplier', 1.0):.2f})"
                f"\nReasons: {', '.join(macro_edge.get('reasons', []))}"
            )

        user_msg = (
            f"MARKET CONTEXT:\n{market_ctx}{edge_text}\n\n"
            f"CANDIDATES (ranked by composite score):\n\n"
            + "\n\n".join(candidate_text) +
            f"\n\nSelect the optimal 3 (high-conviction only) with portfolio "
            f"justification, write a thesis for each, and note what you dropped."
        )

        response = self._call(SYSTEM_PROMPT, user_msg)

        if not response:
            from config import PORTFOLIO_SIZE
            return {
                "selected": [c.get("ticker") for c in candidates[:PORTFOLIO_SIZE]],
                "dropped": [c.get("ticker") for c in candidates[PORTFOLIO_SIZE:]],
                "theses": {},
                "reasoning": "",
            }

        # Parse selected tickers and theses from response
        selected = []
        dropped = []
        theses = {}

        # Parse SELECTED section
        lines = response.split("\n")
        in_dropped = False
        current_ticker = None

        for line in lines:
            upper_line = line.upper().strip()

            if upper_line.startswith("DROPPED"):
                in_dropped = True
                # Parse dropped tickers from this line
                for c in candidates:
                    t = c.get("ticker", "")
                    if t.upper() in upper_line:
                        dropped.append(t)
                continue

            if in_dropped:
                # Check for more dropped tickers on continuation lines
                for c in candidates:
                    t = c.get("ticker", "")
                    if t.upper() in line.upper() and t not in dropped:
                        dropped.append(t)
                continue

            # Check for numbered selection lines (1. TICKER, 2. TICKER, etc.)
            for c in candidates:
                t = c.get("ticker", "")
                if t.upper() in upper_line and t not in selected and not in_dropped:
                    if any(upper_line.startswith(f"{n}.") or upper_line.startswith(f"{n} ") for n in range(1, 10)):
                        selected.append(t)
                        current_ticker = t
                        break
                    elif t.upper() == upper_line.split()[0] if upper_line.split() else "":
                        selected.append(t)
                        current_ticker = t
                        break

            # Parse THESIS lines
            if "THESIS:" in upper_line and current_ticker:
                thesis_text = line.split(":", 1)[-1].strip() if ":" in line else ""
                if thesis_text:
                    theses[current_ticker] = thesis_text

        # Fallback: if parsing fails, use first 3
        from config import PORTFOLIO_SIZE
        if len(selected) < 2:
            selected = [c.get("ticker") for c in candidates[:PORTFOLIO_SIZE]]
            dropped = [c.get("ticker") for c in candidates[PORTFOLIO_SIZE:]]

        # Trim to exactly PORTFOLIO_SIZE
        if len(selected) > PORTFOLIO_SIZE:
            selected = selected[:PORTFOLIO_SIZE]

        logger.info("Portfolio reasoner selected: %s (dropped: %s)",
                     ", ".join(selected), ", ".join(dropped))
        if theses:
            logger.info("Generated theses for %d picks", len(theses))

        return {
            "selected": selected,
            "dropped": dropped,
            "theses": theses,
            "reasoning": response.strip(),
        }
