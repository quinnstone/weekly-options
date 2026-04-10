"""
Market Narrative Agent — runs during Wednesday scan.

Synthesizes structured market data (VIX regime, breadth, credit, COT,
macro surprise, holding window events, sector performance) into a coherent
1-2 paragraph market narrative. This narrative is:

1. Saved to disk for downstream agents to reference
2. Embedded in Discord messages for trader context
3. Fed to determine_direction() and assess_macro_edge() as qualitative input

This replaces the "analyst morning meeting" where a senior trader explains
what the numbers mean in plain English.
"""

import json
import logging
from datetime import datetime

from agents.base import BaseAgent
from config import Config

logger = logging.getLogger(__name__)
config = Config()

SYSTEM_PROMPT = """You are the senior market strategist synthesizing this week's market
data into a coherent narrative. You have full access to the methodology
reference above.

Your job: read the structured market signals and write a 2-3 paragraph
market narrative that answers these questions:

1. **What regime are we in?** Reference the VIX tier (low/normal/elevated/
   high/extreme) and what that means for our weight multipliers. Is the
   regime stable (5+ days) or transitioning?

2. **Where is the edge?** Our backtest shows accuracy varies dramatically
   by regime: 72.3% in high VIX vs 50.5% in normal. Credit at 3-4.5% OAS
   = 57.1% vs tight <3% = 50.4%. Tell the trader where we are on that map.

3. **What's the dominant theme?** Breadth widening or narrowing? Rate
   expectations shifting? Sector rotation in progress? COT extreme
   positioning signaling a contrarian opportunity?

4. **What to watch this week?** Holding window events (FOMC, CPI, earnings
   clusters). How do these events interact with the regime?

End with a ONE-SENTENCE directional lean: "Lean bullish/bearish/neutral
because [specific reason]."

Be terse. No filler. Every sentence should change a trading decision."""


class MarketNarrative(BaseAgent):
    """Generate coherent market narrative from structured data."""

    AGENT_NAME = "market_narrative"
    MAX_TOKENS = 1000

    def narrate(self, market_summary: dict) -> str:
        """Generate market narrative from structured market data.

        Parameters
        ----------
        market_summary : dict
            Full market summary from MarketScanner.get_market_summary().

        Returns
        -------
        str
            Market narrative text (2-3 paragraphs).
        """
        if not self.enabled:
            return ""

        market_ctx = self._format_market_context(market_summary)

        # Add sector detail
        sector_data = market_summary.get("sector_performance", {})
        if sector_data:
            top_sectors = sorted(
                sector_data.items(),
                key=lambda x: x[1].get("return_5d", 0) if isinstance(x[1], dict) else 0,
                reverse=True,
            )[:3]
            bottom_sectors = sorted(
                sector_data.items(),
                key=lambda x: x[1].get("return_5d", 0) if isinstance(x[1], dict) else 0,
            )[:3]
            sector_text = (
                f"\nSector leaders (5d): "
                + ", ".join(f"{s[0]} ({s[1].get('return_5d', 0):+.1f}%)" for s in top_sectors if isinstance(s[1], dict))
                + f"\nSector laggards (5d): "
                + ", ".join(f"{s[0]} ({s[1].get('return_5d', 0):+.1f}%)" for s in bottom_sectors if isinstance(s[1], dict))
            )
        else:
            sector_text = ""

        # Add regime persistence detail
        persistence = market_summary.get("regime_persistence", {})
        persistence_text = ""
        if persistence:
            persistence_text = (
                f"\nRegime persistence: {'stable' if persistence.get('is_stable') else 'unstable'} "
                f"({persistence.get('regime_streak', 0)} day streak, "
                f"transition: {persistence.get('transition_direction', 'none')})"
            )

        # Yield curve
        yc = market_summary.get("yield_curve", {})
        yc_text = ""
        if yc:
            yc_text = (
                f"\nYield curve: {yc.get('curve_signal', '?')} "
                f"(2s10s spread: {yc.get('spread_2s10s', '?')}bps)"
            )

        user_msg = (
            f"MARKET DATA:\n{market_ctx}"
            f"{sector_text}"
            f"{persistence_text}"
            f"{yc_text}\n\n"
            f"Write your market narrative."
        )

        narrative = self._call(SYSTEM_PROMPT, user_msg)

        if narrative:
            # Save to disk for other agents to reference
            date_str = datetime.now().strftime("%Y-%m-%d")
            narrative_path = config.candidates_dir / date_str
            narrative_path.mkdir(parents=True, exist_ok=True)
            with open(narrative_path / "market_narrative.md", "w") as fh:
                fh.write(f"# Market Narrative — {date_str}\n\n")
                fh.write(narrative)
            logger.info("Market narrative saved (%d chars)", len(narrative))

        return narrative or ""
