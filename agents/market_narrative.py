"""
Market Narrative Agent — runs during Wednesday scan.

Synthesizes structured market data (VIX regime, breadth, credit, COT,
macro surprise, holding window events, sector performance) AND recent
market-moving news into a coherent 2-3 paragraph market narrative.

This narrative is:
1. Saved to disk for downstream agents to reference
2. Embedded in Discord messages for trader context
3. Fed to determine_direction() and assess_macro_edge() as qualitative input

This replaces the "analyst morning meeting" where a senior trader explains
what the numbers mean in plain English — including overnight developments,
geopolitical events, and policy changes that structured data alone misses.
"""

import json
import logging
from datetime import datetime

from agents.base import BaseAgent, WEB_SEARCH_TOOL
from config import Config

logger = logging.getLogger(__name__)
config = Config()

SYSTEM_PROMPT = """You are the senior market strategist synthesizing this week's market
data into a coherent narrative. You have full access to the methodology
reference above.

You have a web_search tool. Use it for exactly TWO searches:
1. Search "broad market news this week" to get macro/geopolitical headlines
2. Search for a specific developing story IF the structured data suggests
   something unusual (e.g., VIX spiking, credit widening, cross-asset
   divergence) — search for the *why*

**CRITICAL GUARDRAIL:** Your narrative must be ANCHORED on the structured
data (VIX, breadth, credit, COT, cross-asset signals). News explains
*why* the numbers moved — it does not override them. If news says "markets
panicking" but VIX is 14 and credit is tight, trust the numbers. If VIX
is 28 and news says "tariff escalation," the news explains the regime.

Your job: read the structured signals, check the news, and write a 2-3
paragraph narrative that answers:

1. **What regime are we in?** VIX tier and weight multiplier implications.
   Stable (5+ days) or transitioning? If transitioning, does the news
   explain why?

2. **Where is the edge?** Our backtest: 72.3% accuracy in high VIX vs
   50.5% in normal. Credit 3-4.5% OAS = 57.1% vs tight <3% = 50.4%.

3. **What's the dominant theme?** Lead with what the NUMBERS show (breadth,
   sectors, cross-asset), then cite any news that confirms or complicates
   the picture. Geopolitical events, trade policy, Fed signals, and fiscal
   developments belong here — but only if they're market-moving, not
   speculative commentary.

4. **What to watch this week?** Scheduled events (FOMC, CPI, earnings)
   PLUS any developing stories that could shift the regime mid-week
   (ongoing negotiations, policy announcements, etc.).

End with a ONE-SENTENCE directional lean: "Lean bullish/bearish/neutral
because [specific reason tied to both data and context]."

Be terse. No filler. Every sentence should change a trading decision.
Do NOT let news headlines dominate — the numbers are the primary signal,
news is the explanatory layer."""


class MarketNarrative(BaseAgent):
    """Generate coherent market narrative from structured data + news."""

    AGENT_NAME = "market_narrative"
    MAX_TOKENS = 1500
    TOOLS = [WEB_SEARCH_TOOL]

    def narrate(self, market_summary: dict) -> str:
        """Generate market narrative from structured market data + news.

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

        # Cross-asset detail (helps agent understand *what* moved)
        cross_asset = market_summary.get("cross_asset", {})
        cross_text = ""
        if cross_asset:
            parts = []
            for asset_key in ("bonds_20y", "us_dollar", "crude_oil"):
                asset = cross_asset.get(asset_key, {})
                ret = asset.get("return_5d")
                if ret is not None:
                    parts.append(f"{asset_key}: {ret:+.1f}% 5d ({asset.get('signal', '?')})")
            if parts:
                cross_text = f"\nCross-asset: {', '.join(parts)} → composite: {cross_asset.get('composite', '?')}"

        user_msg = (
            f"MARKET DATA:\n{market_ctx}"
            f"{sector_text}"
            f"{persistence_text}"
            f"{yc_text}"
            f"{cross_text}\n\n"
            f"Use web_search to check current market-moving news (max 2 searches). "
            f"Then write your market narrative anchored on the data above."
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
