"""
Pre-Trade Analyst Agent — runs at Monday 10 AM entry confirmation.

Reads ALL 3 picks + market context in a SINGLE batched call, searches
for overnight developments, and writes a brief per pick. Flags any
picks where the thesis has been invalidated.

Batched architecture: one Opus call covers all picks (with web search
tool for overnight news). This is cheaper and produces more coherent
cross-pick analysis than 5 separate calls.
"""

import logging
from datetime import datetime

from agents.base import BaseAgent, WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the pre-trade analyst reviewing Monday morning picks before
market entry. You have full access to the methodology reference above.

You will receive ALL picks in one batch. For EACH pick, write a 2-3 sentence
assessment. Use the web_search tool to check for overnight news on any ticker
where you suspect a material development (earnings, downgrades, sector events).
You don't need to search every ticker — only those where something might have changed.

For each pick, lead with the signal:
- **GO:** Thesis intact. Note ensemble agreement and the key risk for the week.
- **ADJUST:** Something shifted. Reference specific thresholds (delta drift >0.15,
  gap >2%, IV/RV now >1.3).
- **SKIP:** Thesis invalidated. Cite the specific signal that broke.

Also assess the PORTFOLIO as a whole:
- Are the 3 picks too correlated in macro exposure?
- Does the direction mix make sense for this week's regime?
- Any picks that overlap in event risk (e.g., two picks exposed to FOMC)?

Format your response as:
[TICKER]: [GO/ADJUST/SKIP] — [2-3 sentence assessment]
...
PORTFOLIO: [1-2 sentence overall assessment]"""


class PreTradeAnalyst(BaseAgent):
    """Analyze all picks before Monday entry (batched single call)."""

    AGENT_NAME = "pre_trade_analyst"
    MAX_TOKENS = 2500
    TOOLS = [WEB_SEARCH_TOOL]

    def analyze(self, picks: list, market_summary: dict,
                confirmations: list = None) -> list:
        """Generate briefs for all picks in a single batched call.

        Parameters
        ----------
        picks : list
            Monday picks from the pipeline.
        market_summary : dict
            Current market context.
        confirmations : list or None
            Entry confirmation data (gap, delta drift, etc.).

        Returns
        -------
        list[dict]
            Each with 'ticker', 'signal' (GO/ADJUST/SKIP), 'brief'.
        """
        if not self.enabled or not picks:
            return []

        market_ctx = self._format_market_context(market_summary)

        # Build confirmation map
        conf_map = {}
        if confirmations:
            for c in confirmations:
                conf_map[c.get("ticker", "")] = c

        # Build ALL picks into a single prompt
        picks_text = []
        for i, pick in enumerate(picks, 1):
            ticker = pick.get("ticker", "?")
            pick_text = self._format_pick_data(pick)

            conf = conf_map.get(ticker, {})
            conf_text = ""
            if conf:
                signal = conf.get("signal", "?")
                conf_text = (
                    f"\n  Mechanical check: {signal}"
                    f" | Price: ${conf.get('current_price', '?')}"
                    f" | Gap: {conf.get('gap_pct', '?')}%"
                    f" | {conf.get('detail', '')}"
                )

            picks_text.append(f"### Pick #{i}: {ticker}\n{pick_text}{conf_text}")

        user_msg = (
            f"MARKET CONTEXT:\n{market_ctx}\n\n"
            f"ALL PICKS FOR REVIEW:\n\n"
            + "\n\n".join(picks_text) +
            f"\n\nSearch for overnight news on any tickers where you suspect "
            f"material changes. Then give GO/ADJUST/SKIP per pick, plus a "
            f"portfolio-level assessment."
        )

        response = self._call(SYSTEM_PROMPT, user_msg)
        if not response:
            return []

        # Parse per-ticker signals from batched response
        results = []
        for pick in picks:
            ticker = pick.get("ticker", "?")
            # Find this ticker's section in the response
            signal = "GO"  # default
            brief = ""

            lines = response.split("\n")
            capturing = False
            brief_lines = []
            for line in lines:
                if ticker.upper() in line.upper() and any(s in line.upper() for s in ["GO", "ADJUST", "SKIP"]):
                    capturing = True
                    # Parse signal from this line
                    upper_line = line.upper()
                    if "SKIP" in upper_line:
                        signal = "SKIP"
                    elif "ADJUST" in upper_line:
                        signal = "ADJUST"
                    else:
                        signal = "GO"
                    # Get the text after the signal
                    for marker in ["—", "-", ":"]:
                        if marker in line:
                            brief_lines.append(line.split(marker, 1)[-1].strip())
                            break
                    continue
                if capturing:
                    if line.strip() == "" or "PORTFOLIO" in line.upper():
                        break
                    # Stop if we hit another ticker
                    if any(p.get("ticker", "").upper() in line.upper()
                           for p in picks if p.get("ticker") != ticker):
                        break
                    brief_lines.append(line.strip())

            brief = " ".join(brief_lines).strip()

            results.append({
                "ticker": ticker,
                "signal": signal,
                "brief": brief or f"No specific assessment parsed for {ticker}.",
            })
            logger.info("Pre-trade brief for %s: %s", ticker, signal)

        return results
