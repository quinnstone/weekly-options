"""
Earnings Analyst Agent — runs during Wednesday scan.

For tickers with earnings in the upcoming Mon-Fri holding window,
analyzes: consensus estimates, implied vs historical move, IV richness,
and whether the earnings event creates opportunity or risk.

This is the "event risk specialist" that flags earnings-week setups
that need special handling (tighter stops, IV crush protection, or skip).
"""

import logging

from agents.base import BaseAgent, WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the earnings event specialist for the weekly options pipeline.
You have full access to the methodology reference above.

When a ticker has earnings during the Mon-Fri holding window, assess using our
framework:

**TRADE** — IV is cheap relative to expected move. Our IV/RV ratio < 0.9 AND
IV term structure shows weekly IV below monthly (ratio < 0.85 = discount).
The earnings event creates edge, not just risk.

**CAUTION** — Mixed signals. Trade at reduced size (Kelly cap drops from 3% to
1.5% for earnings-week positions). Consider:
- Where in the week earnings falls: Mon/Tue = full theta exposure post-event,
  Thu/Fri = less time for IV crush to matter
- Our estimated IV crush (35-50%): does the directional move need to exceed
  the crush to profit?
- Pattern library: if we have 5+ observations of this pattern with earnings,
  what's the historical win rate?

**AVOID** — IV/RV > 1.3 (expensive), implied move > historical average, and/or
our event_risk score already penalizing by -20. The holding window has FOMC+earnings
double risk. Skip.

Reference the specific numbers: IV/RV ratio, term structure ratio, estimated
crush %, and how these map to our scoring thresholds. Give a 2-3 sentence verdict."""


class EarningsAnalyst(BaseAgent):
    """Analyze earnings-week tickers for the Wednesday scan."""

    AGENT_NAME = "earnings_analyst"
    MAX_TOKENS = 2000
    TOOLS = [WEB_SEARCH_TOOL]  # Can search for earnings estimates, peer results

    def analyze(self, candidates: list, market_summary: dict = None) -> list:
        """Screen candidates with upcoming earnings in the holding window.

        Parameters
        ----------
        candidates : list
            Wednesday scan candidates (full candidate dicts).
        market_summary : dict or None
            Current market context.

        Returns
        -------
        list[dict]
            Each with 'ticker', 'signal' (TRADE/CAUTION/AVOID),
            'earnings_date', 'brief'.
        """
        if not self.enabled:
            return []

        # Filter to candidates that have earnings in the holding window
        earnings_candidates = []
        for c in candidates:
            opts = c.get("options", {})
            crush = opts.get("earnings_iv_crush", {})
            if crush.get("has_earnings_in_window"):
                earnings_candidates.append(c)

        if not earnings_candidates:
            logger.info("No earnings-week candidates to analyze")
            return []

        logger.info("Analyzing %d earnings-week candidates", len(earnings_candidates))

        market_ctx = ""
        if market_summary:
            market_ctx = self._format_market_context(market_summary)

        results = []
        for c in earnings_candidates:
            ticker = c.get("ticker", "?")
            opts = c.get("options", {})
            crush = opts.get("earnings_iv_crush", {})
            iv_ts = opts.get("iv_term_structure", {})
            chain = opts.get("chain_summary", {})

            # Build context for this ticker
            earnings_date = crush.get("earnings_date", "unknown")
            crush_pct = crush.get("estimated_crush_pct", 0)
            implied_move = chain.get("implied_move_pct", 0)

            tech = c.get("technical", {})
            hist_vol = tech.get("historical_volatility", 0)

            pick_text = self._format_pick_data(c)

            earnings_detail = (
                f"\nEARNINGS DATA:"
                f"\n  Earnings date: {earnings_date}"
                f"\n  Estimated IV crush: {crush_pct:.0%}"
                f"\n  Implied move: {implied_move:.1%}"
                f"\n  IV term structure: {iv_ts.get('state', 'unknown')} "
                f"(weekly/monthly ratio: {iv_ts.get('ratio', 0):.2f})"
                f"\n  Historical vol: {hist_vol:.1%}" if hist_vol else ""
            )

            user_msg = (
                f"MARKET CONTEXT:\n{market_ctx}\n\n"
                f"CANDIDATE:\n{pick_text}"
                f"{earnings_detail}\n\n"
                f"Give your TRADE/CAUTION/AVOID verdict in 2-3 sentences."
            )

            response = self._call(SYSTEM_PROMPT, user_msg)
            if response:
                # Parse signal from response
                signal = "CAUTION"  # default to cautious
                upper = response.upper()
                if "AVOID" in upper[:20]:
                    signal = "AVOID"
                elif "TRADE" in upper[:20]:
                    signal = "TRADE"

                results.append({
                    "ticker": ticker,
                    "signal": signal,
                    "earnings_date": earnings_date,
                    "estimated_crush_pct": crush_pct,
                    "brief": response.strip(),
                })
                logger.info("Earnings brief for %s: %s (earnings %s)",
                            ticker, signal, earnings_date)

        return results
