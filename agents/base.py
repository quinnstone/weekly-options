"""
Base agent infrastructure for the Weekly Options Trading Analysis System.

Uses the Anthropic API to run specialized Claude agents that analyze
trade setups, monitor positions, and generate insights. Each agent
receives structured data and returns structured analysis.

Agents are optional — the pipeline runs without them if ANTHROPIC_API_KEY
is not configured. When available, agent output is embedded in Discord
notifications alongside the quantitative signals.

Context loading hierarchy:
1. METHODOLOGY.md — Static reference (scoring rules, thresholds, architecture)
2. CURRENT_STATE.md — Dynamic snapshot (live weights, pattern stats, recent lessons)
3. playbook.md / known_risks.md — Accumulated strategic knowledge from data/performance/
"""

import json
import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)
config = Config()

# ---------------------------------------------------------------
# Context document loading
# ---------------------------------------------------------------

_AGENTS_DIR = Path(__file__).parent
_METHODOLOGY_PATH = _AGENTS_DIR / "METHODOLOGY.md"
_CURRENT_STATE_PATH = _AGENTS_DIR / "CURRENT_STATE.md"

# Cache loaded docs (cleared per-process; fresh on each cron run)
_DOC_CACHE = {}


def _load_doc(path: Path, label: str) -> str:
    """Load a text file with caching. Returns empty string on failure."""
    if path not in _DOC_CACHE:
        try:
            _DOC_CACHE[path] = path.read_text()
            logger.debug("Loaded %s (%d chars)", label, len(_DOC_CACHE[path]))
        except FileNotFoundError:
            logger.debug("%s not found at %s — skipping", label, path)
            _DOC_CACHE[path] = ""
        except Exception as exc:
            logger.warning("Failed to load %s: %s", label, exc)
            _DOC_CACHE[path] = ""
    return _DOC_CACHE[path]


def _build_system_context() -> str:
    """Assemble the full context block injected into every agent call.

    Loads (in order):
    1. METHODOLOGY.md — static scoring rules and architecture
    2. CURRENT_STATE.md — dynamic live state (weights, patterns, lessons)
    3. playbook.md — accumulated strategic playbook from performance dir
    4. known_risks.md — known system limitations and caveats
    """
    sections = []

    methodology = _load_doc(_METHODOLOGY_PATH, "METHODOLOGY.md")
    if methodology:
        sections.append(methodology)

    current_state = _load_doc(_CURRENT_STATE_PATH, "CURRENT_STATE.md")
    if current_state:
        sections.append(current_state)

    # Load from data/performance/ if they exist
    playbook_path = config.performance_dir / "playbook.md"
    playbook = _load_doc(playbook_path, "playbook.md")
    if playbook:
        sections.append(f"# Strategic Playbook\n\n{playbook}")

    risks_path = config.performance_dir / "known_risks.md"
    risks = _load_doc(risks_path, "known_risks.md")
    if risks:
        sections.append(f"# Known Risks & Limitations\n\n{risks}")

    trade_log_path = _AGENTS_DIR / "TRADE_LOG.md"
    trade_log = _load_doc(trade_log_path, "TRADE_LOG.md")
    if trade_log:
        sections.append(trade_log)

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------
# Web search tool definition (for agents that need it)
# ---------------------------------------------------------------

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for recent news, pre-market moves, earnings reports, "
        "sector developments, or analyst commentary about a specific ticker or "
        "market topic. Use this to check for overnight developments that could "
        "affect the trading thesis. Return the search query as a string."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — be specific. Include ticker symbol and timeframe. E.g., 'AAPL earnings results after hours April 2026' or 'FOMC rate decision impact markets today'",
            },
        },
        "required": ["query"],
    },
}


def _extract_ticker(query: str) -> "str | None":
    """Try to extract a stock ticker from a search query.

    Returns the first plausible ticker found, or None for broad/macro queries.
    """
    # Known non-ticker words that look like tickers
    STOP_WORDS = {
        "THE", "FOR", "AND", "NOT", "ARE", "WAS", "HAS", "HAD", "CAN", "MAY",
        "NOW", "NEW", "ALL", "OUR", "HOW", "ANY", "FED", "GDP", "CPI", "PPI",
        "NFP", "PCE", "OIL", "WAR", "BIG", "LOW", "KEY", "TOP", "USE", "SET",
        "RUN", "CUT", "PUT", "GET", "LET", "SAY", "SEE", "TRY", "BUY", "OAS",
        "NEWS", "RATE", "FOMC", "BOND", "EURO", "JOBS", "DATA", "MOVE",
        "RISK", "WEEK", "LAST", "NEXT", "FROM", "WILL", "WHAT", "THIS",
        "THAT", "WITH", "THAN", "INTO", "OVER", "ALSO", "BEEN", "JUST",
        "IMPACT", "MARKET", "MACRO", "TRADE", "TODAY", "WORLD",
        "TARIFF", "GLOBAL", "BROAD", "GEOPOLITICAL",
    }
    words = query.upper().split()
    for word in words:
        if word.isalpha() and 1 <= len(word) <= 5 and word not in STOP_WORDS:
            return word
    return None


def _execute_web_search(query: str) -> str:
    """Execute a web search using available providers.

    For ticker-specific queries: yfinance news + Finnhub company news.
    For broad/macro queries: Finnhub general market news.
    """
    results = []
    ticker = _extract_ticker(query)

    if ticker:
        # --- Ticker-specific search ---

        # Strategy 1: yfinance news (always available)
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            news = tk.news
            if news:
                for article in news[:5]:
                    title = article.get("title", "")
                    publisher = article.get("publisher", "")
                    results.append(f"- [{publisher}] {title}")
        except Exception:
            pass

        # Strategy 2: Finnhub company news
        try:
            finnhub_key = config.finnhub_api_key
            if finnhub_key:
                import urllib.request
                from datetime import datetime, timedelta
                today = datetime.now().strftime("%Y-%m-%d")
                yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
                url = (
                    f"https://finnhub.io/api/v1/company-news?"
                    f"symbol={ticker}&from={yesterday}&to={today}&"
                    f"token={finnhub_key}"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "WeeklyOptions/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                    for article in data[:5]:
                        headline = article.get("headline", "")
                        source = article.get("source", "")
                        summary = article.get("summary", "")[:150]
                        entry = f"- [{source}] {headline}"
                        if summary:
                            entry += f": {summary}"
                        if entry not in results:
                            results.append(entry)
        except Exception:
            pass
    else:
        # --- Broad market / macro / geopolitical search ---
        results = _fetch_general_news()

    if results:
        return f"Web search results for '{query}':\n" + "\n".join(results[:10])
    return f"No recent news found for '{query}'. The search may have failed or there may be no recent coverage."


def _fetch_general_news(max_articles: int = 10) -> list:
    """Fetch broad market news from Finnhub general news endpoint.

    Returns market-moving headlines — geopolitics, Fed, trade policy,
    macro events. Filters out crypto, lifestyle, and low-relevance noise.
    """
    try:
        finnhub_key = config.finnhub_api_key
        if not finnhub_key:
            return []

        import urllib.request
        url = f"https://finnhub.io/api/v1/news?category=general&token={finnhub_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "WeeklyOptions/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        # Filter for market-relevant news (skip crypto, lifestyle, fluff)
        SKIP_KEYWORDS = {"crypto", "bitcoin", "ethereum", "nft", "meme coin",
                         "celebrity", "kardashian", "reality tv"}
        results = []
        for article in data:
            headline = (article.get("headline") or "").strip()
            source = article.get("source", "")
            summary = (article.get("summary") or "")[:150]
            if not headline:
                continue
            headline_lower = headline.lower()
            if any(kw in headline_lower for kw in SKIP_KEYWORDS):
                continue
            entry = f"- [{source}] {headline}"
            if summary:
                entry += f": {summary}"
            results.append(entry)
            if len(results) >= max_articles:
                break

        return results
    except Exception:
        return []


class BaseAgent:
    """Base class for Claude-powered analysis agents.

    All agents automatically receive:
    1. METHODOLOGY.md — static scoring rules, thresholds, architecture
    2. CURRENT_STATE.md — dynamic live weights, pattern stats, lessons
    3. playbook.md / known_risks.md — accumulated strategic knowledge
    4. TRADE_LOG.md — recent trade outcomes

    Agents with TOOLS defined get tool_use capability (e.g., web search).
    """

    # Subclasses set these
    AGENT_NAME = "base"
    MODEL = "claude-opus-4-20250514"
    MAX_TOKENS = 1500
    TOOLS = []  # Subclasses can add tools (e.g., [WEB_SEARCH_TOOL])

    def __init__(self):
        self.enabled = config.has_anthropic()
        self._client = None
        if self.enabled:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
                logger.info("Agent '%s' initialized", self.AGENT_NAME)
            except Exception as exc:
                logger.warning("Failed to initialize agent '%s': %s", self.AGENT_NAME, exc)
                self.enabled = False

    def _call(self, system_prompt: str, user_message: str) -> str:
        """Call Claude API with full context and return text response.

        Injects methodology + live state + playbook + risks into system prompt.
        If the agent has TOOLS defined, runs a tool-use loop (max 3 rounds).
        """
        if not self.enabled or not self._client:
            return ""

        # Build full system context
        context = _build_system_context()
        if context:
            full_system = (
                f"{context}\n\n"
                f"---\n\n"
                f"# Your Role\n\n"
                f"{system_prompt}"
            )
        else:
            full_system = system_prompt

        try:
            # Simple path: no tools
            if not self.TOOLS:
                response = self._client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=full_system,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text

            # Tool-use path: multi-turn loop (max 3 tool calls)
            messages = [{"role": "user", "content": user_message}]

            for _ in range(3):
                response = self._client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=full_system,
                    messages=messages,
                    tools=self.TOOLS,
                )

                # Check if we got a final text response
                if response.stop_reason == "end_turn":
                    text_parts = [b.text for b in response.content if b.type == "text"]
                    return "\n".join(text_parts)

                # Process tool calls
                tool_results = []
                text_parts = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        result = self._handle_tool_call(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                if not tool_results:
                    # No tool calls and no end_turn — return what we have
                    return "\n".join(text_parts) if text_parts else ""

                # Add assistant response and tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            # Exhausted tool rounds — extract final text
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts) if text_parts else ""

        except Exception as exc:
            logger.error("Agent '%s' API call failed: %s", self.AGENT_NAME, exc)
            return ""

    def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result string."""
        if tool_name == "web_search":
            query = tool_input.get("query", "")
            logger.info("Agent '%s' searching: %s", self.AGENT_NAME, query)
            return _execute_web_search(query)

        return f"Unknown tool: {tool_name}"

    def _format_pick_data(self, pick: dict) -> str:
        """Format a pick dict into a concise text summary for the agent."""
        ticker = pick.get("ticker", "?")
        direction = pick.get("direction", "?")
        strike = pick.get("strike")
        premium = pick.get("premium")
        confidence = pick.get("confidence", pick.get("direction_confidence", 0))
        score = pick.get("composite_score", 0)
        current = pick.get("current_price")
        breakeven = pick.get("breakeven")
        be_move = pick.get("breakeven_move_pct")
        delta = pick.get("estimated_delta")
        expiry = pick.get("expiry")

        # Ensemble info
        ensemble = pick.get("ensemble", {})
        consensus = ensemble.get("consensus", "unknown")

        # Pattern info
        pattern = pick.get("pattern", {})
        pattern_wr = pattern.get("pattern_win_rate")

        # Scores
        scores = pick.get("scores", {})
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        # Fundamental / catalyst signals — surface so agents can validate or
        # invalidate the mechanical thesis on qualitative grounds.
        sentiment = pick.get("sentiment", {}) or {}
        analyst_rev = sentiment.get("analyst_revision", {}) or {}
        sec_8k = sentiment.get("sec_8k", {}) or {}
        social = sentiment.get("social", {}) or {}
        insider = pick.get("insider", {}) or {}
        flow = pick.get("flow", {}) or {}

        catalyst_bits = []
        arev = analyst_rev.get("direction")
        if arev in ("upgrade", "downgrade"):
            catalyst_bits.append(f"analyst {arev}s (Δ={analyst_rev.get('delta', 0):+d})")
        if sec_8k.get("has_recent_8k"):
            catalyst_bits.append(f"{sec_8k.get('filing_count', 1)} recent 8-K")
        ins_signal = insider.get("insider_signal")
        if ins_signal and ins_signal != "neutral":
            catalyst_bits.append(f"insider {ins_signal}")
        if flow.get("unusual_volume"):
            catalyst_bits.append("unusual options flow")

        social_bits = []
        if social:
            flow_consensus = social.get("flow_consensus", "neutral")
            flow_conviction = social.get("flow_conviction", 0)
            if flow_consensus != "neutral" and flow_conviction > 0.3:
                social_bits.append(f"social flow {flow_consensus} ({flow_conviction:.0%} conviction)")
            catalysts = social.get("catalysts", [])
            if catalysts:
                social_bits.append(f"catalysts: {', '.join(catalysts[:2])}")
            risks = social.get("risks", [])
            if risks:
                social_bits.append(f"risks: {', '.join(risks[:2])}")

        lines = [
            f"**{ticker}** — {direction.upper()} ${strike:,.2f}" if strike else f"**{ticker}** — {direction.upper()}",
            f"Price: ${current:,.2f}" if current else "",
            f"Premium: ${premium:.2f} | Delta: {delta:.2f}" if premium and delta else "",
            f"Breakeven: ${breakeven:,.2f} ({be_move:.1f}% move)" if breakeven and be_move else "",
            f"Expiry: {expiry}" if expiry else "",
            f"Score: {score:.1f} | Confidence: {confidence:.0%} | Consensus: {consensus}",
            f"Top signals: {', '.join(f'{k}={v:.0f}' for k,v in top_scores)}" if top_scores else "",
            f"Pattern win rate: {pattern_wr:.0%} ({pattern.get('pattern_observations', 0)} obs)" if pattern_wr else "",
            f"Catalysts: {' | '.join(catalyst_bits)}" if catalyst_bits else "",
            f"Social: {' | '.join(social_bits)}" if social_bits else "",
        ]
        return "\n".join(l for l in lines if l)

    def _format_market_context(self, market_summary: dict) -> str:
        """Format market summary into concise text for the agent."""
        vix = market_summary.get("vix", {})
        regime = market_summary.get("vix_regime", {})
        breadth = market_summary.get("breadth", {})
        credit = market_summary.get("credit_spread", {})
        cot = market_summary.get("cot_positioning", {})
        macro = market_summary.get("macro_surprise", {})
        holding = market_summary.get("holding_window", {})
        cross = market_summary.get("cross_asset", {})

        lines = [
            f"VIX: {vix.get('current', '?')} ({regime.get('regime', '?')})",
            f"Breadth: {breadth.get('breadth_signal', '?')}",
            f"Credit: {credit.get('credit_state', '?')} (HY OAS {credit.get('hy_oas', '?')}%)",
            f"COT: {cot.get('signal', '?')} (percentile {cot.get('percentile', '?')})",
            f"Macro surprise: {macro.get('signal', '?')} (score {macro.get('surprise_score', '?')})",
            f"Holding window risk: {holding.get('risk_level', '?')}",
            f"Cross-asset: {cross.get('composite', '?')} (headwinds={cross.get('headwinds', '?')}, tailwinds={cross.get('tailwinds', '?')})",
        ]

        # Include saved market narrative if available (from Wednesday scan)
        narrative_path = config.candidates_dir
        for days_back in range(7):
            from datetime import datetime as _dt, timedelta
            date_str = (_dt.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            np = narrative_path / date_str / "market_narrative.md"
            if np.exists():
                try:
                    text = np.read_text().strip()
                    # Strip the markdown header
                    if text.startswith("#"):
                        text = "\n".join(text.split("\n")[2:]).strip()
                    if text:
                        lines.append(f"Market narrative ({date_str}): {text[:500]}")
                except Exception:
                    pass
                break

        return "\n".join(lines)
