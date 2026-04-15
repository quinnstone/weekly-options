# Zero DTE Playbook

This document is the agent's evolving knowledge base. It is read before every
Friday pick session and updated after every weekly reflection. Over time it
becomes the primary source of strategic intelligence for 0DTE selection.

---

## Scoring Weights (current)

| Signal | Weight | Notes |
|--------|--------|-------|
| Technical | 0.25 | RSI extremes, ATR%, volume ratio, Bollinger, MACD |
| Options | 0.25 | IV rank, P/C ratio, max pain divergence, volume |
| Sentiment | 0.20 | Finnhub news + VADER, StockTwits when available |
| Flow | 0.15 | Unusual options volume, vol/OI ratio |
| Market Regime | 0.15 | VIX-based directional bias |

## Regime Playbook

### High VIX (>25)
- Bias: puts
- Observations: (none yet — will populate after week 1 results)

### Normal VIX (15-25)
- Bias: neutral, follow technicals
- Observations: (none yet)

### Low VIX (<15)
- Bias: calls, mean-reversion plays
- Observations: (none yet)

## Economic Calendar Awareness

High-impact economic events are the single biggest driver of 0DTE outcomes on
release days. The system now checks the Finnhub economic calendar each week and
flags events that demand special attention.

### Key events and their 0DTE implications

| Event | Typical Impact | 0DTE Notes |
|-------|---------------|------------|
| FOMC Rate Decision | Extreme | 2:00 PM release creates massive directional move. Avoid entering before release unless hedged. Best 0DTE day of the month when timed correctly. |
| CPI / PPI | Very High | 8:30 AM pre-market release. Gap open is common. Favor straddles or wait for the first 30-min candle to settle before picking direction. |
| NFP (first Friday) | Very High | 8:30 AM release. Large gap + follow-through. If trading 0DTE on NFP Friday, expect 2-3x normal ATR. |
| PCE | High | Fed's preferred inflation gauge. Similar dynamics to CPI but slightly less volatile. |
| GDP | High | Quarterly release. Significant gap potential. |
| JOLTS | Medium-High | Labor market data. Can move markets but less dramatic than NFP. |
| Retail Sales | Medium | Consumer spending data. Moderate impact. |
| Jobless Claims | Medium | Every Thursday. Routine but can amplify other signals. |

### How the system uses economic events

- **Score boost (+10):** When a high-impact event falls on trading day (Friday),
  all candidate scores are boosted by 10 points. Big intraday moves create
  better 0DTE profit opportunities regardless of direction.
- **Confidence penalty (-0.05):** Direction confidence is reduced slightly
  because the outcome of economic releases is inherently unpredictable. The
  system acknowledges it cannot forecast the data.
- **Fallback list:** If the Finnhub API is unavailable, the system uses a
  hardcoded list of recurring high-impact events to maintain awareness.

### Strategy notes

- On FOMC days, consider waiting until after the 2:00 PM release to enter.
- On CPI/NFP days, the opening 30 minutes are extremely volatile. Let the
  market digest the number before committing.
- Straddles or strangles are safer than directional plays around data releases.
- If VIX is already elevated AND a high-impact event is scheduled, expect
  outsized moves. Reduce position size accordingly.

## Signal Reliability Log

| Signal | Weeks Tracked | Accuracy | Notes |
|--------|--------------|----------|-------|
| (populated after week 1) | | | |

## Patterns Discovered

(populated as the agent identifies recurring patterns)

## Mistakes to Avoid

(populated from weekly reflections)

## What Works

(populated from weekly reflections)

## Weekly Reflections

### Week 1 — 2026-03-20 (test run)
- **Market:** VIX 27.8 (high), broad selloff, SPY RSI 25
- **Picks:** ULTA $530P, GNRC $200P, URI $700P, STLD $165P, LUV $40P
- **Outcome:** (pending — record after market close)
- **Notes:** First run. All puts due to high VIX. Confidence flat at 0.50 across
  all picks — direction signals splitting (oversold = bullish contrarian vs.
  high VIX downtrend = bearish momentum). Sentiment data was thin due to
  Finnhub rate limiting. Picks were near-ATM strikes.
