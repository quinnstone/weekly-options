# Weekly Options Analysis Pipeline

A systematic weekly options trading system (Monday entry, Friday expiry) combining automated data collection, macro regime awareness, empirically calibrated scoring, and 8 Claude-powered AI agents that provide qualitative reasoning at every decision point.

**This is a research experiment, not investment advice.**

---

## How It Works

The system runs a 3-stage weekly pipeline (Wed scan, Fri refresh, Mon picks) across a dynamic universe of 103+ liquid tickers. Nine cron jobs handle intraday monitoring from Monday through Saturday. Eight AI agents embedded in the pipeline provide qualitative judgment — market narratives, pre-trade analysis, position monitoring, portfolio reasoning, and post-mortem feedback. We run 3 high-conviction picks per week (not 5) — quality over quantity, less leverage, higher confidence bar.

### Weekly Flow

```
Wed 7AM ET          Fri 4PM ET           Mon 8AM ET          Mon 10AM         Mon 1PM
    |                   |                    |                   |                |
[Broad Scan]       [Delta-Aware          [Final Picks]     [Entry Confirm]   [Midday
 103+ tickers       Refresh]              Score & rank       Gap/delta        Monitor]
 + Market          35 → 20               Portfolio           check,           P&L,
 Narrative         candidates             Reasoner →         Pre-Trade        targets]
 Agent]                                    best 3             Analyst +
    |                                      + Theses           web search]
[25 + 10 bench]                            → Discord]

Tue 3PM    Wed 3PM    Thu 10AM/3PM    Fri 10AM        Fri 1:30PM     Sat 10AM
   |          |           |              |                |              |
[Position  [Position  [Position      [Position       [FINAL EXIT    [Scorecard +
 Monitor    Monitor    Monitor        Monitor         Close all      Post-Mortem +
 ROUTINE]   ROUTINE]   ELEVATED]      CRITICAL]       by 2 PM ET]    Deep Reflect]
```

---

## Three-Tier Scoring Architecture

### 10-Factor Weighted Model

| Tier | Weight | Factors |
|------|--------|---------|
| **Direction** (T1) | 60% | momentum (0.20), mean_reversion (0.15), trend_persistence (0.15), regime_bias (0.10) |
| **Edge Quality** (T2) | 25% | iv_mispricing (0.10), flow_conviction (0.08), event_risk (0.07) |
| **Execution** (T3) | 15% | liquidity (0.05), strike_efficiency (0.05), theta_cost (0.05) |

Weights are regime-adaptive — they shift based on VIX tier. In high VIX (25-30), momentum drops to 0.7x while mean_reversion rises to 1.4x. Weights are also self-learning: the pattern library and backtest validation gate can update them weekly (see Learning Loop below).

### Three-Model Ensemble

The composite score is NOT a single model. Three independent models vote, and agreement/disagreement is measured:

| Model | Weight | Focus |
|-------|--------|-------|
| Linear factor | 50% | Full 10-factor weighted sum |
| Momentum-only | 25% | momentum + trend_persistence + regime_bias + flow |
| Mean-reversion + value | 25% | mean_reversion + iv_mispricing + event_risk + theta_cost + liquidity |

- Model std dev < 5: +5% bonus (strong consensus)
- Model std dev > 15: -8% penalty (disagreement)

### Confidence Calibration

Raw model confidence is empirically calibrated against observed win rates. Static calibration penalizes extremes (0.85-1.0 raw confidence maps to only 0.42 calibrated — extreme unanimity is suspicious). After 10+ observations per confidence bucket, live pattern data overrides static calibration (60% live / 40% static blend).

### Regime-Adaptive Multipliers

| VIX Regime | Range | Key Shifts | Backtest Accuracy |
|------------|-------|------------|-------------------|
| Low | 0-15 | momentum 1.3x, mean_reversion 0.8x | ~52-54% |
| Normal | 15-20 | Baseline (all 1.0x) | ~55-57% |
| Elevated | 20-25 | mean_reversion 1.2x, iv_mispricing 1.3x | ~58-60% |
| High | 25-30 | mean_reversion 1.4x, momentum 0.7x | ~56-58% |
| Extreme | 30+ | mean_reversion 1.5x, momentum 0.5x | ~56-58% |

---

## AI Agent System (8 Agents)

All agents receive a layered context hierarchy:
1. **METHODOLOGY.md** — Static scoring rules, thresholds, architecture
2. **CURRENT_STATE.md** — Auto-generated live weights, pattern stats, calibration, recent reflection
3. **TRADE_LOG.md** — Auto-generated rolling 8-week trade outcomes
4. **playbook.md** — Accumulated strategic knowledge
5. **known_risks.md** — Known system limitations

Context is refreshed on every pipeline run and after every learning update.

### Agent Summary

| Agent | When | Tool Use | Purpose |
|-------|------|----------|---------|
| **Market Narrative** | Wed scan | - | Synthesizes structured market data into coherent 2-3 paragraph narrative |
| **Earnings Analyst** | Wed scan | Web search | TRADE/CAUTION/AVOID for tickers with earnings in holding window |
| **Pre-Trade Analyst** | Mon 10AM confirm | Web search | GO/ADJUST/SKIP per pick with overnight news + thesis validation |
| **Portfolio Reasoner** | Mon picks | - | Reviews top 6, selects optimal 3 considering concentration + macro exposure |
| **Thesis Writer** | Mon picks | - | 2-3 sentence trading thesis per pick for Discord |
| **Position Monitor** | Tue-Fri | - | P&L tracking, stop/target alerts, charm decay warnings |
| **Post-Mortem** | Fri scorecard | - | WHY each pick won or lost (mechanism, not just P&L) |
| **Deep Reflection** | Sat | - | CIO weekly review, tactical adjustments, market outlook |

All agents run on **Claude Opus 4** — the strongest available model. At ~$50/month for ~40 calls/week, the cost is negligible relative to capital at risk in options trading.

### Tool Use (Web Search)

Pre-Trade Analyst and Earnings Analyst have Anthropic tool_use capability. They can search for overnight news, earnings results, and sector developments via:
- **yfinance news** — always available, no extra API key
- **Finnhub company news** — when FINNHUB_API_KEY is set

The tool-use loop runs up to 3 rounds per agent call.

### Estimated API Cost

- ~35-40 Opus calls/week across all 8 agents
- ~15K input tokens + ~1K output tokens per call (includes full methodology context)
- **Total: ~$12/week or ~$50/month** at Opus pricing ($15/M input, $75/M output)

---

## Pipeline Stages

### Wednesday — Broad Scan (103+ → 25)

1. Market summary (VIX, breadth, credit, COT, macro surprise, holding window)
2. **Market Narrative Agent** — synthesizes data into trading context
3. Core universe (103 tickers) + dynamic additions (unusual volume, earnings, news)
4. Six data passes: technical, options chain (BSM IV), sentiment, Finviz, EDGAR insider, flow
5. Narrow to 25 + 10 bench via weighted scan_rank
6. **Earnings Analyst Agent** — screens candidates with earnings in Mon-Fri window

### Friday — Delta-Aware Refresh (35 → 20)

1. Load Wednesday's 25 + 10 bench
2. Snapshot Wednesday data, re-run all scans
3. Delta-aware re-ranking: how did each setup evolve? RSI trajectory, MACD change, IV premium movement
4. Promote/demote candidates based on setup evolution
5. Top 20 carry forward to Monday

### Monday — Final Picks (20 → 5)

1. Fresh market summary + technical + options scans
2. Regime gate: assess macro edge (VIX + credit + NFCI)
3. 10-signal direction determination + 3-model ensemble scoring
4. Mechanical diversity filter (sector, correlation dedup, direction balance)
5. **Portfolio Reasoner Agent** — reviews top 6, selects optimal 3 for concentration risk
6. Strike selection (0.35 delta target, BSM Greeks, IV term structure)
7. **Thesis Writer Agent** — generates trading thesis per pick
8. Discord delivery with full execution data + theses
9. Save picks for tracking

### Monday 10AM — Entry Confirmation

1. Fetch fresh quotes 30 min after market open
2. Gap check (>2% = SKIP), delta drift (>0.15 = ADJUST), premium change
3. **Pre-Trade Analyst Agent** — searches overnight news, validates thesis, GO/ADJUST/SKIP
4. Discord confirmation message

### Tue-Fri — Position Monitoring

Runs at 9 scheduled intervals with day-appropriate urgency:
- **ROUTINE** (Mon PM, Tue, Wed): Brief status, flag outliers
- **ELEVATED** (Thu): Theta accelerating, flag weak positions for exit
- **CRITICAL** (Fri): Everything closes by 2 PM ET

Time-decay-aware daily targets: 40% (Mon) → 35% (Tue) → 25% (Wed) → 15% (Thu) → 10% (Fri)

### Friday 1:30PM — Final Exit

Mandatory close reminder. All positions get CLOSE status with explicit exit instructions. Market orders if needed.

### Friday 4PM — Scorecard

1. Grade each pick: WIN/PARTIAL/LOSS with P&L
2. **Post-Mortem Agent** — explains WHY each pick won or lost
3. Update scorecard, database, pattern library
4. Discord scorecard notification

### Saturday — Reflection

1. Mechanical reflection: signal correlations, win/loss analysis, weight adjustments
2. **Deep Reflection Agent** — CIO weekly review through 6 failure modes:
   - Direction wrong (which signals misled?)
   - Direction right, theta killed it (charm decay?)
   - Regime shift mid-week
   - IV mispriced
   - Event dominated
   - Pattern library insight
3. Propose tactical adjustments (must pass validation gate)
4. Apply learnings (if backtest validates)
5. Refresh agent context documents

---

## Greeks & IV Methodology

### BSM Greeks Computed
- **Delta** (0.35 target), **Gamma**, **Theta** (~4%/day for weeklies)
- **Vega** (IV sensitivity)
- **Charm** (delta decay/day — actively scored, critical for weekly holds)
- **Vanna** (delta sensitivity to IV — display-only, pending live validation per ATLAS test-and-revert discipline)

### IV Framework
- **BSM IV solver** — bisection method, no scipy dependency
- **IV/RV ratio** — options cheap (<0.9) or expensive (>1.3) vs realized vol
- **IV term structure** — weekly vs monthly IV comparison (contango/backwardation)
- **Earnings IV crush** — 35-50% estimated crush flagged for holding window

---

## Risk Management

### Stop Loss Rules
| Rule | Threshold | Notes |
|------|-----------|-------|
| Hard stop | -50% option value | Wider than 0DTE (weekly has recovery time) |
| Delta stop | Delta < 0.10 | Option is nearly worthless |
| Gap stop | >2% against thesis | Skip entry at Monday confirmation |

### Position Sizing (Kelly Criterion)
- Half-Kelly formula, capped at 3% per position
- Confidence clipped to 0.30-0.70 range
- Regime gate: multiply by 0.5x if macro_edge multiplier < 0.50

### Portfolio Construction
- Max 2 picks from same sector
- 20-day rolling return correlation dedup (0.75 threshold)
- Direction balance check (avoid all-call/all-put unless high confidence)
- **Portfolio Reasoner Agent** reviews for macro concentration risk

---

## Learning Loop

### Pattern Library
Every trade outcome is recorded with a 6-component pattern key:
`{regime}|{direction}|{dominant_signal}|{rsi_zone}|{trend_state}|{iv_state}`

After 5+ observations, patterns adjust composite scores by up to ±10 points. After 10+ observations, live confidence calibration overrides static defaults.

### Backtest Validation Gate
Weight changes proposed by the learning loop must pass:
1. Accuracy drop ≤ 2% (absolute)
2. Sharpe drop ≤ 0.1
3. At least one metric must improve

If validation fails, proposed weights are rolled back.

### Auto-Updating Agent Context
When weights change or patterns accumulate, `CURRENT_STATE.md` and `TRADE_LOG.md` are automatically regenerated. All agents see the latest system state on their next call.

---

## Data Sources

| Source | Data | Cost | Used In |
|--------|------|------|---------|
| **yfinance** | Price history, options chains, VIX, news | Free | All stages |
| **Finnhub** | Company news, earnings calendar, macro events | Free tier | Sentiment, backtest, agent web search |
| **FRED** | Treasury yields, HY OAS, NFCI | Free | Market scanner, regime gate |
| **CFTC COT** | Speculator positioning (S&P futures) | Free | Contrarian signal in scoring |
| **VADER** | Sentiment scoring on news headlines | Local NLP | Sentiment scanner |
| **Finviz** | Pre-market movers, analyst ratings, short float | Free (scrape) | Finviz scanner |
| **SEC EDGAR** | Insider Form 4 filings | Free | Insider scanner |
| **Anthropic API** | Claude Opus 4 for 8 AI agents | ~$50/mo | All agent analysis |
| **Tradier** | Real-time options chains, Greeks (dormant) | Free w/ account | Optional, activates if key set |

---

## Cron Schedule (GitHub Actions)

| Time (ET) | Day | Stage | Agent(s) |
|-----------|-----|-------|----------|
| 7:00 AM | Wednesday | Broad scan | Market Narrative, Earnings Analyst |
| 4:00 PM | Friday | Delta-aware refresh | — |
| 4:30 PM | Friday | Scorecard grading | Post-Mortem |
| 8:00 AM | Monday | Final picks | Portfolio Reasoner, Thesis Writer |
| 10:00 AM | Monday | Entry confirmation | Pre-Trade Analyst (web search) |
| 1:00 PM | Monday | Midday monitor | Position Monitor (ROUTINE) |
| 3:00 PM | Tue/Wed | EOD check | Position Monitor (ROUTINE) |
| 10:00 AM | Thursday | Morning check | Position Monitor (ELEVATED) |
| 3:00 PM | Thursday | EOD check | Position Monitor (ELEVATED) |
| 10:00 AM | Friday | Morning check | Position Monitor (CRITICAL) |
| 1:30 PM | Friday | Final exit | Position Monitor (CRITICAL) |
| 10:00 AM | Saturday | Deep reflection | Deep Reflection |

---

## Project Structure

```
weekly-options/
├── main.py                      # CLI entry point
├── config.py                    # Environment config (API keys, paths)
├── requirements.txt
├── .github/workflows/
│   └── zero-dte.yml             # 9 cron jobs + workflow dispatch
├── universe/
│   ├── robinhood.py             # 103-ticker core universe
│   └── dynamic.py               # Dynamic expansion (unusual volume, earnings, news)
├── scanners/
│   ├── market.py                # VIX, breadth, yields, credit, COT, macro surprise, events
│   ├── technical.py             # RSI, MACD, Bollinger, ATR, ADX, momentum, SMA slopes
│   ├── options.py               # BSM IV/Greeks, IV term structure, earnings crush, strikes
│   ├── sentiment.py             # News sentiment (Finnhub + VADER)
│   ├── flow.py                  # Unusual options volume detection
│   ├── finviz.py                # Pre-market movers, analyst ratings, short float
│   ├── edgar.py                 # SEC Form 4 insider trades
│   └── tradier.py               # Tradier API client (dormant, activates with key)
├── analysis/
│   ├── scoring.py               # 10-factor scoring, 3-model ensemble, regime gate, calibration
│   ├── narrowing.py             # Stage filters, correlation dedup, Kelly sizing
│   ├── patterns.py              # Pattern library, confidence calibration, validation gate
│   └── backtest.py              # Directional backtester with BSM pricing
├── pipeline/
│   ├── stages.py                # All pipeline stages (Wed/Fri/Mon/confirm/monitor/exit)
│   └── runner.py                # Top-level dispatcher
├── agents/
│   ├── base.py                  # Base agent (context loading, tool use, Anthropic API)
│   ├── METHODOLOGY.md           # Static reference (scoring rules, thresholds, architecture)
│   ├── CURRENT_STATE.md         # Auto-generated live state (weights, patterns, calibration)
│   ├── TRADE_LOG.md             # Auto-generated rolling 8-week trade outcomes
│   ├── state_generator.py       # Generates CURRENT_STATE.md and TRADE_LOG.md
│   ├── market_narrative.py      # Market context synthesis (Wednesday)
│   ├── earnings_analyst.py      # Earnings risk assessment (Wednesday)
│   ├── pre_trade.py             # Entry validation with web search (Monday 10AM)
│   ├── portfolio_reasoner.py    # Portfolio construction reasoning (Monday)
│   ├── thesis_writer.py         # Trading thesis generation (Monday)
│   ├── position_monitor.py      # Intraday P&L monitoring (Tue-Fri)
│   ├── post_mortem.py           # Pick outcome analysis (Friday scorecard)
│   └── deep_reflection.py       # CIO weekly review (Saturday)
├── notifications/
│   └── discord.py               # Discord embeds (picks, monitor, scorecard, exit)
├── tracking/
│   ├── tracker.py               # Pick recording
│   ├── reflector.py             # Weekly reflection + learning loop
│   ├── scorecard.py             # P&L grading + post-mortem integration
│   └── database.py              # SQLite backend
└── data/
    ├── candidates/              # Daily scan results + market narratives
    ├── reports/                  # Final pick reports
    └── performance/             # Weights, patterns, calibration, backtest, scorecard, DB
```

---

## Setup

1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in API keys
4. Add secrets to GitHub Actions:
   - `FINNHUB_API_KEY` — free tier
   - `FRED_API_KEY` — free
   - `DISCORD_WEBHOOK_URL`
   - `ANTHROPIC_API_KEY` — for AI agents (~$10-15/month)
   - `TRADIER_API_KEY` — optional (requires brokerage account)

### Manual Run

```bash
python3 main.py run                          # Auto-detect day, run appropriate stage
python3 main.py stage --stage wednesday      # Run specific stage
python3 main.py stage --stage monday         # Monday picks
python3 main.py stage --stage confirm        # Monday entry confirmation
python3 main.py stage --stage monitor        # Position monitoring
python3 main.py stage --stage final_exit     # Friday mandatory exit
python3 main.py picks                        # Show most recent picks
python3 main.py status                       # Show pipeline status
python3 main.py reflect                      # Weekly reflection + deep reflection
python3 main.py backtest                     # 52-week directional backtest
python3 main.py backtest --weeks 12          # Short-term backtest
python3 main.py scorecard --week 2026-04-07  # Grade a specific week
python3 main.py scorecard                    # Show all-time scorecard
```
