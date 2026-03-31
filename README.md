# 0DTE Options Analysis Pipeline

An experimental zero-days-to-expiration options analysis system that combines automated data collection, macro regime awareness, and empirically calibrated scoring to identify weekly 0DTE opportunities.

**This is a research experiment, not investment advice.**

---

## How It Works

The system runs a weekly pipeline across a dynamic universe of 103+ liquid, weekly-options-eligible tickers spanning 14 sectors — plus dynamically discovered tickers showing unusual options volume, upcoming earnings, or news buzz. Mechanical data collection is automated via GitHub Actions. Intelligent analysis and judgment are provided by Claude Code agents run manually.

### Pipeline Overview

```
Wed 8AM ET          Thu 8AM ET          Fri 8AM ET          Mon 8AM ET          Weekend
    |                   |                   |                   |                   |
[Backtest →         [20+10 candidates]  [20 candidates]     [Grade picks]       [Review results]
 103 core +            |                   |                   |                   |
 dynamic tickers]   Delta-aware        Regime gate →       Fetch closing       Reflect &
    |               Re-evaluation      Score &             prices, calc P&L    Learn
  Scan &               |               Pick Top 3              |                   |
  Narrow            [Re-ranked 20]         |              [Scorecard →        [Update playbook]
    |                                  [3 picks + entry/    Discord]
[20 + 10 bench]                         exit rules →
                                        Discord]
```

### Wednesday — Backtest + Broad Scan (103+ → 20+10)

**Pre-scan backtest**: Before scanning, the system runs a directional backtest (`python3 main.py backtest`) across the full 103-ticker universe using 52 weeks of historical data enriched with FRED macro data and Finnhub earnings calendar. This validates signal weights and flags any signals that have drifted.

The pipeline then starts with the 103-ticker core universe, plus dynamically discovered tickers showing:
- **Unusual options volume** — tickers outside the core with >5,000 contracts traded
- **Earnings this week** — upcoming earnings juice IV and create temporary liquidity
- **News buzz** — 5+ Finnhub articles in 48 hours indicates a catalyst

The expanded universe is scanned with six data passes:

- **Technical scan**: RSI, MACD, Bollinger Bands, ATR, volume ratio, SMA 20/50, intraday gap/VWAP
- **Options chain analysis**: P/C ratios, BSM-derived ATM IV, IV/RV ratio, straddle-based implied move, max pain, OI distribution, ATM premium tracking
- **Sentiment analysis**: Finnhub company news + VADER sentiment scoring
- **Finviz scan**: pre-market movers, analyst ratings, short float, relative volume
- **SEC EDGAR scan**: insider buy/sell activity from Form 4 filings (30-day lookback)
- **Flow scan**: unusual options volume detection with direction bias across all pipeline stages

Candidates are ranked and narrowed to the top 20 + 10 bench alternates.

### Thursday — Delta-Aware Re-Evaluation (30 → 20)

The 20 candidates plus 10 bench tickers get fresh technical and options data. Thursday's stage performs a delta-aware comparison: how has each setup evolved since Wednesday? Candidates that improved (e.g., RSI moved further into extreme territory, volume confirmed direction) get re-ranked above those that deteriorated. The top 20 carry forward to Friday.

### Friday — Regime Gate + Final Picks (20 → 3)

Fresh morning data is collected (including Finviz pre-market scrape, fresh options chains, and flow scan), then:

**1. Macro regime gate** — Before scoring, the system checks whether the current environment gives the model a statistical edge using three FRED-derived indicators:

| Indicator | Source | Best Regime | Accuracy | Worst Regime | Accuracy |
|-----------|--------|-------------|----------|--------------|----------|
| Credit spread | FRED HY OAS | Normal (3-4.5%) | 57.1% | Tight (<3%) | 50.4% |
| Financial conditions | FRED NFCI | Normal (0 to -0.5) | 59.4% | Loose (<-0.5) | 50.8% |
| VIX regime | yfinance | High (25-35) | 72.3% | Normal (15-20) | 50.5% |

If the environment is complacent (normal VIX + tight credit + loose NFCI), all confidence scores are multiplied by 0.5x and the report flags "REGIME GATE: reduce position sizing or skip."

**2. Eight-factor scoring** with regime-adaptive weights:

| Signal | Base Weight | What It Captures |
|--------|-------------|------------------|
| Technical indicators | 18% | RSI, MACD, Bollinger, ATR, gap (regime-aware), VWAP |
| Options data | 18% | BSM IV, IV/RV ratio, P/C ratio, max pain, bid-ask spread quality |
| Expected move (ATR vs straddle) | 18% | Are options cheap relative to actual movement? |
| Market regime (VIX) | 12% | Does the vol environment favour this direction? |
| Sentiment | 12% | News sentiment via Finnhub + VADER |
| Finviz data | 8% | Pre-market moves, analyst ratings, relative volume, short float |
| Unusual flow | 8% | Fresh positioning signals (volume/OI ratio) |
| SEC EDGAR insider trades | 6% | Insider buy/sell clusters from Form 4 filings |

**3. Direction determination** with calibrated confidence:

- Each signal casts a weighted vote for bullish or bearish
- **Gap signal is regime-aware**: weighted 0.3x in low VIX (unreliable — backtest showed -0.080 correlation in calm markets) up to 1.2x in high VIX (reliable momentum signal, +0.088 correlation)
- **Confidence is empirically calibrated**: raw vote ratios are mapped through a calibration curve derived from the 52-week backtest. This fixes an inverted confidence problem where the model's highest raw confidence produced its lowest accuracy (52.4%)

**4. Strike selection** (expected-move-aware):

- Targets ~0.40-0.45 delta (near the money)
- Penalises strikes further than 0.7x ATR from current price (prevents deep OTM picks)
- Computes breakeven price and required move percentage
- Compares breakeven move to expected daily move (ATR) — if breakeven > expected, the trade is flagged as unfavorable risk/reward

**5. Entry/exit guidance** included with every pick:

- **Entry**: Wait 15-30 min after open, limit order at mid price, don't chase
- **Profit target**: 50% gain on premium
- **Stop loss**: 40% loss on premium
- **Time stop**: Close by 12:00 ET if neither target hit — theta accelerates after noon on 0DTE
- **Position sizing**: Max 1-2% of account per trade; halve if regime gate flags reduced edge

**6. Discord delivery**: Two embeds — picks with full execution data (strike, premium, delta, breakeven, expected move, regime gate status) plus a standing execution guide with entry/exit best practices.

### Options Chain Analysis

The options scanner computes quant-grade metrics rather than relying on vendor-provided IV:

- **BSM IV solver** — Implied volatility computed from option mid-prices via bisection method (no scipy dependency). Falls back to yfinance only when BSM fails.
- **Realized volatility** — 20-day close-to-close annualized vol from 1-month price history
- **IV/RV ratio** — Directly measures if options are cheap (ratio < 1.0) or expensive (ratio > 1.5) relative to actual stock movement. Replaces the broken iv_rank metric.
- **Straddle-based implied move** — ATM straddle mid-price as percentage of stock price. More reliable than IV-derived move for near-expiry options.
- **ATM premium tracking** — Call/put mid-prices and bid-ask spread percentages at the money

### VIX Regime Awareness

Strategy adapts to volatility environment — both in direction bias and in which signals carry more weight:

| VIX Level | Regime | Weight Shifts | Backtest Accuracy |
|-----------|--------|---------------|-------------------|
| < 15 | Low | Technicals +30%, sentiment +30%, options -10% | 50.7% |
| 15-20 | Normal | Base weights (no shift) | 50.5% |
| 20-25 | Elevated | Options +20%, flow +30%, technicals -10% | 57.1% |
| 25-35 | High | Options +40%, expected move +40%, technicals -30% | 72.3% |
| > 35 | Extreme | Options +50%, expected move +50%, technicals -50% | 69.9% |

---

## Directional Backtest

The system includes a weekly directional backtester that validates signal weights against historical data before each Wednesday scan.

### What It Tests

**Technical signals** (from price history):
- RSI (best individual signal: 54.7% accuracy, r=+0.081)
- MACD histogram (52.0%, r=+0.047)
- Price vs SMA20 (52.8%, r=+0.058)
- Gap % with regime-aware weighting (49.6% raw, r=+0.088 over 52 weeks)
- Volume ratio (54.1%, r=+0.064)
- Bollinger Band position (46.2%, r=+0.006 — weakest signal)

**Contextual features** (from FRED + Finnhub):
- Earnings proximity — classifies gaps as earnings-driven vs non-earnings (Finnhub earnings calendar)
- High-impact macro events — flags FOMC, CPI, NFP, etc. in the Wed-Fri window
- Yield curve state — 10Y-2Y Treasury spread from FRED (inverted/flat/normal/steep)
- Credit spread level — ICE BofA High Yield OAS from FRED (tight/normal/wide/stressed)
- Financial conditions — Chicago Fed NFCI from FRED (loose/normal/tightening/tight)

### Key Findings (52-week, 5,150 predictions)

- **Overall accuracy**: 53.4% (+3.4% edge over coin flip)
- **Regime is the edge**: 72.3% in high VIX vs 50.5% in normal — the model's value is knowing *when* to trade
- **Credit spreads matter**: 57.1% when normal vs 50.4% when tight
- **Financial conditions matter**: 59.4% when normal vs 50.8% when loose
- **Gap signal is regime-dependent**: correlation flips from -0.080 (12-week) to +0.088 (52-week)
- **BB is dead weight**: 46.2% accuracy, r=+0.006

### Running the Backtest

```bash
python3 main.py backtest              # 52-week lookback (default)
python3 main.py backtest --weeks 12   # Short-term window
```

Results saved to `data/performance/backtest_{date}.json`.

---

## My Weekly Involvement (~11 minutes)

The mechanical pipeline runs automatically. My role is running four Claude Code agents that provide the intelligent judgment layer:

| Day | Agent | What I Do | Time |
|-----|-------|-----------|------|
| **Wednesday** | `zero-dte-context` | Run after the Action finishes. Agent searches for macro narrative, identifies key market themes, checks economic calendar. Writes `market_context.md`. | ~2 min |
| **Thursday** | `zero-dte-catalyst` | Run after the Action finishes. Agent searches for stock-specific catalysts (analyst actions, earnings, FDA, insider trades) across the 20 candidates. Writes `catalyst_report.md`. | ~3 min |
| **Friday** | `zero-dte-friday` | Run after the Action finishes, before market open (9:30 AM ET). Agent reads the playbook, market context, and catalyst report, reviews the pipeline's 3 picks, applies judgment, can override or swap picks, sends final version to Discord. | ~3 min |
| **Weekend** | `zero-dte-reflect` | Run anytime. Agent checks actual outcomes via intraday data, scores each pick (WIN/PARTIAL/LOSS), analyzes which signals were predictive, and updates the playbook with lessons learned. | ~3 min |

---

## Performance Tracking

Every pick is tracked end-to-end: what we recommended, what actually happened, and whether it made money.

### Monday Scorecard (Automatic)

Each Monday at 8 AM ET, the GitHub Action automatically grades last Friday's picks:

1. Fetches the actual closing price for each ticker on expiry day via yfinance
2. Calculates the intrinsic value at expiry
3. Computes P&L per contract: (intrinsic - entry premium) x 100
4. Labels each pick: **WIN** (profitable), **PARTIAL** (ITM but didn't cover premium), **LOSS** (OTM)
5. Sends a scorecard to Discord and commits updated results to the repo

### SQLite Database

All picks and outcomes are stored in a local SQLite database (`data/performance/zero_dte.db`) with three tables:

- **picks** — every individual pick with entry data and graded outcome
- **weekly_results** — aggregated weekly stats (cost, return, P&L, win rate)
- **market_snapshots** — VIX level, regime, credit spread, NFCI at time of picks

---

## Scoring Roadmap

### Implemented
- **Regime-adaptive weights** — VIX regime dynamically shifts which signals carry more weight
- **8-factor composite scoring** — technical, options, sentiment, flow, market regime, expected move, Finviz, insider
- **Intraday signals** — pre-market gap, previous-day VWAP, intraday ATR
- **BSM IV solver** — Proper implied volatility from option mid-prices, replacing broken vendor IV
- **IV/RV ratio** — Options mispricing metric replacing the meaningless cross-strike IV rank
- **Straddle-based implied move** — Direct market pricing of expected movement
- **Directional backtest** — 52-week historical validation with FRED macro data and Finnhub earnings calendar
- **Regime gate** — Macro environment assessment (VIX + credit spreads + NFCI) that flags complacent environments
- **Confidence calibration** — Empirical calibration curve fixing inverted confidence (high raw confidence = low accuracy)
- **Context-aware gap signal** — Gap weight varies 0.3x-1.2x by VIX regime based on backtest evidence
- **Expected-move-aware strike selection** — OTM penalty, breakeven vs ATR comparison
- **Entry/exit execution guidance** — Profit target, stop loss, time stop, entry timing rules in every pick

### Planned (after 4-6 weeks of scorecard data, ~50+ graded picks)
- **Historical options backtest** — With Massive.com/Polygon.io data, backtest the 8 options-derived signals that currently can't be validated historically (P/C ratio, IV, flow, max pain, etc.)
- **Gradient-boosted model** — Train XGBoost/LightGBM on scored picks as features, win/loss as target
- **Ticker-class learning** — Per-sector weight overrides (TSLA responds to sentiment, JPM to macro, NVDA to flow)
- **Kelly criterion position sizing** — Size by edge/confidence instead of equal weight
- **Multi-leg strategies** — Graduate from single-leg directional plays to vertical spreads based on scorecard failure modes

---

## The Playbook

The system maintains an evolving `playbook.md` that serves as institutional memory. After each week's reflection, it gets updated with:

- Which signals were reliable vs misleading
- Patterns discovered (e.g., "VIX above 25 + RSI oversold = strong bounce signal")
- Mistakes to avoid
- Regime-specific observations

The Friday analyst agent reads this playbook before reviewing picks, so the system compounds its learning week over week.

---

## Data Sources

| Source | Data | Endpoint | Used In |
|--------|------|----------|---------|
| **yfinance** | Price history, options chains, VIX | Open source | All stages, backtest |
| **Finnhub** (free tier) | Company news, earnings calendar | `/company-news`, `/calendar/earnings` | Sentiment, backtest gap classification |
| **FRED** | Treasury yields, HY OAS credit spread, NFCI financial conditions | Federal Reserve API | Market scanner, regime gate, backtest |
| **VADER** | Sentiment scoring on news headlines | Local NLP library | Sentiment scanner |
| **Finviz** | Pre-market moves, analyst ratings, short float, insider snapshot | Web scraping (free) | Finviz scanner, scoring |
| **SEC EDGAR** | Insider buy/sell activity (Form 4 filings) | `data.sec.gov` API (free) | Insider scanner |

---

## Curated Universe (103 Tickers)

Organized across 14 sectors, selected for liquid weekly options availability:

**Indices/ETFs**: SPY, QQQ, IWM, DIA, XLF, XLE, XLK, GLD, SLV, TLT, HYG, EEM, ARKK
**Mega Cap Tech**: AAPL, MSFT, GOOG, GOOGL, AMZN, META, NVDA, TSLA
**Semiconductors**: AMD, INTC, MU, QCOM, AVGO, MRVL, ON, SMCI
**Software/Cloud**: CRM, SNOW, PLTR, NET, DDOG, ZS, CRWD, PANW
**Fintech/Payments**: XYZ, PYPL, COIN, HOOD, SOFI, V, MA
**Banks/Finance**: JPM, GS, MS, BAC, WFC, C, SCHW
**Energy**: XOM, CVX, OXY, SLB, HAL, DVN, MPC
**Retail/Consumer**: WMT, COST, TGT, HD, LOW, NKE, SBUX, MCD
**Healthcare/Biotech**: UNH, JNJ, PFE, MRNA, ABBV, LLY, AMGN
**Industrials**: BA, CAT, DE, GE, RTX, LMT, UPS, FDX
**Media/Entertainment**: NFLX, DIS, CMCSA, WBD, PSKY, ROKU
**Travel/Leisure**: DAL, UAL, LUV, AAL, ABNB, MAR, WYNN
**Auto/EV**: F, GM, RIVN, LCID, NIO, LI
**Telecom**: T, VZ, TMUS

---

## Tech Stack

- **Python 3.11** — pipeline logic
- **GitHub Actions** — automated Wed/Thu/Fri data collection + Monday scorecard (free tier)
- **Claude Code agents** — intelligent judgment layer (market context, catalyst scanning, pick review, reflection)
- **Discord webhooks** — pick delivery with execution guidance + weekly scorecard results
- **SQLite** — persistent backend for all picks, outcomes, and performance analytics
- **FRED API** — macro regime data (credit spreads, financial conditions, Treasury yields)
- **Finnhub API** — earnings calendar, economic events, company news
- **macOS launchd** — optional local scheduling (backup)

---

## Project Structure

```
zero-dte/
├── main.py                  # CLI entry point
├── config.py                # Environment config
├── requirements.txt
├── .github/workflows/
│   └── zero-dte.yml         # Automated pipeline schedule
├── universe/
│   ├── robinhood.py         # Curated 103-ticker core universe
│   └── dynamic.py           # Dynamic expansion (unusual volume, earnings, news)
├── scanners/
│   ├── market.py            # VIX, breadth, yields, credit spread, NFCI, economic calendar
│   ├── technical.py         # RSI, MACD, Bollinger, ATR, volume
│   ├── options.py           # BSM IV solver, IV/RV ratio, straddle implied move, optimal strikes
│   ├── sentiment.py         # News sentiment via Finnhub + VADER
│   ├── flow.py              # Unusual options activity detection
│   ├── finviz.py            # Finviz web scraping (pre-market, analysts, short float)
│   └── edgar.py             # SEC EDGAR Form 4 insider trading data
├── analysis/
│   ├── scoring.py           # 8-factor scoring, regime gate, confidence calibration, direction
│   ├── narrowing.py         # Stage-based filtering + diversity
│   └── backtest.py          # 52-week directional backtest with FRED/Finnhub enrichment
├── pipeline/
│   ├── stages.py            # Wed/Thu/Fri stage orchestration with regime gate
│   └── runner.py            # Top-level dispatcher
├── notifications/
│   └── discord.py           # Discord picks + execution guide + scorecard delivery
├── tracking/
│   ├── tracker.py           # Pick recording
│   ├── reflector.py         # Weekly outcome analysis
│   ├── scorecard.py         # P&L grading + Discord scorecard
│   └── database.py          # SQLite backend for all performance data
├── data/
│   ├── candidates/          # Daily scan results (JSON)
│   ├── reports/             # Final pick reports
│   └── performance/         # Backtest results, playbook, scorecard, SQLite DB
└── .claude/agents/
    ├── zero-dte-context.md  # Wednesday market narrative agent
    ├── zero-dte-catalyst.md # Thursday catalyst scanner agent
    ├── zero-dte-friday.md   # Friday analyst agent
    └── zero-dte-reflect.md  # Weekend reflection agent
```

---

## Setup

1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in API keys
4. Add API keys as GitHub Secrets (`FINNHUB_API_KEY`, `FRED_API_KEY`, `DISCORD_WEBHOOK_URL`)
5. The pipeline runs automatically on schedule via GitHub Actions

### Manual Run

```bash
python3 main.py run              # Auto-detect day, run appropriate stage
python3 main.py stage --stage wednesday   # Run specific stage
python3 main.py picks            # Show most recent picks
python3 main.py status           # Show pipeline status
python3 main.py reflect          # Run weekly reflection
python3 main.py backtest         # Run 52-week directional backtest
python3 main.py backtest --weeks 12  # Short-term backtest
python3 main.py scorecard --week 2026-03-22  # Grade a week's picks
python3 main.py scorecard        # Show all-time scorecard
```
