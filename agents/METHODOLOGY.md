# Weekly Options Trading System — Methodology Reference

This document is the canonical reference for all AI agents in the pipeline.
When analyzing trades, generating briefs, or making recommendations, reason
from this methodology — not generic options knowledge.

---

## System Overview

**Strategy:** Directional weekly options (Monday entry, Friday expiry).
**Universe:** ~103 curated tickers + dynamic additions (unusual volume, earnings, news).
**Pipeline cadence:** Wed scan (100+ → 25) → Fri refresh (35 → 20) → Mon picks (20 → 3).
**Position sizing:** Half-Kelly criterion, capped at 3% per trade, regime-gated.
**Risk management:** 50% hard stop, delta-based stops, time-decay-aware daily targets.

---

## Scoring Architecture

### Three-Tier Weighted Model

| Tier | Weight | Purpose | Factors |
|------|--------|---------|---------|
| Direction (T1) | 60% | Is the thesis right? | momentum (0.20), mean_reversion (0.15), regime_bias (0.10), trend_persistence (0.15) |
| Edge Quality (T2) | 25% | Is the trade mispriced? | iv_mispricing (0.10), flow_conviction (0.08), event_risk (0.07) |
| Execution (T3) | 15% | Can we execute cleanly? | liquidity (0.05), strike_efficiency (0.05), theta_cost (0.05) |

### Three-Model Ensemble

The composite score is NOT a single model. Three independent models vote:

1. **Linear factor model** (50% weight): Full 10-factor weighted sum
2. **Momentum-only model** (25%): momentum(0.40) + trend_persistence(0.35) + regime_bias(0.15) + flow_conviction(0.10)
3. **Mean-reversion + value model** (25%): mean_reversion(0.35) + iv_mispricing(0.30) + event_risk(0.15) + theta_cost(0.10) + liquidity(0.10)

**Consensus mechanics:**
- Model std dev < 5: +5% bonus (strong agreement = higher conviction)
- Model std dev > 15: -8% penalty (disagreement = reduce confidence)

When reviewing a trade, check which model(s) drove the score. A pick where all
three models agree is fundamentally different from one where momentum loves it
but mean-reversion hates it.

### Regime-Adaptive Multipliers

Weights shift based on VIX regime. This is critical — a "good momentum score"
means different things in low-VIX vs high-VIX environments.

| VIX Regime | Range | Key Shifts |
|------------|-------|------------|
| Low | 0-15 | momentum 1.3x, mean_reversion 0.8x, theta_cost 1.2x |
| Normal | 15-20 | Baseline (all 1.0x) |
| Elevated | 20-25 | mean_reversion 1.2x, iv_mispricing 1.3x |
| High | 25-30 | mean_reversion 1.4x, momentum 0.7x |
| Extreme | 30+ | mean_reversion 1.5x, momentum 0.5x |

**Implication for agents:** In high-VIX, momentum signals are deweighted by 30%.
A pick that scores well on momentum in a high-VIX week got there despite the
headwind — that's either very strong momentum or a scoring anomaly to investigate.

---

## Direction Determination

Direction (call vs put) comes from a 10-signal voting system, NOT the composite
score. The score tells us *how good* the trade is; direction tells us *which way*.

**Signals and weights:**
1. Multi-day momentum (5d return dominant): up to 1.5 weight
2. Trend strength (ADX-classified): strong = 1.5, moderate = 0.8
3. RSI mean reversion: oversold (<30) bullish = 1.0, overbought (>70) bearish = 1.0
4. Price vs SMA20: >2% above = 0.8 bullish
5. MACD histogram: expanding = 0.5, crossover = 0.8
6. Put/Call ratio: >1.3 = 0.5 bearish, <0.7 = 0.5 bullish
7. Max pain pull: >1% divergence = 0.5
8. Analyst consensus: buy/sell = 0.4
9. Insider signal: bullish = 0.6, bearish = 0.3
10. VIX regime: >25 = 0.4 bearish, <14 = 0.2 bullish

**Confidence:** raw = |bullish_score - bearish_score| / total_weight
Calibrated via 60% live pattern data + 40% static table.

**Static calibration penalizes extremes:**
- 0.85-1.0 raw confidence → 0.42 calibrated (extreme unanimity = suspicious)
- 0.50-0.70 raw → 0.58 calibrated (sweet spot)

---

## Confidence Calibration

The pattern library tracks observed win rates per confidence bucket. After 10+
observations in a bucket, live calibration overrides static. This means our
confidence numbers are empirically grounded, not theoretical.

**Calibrated confidence < 0.25 = skip the trade** (hard gate at Monday picks).

---

## Pattern Library

Every trade outcome is recorded with a 6-component pattern key:
`{regime}|{direction}|{dominant_signal}|{rsi_zone}|{trend_state}|{iv_state}`

Example: `elevated|call|momentum|neutral|trending|cheap`

**Components:**
- regime: low/normal/elevated/high/extreme
- direction: call/put
- dominant: highest-scoring of (momentum, mean_reversion, trend_persistence)
- rsi_zone: oversold (<35), neutral (35-65), overbought (>65)
- trend_state: trending (ADX>25), choppy (ADX≤25)
- iv_state: cheap (IV/RV<0.9), fair (0.9-1.3), expensive (>1.3)

**Usage:** After 5+ observations of a pattern, its historical win rate adjusts
the composite score by up to ±10 points. This is how the system learns that
e.g. "elevated regime + call + momentum-driven + trending + cheap IV" wins 68%
of the time.

---

## Greeks & IV Methodology

### BSM Greeks We Compute
- **Delta:** Price sensitivity (target: 0.35 for entries)
- **Gamma:** Delta acceleration (bonus for winners: 1.10x return estimate)
- **Theta:** Time decay per calendar day, regime-adjusted in PositionMonitor:
    - VIX ≥ 25 (high): ~3.0%/day — vega cushion offsets decay
    - VIX 20-25: ~3.5%/day
    - VIX 15-20: ~4.0%/day — baseline
    - VIX < 15: ~4.8%/day — calm markets bleed faster
- **Vega:** IV sensitivity per 1% move
- **Charm:** Delta decay per day — CRITICAL for weekly holds. A 0.35 delta
  on Monday can be 0.20 by Wednesday from charm alone. Scored actively
  in theta_cost: -10 penalty if charm >2x between day 1 and day 3.
- **Vanna:** Delta sensitivity to IV changes — display-only pending live
  validation (ATLAS test-and-revert discipline). Will be promoted to
  scoring input after 4-6 weeks of data if predictive.

### IV Term Structure
Weekly IV vs Monthly IV comparison:
- Ratio < 0.85: Weekly significantly cheap (favorable entry)
- Ratio 0.85-0.95: Modestly cheap
- Ratio 1.05-1.15: Modestly expensive (event premium)
- Ratio > 1.15: Weekly expensive (avoid or size down)

This feeds into iv_mispricing scoring (±10 points) and theta_cost scoring (±8).

### IV/RV Ratio (Our Core IV Metric)
- IV/RV < 0.8: Excellent (score 80-100)
- IV/RV 0.8-1.0: Good (score 65-80)
- IV/RV 1.0-1.3: Fair (score ~45)
- IV/RV > 1.5: Expensive (score < 30)

We use IV/RV ratio, NOT IV percentile (which requires historical IV surface
data we don't have).

---

## Market Context Signals

### What Each Signal Means for Weekly Holds

**VIX Regime:** Determines weight multipliers (see above). Stable regime for
5+ days = higher confidence that signals hold through Friday.

**Breadth (SPY vs RSP):**
- broad_rally (RSP outperforming): Healthy, supports directional plays
- narrow_leadership (SPY outperforming): Fragile, favor large-cap names

**Credit Spread (HY OAS):**
- Tight (<3%): Complacent market, lower edge (backtest: 50.4% accuracy)
- Normal (3-4.5%): Best accuracy zone (backtest: 57.1%)
- Wide (>4.5%): Stress, favor defensive positions

**COT Positioning:** CFTC speculator positioning, contrarian signal.
- Extreme long → bearish contrarian: +2.5 for puts
- Extreme short → bullish contrarian: +2.5 for calls

**Macro Surprise:** Rolling 30-day beat/miss tracker.
- Beating: economy outperforming → +2.0 for calls
- Missing: economy disappointing → +2.0 for puts

**Holding Window Events:**
- High risk (≥2 events or FOMC): -20 score points, confidence multiplier -0.10
- Medium risk (1 event): -5 points
- FOMC weeks: additional -10 points

**Regime Persistence:**
- Stable (5+ days same regime): Higher confidence signals hold through expiry
- Transitioning: +0.10 penalty on confidence multiplier

---

## Risk Rules

### Daily Profit Targets (Time-Decay-Aware)
| Day | Target | Rationale |
|-----|--------|-----------|
| Monday | 40% | Full time value; hold for home run |
| Tuesday | 35% | Slight theta erosion |
| Wednesday | 25% | Midweek; take what you can |
| Thursday | 15% | Theta accelerating; take any meaningful profit |
| Friday | 10% | Close everything by 2 PM ET regardless |

### Stop Loss Rules
- **Hard stop:** -50% option value (wider than 0DTE because weekly has recovery time)
- **Delta stop:** Exit if delta < 0.10 (option is nearly worthless)
- **Gap stop:** If stock gaps >2% against thesis at open, skip entry

### Position Sizing (Kelly Criterion)
- Formula: f* = (b*p - q) / b, applied at half-Kelly
- p = confidence (clipped 0.30-0.70), b = reward/risk ratio
- Cap: 3% of portfolio per position
- Regime gate: multiply by 0.5 if macro_edge multiplier < 0.50

### Portfolio Construction (3-pick portfolio)
- Max 1 pick per sector (enforced diversification at small portfolio size)
- Correlation dedup: 20-day rolling return correlation > 0.75 = drop lower score
- Direction balance: if all 3 same direction AND avg confidence < 0.75, swap one for best contrarian from bench

---

## Backtest Validation Gate

Weight changes proposed by the pattern library are NOT auto-deployed. They must
pass a validation gate:

1. Accuracy must not drop > 2% (absolute)
2. Sharpe must not drop > 0.1
3. At least one metric must improve

If validation fails, proposed weights are rolled back. This prevents drift from
overfitting to recent results.

---

## Key Thresholds Quick Reference

| Metric | Threshold | Context |
|--------|-----------|---------|
| Delta target | 0.35 | Strike selection for calls and puts |
| Confidence minimum | 0.25 | Hard gate at Monday picks |
| Correlation dedup | 0.75 | 20-day return correlation |
| RSI oversold/overbought | 30/70 | Mean reversion triggers (weekly, wider than daily) |
| ADX trending | > 25 | vs choppy classification |
| VIX regime boundaries | 15/20/25/30 | Five-tier system |
| IV/RV cheap/expensive | 0.9/1.3 | IV state classification |
| IV term structure cheap | < 0.85 | Weekly vs monthly ratio |
| Pattern min observations | 5 | Before pattern adjusts scores |
| Calibration reliable | 10+ obs | Before live calibration overrides static |
| Weekly ATR% minimum | 1.5% | Wednesday scan hard filter |
| Options volume minimum | 500 | Wednesday scan (300 for Friday) |
| Max sector concentration | 2 of 5 | Diversity rule |
| Kelly cap | 3% | Maximum position size |
| Hard stop | -50% | Option value loss trigger |
| Gap skip threshold | 2% | Monday entry confirmation |
| Delta drift skip | > 0.15 | Monday entry confirmation |
| Model consensus bonus | std < 5 | +5% composite |
| Model disagreement | std > 15 | -8% composite |

---

## What Agents Should Do Differently Than Generic Analysis

1. **Reference specific scores and weights** — don't say "the momentum looks good,"
   say "momentum scored 72/100 but mean_reversion only scored 38, and in this
   elevated VIX regime, mean_reversion gets a 1.2x multiplier."

2. **Check ensemble agreement** — a pick where all three models agree is high
   conviction. A pick where momentum-model loves it but MR-model hates it has
   internal tension that may resolve badly.

3. **Account for charm decay** — a 0.35 delta call on Monday might be 0.15 by
   Thursday from charm alone. Factor this into hold/exit decisions.

4. **Reference pattern library** — if we've seen this pattern before (e.g.,
   "elevated|call|momentum|neutral|trending|cheap"), cite the historical win
   rate and number of observations.

5. **Use regime context** — don't analyze trades in a vacuum. The same setup
   in low-VIX (momentum 1.3x) vs high-VIX (momentum 0.7x) has fundamentally
   different expected behavior.

6. **Be specific about thresholds** — "approaching stop" means option P&L is
   near -50%, not a vague warning. "Target hit" means the day-specific target
   (40% Monday, 15% Thursday, etc.) was reached.

7. **Cite the validation gate** — when proposing tactical adjustments in
   reflection, note that weight changes must pass the backtest validation gate
   (accuracy drop ≤2%, Sharpe drop ≤0.1).
