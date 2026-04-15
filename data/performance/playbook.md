# Weekly Options Playbook

This document is the agent's evolving knowledge base for weekly-expiry directional options (Monday entry, Friday expiry). It is read before every pick session (Wednesday scan, Monday confirm) and updated after every weekly reflection. Over time it becomes the primary source of strategic intelligence for weekly selection.

Authoritative methodology lives in `agents/METHODOLOGY.md`. This playbook captures what has been LEARNED from live outcomes — patterns, regime observations, mistakes to avoid, things that worked.

---

## Scoring Architecture (current)

See METHODOLOGY.md for full details. Summary:

- **Tier 1 (Direction, 60% weight):** momentum (0.20), mean_reversion (0.15), regime_bias (0.10), trend_persistence (0.15)
- **Tier 2 (Edge Quality, 25%):** iv_mispricing (0.10), flow_conviction (0.08), event_risk (0.07)
- **Tier 3 (Execution, 15%):** liquidity (0.05), strike_efficiency (0.05), theta_cost (0.05)

Ensemble of three models (linear factor 50%, momentum-only 25%, mean-reversion+value 25%) votes on composite. Consensus bonus/penalty based on model std dev.

Regime-adaptive multipliers modify weights based on VIX regime (see METHODOLOGY §Regime-Adaptive Multipliers).

---

## Regime Playbook (to be populated from live data)

### Low VIX (0-15)
- Bias: momentum plays, trend persistence
- Multiplier profile: momentum 1.3x, mean_reversion 0.8x, theta_cost 1.2x (premium decays faster in calm markets)
- Observations: *(populate after week 1+)*

### Normal VIX (15-20)
- Bias: baseline scoring — signals carry their default weight
- Observations: *(populate)*

### Elevated VIX (20-25)
- Bias: mean_reversion plays gain edge; IV mispricing opportunities appear
- Multiplier profile: mean_reversion 1.2x, iv_mispricing 1.3x
- Observations: *(populate)*

### High VIX (25-30)
- Bias: mean_reversion dominant; momentum deweighted
- Multiplier profile: mean_reversion 1.4x, momentum 0.7x
- Observations: *(populate)*

### Extreme VIX (30+)
- Bias: strong contrarian setups; small size
- Multiplier profile: mean_reversion 1.5x, momentum 0.5x
- Observations: *(populate)*

---

## Economic Calendar Awareness (Weekly Holds)

High-impact events DURING the Mon-Fri holding window demand special attention. For weekly options, mid-week events matter differently than they do for 0DTE — IV shifts through the full holding period, not just a single release.

### Key events and weekly-hold implications

| Event | Typical Day | Weekly-Hold Notes |
|-------|-------------|-------------------|
| FOMC Rate Decision | Wed 2 PM | Splits the holding window. Entered Monday at baseline IV; Fed day re-prices volatility regime. `event_risk` score penalizes by -20 if inside window. |
| CPI / PPI | Tue/Wed 8:30 AM | Gap open reshapes Monday entry; PreTradeAnalyst reassesses delta drift vs gap threshold (2%). |
| NFP | First Fri 8:30 AM | Lands on expiry day. Expected weekly move estimate should account for NFP gap; force-close policy (2 PM Fri) mitigates post-NFP whip. |
| PCE | Last Fri | Same dynamics as NFP for expiry-day risk. |
| GDP | Quarterly | Significant gap potential; treat as high-risk event within holding window. |
| Earnings | Any day | Per-ticker, not calendar-wide. EarningsAnalyst handles with TRADE/CAUTION/AVOID verdict. |

### How the system handles events

- **Event risk penalty:** -20 composite score if high-risk event in window; -5 for medium-risk. FOMC gets additional -10.
- **Confidence multiplier:** -0.10 when holding window contains high-risk events; compounds with VIX regime.
- **Kelly cap adjustment:** sizing cap drops from 3% to 1.5% for earnings-week positions specifically.

---

## Signal Reliability Log

Populated by DeepReflection after each week. Format: `signal | weeks tracked | accuracy contribution | notes`.

| Signal | Weeks Tracked | Accuracy | Notes |
|--------|---------------|----------|-------|
| *(populated after week 1)* | | | |

---

## Patterns Discovered

Populated as the pattern library accumulates 5+ observations per key.

Pattern key format: `{regime}|{direction}|{dominant_signal}|{rsi_zone}|{trend_state}|{iv_state}`

Example placeholder: `elevated|call|momentum|neutral|trending|cheap` — win rate TBD.

*(populate from scorecard)*

---

## Mistakes to Avoid

Populated from weekly reflections after live losses. Each entry: the setup, why it seemed appealing, what actually happened, and the rule that would have prevented it.

*(populate from DeepReflection)*

---

## What Works

Populated from weekly reflections after live wins. Each entry: the setup, the signal convergence, and why the thesis played out.

*(populate from DeepReflection)*

---

## Weekly Reflections

### Pre-live (for reference)

Prior to 2026-04-15, the system ran in test/paper mode. Test-run lessons are in archival files; not authoritative for live decisions.

### Week 1 — 2026-04-15 (first live run)

- **Market:** *(populate Wed AM from market_summary)*
- **Picks:** *(populate Monday after narrowing)*
- **Outcome:** *(populate Friday after close)*
- **Notes:** First live run. Baseline for all future weeks.

---

## How This Playbook Is Used

1. **Every agent call** loads this file into context via `agents/base.py::_build_system_context()`.
2. **DeepReflection (Saturday)** proposes updates to this file based on the week's outcomes.
3. **Validation gate** prevents weight changes from being auto-applied — but playbook entries (patterns, mistakes, what works) update freely because they're qualitative observations, not numerical weights.
4. **Agents should cite this playbook** when a current setup matches a logged pattern: "matches pattern from week 3 observation — elevated VIX + call + momentum-driven worked 2/2 times with avg +34% P&L."
