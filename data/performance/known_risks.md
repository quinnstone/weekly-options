# Known Risks & Weekly-Options Pipeline Limitations

Documented 2026-04-14, rewritten from the 0DTE-fork original. These are known limitations of the current pipeline for **weekly (Mon entry, Fri expiry)** options, not bugs. The system will produce picks, but these gaps may reduce accuracy until live data refines them. Review after 4-6 weeks of scorecard data.

---

## Critical — Core mechanics not fully modeled

### 1. Confidence calibration is placeholder
- `analysis/scoring.py` uses a static table mapping raw confidence to calibrated confidence (e.g., 0.85-1.0 raw → 0.42 calibrated). The table was set theoretically, not empirically.
- Live pattern library overrides static calibration after 10+ observations per bucket — but until ~8-12 weeks of live picks accumulate, confidence numbers are rough.
- **Fix complexity**: Requires data only; no code change needed.
- **Impact**: Confidence-gated decisions (Kelly sizing, Monday hard-gate at 0.25, contrarian swap) may be mis-weighted.
- **When to revisit**: After 8-12 weeks.

### 2. Kelly sizing is confidence-gated on a placeholder
- Kelly fraction f* = (bp - q)/b uses calibrated confidence as p. When p itself is a placeholder (see #1), Kelly output is also a placeholder.
- **Current mitigation**: half-Kelly + 3% hard cap limits downside from miscalibration.
- **When to revisit**: alongside #1.

### 3. IV/RV-based `iv_rank` is not broker IV rank
- The `iv_rank` field in `scanners/options.py` is a 0-100 mapping of ATM IV / 20-day RV ratio, with inverted semantics (high = cheap).
- For weekly options, IV/RV is structurally > 1.5 because weeklies carry concentrated event/weekend risk premium that annualized RV doesn't capture.
- **Do not build hard rules on this metric.** See `project_iv_rank_caveats.md` in memory.
- **Fix**: source a 52-week IV percentile (Polygon or roll our own over 10+ weeks of scans).

---

## High — Signals are suboptimal but not wrong

### 4. Wednesday→Monday data staleness
- Wed scan runs at 7 AM, Mon picks at 8 AM — ~4.5 days of market action between scan and entry.
- Friday 4 PM refresh mitigates this (re-pulls technical + options + sentiment for survivors + bench).
- Residual gap: **Friday 4 PM close → Monday 8 AM open** is still 64 hours of weekend news and Asia/Europe action.
- **Current mitigation**: PreTradeAnalyst agent runs Monday 10 AM with web_search to catch overnight developments.
- **Remaining gap**: dynamic universe additions only fire Wednesday; a new hot ticker emerging Thursday-Monday is missed.

### 5. Delta/gamma assumption is BSM with implied-vol assumption
- Delta and gamma are computed via BSM with the ATM IV as the vol input.
- For weekly expirations, this is materially accurate (5 days to expiry ≈ enough time value for BSM to behave).
- Weakness: during elevated-VIX weeks, the actual vol surface has a significant skew we're not modeling.
- **Fix complexity**: Medium — would require IV surface interpolation across strikes.
- **Impact**: Strike selection may target slightly wrong delta zone when skew is steep.

### 6. Scorecard grades on Friday 4 PM close, not optimal exit
- P&L calculated using Friday closing intrinsic value.
- Real weekly exits happen across the week per the time-decay ladder (40% Mon, 35% Tue, 25% Wed, 15% Thu, 10% Fri).
- If a pick hits +40% Tuesday and reverses to -60% Friday, the scorecard shows a loss even though the trader should have exited Tuesday.
- **Fix complexity**: Medium — need to track intraday high/low throughout the week.
- **Impact**: Scorecard understates system performance; post-mortem misattributes "bad pick" when it was actually "bad exit timing."
- **When to fix**: After week 4, if the "take profit then reverse" pattern shows up in outcomes.

### 7. Mid-week agent blind window (Tue 3 PM → Thu 10 AM)
- PositionMonitor fires at routine urgency (no Opus call) on Wed 3 PM, then escalates at Thu 10 AM.
- 43-hour window where only mechanical thresholds guard held positions.
- **Current mitigation**: mechanical stop-loss (-50%) and target (day-specific) still fire.
- **When to fix**: after 4-6 weeks if scorecard shows positions bled through material midweek news.
- See `project_midweek_agent_gap.md` in memory.

---

## Medium — Tuning issues

### 8. Scoring weights are pre-live-calibrated
Current scoring is the 3-tier 10-factor ensemble from METHODOLOGY.md. Factor weights within each tier were set from the 0DTE fork's empirics and adjusted for weekly holds (momentum multipliers vary by VIX regime).
- **Right answer**: let the pattern library + DeepReflection propose adjustments after 4+ weeks; validation gate (accuracy drop ≤2%, Sharpe drop ≤0.1) prevents drift.
- **Expected first-live behavior**: weights will adjust 2-3 times in the first 2 months.

### 9. Dynamic universe volume threshold is static
- 500-contract options volume floor for Wednesday scan; 300 for Friday. Doesn't adjust by ticker size.
- **Impact**: May include mega-caps as "unusual" that routinely trade 1000 contracts; may miss genuinely unusual activity on small-caps hitting 400.
- **Fix complexity**: Low — switch to percentile of ticker's 20-day average.

### 10. Earnings timing granularity
- Earnings-week picks receive EarningsAnalyst review (TRADE/CAUTION/AVOID) and potential earnings_warning flag.
- Fine distinction missing: earnings BEFORE Wednesday scan vs ON Thursday vs ON Friday morning have different IV-crush dynamics.
- **Current coverage**: EarningsAnalyst sees IV/RV ratio + term structure + estimated crush %, generally catches bad setups.
- **Workaround**: agent has web_search to check post-earnings reaction if earnings already passed.

---

## Weekly-specific edge cases to watch for

### Expiry day (Friday) risks
- **Pin risk**: stock may pin to max-pain strike in final hour, options lose extrinsic value fast.
- **3 PM liquidity decline**: bid-ask spreads widen for weeklies post-3 PM.
- **Current policy**: force-close everything by 2 PM ET Friday per methodology §Risk Rules.

### Mid-week event risks
- **FOMC Wed 2 PM**: if FOMC lands during holding window, the VIX-regime-based weight multipliers shift intra-week. Agents see this via CURRENT_STATE regeneration but mechanical scores were locked Monday.
- **CPI/PPI Tuesday 8:30 AM**: gaps open; Monday entries at mid-price may be +/- 3% off overnight by Tuesday open. PreTradeAnalyst SKIP authority catches the most egregious cases.
- **Triple witching**: quarterly Fridays have abnormal flow; known_risk accepted.

### Data risks
- **yfinance throttling**: scanning 120+ tickers with options chains risks throttling. Staged rate limits in `_batch_scan()` mitigate.
- **Finnhub free tier**: 60 calls/minute; sentiment + economic calendar contends for quota. Currently within limits for 25-30 candidate batches.
- **Tradier options quote latency**: 15-30 minute delay during RTH on free tier. Acceptable for weekly picks; would matter for intraday.

---

## Review schedule

- **After week 2**: sanity-check. Are signals firing? Is scorecard grading producing P&L? Are all 4 agents with override authority actually being invoked?
- **After week 4**: are we winning on direction but losing on exit timing (suggests #6)? Losing on direction (suggests weight tuning per #8)?
- **After week 8**: confidence calibration — is pattern library overriding static table yet? If not, pattern key consistency issue worth investigating.
- **After week 12**: evaluate whether intraday monitoring would have materially changed outcomes. Decide Polygon.io go/no-go per `project_polygon_decision.md`.

---

*Last updated: 2026-04-14 (rewritten from 0DTE fork for weekly options)*
