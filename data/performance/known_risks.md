# Known Risks & 0DTE-Specific Gaps

Documented 2026-03-22. These are known limitations of the current pipeline, not bugs. The system will produce picks, but these gaps may reduce accuracy for 0DTE specifically. Review after 4-6 weeks of scorecard data to prioritize which adjustments actually move the needle.

---

## Critical — Core 0DTE mechanics not modeled

### 1. No intraday monitoring or exit rules
- Pipeline picks in the morning and assumes hold-to-close
- Real 0DTE needs: take profit at 50%, stop loss at 20%, force exit by 3:50 PM
- A winning pick at 11 AM can be a total loss by 4 PM without exit signals
- **Fix complexity**: High — requires an intraday loop or scheduled re-checks
- **Workaround**: Friday agent can set manual exit targets in Discord message

### 2. Gamma/theta blindness
- Options expiring in hours have exponential time decay (not linear)
- Near-binary delta behavior: $0.50 stock move can flip delta 0.3 to 0.8
- `sqrt(252)` expected move formula assumes daily decay — wrong for same-day
- The system doesn't model theta acceleration (3-5% premium lost per hour in final 4 hours)
- **Fix complexity**: Medium — add Black-Scholes with hours-to-expiry for theta/gamma
- **Impact**: Picks may recommend options that are too close to ATM (gamma risk) or too far OTM (theta death)

### 3. Daily indicators applied to intraday game
- RSI(14), MACD(12,26,9), Bollinger(20) measure multi-day trends
- A stock can be "oversold" on daily RSI and still drop 3% intraday
- Missing: VWAP, opening range breakout, 5-min momentum, intraday volume profile
- **Fix complexity**: Medium — add intraday data pull via yfinance `interval="5m"`
- **Workaround**: Friday agent applies intraday judgment manually

---

## High — Signals are suboptimal but not wrong

### 4. Wednesday data is 48+ hours stale by Friday
- Core thesis (scan Wed, pick Fri) works for identifying which stocks to watch
- But scoring needs Friday morning recalculation — currently does this via re-scan
- Thursday earnings miss or gap can invalidate Wednesday rankings
- **Current mitigation**: Thursday refresh + Friday re-scan already address this partially
- **Remaining gap**: Dynamic universe additions only run Wednesday, not Friday

### 5. Delta approximation is crude
- `moneyness * 0.5` as delta proxy is wrong for 0DTE
- 0DTE options are nearly digital — delta jumps dramatically with small moves
- Should use Black-Scholes with actual hours remaining
- **Fix complexity**: Low — standard BS formula, just needs time-to-expiry in hours
- **Impact**: Strike selection may target wrong delta zone

### 6. Vol/OI ratio breaks on expiry day
- Open interest collapses on Friday for weeklies, making ratio meaningless
- Flow scoring should use absolute volume + notional value, not ratios
- **Fix complexity**: Low — change threshold from ratio to absolute + percentile
- **Impact**: Flow signal is unreliable on Fridays specifically

### 7. Scorecard assumes hold-to-close
- P&L calculated using closing price intrinsic value
- Real 0DTE exits happen earlier, bid-ask spread at 3:55 PM can be $0.50 wide
- No tracking of intraday high (could have exited profitable even if closed OTM)
- **Fix complexity**: Low — add intraday high/low tracking via yfinance
- **Impact**: Scorecard may show losses on picks that were profitable intraday

---

## Medium — Tuning issues

### 8. Scoring weights aren't calibrated for 0DTE
Current weights:
```
technical: 0.20, options: 0.20, sentiment: 0.15,
flow: 0.10, market_regime: 0.15, expected_move: 0.20
```
Recommended for 0DTE:
```
technical: 0.10, options: 0.20, sentiment: 0.10,
flow: 0.25, market_regime: 0.05, expected_move: 0.30
```
- Flow should be primary directional signal (25%+)
- Expected move should dominate (premium decay is the P&L driver)
- Market regime matters less intraday
- **Fix complexity**: Trivial — change constants in scoring.py
- **When to fix**: After 4+ weeks of data, let the reflection engine suggest adjustments

### 9. Dynamic universe volume threshold is static
- 5,000 contracts threshold: nothing on SPY, massive on RBLX
- Should be percentile-based relative to ticker's historical average
- **Fix complexity**: Low
- **Impact**: May miss unusual activity on low-volume names, or include noise on high-volume names

### 10. No earnings timing distinction
- Earnings ON Friday = IV crush at expiry (kills 0DTE plays)
- Earnings BEFORE Wednesday = IV already compressing into Friday
- Pipeline treats both the same
- **Fix complexity**: Low — check earnings date vs expiry date, flag or exclude
- **Workaround**: Friday agent and catalyst report should catch this manually

---

## Edge cases to watch for

### Friday-specific risks
- **3:00 PM liquidity cliff**: Bid-ask spreads widen dramatically after 3 PM on weeklies
- **Pin risk**: Stock pins to max pain strike in final hour, options lose all extrinsic value
- **Assignment risk**: ITM options may be exercised early on Friday (though unlikely for 0DTE)

### Market regime risks
- **FOMC Fridays**: If FOMC was Wed/Thu, Friday is post-announcement — different dynamics
- **Triple witching**: Quarterly expiry Fridays have abnormal flow patterns
- **VIX > 30**: Options are expensive, but also more likely to pay off — current system gets cautious when it should potentially get aggressive (contrarian)

### Data risks
- **yfinance rate limits**: Scanning 120+ tickers with options chains may hit throttling
- **Finnhub free tier**: 60 calls/minute — dynamic universe + sentiment scanning may exceed
- **Stale quotes**: yfinance options data can lag 15-30 minutes during market hours

---

## Review schedule

- **After week 4**: Check scorecard results. Are we winning on direction but losing on timing? (Suggests intraday gaps matter most.) Are we losing on direction? (Suggests signal/weight gaps matter most.)
- **After week 8**: Review which signals correlated with wins vs losses. Adjust weights based on data, not theory.
- **After week 12**: Evaluate if intraday monitoring/exit rules would have changed outcomes. If scorecard shows "picks hit 50% profit then reversed to loss" pattern, prioritize exit rules.

---

*Last updated: 2026-03-22*
