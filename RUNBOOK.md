# Weekly-Options Operations Runbook

Quick reference for running the pipeline. The system is automated via GitHub
Actions cron, but **Monday's two entry jobs are run manually** for reliable,
on-time, pre-open execution (GH scheduled crons have run 60–315 min late every
Monday observed — see "Why manual Monday" below).

---

## Monday routine (the only manual steps)

Run these two, in order, from your terminal (or the GitHub Actions UI).

### 1. ~8:00 AM ET — generate the week's picks
```
gh workflow run weekly-options.yml -f stage=monday
```
- Selects the 3 picks and their strikes against pre-open pricing.
- **Run this at/just after 8:00, and NOT past ~9:15 AM ET.** After ~9:15 the
  strike selection starts anchoring to intraday prices that have already moved
  (the problem that produced the bad 2026-06-01 picks). Earlier is safer.

### 2. ~10:00 AM ET — entry confirmation
```
gh workflow run weekly-options.yml -f stage=confirm
```
- Re-checks each pick 30 min after the open: real overnight gap, delta drift,
  live premium. Emits GO / ADJUST / SKIP per pick. **This is the message you
  trade from.**

### Watch a run (optional)
```
gh run list --workflow=weekly-options.yml --limit 3
```

### Prefer clicking?
GitHub → **Actions** → *Weekly Options Pipeline* → **Run workflow** →
choose `monday` (then `confirm` at 10:00). The dropdown lists each stage by
name; pick `monday` first, `confirm` second.

> **Laptop lid:** Safe to close it. These jobs run on GitHub's servers
> (`runs-on: ubuntu-latest`), not your machine. Once `gh workflow run` returns,
> the dispatch is queued server-side; your laptop is not involved in execution.
> You can re-open later and check results with `gh run list`.

---

## Everything else is automatic (no action needed)

| Job | When | Purpose |
|-----|------|---------|
| Wednesday scan | Wed (cron) | Build candidate pool — feeds Friday |
| Friday refresh | Fri 4 PM (cron) | Re-rank pool + bench — **feeds Monday's picks** |
| Tue/Wed/Thu monitors | (cron) | Status updates on open positions |
| Friday final_exit | Fri 1:30 PM (cron) | Close expiring positions |
| Scorecard / Reflection | Fri/Sat (cron) | Grade prior week, agent reflection |
| Directional backtest | Wed 9 AM ET (cron) | Walk-forward validation of direction signals (26-wk lookback); output in data/performance/ |
| Hypothesis tests | Wed (with backtest) | Standing analysis-only tests of PROPOSED process changes (analysis/hypothesis_tests.py); a hypothesis must win here for weeks before touching the live pipeline |

These read **settled** data and have days of slack before they're consumed, so
cron lateness does not corrupt them. They only need to *run* — see the health
check below to confirm they did.

---

## Weekly health check (run anytime Fri–Mon morning, before the 8 AM picks)

Confirms the two jobs that PREPARE Monday's picks actually produced their files.
A *dropped* cron (GH occasionally skips or a run crashes) is the one real risk —
e.g. 2026-04-24's Friday refresh crashed and produced nothing, so that week's
Monday picks silently fell back to stale Wednesday data.

```
# Most recent Wednesday scan + Friday refresh present?
ls -la data/candidates/$(date -v-wed +%Y-%m-%d 2>/dev/null || date -d 'last wednesday' +%Y-%m-%d)/wednesday_scan.json \
       data/candidates/$(date -v-fri +%Y-%m-%d 2>/dev/null || date -d 'last friday' +%Y-%m-%d)/friday_refresh.json 2>/dev/null
```

If **Friday refresh is missing**, backfill before Monday:
```
gh workflow run weekly-options.yml -f stage=friday
```
If **Wednesday scan is missing**, backfill it first:
```
gh workflow run weekly-options.yml -f stage=wednesday
```
(Both read settled data, so running them late over the weekend is fine — they
produce the same result they would have on schedule.)

---

## Why manual Monday (context)

GitHub Actions scheduled crons are best-effort and have been chronically late:
every Monday from 2026-04-13 onward fired 60–315 min after the 8:00 AM target,
and the last 4 fired *after* the 9:30 open. A late picks run selects strikes
against an already-moved intraday price (the 2026-06-01 failure: a $280 DDOG
strike picked against a stock that had run from $252 to $277 that morning).

Manual dispatch starts within seconds because it's an API call, not a polled
schedule — so you control the timing. The scheduled Monday crons remain as an
automatic backstop, and an idempotency guard in `monday_picks` ensures a
late-firing backup cron will **not** overwrite the good picks you generated
manually at 8 AM (it detects the existing file and exits).
