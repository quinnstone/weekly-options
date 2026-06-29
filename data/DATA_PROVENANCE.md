# Data Provenance — read before learning off historical data

Authoritative map of where outcome/learning data lives, what's trustworthy, and
which capabilities each graded week carries. Written 2026-06-29.

## Source of truth: `data/performance/scorecard_data.json`

**This committed JSON is the canonical record** of graded picks and outcomes.
Every learning/analysis path reads it (or other committed files), NOT the DB:

| Consumer | Reads |
|---|---|
| Reflector (`tracking/reflector.py`) | `scorecard_data.json` (via `_load_scorecard_week`) |
| Directional backtest (`analysis/backtest.py`) | yfinance price history + `universe.robinhood` |
| Hypothesis tests (`analysis/hypothesis_tests.py`) | committed candidate pools + yfinance |
| Discord scorecard / `sc.show()` | `scorecard_data.json` (`self.data`) |

## The SQLite DB (`data/performance/zero_dte.db`) is WRITE-ONLY and gitignored

Do **not** trust or learn from the DB. Verified facts (2026-06-29):
- **Gitignored** (`.gitignore: *.db`) — never committed, not shared between your
  clone and CI. Persisted only inside the CI cache (`actions/cache` on
  `data/performance/`), so it accumulates on the runner but is invisible locally
  (your local copy reads 0 rows).
- **Written** by 3 paths: `record_picks`, `grade_picks`, `record_weekly_result`.
- **Read by NOTHING.** All six `get_*` query methods (`get_alltime_stats`,
  `get_weekly_results`, `get_picks_for_week`, `get_ticker_history`,
  `get_streak`, `get_monthly_summary`) have **zero callers**. It is dead-end
  storage kept in sync as a future option, not an active analysis layer.

If the DB is ever promoted to the analysis layer, re-verify this file.

## `grader_version` — which capabilities each graded week carries

Stamped on every week in `scorecard_data.json` (and the unused DB) from
2026-06-29 forward. Tells you EXPLICITLY which fields a week has, so you never
misread an absent field as a feature failure. Map (see `GRADER_VERSION` in
`tracking/scorecard.py`):

| version | date | adds |
|---|---|---|
| 1 | pre-2026-06 | base grading: pnl / result / W-L-P (hold-to-expiry intrinsic) |
| 2 | 2026-06-01 | `entry_signal` (GO / ADJUST / SKIP) per pick |
| 3 | 2026-06-08 | `diagnostics` (pre_run_5d_pct, spy_week_move_pct, rsi_at_entry, premium_pct_of_spot) |
| 4 | 2026-06-29 | `entry_features` (model scores) + `traded_*` (GO-only realized P&L) |

**Historical weeks before each version lack those fields and are NOT
backfilled** (backfills don't survive the CI cache-clobber). Pre-`grader_version`
weeks (graded before 2026-06-29) have no version stamp at all — infer from the
table above by `graded_at` date if needed. Concretely, as of writing:
- 2026-04-27, 2026-05-04, 2026-05-18: v1-equivalent (no entry_signal/diagnostics)
- 2026-06-01: has entry_signal only
- 2026-06-08, 2026-06-22: have entry_signal + diagnostics (no entry_features/traded_*)
- 2026-07-06 onward: full v4

## Directional vs. realized — do not confuse them

Each graded week and the all-time rollup carry TWO P&L sets:
- **Directional** (`total_pnl`, `wins/losses/partials`): grades ALL picks incl.
  SKIP/ADJUST the system told you NOT to enter. This is the **model-accuracy**
  signal — use it for learning which signals predict direction. It is NOT your
  account P&L (all-time was -$1,145 on this basis as of 2026-06-22).
- **Realized / traded** (`traded_pnl`, `traded_wins/...`): GO-only, i.e. what was
  actually entered. This is the **account P&L** (all-time +$729 / +13.2% as of
  2026-06-22). `traded_*` exists only on v4+ weeks; older weeks default GO.

Train signal-quality models on directional outcomes; report performance on
realized.
