"""Standing hypothesis tests — analysis-only, NO impact on the live pipeline.

Runs alongside the Wednesday directional backtest (see workflow `backtest`
stage). Each hypothesis here is a PROPOSED process change that must earn its
way into the live system with data, per the adversarial-review discipline
(memory: feedback_adversarial_self_review — most proposals should die).

Current hypotheses (added 2026-06-11, Qullamaggie-derived — see memory
project_beta_not_exhaustion for why naive exhaustion filters already died):

H1  CONSOLIDATION-ENTRY: among momentum leaders (21d return > +15%), do names
    entered while CONSOLIDATING (paused, tight range near highs) outperform
    names entered while still EXTENDED (5d run > +8%) over the next Mon->Fri
    week? Tests *where-in-structure* entry timing — a different claim than
    the refuted RSI/exhaustion filter.

H2  MA-REGIME PARTICIPATION: do momentum leaders only deliver in ON regimes
    (QQQ close > 20d SMA and 10d SMA > 20d SMA)? The simplest testable form
    of the open book-beta question — the candidate alternative to the
    nearly-inert assess_macro_edge confidence scaling.

Output: data/performance/hypothesis_tests_{date}.json + console report.
Failures are the caller's problem to ignore (workflow runs this non-fatally).
"""
import sys, os, json, logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()

LEADER_21D = 15.0      # % 21d return to qualify as a momentum leader
EXTENDED_5D = 8.0      # % 5d run that marks "still extended / mid-air"
CONSOL_LO, CONSOL_HI = -3.0, 4.0   # 5d move band that marks "paused"
CONSOL_RANGE_MAX = 8.0             # 5d high-low range (% of low) to call it tight


def _mondays(weeks_back: int) -> list:
    """Trading Mondays (skip if market holiday — approximated by data presence)."""
    today = datetime.now().date()
    mons = []
    d = today - timedelta(days=today.weekday())  # this week's Monday
    for _ in range(weeks_back):
        d -= timedelta(days=7)
        mons.append(d)
    return sorted(mons)


def run(weeks: int = 26, tickers: list = None) -> dict:
    if tickers is None:
        from universe.robinhood import get_full_universe
        tickers = get_full_universe()
    tickers = sorted(set(tickers) | {"QQQ", "SPY"})

    start = (datetime.now() - timedelta(weeks=weeks + 8)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    close, openp, high, low = px["Close"], px["Open"], px["High"], px["Low"]
    qqq = close["QQQ"].dropna()
    sma10, sma20 = qqq.rolling(10).mean(), qqq.rolling(20).mean()

    h1_rows, h2_rows = [], []
    for mon in _mondays(weeks):
        m = pd.Timestamp(mon)
        idx = close.index
        locs = idx.get_indexer([m], method="bfill")
        if locs[0] < 30:
            continue
        mloc = locs[0]
        if idx[mloc].date() != mon:      # holiday Monday -> use actual first day
            pass
        fri = min(mloc + 4, len(idx) - 1)
        pre_end = mloc - 1               # prior Friday
        # regime as of prior Friday close (no look-ahead)
        ref = idx[pre_end]
        regime_on = bool(
            ref in sma20.index and not pd.isna(sma20.loc[ref])
            and qqq.loc[ref] > sma20.loc[ref] and sma10.loc[ref] > sma20.loc[ref]
        )

        for t in tickers:
            if t in ("QQQ", "SPY"):
                continue
            try:
                c = close[t]
                c21, c5, c0 = c.iloc[pre_end - 21], c.iloc[pre_end - 5], c.iloc[pre_end]
                if any(pd.isna(v) for v in (c21, c5, c0)):
                    continue
                ret21 = (c0 / c21 - 1) * 100
                if ret21 < LEADER_21D:
                    continue            # not a momentum leader — out of scope
                ret5 = (c0 / c5 - 1) * 100
                rng5 = (high[t].iloc[pre_end - 5:pre_end + 1].max()
                        / low[t].iloc[pre_end - 5:pre_end + 1].min() - 1) * 100
                e, x = openp[t].iloc[mloc], c.iloc[fri]
                if pd.isna(e) or pd.isna(x) or e <= 0:
                    continue
                nxt = (x / e - 1) * 100

                if ret5 > EXTENDED_5D:
                    cohort = "extended"
                elif CONSOL_LO <= ret5 <= CONSOL_HI and rng5 <= CONSOL_RANGE_MAX:
                    cohort = "consolidating"
                else:
                    cohort = "other"
                h1_rows.append((str(mon), t, cohort, round(nxt, 2)))
                h2_rows.append((str(mon), t, regime_on, round(nxt, 2)))
            except Exception:
                continue

    h1 = pd.DataFrame(h1_rows, columns=["week", "ticker", "cohort", "next_wk"])
    h2 = pd.DataFrame(h2_rows, columns=["week", "ticker", "regime_on", "next_wk"])

    def _agg(df, key):
        g = df.groupby(key)["next_wk"].agg(["count", "mean", "median",
                                            lambda s: (s > 0).mean()])
        g.columns = ["n", "avg_next_wk", "median_next_wk", "win_rate"]
        return g.round(3)

    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "weeks": weeks,
        "leader_def": f"21d>{LEADER_21D}%",
        "h1_consolidation_entry": json.loads(_agg(h1, "cohort").to_json(orient="index")),
        "h2_regime_participation": json.loads(
            _agg(h2, "regime_on").rename(index={True: "ON", False: "OFF"}).to_json(orient="index")),
        "h2_weeks_on": int(h2[h2.regime_on].week.nunique()),
        "h2_weeks_off": int(h2[~h2.regime_on].week.nunique()),
    }

    print(f"\n=== HYPOTHESIS TESTS ({weeks}w, leaders={LEADER_21D}%+ 21d) ===")
    print("\nH1 — entry structure among momentum leaders (next Mon->Fri):")
    print(_agg(h1, "cohort").to_string())
    print("\nH2 — QQQ 10/20-SMA regime (ON vs OFF), leaders only:")
    print(_agg(h2, "regime_on").rename(index={True: "ON", False: "OFF"}).to_string())
    print(f"   regime weeks: ON={results['h2_weeks_on']}, OFF={results['h2_weeks_off']}")

    out = config.performance_dir / f"hypothesis_tests_{results['run_date']}.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {out}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=26)
    args = ap.parse_args()
    run(weeks=args.weeks)
