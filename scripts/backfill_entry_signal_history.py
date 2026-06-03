"""One-off: backfill entry_signal onto already-graded weeks that were graded
before the entry_signal feature existed (commit c36751b, 2026-06-01).

Why a standalone injector and NOT a re-grade:
  - The stored P&L / results / totals are already correct; only the
    entry_signal annotation is missing. Re-running grade_week would re-fetch
    closing prices and could re-trigger the (paid) post-mortem agent — wasted
    work to recompute numbers that are right.
  - This reads the existing entry_confirmations.json for each week (the same
    source grade_week now uses) and writes the tag in place. No network, no
    agents, no changes to any P&L field.

Idempotent: only fills entry_signal where it's currently None/missing; picks
with no confirmation file default to "GO" (matching grade_week's behavior).
"""
import json
from config import Config

config = Config()
DATA_PATH = config.performance_dir / "scorecard_data.json"

data = json.load(open(DATA_PATH))
changed = False

for week in data["weeks"]:
    pick_date = week["pick_date"]
    conf_path = config.candidates_dir / pick_date / "entry_confirmations.json"
    signals = {}
    if conf_path.exists():
        try:
            for c in json.load(open(conf_path)):
                if c.get("ticker"):
                    signals[c["ticker"]] = c.get("signal")
        except Exception as exc:
            print(f"  WARN {pick_date}: could not read confirmations ({exc})")

    for p in week.get("picks", []):
        if p.get("entry_signal") in (None, ""):
            p["entry_signal"] = signals.get(p.get("ticker"), "GO")
            changed = True

print("=== entry_signal after backfill ===")
for week in data["weeks"]:
    sigs = {p["ticker"]: p.get("entry_signal") for p in week["picks"]}
    print(f"  {week['pick_date']}: {sigs}")

if changed:
    with open(DATA_PATH, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"\nWrote {DATA_PATH}")
else:
    print("\nNo changes needed (all picks already tagged).")
