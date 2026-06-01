"""One-off backfill: regenerate 2026-06-01 entry_confirmations.json with the
corrected gap-baseline + delta logic (commit fixing the stale-original_price
gap bug).

WHY this is a standalone script and not a live stage re-run:
  - The live monday_entry_confirmation() re-run now (after close) would fetch
    end-of-day prices, make a paid Opus Pre-Trade Analyst call, and re-save
    monday_picks + the report file with EOD premiums — corrupting the
    scorecard's entry basis. We must not do any of that.

WHAT this preserves (faithfulness to the original 15:20 run):
  - current_price: reused verbatim from the existing entry_confirmations.json
    (the intraday snapshot the broken run actually saw). NOT re-fetched.
  - entry premiums in monday_picks/report: left untouched (the original
    gap-SKIP'd before the premium re-estimate, so they're the clean originals).

WHAT this corrects:
  - gap baseline: real prior trading-day close (fixed historical fact) instead
    of the stale pick["current_price"]. This is the bug that false-SKIP'd 3/3.
  - new_delta: now computed (the original skipped the delta block), using the
    strike's real ATM IV from the option chain.
  - Mechanical signal recomputed under the fixed decision tree.

The Opus agent layer is intentionally omitted — the original saved file carried
no agent briefs either (the agent step did not attach for this run), so
mechanical-only is consistent with what was persisted.
"""
import json
from datetime import datetime, date

import yfinance as yf

from config import Config
from scanners.options import _bsm_greeks, _bsm_price, _RISK_FREE_RATE

config = Config()
DATE = "2026-06-01"
cdir = config.candidates_dir / DATE

picks = json.load(open(cdir / "monday_picks.json"))
old_confs = json.load(open(cdir / "entry_confirmations.json"))
old_by_ticker = {c["ticker"]: c for c in old_confs}

GAP_THRESHOLD = 2.0

def atm_iv_from_chain(tk, expiry, strike, direction):
    """Real IV for the pick's strike from the live option chain. Falls back to
    None if unavailable (caller then uses 0.30 like the production code)."""
    try:
        chain = tk.option_chain(expiry)
        df = chain.calls if direction == "call" else chain.puts
        if df is None or df.empty:
            return None
        m = df[df["strike"] == strike]
        if m.empty:
            df = df.copy()
            df["_d"] = (df["strike"] - strike).abs()
            row = df.loc[df["_d"].idxmin()]
        else:
            row = m.iloc[0]
        iv = float(row.get("impliedVolatility", 0))
        return iv if iv > 0.05 else None
    except Exception:
        return None

new_confs = []
for pick in picks:
    ticker = pick["ticker"]
    direction = pick.get("direction", "call")
    strike = pick.get("strike")
    original_premium = pick.get("premium")
    original_price = pick.get("current_price")
    original_delta = pick.get("estimated_delta", 0.35)
    expiry = pick.get("expiry")

    old = old_by_ticker.get(ticker, {})
    # Faithful: reuse the intraday snapshot the broken run captured.
    current_price = old.get("current_price")

    tk = yf.Ticker(ticker)
    daily = tk.history(period="5d")
    prior_close = float(daily["Close"].iloc[-2])
    today_open = float(daily["Open"].iloc[-1])

    # ---- corrected gap (true overnight move) ----
    gap_pct = (today_open - prior_close) / prior_close * 100

    # ---- delta re-estimate at the snapshot price, real IV ----
    iv = atm_iv_from_chain(tk, expiry, strike, direction) or pick.get("iv") or 0.30
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max((expiry_dt - date(2026, 6, 1)).days, 1) / 365.0
    except Exception:
        T = 5 / 365.0
    option_type = "call" if direction == "call" else "put"
    new_delta = abs(_bsm_greeks(current_price, strike, T, _RISK_FREE_RATE, iv, option_type)["delta"])
    delta_shift = abs(new_delta - abs(original_delta))

    # ---- decision tree (mirrors fixed stages.py) ----
    if abs(gap_pct) > GAP_THRESHOLD:
        signal = "SKIP"
        detail = f"Gap {gap_pct:+.1f}% exceeds 2% threshold — move may be priced in"
        new_premium = None
    elif new_delta < 0.10:
        signal = "SKIP"
        detail = f"Delta collapsed to {new_delta:.2f} — option nearly worthless"
        new_premium = None
    elif delta_shift > 0.15:
        signal = "ADJUST"
        detail = (f"Delta shifted from {abs(original_delta):.2f} to {new_delta:.2f} "
                  f"(price moved from ${original_price:.2f} to ${current_price:.2f}). "
                  f"Consider re-selecting strike closer to current price.")
        new_premium = None
    else:
        signal = "GO"
        new_premium = round(_bsm_price(current_price, strike, T, _RISK_FREE_RATE, iv, option_type), 2)
        # direction validation note (section 4) — uses snapshot vs open
        price_move_pct = (current_price - today_open) / today_open * 100 if today_open else 0
        confirmed = (direction == "call" and price_move_pct > 0) or (direction == "put" and price_move_pct < 0)
        if not confirmed and abs(price_move_pct) > 1.0:
            detail = f"Price moving against thesis ({price_move_pct:+.1f}% since open). Proceed with caution or wait for reversal."
        elif confirmed:
            detail = f"Early action confirms {direction} thesis ({price_move_pct:+.1f}% since open)."
        else:
            detail = f"Flat since open ({price_move_pct:+.1f}%) — no strong confirmation yet."
        if new_premium and original_premium and original_premium > 0:
            chg = (new_premium - original_premium) / original_premium * 100
            if abs(chg) > 15:
                detail += (f" Premium moved {chg:+.0f}% (${original_premium:.2f} → ${new_premium:.2f}). "
                           f"Adjust limit order accordingly.")

    new_confs.append({
        "ticker": ticker,
        "direction": direction,
        "strike": strike,
        "original_premium": original_premium,
        "original_price": original_price,
        "signal": signal,
        "detail": detail,
        "current_price": round(current_price, 2) if current_price else None,
        "new_delta": round(new_delta, 3) if new_delta else None,
        "new_premium": new_premium,
        "gap_pct": round(gap_pct, 2),
        "timestamp": old.get("timestamp", datetime.now().isoformat()),
        "backfilled": True,
        "backfill_note": "Regenerated with corrected gap baseline (real prior close) + delta; "
                         "entry premiums in monday_picks/report intentionally unchanged.",
    })

print("=== BEFORE → AFTER ===")
for ticker in [c["ticker"] for c in new_confs]:
    o = old_by_ticker[ticker]
    n = next(c for c in new_confs if c["ticker"] == ticker)
    print(f"  {ticker:5s}  {o['signal']:6s} (gap {o.get('gap_pct')})  ->  "
          f"{n['signal']:6s} (gap {n['gap_pct']}, delta {n['new_delta']})")

out = cdir / "entry_confirmations.json"
with open(out, "w") as fh:
    json.dump(new_confs, fh, indent=2)
print(f"\nWrote {out}")
