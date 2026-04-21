"""
Position Monitor Agent — runs Tue-Fri during market hours.

Fetches current prices for the 3 holdings, calculates P&L, checks
stops/targets, and generates actionable alerts. Solves the "blind
between entry and grading" problem.

Runs at different urgency levels:
- ROUTINE (Tue-Wed): brief status update
- ELEVATED (Thu): theta accelerating, flag weak positions
- CRITICAL (Fri): must close by 2 PM, explicit exit instructions
"""

import logging
from datetime import datetime

import yfinance as yf

from agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the position monitor for 3 weekly options positions (Mon entry
→ Fri expiry). You have full access to the methodology reference above.

Use the time-decay-aware daily targets from the methodology:
- Monday: 40% target | Tuesday: 35% | Wednesday: 25% | Thursday: 15% | Friday: 10%

For each position, assess:
- P&L vs the day-specific target (not a static target)
- Charm decay: a 0.35 delta on Monday could be 0.20 by Wednesday. If estimated
  option P&L includes theta cost of ~4%/day, call out positions where charm is
  eating delta faster than the stock is moving.
- Whether the original ensemble consensus still holds — did the dominant signal
  (momentum, mean_reversion, trend) get confirmed or contradicted by this week's
  price action?

**Risk thresholds to reference:**
- Hard stop: -50% option value
- Delta stop: delta < 0.10 (nearly worthless)
- Warning zone: -30% (approaching stop)

At the end, give ONE clear recommendation. Be terse — trading desk, not research note.

**Urgency levels:**
- ROUTINE (Mon PM, Tue, Wed): Brief status. Flag outliers.
- ELEVATED (Thu): Theta accelerating. Any position below -15% should be flagged
  for exit — recovery through Friday is unlikely with charm acceleration.
- CRITICAL (Fri): Everything closes by 2 PM ET. Market orders if needed. No exceptions."""


class PositionMonitor(BaseAgent):
    """Monitor open positions and generate alerts."""

    AGENT_NAME = "position_monitor"
    MAX_TOKENS = 1500

    # Time-decay-aware profit targets by day of week
    DAILY_TARGETS = {
        0: 40,   # Monday: 40% target
        1: 35,   # Tuesday: 35%
        2: 25,   # Wednesday: 25%
        3: 15,   # Thursday: 15% (theta accelerating)
        4: 10,   # Friday: take any profit
    }

    def monitor(self, picks: list, market_summary: dict = None,
                urgency: str = "ROUTINE") -> dict:
        """Check all positions and generate status report.

        Parameters
        ----------
        picks : list
            Original Monday picks with entry data.
        market_summary : dict or None
            Current market context (optional for intraday checks).
        urgency : str
            ROUTINE, ELEVATED, or CRITICAL.

        Returns
        -------
        dict
            'positions' (list of position dicts), 'alerts' (list),
            'agent_analysis' (str), 'summary' (dict).
        """
        weekday = datetime.now().weekday()
        target_pct = self.DAILY_TARGETS.get(weekday, 25)

        # Regime-adjust the theta model. A flat 4%/day overstates decay in
        # high-VIX regimes (premium inflates, partially offsetting theta) and
        # understates it in low-VIX regimes (calm markets bleed faster).
        vix_current = 0
        if market_summary:
            vix_current = market_summary.get("vix", {}).get("current", 0) or 0
        if vix_current >= 25:
            theta_per_day = 3.0   # high VIX — vega cushion
        elif vix_current >= 20:
            theta_per_day = 3.5
        elif vix_current >= 15:
            theta_per_day = 4.0   # baseline
        elif vix_current > 0:
            theta_per_day = 4.8   # low VIX — faster bleed
        else:
            theta_per_day = 4.0   # no data: baseline

        positions = []
        alerts = []

        for pick in picks:
            ticker = pick.get("ticker", "?")
            direction = pick.get("direction", "call")
            entry_premium = pick.get("premium")
            entry_price = pick.get("current_price")
            strike = pick.get("strike")
            expiry = pick.get("expiry")
            entry_delta = pick.get("estimated_delta", 0.35)

            # Fetch current stock price
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period="1d", interval="5m")
                if hist.empty:
                    hist = tk.history(period="2d")
                if hist.empty:
                    positions.append({
                        "ticker": ticker, "status": "NO_DATA",
                        "detail": "Could not fetch current price",
                    })
                    continue
                current_price = float(hist["Close"].iloc[-1])
            except Exception as exc:
                logger.error("Price fetch failed for %s: %s", ticker, exc)
                positions.append({
                    "ticker": ticker, "status": "ERROR",
                    "detail": str(exc),
                })
                continue

            # Stock move since entry
            if entry_price and entry_price > 0:
                stock_move_pct = ((current_price - entry_price) / entry_price) * 100
                if direction == "put":
                    stock_move_pct = -stock_move_pct
            else:
                stock_move_pct = 0

            days_held = weekday  # Monday=0, so Tuesday=1 day held, etc.

            # --- Live option price lookup ---
            current_option_price = None
            current_delta = None
            live_price_source = "none"

            if strike and expiry:
                try:
                    option_type = "calls" if direction == "call" else "puts"
                    chain = tk.option_chain(expiry)
                    opts_df = getattr(chain, option_type, None)
                    if opts_df is not None and not opts_df.empty:
                        # Find matching strike
                        match = opts_df[opts_df["strike"] == strike]
                        if match.empty:
                            # Nearest strike within $0.50
                            opts_df = opts_df.copy()
                            opts_df["_dist"] = (opts_df["strike"] - strike).abs()
                            nearest = opts_df.loc[opts_df["_dist"].idxmin()]
                            if nearest["_dist"] <= 0.50:
                                match = opts_df[opts_df["strike"] == nearest["strike"]]

                        if not match.empty:
                            row = match.iloc[0]
                            bid = float(row.get("bid", 0))
                            ask = float(row.get("ask", 0))
                            if bid > 0 and ask > 0:
                                current_option_price = round((bid + ask) / 2, 2)
                                live_price_source = "bid_ask_mid"
                            elif float(row.get("lastPrice", 0)) > 0:
                                current_option_price = round(float(row["lastPrice"]), 2)
                                live_price_source = "last_price"
                            # Grab live IV for delta re-estimate
                            live_iv = float(row.get("impliedVolatility", 0))
                            if live_iv > 0.05:
                                current_delta = self._estimate_delta(
                                    current_price, strike, expiry, live_iv, direction,
                                )
                except Exception as exc:
                    logger.debug("Option chain lookup failed for %s: %s", ticker, exc)

            # Compute option P&L from real prices when available
            if current_option_price and entry_premium and entry_premium > 0:
                option_return_pct = ((current_option_price - entry_premium) / entry_premium) * 100
            elif current_option_price and entry_premium:
                # entry_premium is 0 — can't compute %, use absolute
                option_return_pct = 0
            else:
                # Fallback: delta approximation (pre-market or chain lookup failed)
                delta = entry_delta or 0.35
                premium_pct = (entry_premium / entry_price * 100) if entry_premium and entry_price else 3.0
                option_return_pct = (delta * stock_move_pct / premium_pct * 100) if premium_pct > 0 else 0
                if option_return_pct > 0:
                    option_return_pct *= 1.10
                theta_cost = days_held * theta_per_day
                option_return_pct -= theta_cost
                live_price_source = "delta_estimate"

            # If we got live option price but no entry premium, derive it
            if not entry_premium and current_option_price:
                current_option_price_display = current_option_price
            else:
                current_option_price_display = current_option_price

            # Status determination
            status = "HOLD"
            detail = ""

            if option_return_pct >= target_pct:
                status = "TARGET_HIT"
                detail = f"Up {option_return_pct:.0f}% — at or above {target_pct}% day-{days_held+1} target. Consider taking profit."
                alerts.append(f"{ticker}: TARGET HIT ({option_return_pct:.0f}% vs {target_pct}% target)")
            elif option_return_pct <= -50:
                status = "STOP_HIT"
                detail = f"Down {option_return_pct:.0f}% — stop loss triggered at -50%. Close position."
                alerts.append(f"{ticker}: STOP LOSS HIT ({option_return_pct:.0f}%)")
            elif option_return_pct <= -30:
                status = "WARNING"
                detail = f"Down {option_return_pct:.0f}% — approaching stop. Watch closely."
                alerts.append(f"{ticker}: APPROACHING STOP ({option_return_pct:.0f}%)")
            elif urgency == "CRITICAL":
                status = "CLOSE"
                detail = f"Friday expiry — close by 2 PM ET. Current P&L: {option_return_pct:.0f}%"
                alerts.append(f"{ticker}: CLOSE BY 2PM (P&L {option_return_pct:.0f}%)")
            elif urgency == "ELEVATED" and option_return_pct < 0:
                detail = f"Theta accelerating. Down {option_return_pct:.0f}%. Consider closing to limit further decay."
            else:
                detail = f"P&L: {option_return_pct:.0f}% (target: {target_pct}% today). Stock moved {stock_move_pct:+.1f}%."

            positions.append({
                "ticker": ticker,
                "direction": direction,
                "strike": strike,
                "entry_price": entry_price,
                "current_price": round(current_price, 2),
                "entry_premium": entry_premium,
                "current_option_price": current_option_price_display,
                "current_delta": round(current_delta, 3) if current_delta else None,
                "stock_move_pct": round(stock_move_pct, 2),
                "option_return_pct": round(option_return_pct, 1),
                "days_held": days_held,
                "today_target_pct": target_pct,
                "price_source": live_price_source,
                "status": status,
                "detail": detail,
            })

        # Summary stats
        active = [p for p in positions if p.get("status") not in ("NO_DATA", "ERROR")]
        avg_return = sum(p.get("option_return_pct", 0) for p in active) / max(len(active), 1)
        targets_hit = sum(1 for p in active if p["status"] == "TARGET_HIT")
        stops_hit = sum(1 for p in active if p["status"] == "STOP_HIT")

        summary = {
            "total_positions": len(active),
            "avg_return_pct": round(avg_return, 1),
            "targets_hit": targets_hit,
            "stops_hit": stops_hit,
            "urgency": urgency,
            "day": datetime.now().strftime("%A"),
            "days_to_expiry": 5 - weekday,
        }

        # Agent analysis — only for ELEVATED/CRITICAL (Thu/Fri).
        # ROUTINE days (Mon PM, Tue, Wed) use mechanical alerts only.
        # This saves ~4 Opus calls/week without losing decision quality.
        agent_analysis = ""
        if self.enabled and positions and urgency in ("ELEVATED", "CRITICAL"):
            pos_text = "\n\n".join(
                f"{p['ticker']} ({p.get('direction','?').upper()}): "
                f"stock {p.get('stock_move_pct',0):+.1f}%, "
                f"option P&L {p.get('option_return_pct',0):+.0f}%, "
                f"status: {p['status']} — {p.get('detail','')}"
                for p in positions if p.get("status") not in ("NO_DATA", "ERROR")
            )

            market_ctx = self._format_market_context(market_summary) if market_summary else "No market data available."

            user_msg = (
                f"URGENCY: {urgency}\n"
                f"DAY: {summary['day']} (day {weekday+1} of 5, {summary['days_to_expiry']} to expiry)\n\n"
                f"MARKET:\n{market_ctx}\n\n"
                f"POSITIONS:\n{pos_text}\n\n"
                f"Give a brief status summary and ONE clear recommendation."
            )

            agent_analysis = self._call(SYSTEM_PROMPT, user_msg)

        return {
            "positions": positions,
            "alerts": alerts,
            "agent_analysis": agent_analysis,
            "summary": summary,
        }

    @staticmethod
    def _estimate_delta(stock_price, strike, expiry_str, iv, direction):
        """Compute BSM delta for current position state."""
        try:
            from scanners.options import _bsm_greeks, _RISK_FREE_RATE
            expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            T = max((expiry_dt - datetime.now().date()).days, 1) / 365.0
            option_type = "call" if direction == "call" else "put"
            greeks = _bsm_greeks(stock_price, strike, T, _RISK_FREE_RATE, iv, option_type)
            return abs(greeks["delta"])
        except Exception:
            return None
