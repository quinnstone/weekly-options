"""
Scorecard for the Weekly Options Trading Analysis System.

Tracks hypothetical P&L assuming one contract purchased per pick.
Maintains weekly results (Mon entry → Fri expiry) and an all-time
running total, rendered as a markdown file committed to the repo.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()

SCORECARD_PATH = config.performance_dir / "scorecard.md"
SCORECARD_DATA_PATH = config.performance_dir / "scorecard_data.json"


class Scorecard:
    """Track hypothetical 1-contract P&L for each week's picks."""

    def __init__(self):
        self.data = self._load_data()
        from tracking.database import Database
        self.db = Database()

    # ------------------------------------------------------------------
    #  Core: grade a week's picks
    # ------------------------------------------------------------------

    def grade_week(self, pick_date: str) -> dict:
        """Grade a week's picks by comparing entry premium to expiry outcome.

        Parameters
        ----------
        pick_date : str
            The date the picks were generated (YYYY-MM-DD).

        Returns
        -------
        dict
            Weekly result with per-pick P&L and summary stats.
        """
        # Load picks from the report file (has strike/premium/expiry)
        report_path = config.reports_dir / f"{pick_date}_picks.json"
        if not report_path.exists():
            logger.error("No report found for %s", pick_date)
            return {}

        with open(report_path) as fh:
            report = json.load(fh)

        picks = report.get("picks", [])
        if not picks:
            logger.warning("No picks in report for %s", pick_date)
            return {}

        # Determine expiry date from picks
        expiry = picks[0].get("expiry")
        if not expiry:
            logger.error("No expiry date in picks for %s", pick_date)
            return {}

        # Grade each pick
        graded = []
        for pick in picks:
            result = self._grade_pick(pick, expiry)
            graded.append(result)

        # Weekly summary
        total_cost = sum(r["entry_cost"] for r in graded)
        total_exit = sum(r["exit_value_total"] for r in graded)
        total_pnl = sum(r["pnl"] for r in graded)
        wins = sum(1 for r in graded if r["result"] == "WIN")
        losses = sum(1 for r in graded if r["result"] == "LOSS")
        partials = sum(1 for r in graded if r["result"] == "PARTIAL")

        weekly = {
            "pick_date": pick_date,
            "expiry": expiry,
            "picks": graded,
            "total_cost": round(total_cost, 2),
            "total_exit": round(total_exit, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((total_pnl / total_cost * 100) if total_cost > 0 else 0, 2),
            "wins": wins,
            "losses": losses,
            "partials": partials,
            "win_rate": round(wins / len(graded), 4) if graded else 0,
            "graded_at": datetime.now().isoformat(),
        }

        # Save to running data (JSON + DB)
        self._save_week(weekly)
        self._render_scorecard()

        # Save to database
        try:
            self.db.grade_picks(pick_date, graded)
            self.db.record_weekly_result(weekly)
        except Exception as exc:
            logger.error("Failed to save to database: %s", exc)

        # Run post-mortem analysis (agent-powered)
        try:
            from agents.post_mortem import PostMortemAgent
            pm = PostMortemAgent()
            if pm.enabled and picks and graded:
                # Load market summary for context
                market_summary = None
                try:
                    from config import Config as _Cfg
                    _cfg = _Cfg()
                    summary_path = _cfg.candidates_dir / pick_date / "market_summary.json"
                    if summary_path.exists():
                        import json as _json
                        with open(summary_path) as fh:
                            market_summary = _json.load(fh)
                except Exception:
                    pass

                post_mortems = pm.analyze_week(picks, graded, market_summary)
                if post_mortems:
                    # Attach post-mortems to weekly data
                    weekly["post_mortems"] = post_mortems
                    pm_map = {p["ticker"]: p["post_mortem"] for p in post_mortems}
                    for g in graded:
                        pm_text = pm_map.get(g.get("ticker"))
                        if pm_text:
                            g["post_mortem"] = pm_text
                    logger.info("Post-mortems generated for %d picks", len(post_mortems))
        except Exception as exc:
            logger.error("Post-mortem agent failed: %s", exc)

        # Link agent decisions to outcomes for audit tracking
        try:
            from tracking.agent_tracker import link_outcomes
            week_str = datetime.now().strftime("%Y-W%W")
            outcome_map = {
                g.get("ticker"): {
                    "result": g.get("result", "unknown"),
                    "pnl": g.get("pnl", 0),
                }
                for g in graded if g.get("ticker")
            }
            link_outcomes(week_str, outcome_map)
        except Exception as exc:
            logger.error("Failed to link agent outcomes: %s", exc)

        # Send to Discord
        try:
            from notifications.discord import DiscordNotifier
            notifier = DiscordNotifier()
            notifier.send_scorecard(weekly, self.data["all_time"])
        except Exception as exc:
            logger.error("Failed to send scorecard to Discord: %s", exc)

        return weekly

    def _grade_pick(self, pick: dict, expiry: str) -> dict:
        """Grade a single pick against actual closing price on expiry.

        Returns dict with entry/exit values and P&L.
        """
        ticker = pick.get("ticker", "???")
        direction = pick.get("direction", "call")
        strike = pick.get("strike")
        premium = pick.get("premium")

        # Default result for missing data
        result = {
            "ticker": ticker,
            "direction": direction,
            "strike": strike,
            "entry_premium": premium,
            "entry_cost": (premium or 0) * 100,
            "closing_price": None,
            "exit_value_per_share": 0,
            "exit_value_total": 0,
            "pnl": -(premium or 0) * 100,
            "pnl_pct": -100.0,
            "result": "LOSS",
        }

        if not strike or not premium:
            result["note"] = "Missing strike or premium data"
            return result

        # Fetch closing price on expiry date
        closing_price = self._get_closing_price(ticker, expiry)
        if closing_price is None:
            result["note"] = "Could not fetch closing price (market may not have closed yet)"
            return result

        # Calculate intrinsic value at expiry
        if direction == "call":
            intrinsic = max(0, closing_price - strike)
        else:  # put
            intrinsic = max(0, strike - closing_price)

        exit_total = intrinsic * 100
        entry_cost = premium * 100
        pnl = exit_total - entry_cost
        pnl_pct = (pnl / entry_cost * 100) if entry_cost > 0 else 0

        # Classify result
        if pnl > 0:
            label = "WIN"
        elif intrinsic > 0 and pnl < 0:
            label = "PARTIAL"  # ITM but didn't cover premium
        else:
            label = "LOSS"

        result.update({
            "closing_price": round(closing_price, 2),
            "exit_value_per_share": round(intrinsic, 2),
            "exit_value_total": round(exit_total, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "result": label,
        })

        return result

    def _get_closing_price(self, ticker: str, date_str: str) -> "float | None":
        """Fetch the closing price for a ticker on a specific date."""
        try:
            expiry_date = datetime.strptime(date_str, "%Y-%m-%d")
            # Fetch a small window around the expiry date
            start = (expiry_date - timedelta(days=1)).strftime("%Y-%m-%d")
            end = (expiry_date + timedelta(days=1)).strftime("%Y-%m-%d")

            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                return None

            # Get the closing price (handle MultiIndex from yfinance)
            if isinstance(data.columns, __import__('pandas').MultiIndex):
                close_col = data["Close"]
                if isinstance(close_col, __import__('pandas').DataFrame):
                    close_col = close_col.iloc[:, 0]
            else:
                close_col = data["Close"]

            return float(close_col.iloc[-1])
        except Exception as exc:
            logger.error("Failed to get closing price for %s on %s: %s", ticker, date_str, exc)
            return None

    # ------------------------------------------------------------------
    #  Data persistence
    # ------------------------------------------------------------------

    def _load_data(self) -> dict:
        """Load the running scorecard data."""
        if SCORECARD_DATA_PATH.exists():
            with open(SCORECARD_DATA_PATH) as fh:
                return json.load(fh)
        return {"weeks": [], "all_time": self._empty_alltime()}

    def _save_week(self, weekly: dict):
        """Add or update a week in the running data."""
        # Remove existing entry for this date if re-grading
        self.data["weeks"] = [
            w for w in self.data["weeks"]
            if w["pick_date"] != weekly["pick_date"]
        ]
        self.data["weeks"].append(weekly)
        self.data["weeks"].sort(key=lambda w: w["pick_date"])

        # Recompute all-time totals
        self._recompute_alltime()

        # Save
        with open(SCORECARD_DATA_PATH, "w") as fh:
            json.dump(self.data, fh, indent=2, default=str)

    def _recompute_alltime(self):
        """Recompute all-time stats from all weeks."""
        total_picks = 0
        total_wins = 0
        total_losses = 0
        total_partials = 0
        total_invested = 0
        total_returned = 0
        total_pnl = 0
        best_pick = None
        worst_pick = None
        best_pnl = float("-inf")
        worst_pnl = float("inf")

        for week in self.data["weeks"]:
            for pick in week.get("picks", []):
                total_picks += 1
                total_invested += pick.get("entry_cost", 0)
                total_returned += pick.get("exit_value_total", 0)
                total_pnl += pick.get("pnl", 0)

                if pick["result"] == "WIN":
                    total_wins += 1
                elif pick["result"] == "LOSS":
                    total_losses += 1
                else:
                    total_partials += 1

                if pick["pnl"] > best_pnl:
                    best_pnl = pick["pnl"]
                    best_pick = {
                        "ticker": pick["ticker"],
                        "date": week["pick_date"],
                        "pnl": pick["pnl"],
                        "pnl_pct": pick["pnl_pct"],
                    }
                if pick["pnl"] < worst_pnl:
                    worst_pnl = pick["pnl"]
                    worst_pick = {
                        "ticker": pick["ticker"],
                        "date": week["pick_date"],
                        "pnl": pick["pnl"],
                        "pnl_pct": pick["pnl_pct"],
                    }

        self.data["all_time"] = {
            "total_weeks": len(self.data["weeks"]),
            "total_picks": total_picks,
            "wins": total_wins,
            "losses": total_losses,
            "partials": total_partials,
            "win_rate": round(total_wins / total_picks, 4) if total_picks > 0 else 0,
            "total_invested": round(total_invested, 2),
            "total_returned": round(total_returned, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((total_pnl / total_invested * 100) if total_invested > 0 else 0, 2),
            "best_pick": best_pick,
            "worst_pick": worst_pick,
        }

    def _empty_alltime(self) -> dict:
        return {
            "total_weeks": 0,
            "total_picks": 0,
            "wins": 0,
            "losses": 0,
            "partials": 0,
            "win_rate": 0,
            "total_invested": 0,
            "total_returned": 0,
            "total_pnl": 0,
            "total_return_pct": 0,
            "best_pick": None,
            "worst_pick": None,
        }

    # ------------------------------------------------------------------
    #  Markdown rendering
    # ------------------------------------------------------------------

    def _render_scorecard(self):
        """Render the full scorecard as markdown and save to repo."""
        lines = []
        at = self.data.get("all_time", {})

        lines.append("# Weekly Options Scorecard")
        lines.append("")
        lines.append("Hypothetical results assuming 1 contract purchased per pick.")
        lines.append("")

        # All-time summary box
        lines.append("## All-Time Performance")
        lines.append("")
        pnl = at.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Weeks Tracked | {at.get('total_weeks', 0)} |")
        lines.append(f"| Total Picks | {at.get('total_picks', 0)} |")
        lines.append(f"| Record | {at.get('wins', 0)}W - {at.get('losses', 0)}L - {at.get('partials', 0)}P |")
        lines.append(f"| Win Rate | {at.get('win_rate', 0):.0%} |")
        lines.append(f"| Total Invested | ${at.get('total_invested', 0):,.2f} |")
        lines.append(f"| Total Returned | ${at.get('total_returned', 0):,.2f} |")
        lines.append(f"| **Net P&L** | **{pnl_sign}${pnl:,.2f}** |")
        lines.append(f"| **ROI** | **{pnl_sign}{at.get('total_return_pct', 0):.2f}%** |")
        lines.append("")

        # Best/worst picks
        best = at.get("best_pick")
        worst = at.get("worst_pick")
        if best or worst:
            lines.append("| | Ticker | Date | P&L |")
            lines.append("|--|--------|------|-----|")
            if best:
                lines.append(f"| Best | {best['ticker']} | {best['date']} | +${best['pnl']:,.2f} ({best['pnl_pct']:+.1f}%) |")
            if worst:
                lines.append(f"| Worst | {worst['ticker']} | {worst['date']} | ${worst['pnl']:,.2f} ({worst['pnl_pct']:+.1f}%) |")
            lines.append("")

        # Weekly breakdown (newest first)
        lines.append("---")
        lines.append("")
        lines.append("## Weekly Results")
        lines.append("")

        for week in reversed(self.data.get("weeks", [])):
            week_pnl = week.get("total_pnl", 0)
            week_sign = "+" if week_pnl >= 0 else ""
            lines.append(f"### Week of {week['pick_date']} (exp. {week.get('expiry', 'N/A')})")
            lines.append("")
            lines.append(
                f"**{week.get('wins', 0)}W-{week.get('losses', 0)}L-{week.get('partials', 0)}P** | "
                f"Cost: ${week.get('total_cost', 0):,.2f} | "
                f"Return: ${week.get('total_exit', 0):,.2f} | "
                f"**P&L: {week_sign}${week_pnl:,.2f} ({week_sign}{week.get('total_return_pct', 0):.1f}%)**"
            )
            lines.append("")

            # Pick table
            lines.append("| # | Ticker | Dir | Strike | Entry | Close | Exit Val | P&L | Result |")
            lines.append("|---|--------|-----|--------|-------|-------|----------|-----|--------|")

            for i, pick in enumerate(week.get("picks", []), 1):
                dir_label = pick["direction"].upper()
                strike = f"${pick['strike']:,.2f}" if pick.get("strike") else "N/A"
                entry = f"${pick['entry_premium']:.2f}" if pick.get("entry_premium") else "N/A"
                close = f"${pick['closing_price']:,.2f}" if pick.get("closing_price") else "N/A"
                exit_val = f"${pick['exit_value_per_share']:.2f}" if pick.get("closing_price") else "N/A"
                pnl_val = pick.get("pnl", 0)
                pnl_str = f"+${pnl_val:,.2f}" if pnl_val >= 0 else f"-${abs(pnl_val):,.2f}"
                result = pick.get("result", "?")

                # Result emoji-free markers
                if result == "WIN":
                    marker = "WIN"
                elif result == "PARTIAL":
                    marker = "PARTIAL"
                else:
                    marker = "LOSS"

                lines.append(
                    f"| {i} | {pick['ticker']} | {dir_label} | {strike} | "
                    f"{entry} | {close} | {exit_val} | {pnl_str} | {marker} |"
                )

            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}*")

        with open(SCORECARD_PATH, "w") as fh:
            fh.write("\n".join(lines))

        logger.info("Scorecard rendered to %s", SCORECARD_PATH)

    # ------------------------------------------------------------------
    #  CLI display
    # ------------------------------------------------------------------

    def show(self):
        """Print scorecard summary to stdout."""
        at = self.data.get("all_time", {})

        if at.get("total_picks", 0) == 0:
            print("No graded picks yet. Run: python3 main.py scorecard --week YYYY-MM-DD")
            return

        pnl = at.get("total_pnl", 0)
        sign = "+" if pnl >= 0 else ""

        print()
        print("=" * 60)
        print("  WEEKLY OPTIONS SCORECARD — ALL TIME")
        print("=" * 60)
        print(f"  Weeks:     {at.get('total_weeks', 0)}")
        print(f"  Picks:     {at.get('total_picks', 0)}")
        print(f"  Record:    {at.get('wins', 0)}W - {at.get('losses', 0)}L - {at.get('partials', 0)}P")
        print(f"  Win Rate:  {at.get('win_rate', 0):.0%}")
        print(f"  Invested:  ${at.get('total_invested', 0):,.2f}")
        print(f"  Returned:  ${at.get('total_returned', 0):,.2f}")
        print(f"  Net P&L:   {sign}${pnl:,.2f}")
        print(f"  ROI:       {sign}{at.get('total_return_pct', 0):.2f}%")
        print()

        # Show most recent week
        weeks = self.data.get("weeks", [])
        if weeks:
            latest = weeks[-1]
            week_pnl = latest.get("total_pnl", 0)
            ws = "+" if week_pnl >= 0 else ""
            print(f"  Latest week: {latest['pick_date']}")
            print(f"  {latest.get('wins', 0)}W-{latest.get('losses', 0)}L-{latest.get('partials', 0)}P  "
                  f"P&L: {ws}${week_pnl:,.2f} ({ws}{latest.get('total_return_pct', 0):.1f}%)")
            print()
            for i, p in enumerate(latest.get("picks", []), 1):
                pnl_val = p.get("pnl") or 0
                ps = "+" if pnl_val >= 0 else ""
                ticker = p.get("ticker") or "?"
                direction = (p.get("direction") or "?").upper()
                strike = p.get("strike")
                strike_str = f"${strike:>8,.2f}" if strike is not None else "       —"
                print(f"    {i}. {ticker:6s} {direction:4s}  "
                      f"Strike: {strike_str}  "
                      f"{ps}${pnl_val:,.2f}  "
                      f"{p.get('result', '?')}")
            print()
        print("=" * 60)
