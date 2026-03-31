"""
Discord notification module for the Zero-DTE Options Trading Analysis System.

Sends formatted messages to a Discord channel via webhook, including
final pick embeds and weekly reflection summaries.
"""

import sys
import os
import json
import logging
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()


class DiscordNotifier:
    """Send notifications to a Discord channel via webhook."""

    def __init__(self):
        """Load webhook URL from config."""
        self.webhook_url = config.discord_webhook_url
        self.enabled = config.has_discord()
        if not self.enabled:
            logger.info("Discord notifications disabled — no webhook configured")

    # ------------------------------------------------------------------
    #  Simple message
    # ------------------------------------------------------------------

    def send_message(self, content: str) -> bool:
        """Send a simple text message to Discord.

        Parameters
        ----------
        content : str
            Plain text message (max 2000 characters for Discord).

        Returns
        -------
        bool
            True if the message was sent successfully.
        """
        if not self.enabled:
            logger.info("Discord disabled — would have sent: %s", content[:100])
            return False

        payload = {"content": content[:2000]}
        return self._post_webhook(payload)

    # ------------------------------------------------------------------
    #  Picks embed
    # ------------------------------------------------------------------

    def send_picks(self, picks: list, market_summary: dict) -> bool:
        """Send formatted embeds with Friday picks including execution guidance.

        Sends two embeds:
        1. The picks with strike, premium, breakeven, entry/exit rules
        2. Best practices for 0DTE execution (always included as a reminder)

        Parameters
        ----------
        picks : list[dict]
            Final pick candidates, each with ticker, direction, strike,
            entry/exit guidance, etc.
        market_summary : dict
            Market context (VIX, regime, breadth, credit, conditions).

        Returns
        -------
        bool
            True if sent successfully.
        """
        if not self.enabled:
            logger.info("Discord disabled — %d picks not sent", len(picks))
            return False

        date_str = datetime.now().strftime("%A %B %d, %Y")

        # Build market context string
        vix_data = market_summary.get("vix", {})
        vix_regime = market_summary.get("vix_regime", {})
        vix_level = vix_data.get("current", "N/A")
        regime = vix_regime.get("regime", "unknown")
        breadth = market_summary.get("breadth", {}).get("breadth_signal", "unknown")
        credit = market_summary.get("credit_spread", {})
        conditions = market_summary.get("financial_conditions", {})

        # Macro edge assessment
        macro_edge = picks[0].get("macro_edge", {}) if picks else {}
        edge_status = "MODEL HAS EDGE" if macro_edge.get("has_edge", True) else "REDUCED EDGE — size down"
        conf_mult = macro_edge.get("confidence_multiplier", 1.0)

        market_context = (
            f"**VIX:** {vix_level} ({regime})\n"
            f"**Breadth:** {breadth}\n"
            f"**Credit Spread:** {credit.get('credit_state', '?')} "
            f"(HY OAS {credit.get('hy_oas', '?')}%)\n"
            f"**Fin. Conditions:** {conditions.get('conditions_state', '?')} "
            f"(NFCI {conditions.get('nfci', '?')})\n"
            f"**Regime Gate:** {edge_status} (conf x{conf_mult:.2f})"
        )

        # Build fields for each pick
        fields = []
        for i, pick in enumerate(picks, 1):
            ticker = pick.get("ticker", "?")
            direction = pick.get("direction", "?").upper()
            strike = pick.get("strike")
            premium = pick.get("premium")
            score = pick.get("composite_score", 0)
            confidence = pick.get("confidence", 0)
            current = pick.get("current_price")
            breakeven = pick.get("breakeven")
            be_move = pick.get("breakeven_move_pct")
            exp_move = pick.get("expected_daily_move_pct")
            delta = pick.get("estimated_delta")

            # Format strike info
            strike_str = f"${strike:,.2f}" if strike else "TBD"
            prem_str = f"${premium:.2f}" if premium else "N/A"
            price_str = f"${current:,.2f}" if current else "?"
            be_str = f"${breakeven:,.2f}" if breakeven else "?"
            be_move_str = f"{be_move:.1f}%" if be_move else "?"
            exp_str = f"{exp_move:.1f}%" if exp_move else "?"
            delta_str = f"{delta:.2f}" if delta else "?"

            # Entry/exit from strike selector
            entry_info = pick.get("entry", {})
            exit_info = pick.get("exit", {})

            # Build one-sentence reasoning
            reasoning = self._build_reasoning(pick)

            value = (
                f"**{direction}** {strike_str} @ {prem_str}\n"
                f"Price: {price_str} | Delta: {delta_str}\n"
                f"Breakeven: {be_str} ({be_move_str} move needed)\n"
                f"Expected daily move: {exp_str}\n"
                f"Score: {score:.1f} | Confidence: {confidence:.0%}\n"
                f"_{reasoning}_"
            )

            fields.append({
                "name": f"{i}. {ticker}",
                "value": value,
                "inline": False,
            })

        # Picks embed
        picks_embed = {
            "title": f"\U0001f3af Zero DTE Picks \u2014 {date_str}",
            "color": 0x00AAFF if macro_edge.get("has_edge", True) else 0xFFAA00,
            "description": market_context,
            "fields": fields,
            "footer": {
                "text": "Experimental \u2014 Not Investment Advice",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Execution best practices embed (always sent as a reminder)
        practices_embed = self._build_execution_guide(picks)

        payload = {"embeds": [picks_embed, practices_embed]}
        return self._post_webhook(payload)

    @staticmethod
    def _build_execution_guide(picks: list) -> dict:
        """Build a Discord embed with 0DTE entry/exit best practices."""
        # Gather exit rules from first pick (they're the same for all)
        exit_info = picks[0].get("exit", {}) if picks else {}
        profit_target = exit_info.get("profit_target_pct", 50)
        stop_loss = exit_info.get("stop_loss_pct", 40)
        time_stop = exit_info.get("time_stop", "12:00 ET")

        guide_text = (
            "**ENTRY**\n"
            "- Wait **15-30 min** after market open for the range to establish\n"
            "- Enter via **limit order at the mid price** or better\n"
            "- Do NOT chase — if the move started without you, **skip it**\n"
            "- Confirm direction with first 15-min candle before entering\n"
            "\n"
            "**EXIT RULES**\n"
            f"- **Take profit** at {profit_target}% gain on the premium\n"
            f"- **Stop loss** at {stop_loss}% loss on the premium\n"
            f"- **Time stop** at {time_stop} — close if neither target hit\n"
            "- Theta accelerates sharply after noon on 0DTE\n"
            "\n"
            "**STRIKE SELECTION**\n"
            "- Strikes are chosen near the money (~0.40-0.45 delta)\n"
            "- Breakeven move should be < expected daily move (ATR)\n"
            "- If breakeven requires a larger move than the expected daily,\n"
            "  the risk/reward is unfavorable — consider skipping\n"
            "\n"
            "**POSITION SIZING**\n"
            "- Max 1-2% of account per trade on 0DTE\n"
            "- If regime gate shows REDUCED EDGE, cut size in half\n"
            "- Never average down on a 0DTE position"
        )

        return {
            "title": "\U0001f4cb 0DTE Execution Guide",
            "color": 0x888888,
            "description": guide_text,
            "footer": {
                "text": "These rules are based on 52-week backtest data",
            },
        }

    # ------------------------------------------------------------------
    #  Weekly reflection
    # ------------------------------------------------------------------

    def send_weekly_reflection(self, reflection: dict) -> bool:
        """Send the weekly reflection summary as a Discord embed.

        Parameters
        ----------
        reflection : dict
            Reflection data from WeeklyReflector.

        Returns
        -------
        bool
            True if sent successfully.
        """
        if not self.enabled:
            logger.info("Discord disabled — reflection not sent")
            return False

        week = reflection.get("week", "unknown")
        win_rate = reflection.get("win_rate", 0)
        total_picks = reflection.get("total_picks", 0)
        wins = reflection.get("wins", 0)

        # Top lessons
        lessons = reflection.get("lessons", [])
        lessons_text = "\n".join(f"- {l}" for l in lessons[:5]) if lessons else "No lessons recorded."

        # Weight changes
        weight_changes = reflection.get("weight_adjustments", {})
        if weight_changes:
            changes_text = "\n".join(
                f"- {k}: {v:+.3f}" for k, v in weight_changes.items()
            )
        else:
            changes_text = "No adjustments."

        embed = {
            "title": f"\U0001f4ca Weekly Reflection \u2014 {week}",
            "color": 0x22CC44 if win_rate >= 0.5 else 0xCC4422,
            "fields": [
                {
                    "name": "Performance",
                    "value": f"Win Rate: **{win_rate:.0%}** ({wins}/{total_picks})",
                    "inline": True,
                },
                {
                    "name": "Avg Return",
                    "value": f"{reflection.get('avg_return', 0):.1%}",
                    "inline": True,
                },
                {
                    "name": "Key Lessons",
                    "value": lessons_text,
                    "inline": False,
                },
                {
                    "name": "Weight Adjustments",
                    "value": changes_text,
                    "inline": False,
                },
            ],
            "footer": {
                "text": "Zero DTE Analysis System",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = {"embeds": [embed]}
        return self._post_webhook(payload)

    # ------------------------------------------------------------------
    #  Weekly scorecard
    # ------------------------------------------------------------------

    def send_scorecard(self, weekly: dict, alltime: dict) -> bool:
        """Send the weekly scorecard results as a Discord embed.

        Parameters
        ----------
        weekly : dict
            This week's graded results from Scorecard.grade_week().
        alltime : dict
            All-time running totals from scorecard data.

        Returns
        -------
        bool
            True if sent successfully.
        """
        if not self.enabled:
            logger.info("Discord disabled — scorecard not sent")
            return False

        pick_date = weekly.get("pick_date", "?")
        expiry = weekly.get("expiry", "?")
        week_pnl = weekly.get("total_pnl", 0)
        week_cost = weekly.get("total_cost", 0)
        week_return = weekly.get("total_return_pct", 0)
        wins = weekly.get("wins", 0)
        losses = weekly.get("losses", 0)
        partials = weekly.get("partials", 0)

        # Color: green if profitable, red if not
        color = 0x22CC44 if week_pnl >= 0 else 0xCC4422

        # Build per-pick results
        pick_lines = []
        for i, p in enumerate(weekly.get("picks", []), 1):
            ticker = p.get("ticker", "?")
            direction = p.get("direction", "?").upper()
            strike = p.get("strike")
            entry = p.get("entry_premium")
            close = p.get("closing_price")
            pnl = p.get("pnl", 0)
            result = p.get("result", "?")

            # Result markers
            if result == "WIN":
                marker = "++"
            elif result == "PARTIAL":
                marker = "+-"
            else:
                marker = "--"

            strike_str = f"${strike:,.0f}" if strike else "N/A"
            entry_str = f"${entry:.2f}" if entry else "N/A"
            close_str = f"${close:,.2f}" if close else "N/A"
            pnl_sign = "+" if pnl >= 0 else ""

            pick_lines.append(
                f"`{marker}` **{ticker}** {direction} @ {strike_str}\n"
                f"Entry: {entry_str} | Close: {close_str} | **{pnl_sign}${pnl:,.2f}**"
            )

        picks_text = "\n\n".join(pick_lines) if pick_lines else "No picks graded."

        # Week summary line
        week_sign = "+" if week_pnl >= 0 else ""
        week_summary = (
            f"**{wins}W - {losses}L - {partials}P**\n"
            f"Invested: ${week_cost:,.2f}\n"
            f"Net P&L: **{week_sign}${week_pnl:,.2f} ({week_sign}{week_return:.1f}%)**"
        )

        # All-time summary
        at_pnl = alltime.get("total_pnl", 0)
        at_sign = "+" if at_pnl >= 0 else ""
        at_wins = alltime.get("wins", 0)
        at_losses = alltime.get("losses", 0)
        at_partials = alltime.get("partials", 0)
        at_weeks = alltime.get("total_weeks", 0)
        at_roi = alltime.get("total_return_pct", 0)

        alltime_text = (
            f"**{at_weeks} weeks** | "
            f"{at_wins}W-{at_losses}L-{at_partials}P | "
            f"**{at_sign}${at_pnl:,.2f} ({at_sign}{at_roi:.1f}%)**"
        )

        embed = {
            "title": f"Weekly Scorecard — {pick_date}",
            "color": color,
            "fields": [
                {
                    "name": "Pick Results",
                    "value": picks_text,
                    "inline": False,
                },
                {
                    "name": "Week Total",
                    "value": week_summary,
                    "inline": True,
                },
                {
                    "name": "All-Time",
                    "value": alltime_text,
                    "inline": True,
                },
            ],
            "footer": {
                "text": f"Expiry: {expiry} | Experimental — Not Investment Advice",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = {"embeds": [embed]}
        return self._post_webhook(payload)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _build_reasoning(self, pick: dict) -> str:
        """Build a one-sentence reasoning string for a pick."""
        parts = []

        tech = pick.get("technical", {})
        rsi = tech.get("rsi")
        if rsi is not None:
            if rsi <= 30:
                parts.append("oversold RSI")
            elif rsi >= 70:
                parts.append("overbought RSI")

        rel_vol = tech.get("relative_volume", 1.0)
        if rel_vol >= 2.0:
            parts.append(f"{rel_vol:.1f}x volume")

        opts = pick.get("options", {})
        iv_rank = opts.get("iv_rank")
        if iv_rank is not None and iv_rank >= 70:
            parts.append(f"IV rank {iv_rank}")

        flow = pick.get("flow", {})
        if flow.get("unusual_volume"):
            parts.append("unusual options flow")

        sent = pick.get("sentiment", {})
        composite = sent.get("composite", 0)
        if composite > 0.3:
            parts.append("bullish sentiment")
        elif composite < -0.3:
            parts.append("bearish sentiment")

        if not parts:
            return "Multi-signal convergence"

        return "Driven by " + ", ".join(parts)

    def _post_webhook(self, payload: dict) -> bool:
        """POST a JSON payload to the Discord webhook URL.

        Parameters
        ----------
        payload : dict
            The Discord webhook payload (with 'content' and/or 'embeds').

        Returns
        -------
        bool
            True if the response indicates success (2xx).
        """
        if not self.webhook_url:
            logger.warning("No webhook URL configured")
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code in (200, 204):
                logger.info("Discord message sent successfully")
                return True
            else:
                logger.warning(
                    "Discord webhook returned %d: %s",
                    response.status_code,
                    response.text[:200],
                )
                return False

        except requests.RequestException as exc:
            logger.error("Discord webhook POST failed: %s", exc)
            return False
