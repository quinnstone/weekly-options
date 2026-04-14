"""
Discord notification module for the Weekly Options Trading Analysis System.

Sends explicit, actionable messages to Discord — every message tells Quinn
exactly what to do, with dollar amounts, strike prices, and exit targets.
"""

import sys
import os
import json
import logging
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, CONTRACTS_PER_TRADE

logger = logging.getLogger(__name__)
config = Config()


class DiscordNotifier:
    """Send notifications to a Discord channel via webhook."""

    def __init__(self):
        self.webhook_url = config.discord_webhook_url
        self.enabled = config.has_discord()
        if not self.enabled:
            logger.info("Discord notifications disabled — no webhook configured")

    # ------------------------------------------------------------------
    #  Simple message
    # ------------------------------------------------------------------

    def send_message(self, content: str) -> bool:
        if not self.enabled:
            logger.info("Discord disabled — would have sent: %s", content[:100])
            return False
        payload = {"content": content[:2000]}
        return self._post_webhook(payload)

    # ------------------------------------------------------------------
    #  Monday Picks — explicit entry instructions
    # ------------------------------------------------------------------

    def send_picks(self, picks: list, market_summary: dict) -> bool:
        if not self.enabled:
            logger.info("Discord disabled — %d picks not sent", len(picks))
            return False

        date_str = datetime.now().strftime("%A %B %d, %Y")
        contracts = CONTRACTS_PER_TRADE

        # Market context
        vix_data = market_summary.get("vix", {})
        vix_regime = market_summary.get("vix_regime", {})
        vix_level = vix_data.get("current", "N/A")
        regime = vix_regime.get("regime", "unknown")
        breadth = market_summary.get("breadth", {}).get("breadth_signal", "unknown")
        credit = market_summary.get("credit_spread", {})
        conditions = market_summary.get("financial_conditions", {})

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
        total_cost = 0

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
            expiry = pick.get("expiry", "Friday")
            earnings_warn = pick.get("earnings_warning", False)

            strike_str = f"${strike:,.2f}" if strike else "TBD"
            prem_str = f"${premium:.2f}" if premium else "N/A"
            price_str = f"${current:,.2f}" if current else "?"
            be_str = f"${breakeven:,.2f}" if breakeven else "?"
            be_move_str = f"{be_move:.1f}%" if be_move else "?"
            exp_str = f"{exp_move:.1f}%" if exp_move else "?"
            delta_str = f"{delta:.2f}" if delta else "?"

            # Dollar amounts
            cost_per_contract = (premium * 100) if premium else 0
            total_entry_cost = cost_per_contract * contracts
            total_cost += total_entry_cost

            # Exit targets in dollars — full weekday ladder
            target_40 = cost_per_contract * 1.40 * contracts if cost_per_contract else 0
            target_35 = cost_per_contract * 1.35 * contracts if cost_per_contract else 0
            target_25 = cost_per_contract * 1.25 * contracts if cost_per_contract else 0
            target_15 = cost_per_contract * 1.15 * contracts if cost_per_contract else 0
            stop_loss_val = cost_per_contract * 0.50 * contracts if cost_per_contract else 0
            profit_40 = target_40 - total_entry_cost
            profit_35 = target_35 - total_entry_cost
            profit_25 = target_25 - total_entry_cost
            profit_15 = target_15 - total_entry_cost
            loss_50 = total_entry_cost - stop_loss_val

            # Strategic entry ladder — walk the limit over 10 min
            entry_mid = premium if premium else 0
            entry_aggressive = round(entry_mid * 1.02, 2) if entry_mid else 0   # +2% above mid
            entry_skip = round(entry_mid * 1.05, 2) if entry_mid else 0          # skip above +5%

            # IV rank context — NOTE: this metric is an IV/RV-ratio proxy, NOT
            # broker-equivalent IV rank. Semantics are inverted from industry
            # standard: HIGHER rank = CHEAPER options relative to realized vol.
            iv_rank = pick.get("options", {}).get("iv_rank")
            if iv_rank is not None:
                if iv_rank > 70:
                    iv_line = f"IV/RV rank: {iv_rank:.0f} — **cheap vs realized vol (potential buying opportunity)**"
                elif iv_rank >= 30:
                    iv_line = f"IV/RV rank: {iv_rank:.0f} — fair value"
                else:
                    iv_line = f"IV/RV rank: {iv_rank:.0f} — **expensive vs realized vol; size down if entering**"
            else:
                iv_line = "IV/RV rank: n/a"

            # Reasoning + social narrative
            reasoning = self._build_reasoning(pick)
            social = pick.get("sentiment", {}).get("social", {})
            social_line = ""
            if social:
                narrative = social.get("narrative", "")
                if narrative and narrative != "Minimal social signal":
                    social_line = f"\n**Social Intel:** {narrative[:180]}"

            earnings_line = "\n**[EARNINGS THIS WEEK]** — IV crush risk, size down" if earnings_warn else ""

            value = (
                f"**ACTION: BUY {contracts} {ticker} {direction} {strike_str} exp {expiry}**\n"
                f"\n"
                f"**Entry ladder** (walk the limit; total cost ~**${total_entry_cost:,.0f}**):\n"
                f"- Start: limit **${entry_mid:.2f}** (mid) — wait 5 min\n"
                f"- Walk to: **${entry_aggressive:.2f}** (mid +2%) — wait 3 min\n"
                f"- Skip if above **${entry_skip:.2f}** (mid +5%) — move on\n"
                f"\n"
                f"Stock price: {price_str} | Delta: {delta_str}\n"
                f"Breakeven: {be_str} (stock needs to move {be_move_str})\n"
                f"Expected weekly move: {exp_str}\n"
                f"{iv_line}\n"
                f"Score: {score:.1f} | Confidence: {confidence:.0%}\n"
                f"\n"
                f"**Exit targets (time-decay ladder):**\n"
                f"Mon-Tue: sell at **${target_40:,.0f}** (+40% = **+${profit_40:,.0f}**)\n"
                f"Wed: sell at **${target_35:,.0f}** (+35% = **+${profit_35:,.0f}**)\n"
                f"Thu: sell at **${target_25:,.0f}** (+25% = **+${profit_25:,.0f}**)\n"
                f"Fri: sell at **${target_15:,.0f}** (+15% = **+${profit_15:,.0f}**)\n"
                f"Stop loss: sell at **${stop_loss_val:,.0f}** (-50% = **-${loss_50:,.0f}**)\n"
                f"\n"
                f"_{reasoning}_"
                f"{social_line}"
                f"{earnings_line}"
            )

            fields.append({
                "name": f"{i}. {ticker} — {direction}",
                "value": value,
                "inline": False,
            })

        # Summary field
        fields.append({
            "name": "Total Capital Required",
            "value": f"**${total_cost:,.0f}** for {len(picks)} positions ({contracts} contract each)",
            "inline": False,
        })

        picks_embed = {
            "title": f"WEEKLY PICKS — {date_str}",
            "color": 0x00AAFF if macro_edge.get("has_edge", True) else 0xFFAA00,
            "description": (
                f"**Wait for the 10:00 AM entry confirmation before placing trades.**\n"
                f"Enter via **limit order at the mid price** or better.\n\n"
                f"{market_context}"
            ),
            "fields": fields,
            "footer": {"text": "Experimental — Not Investment Advice"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Execution guide
        practices_embed = self._build_execution_guide(picks)

        payload = {"embeds": [picks_embed, practices_embed]}
        return self._post_webhook(payload)

    @staticmethod
    def _build_execution_guide(picks: list) -> dict:
        exit_info = picks[0].get("exit", {}) if picks else {}
        stop_loss = exit_info.get("stop_loss_pct", 50)

        guide_text = (
            "**ENTRY (Monday after 10 AM confirmation)**\n"
            "- Wait for the **entry confirmation message** before buying\n"
            "- Enter via **limit order at the mid price** or better\n"
            "- If confirmation says SKIP — do not enter that trade\n"
            "- If confirmation says ADJUST — re-check strike before entering\n"
            "\n"
            "**EXIT RULES (time-decay aware)**\n"
            "- **Mon-Tue:** Take profit at **40%** gain on premium\n"
            "- **Wed:** Take profit at **35%** gain\n"
            "- **Thu:** Take profit at **25%** gain (theta accelerating)\n"
            "- **Fri:** Take profit at **15%** gain\n"
            f"- **Stop loss:** {stop_loss}% loss on premium (any day)\n"
            "- **Hard stop:** Close by **2:00 PM ET Friday** no matter what\n"
            "\n"
            "**WHAT TO DO EACH DAY**\n"
            "- **Mon 10 AM:** Read confirmation message, enter GO trades\n"
            "- **Tue-Thu:** Check Discord for alerts. Close if TARGET or STOP hit\n"
            "- **Fri 1:30 PM:** Close ALL remaining positions before 2 PM\n"
            "- If no alerts, do nothing — hold until target, stop, or Friday"
        )

        return {
            "title": "EXECUTION GUIDE — Read Before Trading",
            "color": 0x888888,
            "description": guide_text,
            "footer": {"text": "Mon entry -> Fri expiry | Max 3 positions | Not Investment Advice"},
        }

    # ------------------------------------------------------------------
    #  Entry Confirmation — GO / SKIP / ADJUST
    # ------------------------------------------------------------------

    def send_entry_confirmations(self, confirmations: list) -> None:
        """Public method for sending entry confirmations from stages."""
        if not self.enabled:
            logger.info("Discord disabled — confirmations not sent")
            return
        self._send_entry_confirmations_impl(confirmations)

    def _send_entry_confirmations_impl(self, confirmations: list) -> None:
        date_str = datetime.now().strftime("%A %B %d, %Y")
        contracts = CONTRACTS_PER_TRADE

        fields = []
        for conf in confirmations:
            ticker = conf.get("ticker", "?")
            signal = conf.get("signal", "?")
            direction = conf.get("direction", "?").upper()
            strike = conf.get("strike")
            detail = conf.get("detail", "")
            current = conf.get("current_price")
            new_premium = conf.get("new_premium")
            agent_brief = conf.get("agent_brief", "")

            strike_str = f"${strike:,.2f}" if strike else "N/A"
            price_str = f"${current:,.2f}" if current else "?"
            delta_str = f"{conf['new_delta']:.2f}" if conf.get("new_delta") else "?"
            prem_str = f"${new_premium:.2f}" if new_premium else "?"

            cost = (new_premium * 100 * contracts) if new_premium else 0

            if signal == "GO":
                action = f"**ENTER NOW:** Buy {contracts} {ticker} {direction} {strike_str} at {prem_str} (${cost:,.0f})"
            elif signal == "ADJUST":
                action = f"**RE-CHECK STRIKE** before entering {ticker} — {detail[:150]}"
            else:
                action = f"**DO NOT ENTER** {ticker} — {detail[:150]}"

            agent_line = f"\n_Agent: {agent_brief[:150]}_" if agent_brief else ""

            value = (
                f"{action}\n"
                f"Stock: {price_str} | Delta: {delta_str} | Premium: {prem_str}"
                f"{agent_line}"
            )

            fields.append({
                "name": f"{'GO' if signal == 'GO' else 'SKIP' if signal == 'SKIP' else 'ADJUST'} — {ticker}",
                "value": value,
                "inline": False,
            })

        go_count = sum(1 for c in confirmations if c["signal"] == "GO")
        total = len(confirmations)
        color = 0x22CC44 if go_count == total else (0xFFAA00 if go_count > 0 else 0xCC4422)

        embed = {
            "title": f"ENTRY CONFIRMATION — {date_str} (10:00 AM ET)",
            "color": color,
            "description": (
                f"**{go_count}/{total} picks confirmed for entry.**\n"
                f"Place limit orders for GO picks now. Skip anything marked SKIP."
            ),
            "fields": fields,
            "footer": {"text": "Enter via limit order at mid price | Not Investment Advice"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._post_webhook({"embeds": [embed]})

    # ------------------------------------------------------------------
    #  Position Monitor — explicit hold/close instructions
    # ------------------------------------------------------------------

    def send_monitor_update(self, result: dict, urgency: str) -> None:
        """Public method for sending monitor updates from stages."""
        if not self.enabled:
            logger.info("Discord disabled — monitor update not sent")
            return
        self._send_monitor_update_impl(result, urgency)

    def _send_monitor_update_impl(self, result: dict, urgency: str) -> None:
        positions = result.get("positions", [])
        alerts = result.get("alerts", [])
        summary = result.get("summary", {})
        agent_analysis = result.get("agent_analysis", "")
        contracts = CONTRACTS_PER_TRADE

        if not positions:
            return

        urgency_colors = {"ROUTINE": 0x3498DB, "ELEVATED": 0xFFAA00, "CRITICAL": 0xCC4422}
        color = urgency_colors.get(urgency, 0x3498DB)

        day = summary.get("day", datetime.now().strftime("%A"))
        dte = summary.get("days_to_expiry", "?")

        # Day-specific profit target
        weekday = datetime.now().weekday()
        if weekday <= 1:
            target_pct = 40
        elif weekday == 2:
            target_pct = 35
        elif weekday == 3:
            target_pct = 25
        else:
            target_pct = 15

        fields = []
        total_pnl = 0

        for p in positions:
            ticker = p.get("ticker", "?")

            if p.get("status") in ("NO_DATA", "ERROR"):
                fields.append({
                    "name": ticker,
                    "value": f"_{p.get('detail', 'No data available')}_",
                    "inline": False,
                })
                continue

            status = p.get("status", "HOLD")
            ret = p.get("option_return_pct", 0)
            stock_move = p.get("stock_move_pct", 0)
            direction = p.get("direction", "?").upper()
            entry_premium = p.get("entry_premium") or 0
            current_option = p.get("current_option_price") or 0
            has_dollars = bool(entry_premium)

            # Dollar P&L (only meaningful when we have entry premium)
            entry_cost = entry_premium * 100 * contracts
            current_val = current_option * 100 * contracts
            dollar_pnl = current_val - entry_cost
            if has_dollars:
                total_pnl += dollar_pnl

            # Target and stop in dollars
            target_val = entry_cost * (1 + target_pct / 100)
            stop_val = entry_cost * 0.50

            # Explicit action instruction — no percentage here (it's on the next line)
            if status == "TARGET_HIT":
                action = (f"**CLOSE NOW — TARGET HIT.** Sell for ~${current_val:,.0f} (+${dollar_pnl:,.0f})"
                          if has_dollars else "**CLOSE NOW — TARGET HIT.** Sell at market or mid limit.")
            elif status == "STOP_HIT":
                action = (f"**CLOSE NOW — STOP HIT.** Sell to limit loss at ~${current_val:,.0f} ({'-' if dollar_pnl < 0 else '+'}${abs(dollar_pnl):,.0f})"
                          if has_dollars else "**CLOSE NOW — STOP HIT.** Cut the loss at market.")
            elif status == "CLOSE":
                action = (f"**CLOSE NOW.** Sell at market if needed. Current value ~${current_val:,.0f}"
                          if has_dollars else "**CLOSE NOW.** Sell at market if needed.")
            elif status == "WARNING":
                action = (f"**WATCH CLOSELY.** Approaching stop. Current value ~${current_val:,.0f}"
                          if has_dollars else "**WATCH CLOSELY.** Approaching -50% stop.")
            else:
                action = (f"**HOLD.** Target: ${target_val:,.0f} (+{target_pct}%) | Stop: ${stop_val:,.0f} (-50%)"
                          if has_dollars else f"**HOLD.** Target +{target_pct}% | Stop -50%")

            pnl_sign = "+" if dollar_pnl >= 0 else ""

            if has_dollars:
                # Dollar line already includes %, so middle line only shows stock
                middle_line = f"{direction} | Stock: {stock_move:+.1f}%"
                pnl_line = f"Entry: ${entry_cost:,.0f} | Now: ${current_val:,.0f} | **P&L: {pnl_sign}${dollar_pnl:,.0f}** ({ret:+.0f}%)"
            else:
                # No dollars → middle line carries the option %
                middle_line = f"{direction} | Stock: {stock_move:+.1f}% | **Option P&L: {ret:+.0f}%**"
                pnl_line = "_Entry premium unavailable — dollar P&L skipped_"

            value = f"{action}\n{middle_line}\n{pnl_line}"

            fields.append({
                "name": f"{'!!' if status in ('TARGET_HIT', 'STOP_HIT', 'CLOSE') else '..'} {ticker}",
                "value": value,
                "inline": False,
            })

        # Portfolio total
        pnl_sign = "+" if total_pnl >= 0 else ""
        fields.append({
            "name": "Portfolio Total",
            "value": f"**{pnl_sign}${total_pnl:,.0f}** across {len([p for p in positions if p.get('status') not in ('NO_DATA', 'ERROR')])} positions | DTE: {dte}",
            "inline": False,
        })

        # Alert summary
        if alerts:
            alert_text = "\n".join(f"- {a}" for a in alerts)
            fields.append({
                "name": "Alerts",
                "value": alert_text,
                "inline": False,
            })

        # Agent analysis
        if agent_analysis:
            truncated = agent_analysis[:900] + "..." if len(agent_analysis) > 900 else agent_analysis
            fields.append({
                "name": "Agent Analysis",
                "value": truncated,
                "inline": False,
            })

        what_to_do = {
            "ROUTINE": f"No action needed unless an alert says CLOSE. Today's target: +{target_pct}% on premium.",
            "ELEVATED": f"Theta is accelerating. Close any position that hits +{target_pct}% today. Tighten mental stops.",
            "CRITICAL": "**Close ALL positions by 2:00 PM ET.** Use market orders if limits aren't filling.",
        }

        embed = {
            "title": f"POSITION UPDATE — {day} [{urgency}]",
            "color": color,
            "description": what_to_do.get(urgency, ""),
            "fields": fields,
            "footer": {"text": "Weekly Options Monitor | Not Investment Advice"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._post_webhook({"embeds": [embed]})

    # ------------------------------------------------------------------
    #  Final Exit — close everything
    # ------------------------------------------------------------------

    def send_final_exit(self, result: dict) -> None:
        """Public method for sending final exit from stages."""
        if not self.enabled:
            logger.info("Discord disabled — final exit not sent")
            return
        self._send_final_exit_impl(result)

    def _send_final_exit_impl(self, result: dict) -> None:
        positions = result.get("positions", [])
        contracts = CONTRACTS_PER_TRADE

        if not positions:
            return

        fields = []
        total_pnl = 0

        for p in positions:
            if p.get("status") in ("NO_DATA", "ERROR"):
                continue

            ticker = p.get("ticker", "?")
            ret = p.get("option_return_pct", 0)
            direction = p.get("direction", "?").upper()
            current = p.get("current_price", "?")
            strike = p.get("strike", "?")
            entry_premium = p.get("entry_premium") or 0
            current_option = p.get("current_option_price") or 0
            has_dollars = bool(entry_premium)

            entry_cost = entry_premium * 100 * contracts
            current_val = current_option * 100 * contracts
            dollar_pnl = current_val - entry_cost
            if has_dollars:
                total_pnl += dollar_pnl

            pnl_sign = "+" if dollar_pnl >= 0 else ""

            if has_dollars:
                pnl_line = f"Stock: ${current} | Option P&L: {ret:+.0f}% (**{pnl_sign}${dollar_pnl:,.0f}**)"
            else:
                pnl_line = f"Stock: ${current} | Option P&L: {ret:+.0f}%"

            value = (
                f"**CLOSE NOW** — Sell {contracts} {direction} ${strike} at market\n"
                f"{pnl_line}\n"
                f"_Use market order if limit doesn't fill in 5 min. Do NOT hold past 2 PM._"
            )

            fields.append({
                "name": f"!! {ticker}",
                "value": value,
                "inline": False,
            })

        pnl_sign = "+" if total_pnl >= 0 else ""
        agent_analysis = result.get("agent_analysis", "")
        if agent_analysis:
            truncated = agent_analysis[:900] + "..." if len(agent_analysis) > 900 else agent_analysis
            fields.append({
                "name": "Agent Analysis",
                "value": truncated,
                "inline": False,
            })

        embed = {
            "title": "FINAL EXIT — Close ALL Positions by 2:00 PM ET",
            "color": 0xFF0000,
            "description": (
                f"**{len(positions)} positions must be closed NOW.**\n"
                f"Estimated week P&L: **{pnl_sign}${total_pnl:,.0f}**\n"
                f"Weekly options expire today — use market orders if needed."
            ),
            "fields": fields,
            "footer": {"text": "MANDATORY EXIT | Not Investment Advice"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._post_webhook({"embeds": [embed]})

    # ------------------------------------------------------------------
    #  Weekly Scorecard — dollar P&L breakdown
    # ------------------------------------------------------------------

    def send_scorecard(self, weekly: dict, alltime: dict) -> bool:
        if not self.enabled:
            logger.info("Discord disabled — scorecard not sent")
            return False

        contracts = CONTRACTS_PER_TRADE
        pick_date = weekly.get("pick_date", "?")
        expiry = weekly.get("expiry", "?")
        week_pnl = weekly.get("total_pnl", 0)
        week_cost = weekly.get("total_cost", 0)
        week_return = weekly.get("total_return_pct", 0)
        wins = weekly.get("wins", 0)
        losses = weekly.get("losses", 0)
        partials = weekly.get("partials", 0)

        color = 0x22CC44 if week_pnl >= 0 else 0xCC4422

        pick_lines = []
        for i, p in enumerate(weekly.get("picks", []), 1):
            ticker = p.get("ticker", "?")
            direction = p.get("direction", "?").upper()
            strike = p.get("strike")
            entry = p.get("entry_premium")
            close = p.get("closing_price")
            pnl = p.get("pnl", 0)
            result = p.get("result", "?")

            if result == "WIN":
                marker = "++"
            elif result == "PARTIAL":
                marker = "+-"
            else:
                marker = "--"

            strike_str = f"${strike:,.0f}" if strike else "N/A"
            entry_str = f"${entry:.2f}" if entry else "N/A"
            close_str = f"${close:.2f}" if close else "N/A"

            # Dollar amounts
            entry_cost = (entry * 100 * contracts) if entry else 0
            close_val = (close * 100 * contracts) if close else 0
            dollar_pnl = close_val - entry_cost
            pnl_sign = "+" if dollar_pnl >= 0 else ""

            pick_lines.append(
                f"`{marker}` **{ticker}** {direction} @ {strike_str}\n"
                f"Bought: {entry_str} (${entry_cost:,.0f}) | Sold: {close_str} (${close_val:,.0f})\n"
                f"**P&L: {pnl_sign}${dollar_pnl:,.0f}** ({pnl_sign}{((dollar_pnl / entry_cost * 100) if entry_cost else 0):,.0f}%)"
            )

        picks_text = "\n\n".join(pick_lines) if pick_lines else "No picks graded."

        week_sign = "+" if week_pnl >= 0 else ""
        week_summary = (
            f"**{wins}W - {losses}L - {partials}P**\n"
            f"Invested: ${week_cost:,.2f}\n"
            f"Net P&L: **{week_sign}${week_pnl:,.2f} ({week_sign}{week_return:.1f}%)**"
        )

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
            "title": f"WEEKLY SCORECARD — {pick_date}",
            "color": color,
            "fields": [
                {"name": "Pick Results", "value": picks_text, "inline": False},
                {"name": "Week Total", "value": week_summary, "inline": True},
                {"name": "All-Time", "value": alltime_text, "inline": True},
            ],
            "footer": {"text": f"Expiry: {expiry} | Not Investment Advice"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = {"embeds": [embed]}
        return self._post_webhook(payload)

    # ------------------------------------------------------------------
    #  Weekly Reflection
    # ------------------------------------------------------------------

    def send_weekly_reflection(self, reflection: dict) -> bool:
        if not self.enabled:
            logger.info("Discord disabled — reflection not sent")
            return False

        week = reflection.get("week", "unknown")
        win_rate = reflection.get("win_rate", 0)
        total_picks = reflection.get("total_picks", 0)
        wins = reflection.get("wins", 0)

        lessons = reflection.get("lessons", [])
        lessons_text = "\n".join(f"- {l}" for l in lessons[:5]) if lessons else "No lessons recorded."

        weight_changes = reflection.get("weight_adjustments", {})
        if weight_changes:
            changes_text = "\n".join(f"- {k}: {v:+.3f}" for k, v in weight_changes.items())
        else:
            changes_text = "No adjustments."

        embed = {
            "title": f"WEEKLY REFLECTION — {week}",
            "color": 0x22CC44 if win_rate >= 0.5 else 0xCC4422,
            "fields": [
                {"name": "Performance", "value": f"Win Rate: **{win_rate:.0%}** ({wins}/{total_picks})", "inline": True},
                {"name": "Avg Return", "value": f"{reflection.get('avg_return', 0):.1%}", "inline": True},
                {"name": "Key Lessons", "value": lessons_text, "inline": False},
                {"name": "Weight Adjustments", "value": changes_text, "inline": False},
                {"name": "What This Means", "value": (
                    "These weight adjustments are applied automatically to next week's scoring. "
                    "Positive = signal worked well, getting more weight. "
                    "Negative = signal underperformed, reduced."
                ), "inline": False},
            ],
            "footer": {"text": "Weekly Options Analysis System"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        payload = {"embeds": [embed]}
        return self._post_webhook(payload)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _build_reasoning(self, pick: dict) -> str:
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

        social = sent.get("social", {})
        if social:
            catalysts = social.get("catalysts", [])
            if catalysts:
                parts.append(f"catalysts: {', '.join(catalysts[:2])}")
            flow_consensus = social.get("flow_consensus", "neutral")
            if flow_consensus != "neutral" and social.get("flow_conviction", 0) > 0.3:
                parts.append(f"{flow_consensus} institutional flow")
            risks = social.get("risks", [])
            if risks:
                parts.append(f"risks: {', '.join(risks[:2])}")

        if not parts:
            return "Multi-signal convergence"

        return "Driven by " + ", ".join(parts)

    def _post_webhook(self, payload: dict) -> bool:
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
