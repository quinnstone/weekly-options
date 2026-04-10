"""
Dynamic state generator for agent context.

Reads live system state (weights, patterns, calibration, recent reflections,
trade log) and generates CURRENT_STATE.md and TRADE_LOG.md so agents always
have up-to-date context about how the system is actually performing — not
just the static methodology rules.

Called before agent runs and during the reflection flow.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)
config = Config()

_AGENTS_DIR = Path(__file__).parent


def generate_current_state() -> str:
    """Generate CURRENT_STATE.md from live data files.

    Reads:
    - weights.json — current scoring weights (may differ from defaults)
    - patterns.json — pattern library stats
    - confidence_calibration.json — live calibration data
    - Most recent *_reflection.json — last week's lessons
    - scorecard_data.json — all-time performance stats

    Returns the markdown content and writes to agents/CURRENT_STATE.md.
    """
    sections = ["# Current System State\n\n_Auto-generated from live data. "
                 f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}_\n"]

    # 1. Current weights
    weights_path = config.performance_dir / "weights.json"
    try:
        if weights_path.exists():
            weights = json.loads(weights_path.read_text())
            sections.append("## Live Scoring Weights\n")
            sections.append("These are the CURRENT weights (may differ from defaults "
                            "in METHODOLOGY.md if learning has updated them):\n")
            sections.append("| Signal | Weight | Default | Delta |")
            sections.append("|--------|--------|---------|-------|")

            defaults = {
                "momentum": 0.20, "mean_reversion": 0.15, "regime_bias": 0.10,
                "trend_persistence": 0.15, "iv_mispricing": 0.10, "flow_conviction": 0.08,
                "event_risk": 0.07, "liquidity": 0.05, "strike_efficiency": 0.05,
                "theta_cost": 0.05,
            }
            for signal, weight in sorted(weights.items(), key=lambda x: -x[1]):
                default = defaults.get(signal, 0)
                delta = weight - default
                marker = ""
                if abs(delta) > 0.005:
                    marker = f"{delta:+.3f}"
                sections.append(f"| {signal} | {weight:.3f} | {default:.3f} | {marker} |")
            sections.append("")
        else:
            sections.append("## Live Scoring Weights\n\nUsing defaults (no weights.json yet).\n")
    except Exception as exc:
        logger.warning("Failed to load weights: %s", exc)

    # 2. Pattern library summary
    patterns_path = config.performance_dir / "patterns.json"
    try:
        if patterns_path.exists():
            patterns = json.loads(patterns_path.read_text())
            total_patterns = len(patterns)
            reliable = {k: v for k, v in patterns.items()
                        if v.get("observations", 0) >= 5}

            sections.append("## Pattern Library Summary\n")
            sections.append(f"Total patterns recorded: {total_patterns}")
            sections.append(f"Reliable patterns (5+ obs): {len(reliable)}\n")

            if reliable:
                sections.append("### Top Performing Patterns\n")
                sections.append("| Pattern | Win Rate | Obs | Avg Return |")
                sections.append("|---------|----------|-----|------------|")

                sorted_patterns = sorted(
                    reliable.items(),
                    key=lambda x: (x[1].get("wins", 0) + 0.5 * x[1].get("partials", 0))
                    / max(x[1].get("observations", 1), 1),
                    reverse=True,
                )
                for key, data in sorted_patterns[:8]:
                    obs = data.get("observations", 0)
                    wins = data.get("wins", 0)
                    partials = data.get("partials", 0)
                    wr = (wins + 0.5 * partials) / max(obs, 1)
                    avg_ret = data.get("total_return", 0) / max(obs, 1)
                    sections.append(f"| `{key}` | {wr:.0%} | {obs} | {avg_ret:+.1%} |")

                # Worst patterns
                sections.append("\n### Worst Performing Patterns\n")
                sections.append("| Pattern | Win Rate | Obs | Avg Return |")
                sections.append("|---------|----------|-----|------------|")
                for key, data in sorted_patterns[-5:]:
                    obs = data.get("observations", 0)
                    wins = data.get("wins", 0)
                    partials = data.get("partials", 0)
                    wr = (wins + 0.5 * partials) / max(obs, 1)
                    avg_ret = data.get("total_return", 0) / max(obs, 1)
                    sections.append(f"| `{key}` | {wr:.0%} | {obs} | {avg_ret:+.1%} |")
                sections.append("")
        else:
            sections.append("## Pattern Library\n\nNo patterns recorded yet.\n")
    except Exception as exc:
        logger.warning("Failed to load patterns: %s", exc)

    # 3. Confidence calibration
    cal_path = config.performance_dir / "confidence_calibration.json"
    try:
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            sections.append("## Live Confidence Calibration\n")
            sections.append("| Bucket | Observed WR | Obs | Reliable? |")
            sections.append("|--------|-------------|-----|-----------|")

            for bucket in sorted(cal.keys(), key=float):
                data = cal[bucket]
                total = data.get("total", 0)
                wr = data.get("win_rate", 0)
                reliable = "Yes" if total >= 10 else "No"
                sections.append(f"| {bucket} | {wr:.0%} | {total} | {reliable} |")
            sections.append("")
        else:
            sections.append("## Confidence Calibration\n\nUsing static defaults (no live data yet).\n")
    except Exception as exc:
        logger.warning("Failed to load calibration: %s", exc)

    # 4. Most recent reflection
    try:
        reflection_files = sorted(config.performance_dir.glob("*_reflection.json"), reverse=True)
        if reflection_files:
            latest = json.loads(reflection_files[0].read_text())
            week = latest.get("week", "unknown")
            sections.append(f"## Most Recent Reflection (Week of {week})\n")
            sections.append(f"- Win rate: {latest.get('win_rate', 0):.0%} "
                            f"({latest.get('wins', 0)}W / {latest.get('losses', 0)}L)")
            sections.append(f"- Avg return: {latest.get('avg_return', 0):.1%}")

            lessons = latest.get("lessons", [])
            if lessons:
                sections.append("\n**Lessons:**")
                for lesson in lessons[:5]:
                    sections.append(f"- {lesson}")

            adj = latest.get("weight_adjustments", {})
            if adj:
                sections.append("\n**Weight adjustments applied:**")
                for signal, delta in sorted(adj.items(), key=lambda x: abs(x[1]), reverse=True):
                    if abs(delta) > 0.001:
                        sections.append(f"- {signal}: {delta:+.3f}")
            sections.append("")
    except Exception as exc:
        logger.warning("Failed to load reflection: %s", exc)

    # 5. All-time scorecard summary
    sc_path = config.performance_dir / "scorecard_data.json"
    try:
        if sc_path.exists():
            sc = json.loads(sc_path.read_text())
            alltime = sc.get("all_time", {})
            if alltime:
                sections.append("## All-Time Performance\n")
                sections.append(f"- Total P&L: ${alltime.get('total_pnl', 0):,.2f}")
                sections.append(f"- Win rate: {alltime.get('win_rate', 0):.0%}")
                sections.append(f"- Total trades: {alltime.get('total_trades', 0)}")
                sections.append(f"- Best pick: {alltime.get('best_pick', 'N/A')}")
                sections.append(f"- Worst pick: {alltime.get('worst_pick', 'N/A')}")
                sections.append("")
    except Exception as exc:
        logger.warning("Failed to load scorecard: %s", exc)

    content = "\n".join(sections)

    # Write to file
    output_path = _AGENTS_DIR / "CURRENT_STATE.md"
    try:
        output_path.write_text(content)
        logger.info("CURRENT_STATE.md updated (%d chars)", len(content))
    except Exception as exc:
        logger.error("Failed to write CURRENT_STATE.md: %s", exc)

    return content


def generate_trade_log() -> str:
    """Generate TRADE_LOG.md from recent scorecard and outcome data.

    Rolling 8-week window of trade outcomes with pattern matches,
    win/loss reasons, and P&L. Gives agents historical context for
    what's been working and failing recently.
    """
    sections = ["# Recent Trade Log\n\n_Auto-generated from scorecard data. "
                 f"Rolling 8-week window. Updated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}_\n"]

    # Load scorecard data
    sc_path = config.performance_dir / "scorecard_data.json"
    try:
        if not sc_path.exists():
            sections.append("No trade history yet.\n")
            content = "\n".join(sections)
            (_AGENTS_DIR / "TRADE_LOG.md").write_text(content)
            return content

        sc = json.loads(sc_path.read_text())
        weeks = sc.get("weeks", [])

        if not weeks:
            sections.append("No completed weeks yet.\n")
            content = "\n".join(sections)
            (_AGENTS_DIR / "TRADE_LOG.md").write_text(content)
            return content

        # Last 8 weeks
        recent_weeks = weeks[-8:]

        # Summary stats across recent weeks
        total_trades = 0
        total_wins = 0
        total_pnl = 0
        direction_stats = {"call": {"wins": 0, "total": 0}, "put": {"wins": 0, "total": 0}}

        for week in recent_weeks:
            week_date = week.get("week", "?")
            picks = week.get("picks", [])
            wins = week.get("wins", 0)
            losses = week.get("losses", 0)
            week_pnl = week.get("total_pnl", 0)

            total_trades += len(picks)
            total_wins += wins
            total_pnl += week_pnl

            sections.append(f"## Week of {week_date} — "
                            f"{wins}W/{losses}L (${week_pnl:+,.2f})\n")

            for p in picks:
                ticker = p.get("ticker", "?")
                direction = p.get("direction", "?")
                result = p.get("result", "?")
                pnl = p.get("pnl", 0)
                entry = p.get("entry_premium", "?")
                close = p.get("closing_price", "?")

                is_win = result == "win"
                marker = "W" if is_win else "L" if result == "loss" else "P"

                direction_stats.setdefault(direction, {"wins": 0, "total": 0})
                direction_stats[direction]["total"] += 1
                if is_win:
                    direction_stats[direction]["wins"] += 1

                sections.append(
                    f"- `[{marker}]` **{ticker}** {direction.upper()} — "
                    f"${pnl:+,.2f} (entry ${entry}, close ${close})"
                )
            sections.append("")

        # Aggregate stats
        sections.append("## 8-Week Aggregate\n")
        wr = total_wins / max(total_trades, 1)
        sections.append(f"- Win rate: {wr:.0%} ({total_wins}/{total_trades})")
        sections.append(f"- Total P&L: ${total_pnl:+,.2f}")

        for direction, stats in direction_stats.items():
            d_wr = stats["wins"] / max(stats["total"], 1)
            sections.append(f"- {direction.upper()}: {d_wr:.0%} ({stats['wins']}/{stats['total']})")
        sections.append("")

    except Exception as exc:
        logger.warning("Failed to generate trade log: %s", exc)
        sections.append(f"Error loading trade data: {exc}\n")

    content = "\n".join(sections)

    # Write to file
    try:
        (_AGENTS_DIR / "TRADE_LOG.md").write_text(content)
        logger.info("TRADE_LOG.md updated (%d chars)", len(content))
    except Exception as exc:
        logger.error("Failed to write TRADE_LOG.md: %s", exc)

    return content


def refresh_agent_context():
    """Regenerate all dynamic context documents.

    Call this before agent runs to ensure fresh state. Also called
    by the reflection flow after apply_learnings().
    """
    generate_current_state()
    generate_trade_log()
    logger.info("Agent context documents refreshed")
