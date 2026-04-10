"""
Agent Impact Tracker — measures whether each agent is earning its keep.

Logs every agent decision alongside the mechanical signal. After each week's
grading, links agent decisions to actual outcomes. The evaluator then computes
override rates, value-add metrics, and flags zero-value agents for removal.

Audit cadence: review after 5 weeks, cut decision at 10 weeks.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)
config = Config()

TRACKER_PATH = config.performance_dir / "agent_tracker.json"


def _load_tracker() -> dict:
    """Load tracker data from disk."""
    if TRACKER_PATH.exists():
        try:
            return json.loads(TRACKER_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt tracker file, starting fresh")
    return {"decisions": [], "weekly_summaries": []}


def _save_tracker(data: dict):
    """Persist tracker data to disk."""
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(data, indent=2, default=str))


def log_decision(
    agent_name: str,
    ticker: str,
    mechanical_signal: str,
    agent_signal: str,
    override_occurred: bool,
    context: dict = None,
):
    """Record a single agent decision for later audit.

    Parameters
    ----------
    agent_name : str
        Which agent made the call (pre_trade, position_monitor, etc.).
    ticker : str
        The ticker this decision applies to (or 'PORTFOLIO' for portfolio-level).
    mechanical_signal : str
        What the rules-based system would have done (GO, HOLD, CLOSE, etc.).
    agent_signal : str
        What the agent recommended (GO, SKIP, ADJUST, CLOSE, etc.).
    override_occurred : bool
        Whether the agent's signal changed the final action.
    context : dict or None
        Optional context (urgency level, confidence, reasoning snippet).
    """
    data = _load_tracker()

    decision = {
        "timestamp": datetime.now().isoformat(),
        "week": datetime.now().strftime("%Y-W%W"),
        "agent": agent_name,
        "ticker": ticker,
        "mechanical_signal": mechanical_signal,
        "agent_signal": agent_signal,
        "override": override_occurred,
        "outcome": None,  # filled in by link_outcomes after grading
        "context": context or {},
    }

    data["decisions"].append(decision)
    _save_tracker(data)
    logger.info(
        "Agent tracker: %s on %s — mechanical=%s, agent=%s, override=%s",
        agent_name, ticker, mechanical_signal, agent_signal, override_occurred,
    )


def link_outcomes(week_str: str, outcomes: dict):
    """After Friday grading, link agent decisions to actual P&L outcomes.

    Parameters
    ----------
    week_str : str
        Week identifier (e.g., '2026-W15').
    outcomes : dict
        Mapping of ticker -> {'result': 'WIN/LOSS/PARTIAL', 'pnl': float}.
    """
    data = _load_tracker()
    linked = 0

    for decision in data["decisions"]:
        if decision["week"] != week_str:
            continue
        if decision["outcome"] is not None:
            continue

        ticker = decision["ticker"]
        if ticker in outcomes:
            decision["outcome"] = outcomes[ticker]
            linked += 1

    _save_tracker(data)
    logger.info("Linked %d agent decisions to outcomes for %s", linked, week_str)


def evaluate(min_weeks: int = 5) -> dict:
    """Evaluate each agent's impact and generate audit report.

    Parameters
    ----------
    min_weeks : int
        Minimum weeks of data before generating recommendations.

    Returns
    -------
    dict
        Per-agent metrics and recommendations.
    """
    data = _load_tracker()
    decisions = data.get("decisions", [])

    if not decisions:
        return {"status": "no_data", "message": "No agent decisions recorded yet."}

    # Group by agent
    by_agent = {}
    for d in decisions:
        agent = d["agent"]
        if agent not in by_agent:
            by_agent[agent] = []
        by_agent[agent].append(d)

    # Count unique weeks
    all_weeks = set(d["week"] for d in decisions)
    weeks_active = len(all_weeks)

    report = {
        "weeks_tracked": weeks_active,
        "total_decisions": len(decisions),
        "sufficient_data": weeks_active >= min_weeks,
        "agents": {},
    }

    for agent, agent_decisions in by_agent.items():
        total = len(agent_decisions)
        overrides = [d for d in agent_decisions if d["override"]]
        override_count = len(overrides)
        override_rate = override_count / total if total > 0 else 0

        # Measure value of overrides — did overridden decisions lead to
        # better outcomes than the mechanical signal would have?
        valuable_overrides = 0
        harmful_overrides = 0
        neutral_overrides = 0

        for d in overrides:
            outcome = d.get("outcome")
            if outcome is None:
                neutral_overrides += 1
                continue

            result = outcome.get("result", "").upper()
            mech = d["mechanical_signal"].upper()
            agent_sig = d["agent_signal"].upper()

            # Agent SKIP on mechanical GO — valuable if outcome was LOSS
            if agent_sig == "SKIP" and mech == "GO":
                if result == "LOSS":
                    valuable_overrides += 1  # Saved us from a loss
                elif result == "WIN":
                    harmful_overrides += 1  # Blocked a winner
                else:
                    neutral_overrides += 1

            # Agent CLOSE on mechanical HOLD — valuable if position deteriorated
            elif agent_sig == "CLOSE" and mech == "HOLD":
                pnl = outcome.get("pnl", 0)
                if pnl < 0:
                    valuable_overrides += 1  # Cut a loser early
                elif pnl > 0:
                    harmful_overrides += 1  # Closed a winner early
                else:
                    neutral_overrides += 1

            # Agent changed portfolio selection
            elif agent_sig != mech:
                if result in ("WIN", "PARTIAL"):
                    valuable_overrides += 1
                elif result == "LOSS":
                    harmful_overrides += 1
                else:
                    neutral_overrides += 1

        # Agreements — agent confirmed what mechanical said
        agreements = [d for d in agent_decisions if not d["override"]]
        agreement_count = len(agreements)

        # Cost estimate (~$0.20 per call, rough)
        unique_calls = len(set(
            (d["week"], d["agent"]) for d in agent_decisions
        ))
        estimated_cost = unique_calls * 0.20

        # Value score: net valuable overrides / total decisions
        value_score = (valuable_overrides - harmful_overrides) / max(total, 1)

        # Recommendation
        recommendation = "KEEP"
        reason = ""

        if weeks_active >= 10 and override_count == 0:
            recommendation = "CUT"
            reason = (
                f"Zero overrides in {weeks_active} weeks. Agent always agrees "
                f"with mechanical signal — provides no additional value."
            )
        elif weeks_active >= 10 and harmful_overrides > valuable_overrides:
            recommendation = "CUT"
            reason = (
                f"Agent overrides are net harmful: {harmful_overrides} bad vs "
                f"{valuable_overrides} good. Actively hurting performance."
            )
        elif weeks_active >= 5 and override_count == 0:
            recommendation = "WATCH"
            reason = (
                f"No overrides yet in {weeks_active} weeks. May be redundant. "
                f"Re-evaluate at 10 weeks."
            )
        elif weeks_active >= 5 and value_score > 0:
            recommendation = "KEEP"
            reason = (
                f"Net positive: {valuable_overrides} valuable overrides vs "
                f"{harmful_overrides} harmful. Agent is earning its cost."
            )
        elif weeks_active >= 5 and value_score == 0 and override_count > 0:
            recommendation = "TUNE"
            reason = (
                f"Overrides are neutral ({neutral_overrides} unresolved). "
                f"Agent is active but not clearly helping. Review prompts."
            )
        else:
            recommendation = "COLLECTING"
            reason = f"Only {weeks_active} weeks of data. Need {min_weeks} minimum."

        report["agents"][agent] = {
            "total_decisions": total,
            "overrides": override_count,
            "override_rate": round(override_rate, 3),
            "valuable_overrides": valuable_overrides,
            "harmful_overrides": harmful_overrides,
            "neutral_overrides": neutral_overrides,
            "agreements": agreement_count,
            "value_score": round(value_score, 4),
            "estimated_cost_total": round(estimated_cost, 2),
            "estimated_cost_per_week": round(estimated_cost / max(weeks_active, 1), 2),
            "recommendation": recommendation,
            "reason": reason,
        }

    return report


def generate_audit_report() -> str:
    """Generate a human-readable audit report for all agents.

    Returns
    -------
    str
        Markdown-formatted audit report.
    """
    report = evaluate()

    if report.get("status") == "no_data":
        return "# Agent Audit Report\n\nNo agent decisions recorded yet. Run the pipeline first."

    lines = [
        "# Agent Audit Report",
        f"**Weeks tracked:** {report['weeks_tracked']}",
        f"**Total decisions logged:** {report['total_decisions']}",
        f"**Sufficient data (5+ weeks):** {'Yes' if report['sufficient_data'] else 'No'}",
        "",
    ]

    # Sort agents by recommendation priority: CUT first, then TUNE, WATCH, KEEP
    priority = {"CUT": 0, "TUNE": 1, "WATCH": 2, "KEEP": 3, "COLLECTING": 4}
    sorted_agents = sorted(
        report["agents"].items(),
        key=lambda x: priority.get(x[1]["recommendation"], 5),
    )

    for agent, metrics in sorted_agents:
        rec = metrics["recommendation"]
        icon = {"CUT": "X", "TUNE": "~", "WATCH": "?", "KEEP": "+", "COLLECTING": "..."}
        lines.append(f"## [{icon.get(rec, '?')}] {agent} — {rec}")
        lines.append(f"- Decisions: {metrics['total_decisions']}")
        lines.append(f"- Override rate: {metrics['override_rate']:.1%} ({metrics['overrides']} overrides)")
        lines.append(
            f"- Value: {metrics['valuable_overrides']} saves, "
            f"{metrics['harmful_overrides']} mistakes, "
            f"{metrics['neutral_overrides']} neutral"
        )
        lines.append(f"- Cost: ~${metrics['estimated_cost_total']:.2f} total (${metrics['estimated_cost_per_week']:.2f}/week)")
        lines.append(f"- **{metrics['reason']}**")
        lines.append("")

    return "\n".join(lines)
