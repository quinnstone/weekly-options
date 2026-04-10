#!/usr/bin/env python3
"""Weekly Options Analysis Pipeline"""
import argparse
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.runner import PipelineRunner
from tracking.reflector import WeeklyReflector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Weekly Options Analysis Pipeline')
    parser.add_argument('command', choices=['run', 'stage', 'picks', 'reflect', 'status', 'backtest', 'scorecard', 'audit-agents'],
                       help='Command to run')
    parser.add_argument('--stage', choices=['wednesday', 'friday', 'monday', 'confirm', 'monitor', 'final_exit'],
                       help='Specific stage to run')
    parser.add_argument('--week', help='Week date (YYYY-MM-DD)')
    parser.add_argument('--weeks', type=int, default=52, help='Backtest lookback in weeks (default 52)')
    parser.add_argument('--dry-run', action='store_true', help='Run without sending notifications')

    args = parser.parse_args()
    runner = PipelineRunner()

    if args.command == 'run':
        runner.run_day()
    elif args.command == 'stage':
        if not args.stage:
            print("--stage required (wednesday, friday, monday)")
            sys.exit(1)
        runner.run_stage(args.stage)
    elif args.command == 'picks':
        runner.show_current_picks()
    elif args.command == 'reflect':
        reflector = WeeklyReflector()
        reflection = reflector.reflect(args.week or datetime.now().strftime('%Y-%m-%d'))
        print(reflector.format_reflection(reflection))

        # Run Deep Reflection agent for qualitative CIO analysis
        try:
            from agents.deep_reflection import DeepReflectionAgent
            deep = DeepReflectionAgent()
            if deep.enabled:
                # Load scorecard if available
                scorecard = None
                try:
                    from tracking.scorecard import Scorecard
                    sc = Scorecard()
                    week_date = args.week or datetime.now().strftime('%Y-%m-%d')
                    scorecard = sc.grade_week(week_date)
                except Exception:
                    pass

                # Load market summary
                market_summary = None
                try:
                    from config import Config as _Cfg
                    _cfg = _Cfg()
                    import json as _json
                    from datetime import timedelta
                    for days_back in range(7):
                        date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                        summary_path = _cfg.candidates_dir / date_str / "market_summary.json"
                        if summary_path.exists():
                            with open(summary_path) as fh:
                                market_summary = _json.load(fh)
                            break
                except Exception:
                    pass

                deep_result = deep.reflect(reflection, scorecard=scorecard,
                                           market_summary=market_summary)
                if deep_result.get("analysis"):
                    print("\n" + "=" * 65)
                    print("DEEP REFLECTION (CIO Analysis)")
                    print("=" * 65)
                    print(deep_result["analysis"])

                    # Track deep reflection decisions
                    from tracking.agent_tracker import log_decision
                    proposals = deep_result.get("proposals", [])
                    log_decision(
                        agent_name="deep_reflection",
                        ticker="SYSTEM",
                        mechanical_signal="current_weights",
                        agent_signal=f"{len(proposals)} proposals",
                        override_occurred=bool(proposals),
                        context={
                            "proposals": [p[:100] if isinstance(p, str) else str(p)[:100] for p in proposals[:5]],
                            "analysis_snippet": deep_result["analysis"][:300],
                        },
                    )
        except Exception as exc:
            logging.getLogger(__name__).error("Deep reflection agent failed: %s", exc)
    elif args.command == 'status':
        runner.show_status()
    elif args.command == 'backtest':
        from analysis.backtest import DirectionalBacktester
        bt = DirectionalBacktester(lookback_weeks=args.weeks)
        results = bt.run()
        bt.print_report(results)
        filepath = bt.save_results(results)
        print(f"\nResults saved to {filepath}")
    elif args.command == 'scorecard':
        from tracking.scorecard import Scorecard
        sc = Scorecard()
        if args.week:
            print(f"Grading picks for {args.week}...")
            result = sc.grade_week(args.week)
            if result:
                print(f"Week graded: {result.get('wins', 0)}W-{result.get('losses', 0)}L-{result.get('partials', 0)}P  "
                      f"P&L: ${result.get('total_pnl', 0):,.2f}")
        sc.show()
    elif args.command == 'audit-agents':
        from tracking.agent_tracker import generate_audit_report, evaluate
        report = generate_audit_report()
        print(report)
        # Also print actionable summary
        eval_data = evaluate()
        if eval_data.get("agents"):
            cuts = [name for name, m in eval_data["agents"].items() if m["recommendation"] == "CUT"]
            watches = [name for name, m in eval_data["agents"].items() if m["recommendation"] == "WATCH"]
            if cuts:
                print(f"\nACTION REQUIRED: Cut these agents (zero value): {', '.join(cuts)}")
            if watches:
                print(f"\nWATCH LIST (re-evaluate at 10 weeks): {', '.join(watches)}")


if __name__ == '__main__':
    main()
