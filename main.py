#!/usr/bin/env python3
"""Zero DTE Options Analysis Pipeline"""
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
    parser = argparse.ArgumentParser(description='Zero DTE Options Analysis Pipeline')
    parser.add_argument('command', choices=['run', 'stage', 'picks', 'reflect', 'status', 'backtest', 'scorecard'],
                       help='Command to run')
    parser.add_argument('--stage', choices=['wednesday', 'thursday', 'friday', 'reflect'],
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
            print("--stage required")
            sys.exit(1)
        runner.run_stage(args.stage)
    elif args.command == 'picks':
        # Show current picks
        runner.show_current_picks()
    elif args.command == 'reflect':
        reflector = WeeklyReflector()
        reflection = reflector.reflect(args.week or datetime.now().strftime('%Y-%m-%d'))
        print(reflector.format_reflection(reflection))
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


if __name__ == '__main__':
    main()
