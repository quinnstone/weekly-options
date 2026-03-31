"""
Pipeline runner for the Zero-DTE Options Trading Analysis System.

Provides the top-level orchestration layer that maps day-of-week to the
correct pipeline stage and manages full-week / backtest runs.
"""

import sys
import os
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from pipeline.stages import DailyStages
from analysis.narrowing import NarrowingPipeline

logger = logging.getLogger(__name__)
config = Config()

# Map weekday number (0=Mon) to stage method name
DAY_STAGE_MAP = {
    2: "wednesday",
    4: "friday",
}


class PipelineRunner:
    """High-level runner that dispatches pipeline stages."""

    def __init__(self):
        self.stages = DailyStages()
        self.narrower = NarrowingPipeline()

    # ------------------------------------------------------------------
    #  Stage dispatch
    # ------------------------------------------------------------------

    def run_stage(self, stage_name: str) -> list:
        """Run a specific stage by name.

        Parameters
        ----------
        stage_name : str
            One of: wednesday, friday.

        Returns
        -------
        list[dict]
            The candidates produced by the stage.
        """
        dispatch = {
            "wednesday": self.stages.wednesday_scan,
            "thursday": self.stages.thursday_refresh,
            "friday": self.stages.friday_picks,
        }

        handler = dispatch.get(stage_name.lower())
        if handler is None:
            logger.error("Unknown stage: %s", stage_name)
            raise ValueError(
                f"Unknown stage '{stage_name}'. "
                f"Valid stages: {', '.join(dispatch.keys())}"
            )

        logger.info("Running stage: %s", stage_name)
        return handler()

    # ------------------------------------------------------------------
    #  Day-based execution
    # ------------------------------------------------------------------

    def run_day(self, day: str = None) -> list:
        """Auto-detect day of week and run the appropriate stage.

        Parameters
        ----------
        day : str or None
            Explicit stage name (e.g. 'wednesday', 'friday'). If None,
            uses today's actual weekday.

        Returns
        -------
        list[dict]
            The candidates produced by the stage.
        """
        if day is not None:
            return self.run_stage(day)

        weekday_num = datetime.now().weekday()
        day = DAY_STAGE_MAP.get(weekday_num)
        if day is None:
            logger.info(
                "Today is weekday %d — no pipeline stage scheduled (Wed=2, Fri=4).",
                weekday_num,
            )
            return []

        logger.info("Running pipeline for day: %s", day)
        return self.run_stage(day)

    # ------------------------------------------------------------------
    #  Full-week run (backtesting)
    # ------------------------------------------------------------------

    def run_full_week(self) -> dict:
        """Run both stages sequentially (for backtesting).

        Returns
        -------
        dict
            Mapping stage name -> candidate list for each stage.
        """
        logger.info("=== RUNNING FULL WEEK PIPELINE ===")
        results = {}
        for stage_name in ("wednesday", "friday"):
            try:
                candidates = self.run_stage(stage_name)
                results[stage_name] = candidates
                logger.info(
                    "Stage %s complete — %d candidates",
                    stage_name, len(candidates),
                )
            except Exception as exc:
                logger.error("Stage %s failed: %s", stage_name, exc)
                results[stage_name] = []
        logger.info("=== FULL WEEK PIPELINE COMPLETE ===")
        return results

    # ------------------------------------------------------------------
    #  Week directory
    # ------------------------------------------------------------------

    def get_current_week_dir(self) -> str:
        """Return data/candidates/YYYY-WNN/ path for the current ISO week."""
        now = datetime.now()
        week_str = now.strftime("%Y-W%W")
        week_dir = config.candidates_dir / week_str
        week_dir.mkdir(parents=True, exist_ok=True)
        return str(week_dir)

    # ------------------------------------------------------------------
    #  Status & display
    # ------------------------------------------------------------------

    def show_current_picks(self) -> None:
        """Print the most recent final picks to stdout."""
        from datetime import timedelta

        for days_back in range(7):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            picks = self.narrower.load_stage_results("friday_picks", date_str)
            if picks:
                print(f"\nFinal picks from {date_str}:")
                print("-" * 60)
                for i, p in enumerate(picks, 1):
                    direction = p.get("direction", "?").upper()
                    ticker = p.get("ticker", "?")
                    score = p.get("composite_score", 0)
                    strike = p.get("strike", "TBD")
                    conf = p.get("direction_confidence", p.get("confidence", 0))
                    print(
                        f"  {i}. {ticker:6s} {direction:4s}  "
                        f"Strike: {strike}  "
                        f"Score: {score:.1f}  "
                        f"Confidence: {conf:.0%}"
                    )
                print()
                return

        print("No recent picks found. Run the pipeline first.")

    def show_status(self) -> None:
        """Display current pipeline status — which stages have run this week."""
        from datetime import timedelta

        print("\nPipeline Status")
        print("=" * 60)

        stages = ["wednesday_scan", "friday_picks"]

        for days_back in range(7):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            found_any = False
            for stage in stages:
                results = self.narrower.load_stage_results(stage, date_str)
                if results:
                    found_any = True
                    print(f"  [{date_str}] {stage:20s} — {len(results)} candidates")
            if found_any:
                print()

        # Show week directory
        week_dir = self.get_current_week_dir()
        print(f"Current week directory: {week_dir}")
        print()
