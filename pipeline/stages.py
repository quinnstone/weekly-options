"""
Pipeline stages for the Zero-DTE Options Trading Analysis System.

Three-stage weekly analysis cycle:
    Wednesday - Broad scan of curated universe, narrow to 20 candidates + 10 bench
    Thursday  - Delta-aware re-evaluation: compare Wed vs Thu data on 30 candidates,
                re-rank to best 20 based on setup evolution toward Friday
    Friday    - Fresh data on 20 candidates, score, diversify, pick top 3
"""

import copy
import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from scanners.market import MarketScanner
from analysis.scoring import CandidateScorer
from analysis.narrowing import NarrowingPipeline, enforce_diversity
from notifications.discord import DiscordNotifier
from tracking.tracker import PerformanceTracker

logger = logging.getLogger(__name__)
config = Config()


class DailyStages:
    """Orchestrates the Wednesday/Friday two-pass pipeline."""

    def __init__(self):
        self.market_scanner = MarketScanner()
        self.scorer = CandidateScorer()
        self.narrower = NarrowingPipeline()
        self.notifier = DiscordNotifier()
        self.tracker = PerformanceTracker()

        # Scanners are imported lazily because some may not exist yet
        self._technical_scanner = None
        self._sentiment_scanner = None
        self._options_scanner = None
        self._flow_scanner = None
        self._finviz_scanner = None
        self._edgar_scanner = None

    # ------------------------------------------------------------------
    #  Lazy scanner accessors
    # ------------------------------------------------------------------

    def _get_technical_scanner(self):
        if self._technical_scanner is None:
            try:
                from scanners.technical import TechnicalScanner
                self._technical_scanner = TechnicalScanner()
            except ImportError:
                logger.warning("TechnicalScanner not available")
        return self._technical_scanner

    def _get_sentiment_scanner(self):
        if self._sentiment_scanner is None:
            try:
                from scanners.sentiment import SentimentScanner
                self._sentiment_scanner = SentimentScanner()
            except ImportError:
                logger.warning("SentimentScanner not available")
        return self._sentiment_scanner

    def _get_options_scanner(self):
        if self._options_scanner is None:
            try:
                from scanners.options import OptionsScanner
                self._options_scanner = OptionsScanner()
            except ImportError:
                logger.warning("OptionsScanner not available")
        return self._options_scanner

    def _get_flow_scanner(self):
        if self._flow_scanner is None:
            try:
                from scanners.flow import FlowScanner
                self._flow_scanner = FlowScanner()
            except ImportError:
                logger.warning("FlowScanner not available")
        return self._flow_scanner

    def _get_finviz_scanner(self):
        if self._finviz_scanner is None:
            try:
                from scanners.finviz import FinvizScanner
                self._finviz_scanner = FinvizScanner()
            except ImportError:
                logger.warning("FinvizScanner not available")
        return self._finviz_scanner

    def _get_edgar_scanner(self):
        if self._edgar_scanner is None:
            try:
                from scanners.edgar import EdgarScanner
                self._edgar_scanner = EdgarScanner()
            except ImportError:
                logger.warning("EdgarScanner not available")
        return self._edgar_scanner

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _current_date_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _map_flow_data(raw_flow: dict) -> dict:
        """Map flow scanner output to field names expected by scoring."""
        if not raw_flow:
            return {}
        return {
            "unusual": raw_flow.get("is_unusual", False),
            "unusual_volume": raw_flow.get("is_unusual", False),
            "direction": raw_flow.get("direction_bias"),
            "volume_oi_ratio": raw_flow.get("vol_oi_ratio", 0),
            "total_vol": raw_flow.get("total_vol", 0),
        }

    def _batch_scan(self, scanner, method_name: str, tickers: list,
                    rate_limit: float = 0.2) -> dict:
        """Run *scanner.method_name(ticker)* for every ticker with rate limiting.

        Returns dict mapping ticker -> result.
        """
        results = {}
        total = len(tickers)
        for idx, ticker in enumerate(tickers, 1):
            try:
                method = getattr(scanner, method_name)
                results[ticker] = method(ticker)
            except Exception as exc:
                logger.error("Scanner error for %s: %s", ticker, exc)
                results[ticker] = {}
            if rate_limit > 0:
                time.sleep(rate_limit)
            if idx % 50 == 0:
                logger.info("Batch scan progress: %d / %d", idx, total)
        return results

    def _load_previous_stage(self, stage_name: str) -> list:
        """Try to load a previous stage's results, checking today and recent days."""
        for days_back in range(5):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            results = self.narrower.load_stage_results(stage_name, date_str)
            if results:
                logger.info(
                    "Loaded %d candidates from '%s' (%s)",
                    len(results), stage_name, date_str,
                )
                return results
        return []

    # ------------------------------------------------------------------
    #  Wednesday — Broad scan
    # ------------------------------------------------------------------

    def wednesday_scan(self) -> list:
        """Run the Wednesday broad-scan stage.

        1. Get market summary
        2. Get core universe (~103) + dynamic additions (unusual volume, earnings, news)
        3. Get sector map
        4. Run technical scan batch on all tickers
        5. Run options chain analysis batch on all tickers (rate_limit=0.3)
        6. Run sentiment analysis batch on all tickers (rate_limit=1.5)
        7. Build complete data dict for each candidate
        8. Use NarrowingPipeline to narrow to 20
        9. Save to data/candidates/{date}/wednesday_scan.json
        10. Save market summary to data/candidates/{date}/market_summary.json

        Returns
        -------
        list[dict]
            The narrowed candidate list (20 candidates).
        """
        date_str = self._current_date_str()
        logger.info("=== WEDNESDAY SCAN — %s ===", date_str)

        # 1. Market summary
        market_summary = self.market_scanner.get_market_summary()
        summary_path = config.candidates_dir / date_str
        summary_path.mkdir(parents=True, exist_ok=True)
        with open(summary_path / "market_summary.json", "w") as fh:
            json.dump(
                {k: v for k, v in market_summary.items()
                 if k != "sectors"},  # sectors is a DataFrame
                fh, indent=2, default=str,
            )
        logger.info("Market summary saved")

        # 2. Core curated universe + dynamic expansion
        from universe.robinhood import get_full_universe, get_universe_by_sector, get_sector_for_ticker
        core_universe = get_full_universe()
        logger.info("Core universe: %d tickers", len(core_universe))

        # Discover dynamic additions (unusual volume, earnings, news buzz)
        dynamic_additions = []
        try:
            from universe.dynamic import discover_dynamic_tickers
            dynamic_additions = discover_dynamic_tickers()
            if dynamic_additions:
                dynamic_tickers = [d["ticker"] for d in dynamic_additions]
                logger.info(
                    "Dynamic additions: %d tickers — %s",
                    len(dynamic_tickers), ", ".join(dynamic_tickers),
                )
                # Save dynamic metadata for reporting
                with open(summary_path / "dynamic_additions.json", "w") as fh:
                    json.dump(dynamic_additions, fh, indent=2)
        except Exception as exc:
            logger.warning("Dynamic universe scan failed: %s", exc)

        # Merge into full universe
        dynamic_ticker_set = {d["ticker"] for d in dynamic_additions}
        universe = sorted(set(core_universe) | dynamic_ticker_set)
        logger.info("Expanded universe: %d tickers (%d core + %d dynamic)",
                     len(universe), len(core_universe), len(dynamic_ticker_set))

        # 3. Sector map
        sector_by_ticker = {t: get_sector_for_ticker(t) for t in universe}
        # Dynamic tickers get "dynamic" as sector
        for d in dynamic_additions:
            sector_by_ticker[d["ticker"]] = "dynamic"

        # 4. Technical scan batch
        tickers = list(universe)
        tech_results = {}
        tech_scanner = self._get_technical_scanner()
        if tech_scanner:
            tech_results = self._batch_scan(tech_scanner, "scan_ticker", tickers)
        else:
            logger.warning("Skipping technical scan — scanner unavailable")

        # 5. Options chain analysis batch
        opts_results = {}
        opts_scanner = self._get_options_scanner()
        if opts_scanner:
            opts_results = self._batch_scan(opts_scanner, "analyze_chain", tickers, rate_limit=0.3)
        else:
            logger.warning("Skipping options scan — scanner unavailable")

        # 6. Sentiment analysis batch
        sent_results = {}
        sent_scanner = self._get_sentiment_scanner()
        if sent_scanner:
            sent_results = self._batch_scan(sent_scanner, "analyze_ticker_sentiment", tickers, rate_limit=1.5)
        else:
            logger.warning("Skipping sentiment scan — scanner unavailable")

        # 6b. Finviz scan (pre-market movers, analyst ratings, insider activity, news)
        finviz_results = {}
        finviz_scanner = self._get_finviz_scanner()
        if finviz_scanner:
            finviz_results = self._batch_scan(finviz_scanner, "scan_ticker", tickers, rate_limit=0.5)
        else:
            logger.warning("Skipping Finviz scan — scanner unavailable")

        # 6c. SEC EDGAR insider trading scan
        edgar_results = {}
        edgar_scanner = self._get_edgar_scanner()
        if edgar_scanner:
            edgar_results = self._batch_scan(edgar_scanner, "get_insider_trades", tickers, rate_limit=0.3)
        else:
            logger.warning("Skipping EDGAR scan — scanner unavailable")

        # 6d. Unusual options flow detection
        flow_results = {}
        flow_scanner = self._get_flow_scanner()
        if flow_scanner:
            flow_results = self._batch_scan(flow_scanner, "detect_unusual_volume", tickers, rate_limit=0.3)
        else:
            logger.warning("Skipping flow scan — scanner unavailable")

        # 7. Build complete candidate dicts
        candidates = []
        for ticker in tickers:
            candidate = {
                "ticker": ticker,
                "technical": tech_results.get(ticker, {}),
                "options": opts_results.get(ticker, {}),
                "sentiment": sent_results.get(ticker, {}),
                "finviz": finviz_results.get(ticker, {}),
                "insider": edgar_results.get(ticker, {}),
                "flow": self._map_flow_data(flow_results.get(ticker, {})),
                "sector": sector_by_ticker.get(ticker, "unknown"),
            }
            candidates.append(candidate)

        # 8. Narrow to 20 + save bench for Thursday re-evaluation
        candidates, bench = self.narrower.narrow_with_bench(candidates, "wednesday_scan")

        # 9. Save wednesday_scan results and bench
        self.narrower.save_stage_results("wednesday_scan", candidates, date_str)
        if bench:
            self.narrower.save_stage_results("wednesday_bench", bench, date_str)
            logger.info("Saved %d bench candidates for Thursday re-evaluation", len(bench))

        logger.info("Wednesday scan complete: %d candidates", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    #  Thursday — Data refresh on candidates
    # ------------------------------------------------------------------

    def thursday_refresh(self) -> list:
        """Delta-aware re-evaluation of Wednesday's 20 candidates + bench.

        Compares Wednesday's data snapshot against Thursday's fresh scans
        to measure how each setup is evolving toward Friday's 0DTE play.
        Candidates with strengthening setups rise; those deteriorating fall.
        Bench candidates can be promoted if their evolution outpaces incumbents.

        1. Load Wednesday's 20 candidates + bench (~10 alternates)
        2. Snapshot Wednesday's data for delta comparison
        3. Re-run technical, options, sentiment, finviz scans on all ~30
        4. Merge fresh data (preserving Wednesday snapshot)
        5. Re-rank using delta-aware Thursday scoring
        6. Take best 20, save new bench from remainder
        7. Log promotions / demotions

        Returns
        -------
        list[dict]
            The re-evaluated candidate list (best 20 from the pool).
        """
        date_str = self._current_date_str()
        logger.info("=== THURSDAY RE-EVALUATION — %s ===", date_str)

        # 1. Load Wednesday's candidates + bench
        candidates = self._load_previous_stage("wednesday_scan")
        if not candidates:
            logger.error("No wednesday_scan results found — run Wednesday stage first")
            return []

        bench = self._load_previous_stage("wednesday_bench")
        wednesday_tickers = {c["ticker"] for c in candidates}

        # Merge into a single pool, avoiding duplicates
        pool = list(candidates)
        bench_tickers_added = set()
        for b in bench:
            if b["ticker"] not in wednesday_tickers:
                pool.append(b)
                bench_tickers_added.add(b["ticker"])

        logger.info(
            "Thursday pool: %d incumbents + %d bench = %d total",
            len(candidates), len(bench_tickers_added), len(pool),
        )

        # 2. Snapshot Wednesday's data before overwriting with fresh scans
        for c in pool:
            c["wednesday_snapshot"] = {
                "technical": copy.deepcopy(c.get("technical", {})),
                "options": copy.deepcopy(c.get("options", {})),
                "sentiment": copy.deepcopy(c.get("sentiment", {})),
                "finviz": copy.deepcopy(c.get("finviz", {})),
            }

        tickers = [c["ticker"] for c in pool]

        # 3. Fresh scans — technical, options, sentiment, finviz
        tech_scanner = self._get_technical_scanner()
        if tech_scanner:
            tech_results = self._batch_scan(tech_scanner, "scan_ticker", tickers)
        else:
            tech_results = {}
            logger.warning("Skipping technical scan — scanner unavailable")

        opts_scanner = self._get_options_scanner()
        if opts_scanner:
            opts_results = self._batch_scan(opts_scanner, "analyze_chain", tickers, rate_limit=0.3)
        else:
            opts_results = {}
            logger.warning("Skipping options scan — scanner unavailable")

        sent_scanner = self._get_sentiment_scanner()
        if sent_scanner:
            sent_results = self._batch_scan(sent_scanner, "analyze_ticker_sentiment", tickers, rate_limit=1.5)
        else:
            sent_results = {}
            logger.warning("Skipping sentiment scan — scanner unavailable")

        finviz_scanner = self._get_finviz_scanner()
        if finviz_scanner:
            finviz_results = self._batch_scan(finviz_scanner, "scan_ticker", tickers, rate_limit=0.5)
        else:
            finviz_results = {}
            logger.warning("Skipping Finviz scan — scanner unavailable")

        flow_scanner = self._get_flow_scanner()
        if flow_scanner:
            flow_results = self._batch_scan(flow_scanner, "detect_unusual_volume", tickers, rate_limit=0.3)
        else:
            flow_results = {}
            logger.warning("Skipping flow scan — scanner unavailable")

        # 4. Merge fresh data (wednesday_snapshot is already preserved)
        for c in pool:
            ticker = c["ticker"]
            if ticker in tech_results:
                c["technical"] = tech_results[ticker]
            if ticker in opts_results:
                c["options"] = opts_results[ticker]
            if ticker in sent_results:
                c["sentiment"] = sent_results[ticker]
            if ticker in finviz_results:
                c["finviz"] = finviz_results[ticker]
            if ticker in flow_results:
                c["flow"] = self._map_flow_data(flow_results[ticker])

        # 5. Re-rank using delta-aware Thursday scoring
        top20, new_bench = self.narrower.narrow_with_bench(pool, "thursday_reeval")

        # 6. Log promotions and demotions
        new_tickers = {c["ticker"] for c in top20}
        promoted = new_tickers - wednesday_tickers
        demoted = wednesday_tickers - new_tickers

        if promoted:
            promoted_details = []
            for c in top20:
                if c["ticker"] in promoted:
                    score = c.get("thursday_score", 0)
                    promoted_details.append(f"{c['ticker']} ({score:.3f})")
            logger.info("PROMOTED from bench: %s", ", ".join(promoted_details))
        if demoted:
            logger.info("DEMOTED to bench: %s", ", ".join(sorted(demoted)))
        if not promoted and not demoted:
            logger.info("No changes — same 20 candidates held their positions")

        # Log top 5 by thursday_score for visibility
        for i, c in enumerate(top20[:5], 1):
            components = c.get("thursday_components", {})
            logger.info(
                "  #%d %s  score=%.3f  [momentum=%.2f iv/prem=%.2f opts=%.2f catalyst=%.2f quality=%.2f sent=%.2f]",
                i, c["ticker"], c.get("thursday_score", 0),
                components.get("setup_momentum", 0),
                components.get("iv_premium_trajectory", 0),
                components.get("options_positioning", 0),
                components.get("fresh_catalysts", 0),
                components.get("base_quality", 0),
                components.get("sentiment_evolution", 0),
            )

        # 7. Save
        self.narrower.save_stage_results("thursday_refresh", top20, date_str)
        if new_bench:
            self.narrower.save_stage_results("thursday_bench", new_bench, date_str)

        logger.info("Thursday re-evaluation complete: %d candidates (%d promoted, %d demoted)",
                     len(top20), len(promoted), len(demoted))
        return top20

    # ------------------------------------------------------------------
    #  Friday — Final picks
    # ------------------------------------------------------------------

    def friday_picks(self) -> list:
        """Run the Friday final-pick stage.

        1. Load Wednesday's scan results (check today, yesterday, 2 days back)
        2. Get fresh market summary
        3. Re-run technical scan on the 20 candidates (fresh data)
        4. Re-run options chain analysis (fresh Friday morning data)
        5. Merge fresh data into candidates
        6. Score each candidate with expected move analysis
        7. Determine direction for each
        8. Apply diversity filter
        9. For top 3, call find_optimal_strikes
        10. Generate report and send to Discord
        11. Save picks for tracking

        Returns
        -------
        list[dict]
            The final picks.
        """
        date_str = self._current_date_str()
        logger.info("=== FRIDAY PICKS — %s ===", date_str)

        # 1. Load Thursday's refreshed data (falls back to Wednesday if Thursday didn't run)
        candidates = self._load_previous_stage("thursday_refresh")
        if not candidates:
            candidates = self._load_previous_stage("wednesday_scan")
        if not candidates:
            logger.error("No candidates found — run Wednesday or Thursday stage first")
            return []

        # 2. Fresh market summary
        market_summary = self.market_scanner.get_market_summary()
        vix_regime = market_summary.get("vix_regime", {})

        # 3. Re-run technical scan (fresh data)
        tech_scanner = self._get_technical_scanner()
        if tech_scanner:
            tickers = [c["ticker"] for c in candidates]
            tech_results = self._batch_scan(tech_scanner, "scan_ticker", tickers)
        else:
            tech_results = {}
            logger.warning("Skipping fresh technical scan — scanner unavailable")

        # 4. Re-run options chain analysis (fresh Friday morning data)
        opts_scanner = self._get_options_scanner()
        if opts_scanner:
            tickers = [c["ticker"] for c in candidates]
            opts_results = self._batch_scan(opts_scanner, "analyze_chain", tickers, rate_limit=0.3)
        else:
            opts_results = {}
            logger.warning("Skipping fresh options scan — scanner unavailable")

        # 4b. Fresh Finviz scan (pre-market data is most valuable on Friday morning)
        finviz_results = {}
        finviz_scanner = self._get_finviz_scanner()
        if finviz_scanner:
            tickers = [c["ticker"] for c in candidates]
            finviz_results = self._batch_scan(finviz_scanner, "scan_ticker", tickers, rate_limit=0.5)
        else:
            logger.warning("Skipping fresh Finviz scan — scanner unavailable")

        # 4c. Fresh flow scan (Friday morning unusual volume = institutional positioning)
        flow_results = {}
        flow_scanner = self._get_flow_scanner()
        if flow_scanner:
            tickers = [c["ticker"] for c in candidates]
            flow_results = self._batch_scan(flow_scanner, "detect_unusual_volume", tickers, rate_limit=0.3)
        else:
            logger.warning("Skipping flow scan — scanner unavailable")

        # 5. Merge fresh data into candidates
        for c in candidates:
            ticker = c["ticker"]
            if ticker in tech_results:
                c["technical"] = tech_results[ticker]
            if ticker in opts_results:
                c["options"] = opts_results[ticker]
            if ticker in finviz_results:
                c["finviz"] = finviz_results[ticker]
            if ticker in flow_results:
                c["flow"] = self._map_flow_data(flow_results[ticker])
            c["market_regime"] = vix_regime

        # 6. Regime gate — assess whether macro environment gives us an edge
        self.scorer.set_market_summary(market_summary)
        macro_edge = self.scorer.assess_macro_edge(market_summary)
        confidence_mult = macro_edge["confidence_multiplier"]

        for reason in macro_edge["reasons"]:
            logger.info("Macro assessment: %s", reason)
        if not macro_edge["has_edge"]:
            logger.warning("REGIME GATE: model has reduced edge in current environment "
                           "(confidence multiplier: %.2f)", confidence_mult)

        # 7. Score each candidate with expected move analysis
        scored = []
        for c in candidates:
            # 8. Determine direction for each
            direction_info = self.scorer.determine_direction(c)
            c["direction"] = direction_info["direction"]
            # Apply macro regime gate to confidence
            c["direction_confidence"] = round(direction_info["confidence"] * confidence_mult, 3)
            c["raw_confidence"] = direction_info.get("raw_confidence", direction_info["confidence"])
            c["direction_hint"] = direction_info["direction"]
            c["macro_edge"] = macro_edge
            scored_c = self.scorer.score_candidate(c)
            scored.append(scored_c)

        # Sort by composite score
        scored.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        # 8. Apply diversity filter
        # Build sector map from candidates
        sector_map = {}
        for c in scored:
            ticker = c.get("ticker", "")
            sector = c.get("sector", "unknown")
            sector_map[ticker] = sector

        # Use the narrowing pipeline's friday_picks filter (handles diversity + direction mix)
        top3 = self.narrower.narrow(scored, "friday_picks")

        # 9. For top 3, call find_optimal_strikes
        picks = []
        for c in top3:
            ticker = c.get("ticker")
            direction = c.get("direction", "call")
            opts = c.get("options", {})

            # Map call/put to bullish/bearish for the options scanner
            scanner_direction = "bullish" if direction == "call" else "bearish"

            # Get optimal strike data
            strike_data = {}
            if opts_scanner:
                try:
                    strike_data = opts_scanner.find_optimal_strikes(ticker, scanner_direction)
                except Exception as exc:
                    logger.error("Failed to get strikes for %s: %s", ticker, exc)
                    strike_data = {}

            pick = dict(c)
            pick["ticker"] = ticker
            pick["direction"] = direction
            pick["strike"] = strike_data.get("strike")
            pick["expiry"] = opts.get("expiry") or strike_data.get("expiry")
            pick["premium"] = strike_data.get("premium") or strike_data.get("mid_price")
            pick["composite_score"] = c.get("composite_score", 0)
            pick["confidence"] = c.get("direction_confidence", c.get("confidence", 0))
            # Execution guidance from improved strike selector
            pick["current_price"] = strike_data.get("current_price")
            pick["breakeven"] = strike_data.get("breakeven")
            pick["breakeven_move_pct"] = strike_data.get("breakeven_move_pct")
            pick["expected_daily_move_pct"] = strike_data.get("expected_daily_move_pct")
            pick["estimated_delta"] = strike_data.get("estimated_delta")
            pick["entry"] = strike_data.get("entry", {})
            pick["exit"] = strike_data.get("exit", {})
            picks.append(pick)

        # 10. Generate report and send to Discord
        report = self._generate_report(picks, market_summary, date_str)

        # Save report
        report_path = config.reports_dir / f"{date_str}_picks.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Report saved to %s", report_path)

        try:
            self.notifier.send_picks(picks, market_summary)
        except Exception as exc:
            logger.error("Failed to send Discord notification: %s", exc)

        # 11. Save picks for tracking
        self.tracker.record_picks(date_str, picks)
        self.narrower.save_stage_results("friday_picks", picks, date_str)

        # 12. Record to database
        try:
            from tracking.database import Database
            db = Database()
            db.record_picks(date_str, picks, market_summary)
        except Exception as exc:
            logger.error("Failed to record picks to database: %s", exc)

        logger.info("Friday picks complete: %d picks", len(picks))
        return picks

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
            "wednesday": self.wednesday_scan,
            "thursday": self.thursday_refresh,
            "friday": self.friday_picks,
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
        # 2 = Wednesday, 3 = Thursday, 4 = Friday
        if weekday_num == 2:
            return self.wednesday_scan()
        elif weekday_num == 3:
            return self.thursday_refresh()
        elif weekday_num == 4:
            return self.friday_picks()
        else:
            logger.info(
                "Today is weekday %d — no pipeline stage scheduled (Wed=2, Thu=3, Fri=4).",
                weekday_num,
            )
            return []

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _generate_report(self, picks: list, market_summary: dict,
                         date_str: str) -> dict:
        """Build a structured report dict for the final picks."""
        vix = market_summary.get("vix", {})
        regime = market_summary.get("vix_regime", {})

        pick_summaries = []
        for p in picks:
            pick_summaries.append({
                "ticker": p.get("ticker"),
                "direction": p.get("direction", "unknown"),
                "strike": p.get("strike"),
                "expiry": p.get("expiry"),
                "premium": p.get("premium"),
                "composite_score": p.get("composite_score", 0),
                "confidence": p.get("confidence", 0),
                "current_price": p.get("current_price"),
                "breakeven": p.get("breakeven"),
                "breakeven_move_pct": p.get("breakeven_move_pct"),
                "expected_daily_move_pct": p.get("expected_daily_move_pct"),
                "estimated_delta": p.get("estimated_delta"),
                "entry": p.get("entry", {}),
                "exit": p.get("exit", {}),
            })

        # Include macro edge assessment if available on picks
        macro_edge = picks[0].get("macro_edge") if picks else {}

        return {
            "date": date_str,
            "market_context": {
                "vix": vix.get("current"),
                "regime": regime.get("regime"),
                "trend": vix.get("trend_direction"),
                "macro_edge": macro_edge,
            },
            "picks": pick_summaries,
            "generated_at": datetime.now().isoformat(),
        }
