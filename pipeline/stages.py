"""
Pipeline stages for the Weekly Options Trading Analysis System.

Three-stage weekly analysis cycle (Wed/Fri/Mon cadence):
    Wednesday scan    - Broad universe scan, narrow to 25 candidates + 10 bench
    Friday refresh    - Delta-aware re-ranking: compare Wed vs Fri data on 35
                        (25 + bench), re-rank to best 20 based on setup evolution
    Monday picks      - Fresh data on 20 candidates, score, diversify, pick top 3
                        high-conviction with strike selection for weekly (Mon→Fri)
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
from tracking.agent_tracker import log_decision
from tracking.tracker import PerformanceTracker

logger = logging.getLogger(__name__)
config = Config()


class DailyStages:
    """Orchestrates the Wednesday/Friday/Monday weekly pipeline."""

    def __init__(self):
        self.market_scanner = MarketScanner()
        self.scorer = CandidateScorer()
        self.narrower = NarrowingPipeline()
        self.notifier = DiscordNotifier()
        self.tracker = PerformanceTracker()

        # Refresh dynamic agent context (CURRENT_STATE.md, TRADE_LOG.md)
        try:
            from agents.state_generator import refresh_agent_context
            refresh_agent_context()
        except Exception as exc:
            logger.debug("Agent context refresh skipped: %s", exc)

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
        for days_back in range(7):
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

        1. Get market summary (including holding window events for Mon-Fri)
        2. Get core universe (~103) + dynamic additions
        3. Get sector map
        4. Run technical scan batch (with weekly momentum signals)
        5. Run options chain analysis batch (rate_limit=0.3)
        6. Run sentiment, finviz, EDGAR, flow scans
        7. Build complete data dict for each candidate
        8. Narrow to 25 + 10 bench using weekly scan_rank
        9. Save results and market summary

        Returns
        -------
        list[dict]
            The narrowed candidate list (25 candidates).
        """
        date_str = self._current_date_str()
        logger.info("=== WEDNESDAY SCAN — %s ===", date_str)

        # 1. Market summary (includes holding window events, regime persistence)
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

        # 1b. Generate market narrative (agent-powered)
        market_narrative = ""
        try:
            from agents.market_narrative import MarketNarrative
            narrator = MarketNarrative()
            market_narrative = narrator.narrate(market_summary)
            if market_narrative:
                logger.info("Market narrative generated (%d chars)", len(market_narrative))
                log_decision(
                    agent_name="market_narrative",
                    ticker="MARKET",
                    mechanical_signal="raw_data",
                    agent_signal="narrative_generated",
                    override_occurred=False,  # Narrative is additive, not override
                    context={"length": len(market_narrative)},
                )
        except Exception as exc:
            logger.error("Market narrative agent failed: %s", exc)

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
        for d in dynamic_additions:
            sector_by_ticker[d["ticker"]] = "dynamic"

        # 4. Technical scan batch (3mo lookback for weekly signals)
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

        # 6. Sentiment, Finviz, EDGAR, flow scans
        sent_results = {}
        sent_scanner = self._get_sentiment_scanner()
        if sent_scanner:
            # Pre-crawl social media once (Reddit + Twitter/UW) before per-ticker scans
            try:
                sent_scanner.load_social_data(tickers=tickers)
            except Exception as exc:
                logger.warning("Social crawl failed (non-fatal): %s", exc)
            sent_results = self._batch_scan(sent_scanner, "analyze_ticker_sentiment", tickers, rate_limit=1.5)
        else:
            logger.warning("Skipping sentiment scan — scanner unavailable")

        finviz_results = {}
        finviz_scanner = self._get_finviz_scanner()
        if finviz_scanner:
            finviz_results = self._batch_scan(finviz_scanner, "scan_ticker", tickers, rate_limit=0.5)
        else:
            logger.warning("Skipping Finviz scan — scanner unavailable")

        edgar_results = {}
        edgar_scanner = self._get_edgar_scanner()
        if edgar_scanner:
            edgar_results = self._batch_scan(edgar_scanner, "get_insider_trades", tickers, rate_limit=0.3)
        else:
            logger.warning("Skipping EDGAR scan — scanner unavailable")

        flow_results = {}
        flow_scanner = self._get_flow_scanner()
        if flow_scanner:
            flow_results = self._batch_scan(flow_scanner, "detect_unusual_volume", tickers, rate_limit=0.3)
        else:
            logger.warning("Skipping flow scan — scanner unavailable")

        # 7. Build complete candidate dicts
        candidates = []
        for ticker in tickers:
            sector = sector_by_ticker.get(ticker, "unknown")
            candidate = {
                "ticker": ticker,
                "technical": tech_results.get(ticker, {}),
                "options": opts_results.get(ticker, {}),
                "sentiment": sent_results.get(ticker, {}),
                "finviz": finviz_results.get(ticker, {}),
                "insider": edgar_results.get(ticker, {}),
                "flow": self._map_flow_data(flow_results.get(ticker, {})),
                "sector": sector,
            }
            candidates.append(candidate)

        # 8. Narrow to 25 + save bench for Friday refresh
        candidates, bench = self.narrower.narrow_with_bench(candidates, "wednesday_scan")

        # 9. Save results
        self.narrower.save_stage_results("wednesday_scan", candidates, date_str)
        if bench:
            self.narrower.save_stage_results("wednesday_bench", bench, date_str)
            logger.info("Saved %d bench candidates for Friday refresh", len(bench))

        # 10. Run Earnings Analyst on candidates with earnings in holding window
        try:
            from agents.earnings_analyst import EarningsAnalyst
            earnings_agent = EarningsAnalyst()
            earnings_briefs = earnings_agent.analyze(candidates, market_summary)
            if earnings_briefs:
                # Attach earnings assessments to candidates
                brief_map = {b["ticker"]: b for b in earnings_briefs}
                for c in candidates:
                    brief = brief_map.get(c.get("ticker"))
                    if brief:
                        c["earnings_agent"] = brief
                        avoid = brief["signal"] == "AVOID"
                        if avoid:
                            c["earnings_warning"] = True
                            logger.warning("Earnings AVOID flag for %s: %s",
                                           brief["ticker"], brief["brief"][:80])
                        log_decision(
                            agent_name="earnings_analyst",
                            ticker=brief["ticker"],
                            mechanical_signal="INCLUDED",
                            agent_signal=brief["signal"],
                            override_occurred=avoid,
                            context={"brief": brief["brief"][:200]},
                        )
                logger.info("Earnings analyst assessed %d tickers", len(earnings_briefs))
        except Exception as exc:
            logger.error("Earnings analyst failed: %s", exc)

        logger.info("Wednesday scan complete: %d candidates", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    #  Friday — Delta-aware refresh
    # ------------------------------------------------------------------

    def friday_refresh(self) -> list:
        """Delta-aware re-evaluation of Wednesday's 25 candidates + bench.

        Compares Wednesday's data snapshot against Friday's fresh scans
        to measure how each setup is evolving toward Monday's entry.
        Candidates with strengthening setups rise; those deteriorating fall.
        Bench candidates can be promoted if their evolution outpaces incumbents.

        1. Load Wednesday's 25 candidates + bench (~10 alternates)
        2. Snapshot Wednesday's data for delta comparison
        3. Re-run technical, options, sentiment, finviz scans on all ~35
        4. Merge fresh data (preserving Wednesday snapshot)
        5. Re-rank using delta-aware Friday scoring
        6. Take best 20, save new bench from remainder
        7. Log promotions / demotions

        Returns
        -------
        list[dict]
            The re-evaluated candidate list (best 20 from the pool).
        """
        date_str = self._current_date_str()
        logger.info("=== FRIDAY REFRESH — %s ===", date_str)

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
            "Friday pool: %d incumbents + %d bench = %d total",
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

        # 3. Fresh scans — technical, options, sentiment, finviz, flow
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
            try:
                sent_scanner.load_social_data(tickers=tickers)
            except Exception as exc:
                logger.warning("Social crawl failed (non-fatal): %s", exc)
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

        # 5. Re-rank using delta-aware Friday scoring
        top20, new_bench = self.narrower.narrow_with_bench(pool, "friday_refresh")

        # 6. Log promotions and demotions
        new_tickers = {c["ticker"] for c in top20}
        promoted = new_tickers - wednesday_tickers
        demoted = wednesday_tickers - new_tickers

        if promoted:
            promoted_details = []
            for c in top20:
                if c["ticker"] in promoted:
                    score = c.get("friday_score", 0)
                    promoted_details.append(f"{c['ticker']} ({score:.3f})")
            logger.info("PROMOTED from bench: %s", ", ".join(promoted_details))
        if demoted:
            logger.info("DEMOTED to bench: %s", ", ".join(sorted(demoted)))
        if not promoted and not demoted:
            logger.info("No changes — same candidates held their positions")

        # Log top candidates by friday_score
        for i, c in enumerate(top20[:5], 1):
            components = c.get("friday_components", {})
            logger.info(
                "  #%d %s  score=%.3f  [setup=%.2f trend=%.2f iv=%.2f opts=%.2f quality=%.2f sent=%.2f]",
                i, c["ticker"], c.get("friday_score", 0),
                components.get("setup_momentum", 0),
                components.get("trend_evolution", 0),
                components.get("iv_premium_trajectory", 0),
                components.get("options_positioning", 0),
                components.get("base_quality", 0),
                components.get("sentiment_evolution", 0),
            )

        # 7. Save
        self.narrower.save_stage_results("friday_refresh", top20, date_str)
        if new_bench:
            self.narrower.save_stage_results("friday_bench", new_bench, date_str)

        logger.info("Friday refresh complete: %d candidates (%d promoted, %d demoted)",
                     len(top20), len(promoted), len(demoted))
        return top20

    # ------------------------------------------------------------------
    #  Monday — Final picks (entry day)
    # ------------------------------------------------------------------

    def monday_picks(self) -> list:
        """Run the Monday final-pick stage (entry day for weekly options).

        1. Load Friday's refreshed data (falls back to Wednesday if Friday didn't run)
        2. Get fresh market summary (including holding window Mon→Fri events)
        3. Re-run technical scan on 20 candidates (fresh Monday morning data)
        4. Re-run options chain analysis (Monday open data, weekly expiry)
        5. Run finviz and flow scans (Monday pre-market/open data)
        6. Merge fresh data into candidates
        7. Assess macro regime gate
        8. Determine direction and score each candidate
        9. Apply monday_picks filter (diversity, direction balance, earnings guard)
        10. For top 3, call find_optimal_strikes with weekly expiry
        11. Generate report and send to Discord
        12. Save picks for tracking

        Returns
        -------
        list[dict]
            The final 3 high-conviction picks for the week.
        """
        date_str = self._current_date_str()
        logger.info("=== MONDAY PICKS — %s ===", date_str)

        # 1. Load Friday's refreshed data (falls back to Wednesday)
        candidates = self._load_previous_stage("friday_refresh")
        if not candidates:
            candidates = self._load_previous_stage("wednesday_scan")
        if not candidates:
            logger.error("No candidates found — run Wednesday or Friday stage first")
            return []

        # 2. Fresh market summary with holding window for the week ahead
        market_summary = self.market_scanner.get_market_summary()
        vix_regime = market_summary.get("vix_regime", {})

        # 2b. Load most recent market narrative lean (from Wednesday scan).
        # NOT weighted into scoring math — surfaced as a conflict flag to
        # reviewing agents and logged for future empirical evaluation.
        from agents.market_narrative import load_recent_narrative_lean
        narrative_lean, narrative_date = load_recent_narrative_lean()
        if narrative_lean:
            logger.info("Market narrative lean: %s (from %s)", narrative_lean, narrative_date)
            market_summary["narrative_lean"] = narrative_lean
            market_summary["narrative_lean_date"] = narrative_date
        else:
            logger.info("No recent market narrative lean available")

        # Save Monday's market summary
        summary_path = config.candidates_dir / date_str
        summary_path.mkdir(parents=True, exist_ok=True)
        with open(summary_path / "market_summary.json", "w") as fh:
            json.dump(
                {k: v for k, v in market_summary.items()
                 if k != "sectors"},
                fh, indent=2, default=str,
            )

        # 3. Re-run technical scan (fresh Monday data)
        tickers = [c["ticker"] for c in candidates]
        tech_scanner = self._get_technical_scanner()
        if tech_scanner:
            tech_results = self._batch_scan(tech_scanner, "scan_ticker", tickers)
        else:
            tech_results = {}
            logger.warning("Skipping fresh technical scan — scanner unavailable")

        # 4. Re-run options chain analysis (Monday open, weekly expiry)
        opts_scanner = self._get_options_scanner()
        if opts_scanner:
            opts_results = self._batch_scan(opts_scanner, "analyze_chain", tickers, rate_limit=0.3)
        else:
            opts_results = {}
            logger.warning("Skipping fresh options scan — scanner unavailable")

        # 5. Fresh Finviz + flow scans (Monday pre-market/open data)
        finviz_results = {}
        finviz_scanner = self._get_finviz_scanner()
        if finviz_scanner:
            finviz_results = self._batch_scan(finviz_scanner, "scan_ticker", tickers, rate_limit=0.5)
        else:
            logger.warning("Skipping fresh Finviz scan — scanner unavailable")

        flow_results = {}
        flow_scanner = self._get_flow_scanner()
        if flow_scanner:
            flow_results = self._batch_scan(flow_scanner, "detect_unusual_volume", tickers, rate_limit=0.3)
        else:
            logger.warning("Skipping flow scan — scanner unavailable")

        # 5b. Fresh social sentiment (Monday morning — most relevant for entry)
        sent_scanner = self._get_sentiment_scanner()
        if sent_scanner:
            try:
                sent_scanner.load_social_data(tickers=tickers)
                # Update candidates with fresh social intelligence
                social_data = (sent_scanner._social_data or {}).get("ticker_sentiment", {})
                for c in candidates:
                    ticker = c.get("ticker", "")
                    social = social_data.get(ticker, {})
                    if social and social.get("mentions", 0) > 0:
                        c.setdefault("sentiment", {})["social"] = {
                            "narrative": social.get("narrative", ""),
                            "catalysts": social.get("catalysts", []),
                            "risks": social.get("risks", []),
                            "flow_consensus": social.get("flow_consensus", "neutral"),
                            "flow_conviction": social.get("flow_conviction", 0),
                            "flow_signals": social.get("flow_signals", []),
                            "signal_strength": social.get("signal_strength", 0),
                            "top_posts": social.get("top_posts", []),
                            "price_targets": social.get("price_targets", []),
                            "post_types": social.get("post_types", {}),
                            "avg_sentiment": social.get("avg_sentiment", 0),
                            "consensus": social.get("consensus", "unknown"),
                            "mentions": social.get("mentions", 0),
                        }
            except Exception as exc:
                logger.warning("Monday social crawl failed (non-fatal): %s", exc)

        # 6. Merge fresh data into candidates
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
            # Attach holding window, regime persistence, and cross-asset for scoring
            c["holding_window"] = market_summary.get("holding_window", {})
            c["regime_persistence"] = market_summary.get("regime_persistence", {})
            c["cross_asset"] = market_summary.get("cross_asset", {})

        # 7. Regime gate — assess whether macro environment gives us an edge
        self.scorer.set_market_summary(market_summary)
        macro_edge = self.scorer.assess_macro_edge(market_summary)
        confidence_mult = macro_edge["confidence_multiplier"]

        for reason in macro_edge["reasons"]:
            logger.info("Macro assessment: %s", reason)
        if not macro_edge["has_edge"]:
            logger.warning("REGIME GATE: model has reduced edge in current environment "
                           "(confidence multiplier: %.2f)", confidence_mult)

        # 8. Determine direction and score each candidate
        scored = []
        for c in candidates:
            direction_info = self.scorer.determine_direction(c)
            c["direction"] = direction_info["direction"]
            c["direction_confidence"] = round(direction_info["confidence"] * confidence_mult, 3)
            c["raw_confidence"] = direction_info.get("raw_confidence", direction_info["confidence"])
            c["direction_hint"] = direction_info["direction"]
            c["macro_edge"] = macro_edge

            # Tag narrative lean and compute conflict flag (information only —
            # NOT weighted into scoring; reviewed by agents and logged for
            # future empirical decision on whether to integrate as a signal).
            if narrative_lean:
                c["narrative_lean"] = narrative_lean
                direction = c["direction"]
                conflict = (
                    (narrative_lean == "bearish" and direction == "call")
                    or (narrative_lean == "bullish" and direction == "put")
                )
                c["narrative_scoring_conflict"] = conflict

            scored_c = self.scorer.score_candidate(c)
            scored.append(scored_c)

        scored.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        # Log narrative-vs-scoring conflicts for top-ranked candidates only
        if narrative_lean:
            conflicts = [c for c in scored[:10] if c.get("narrative_scoring_conflict")]
            for c in conflicts:
                log_decision(
                    agent_name="market_narrative",
                    ticker=c.get("ticker", "?"),
                    mechanical_signal=f"direction={c['direction']}",
                    agent_signal=f"narrative_lean={narrative_lean}",
                    override_occurred=False,
                    context={"reason": "scoring direction conflicts with narrative lean"},
                )
            if conflicts:
                logger.warning(
                    "Narrative-vs-scoring conflicts in top 10: %s (narrative %s)",
                    ", ".join(c.get("ticker", "?") for c in conflicts),
                    narrative_lean,
                )

        # 9. Apply monday_picks filter (diversity + direction balance + earnings guard)
        top5 = self.narrower.narrow(scored, "monday_picks")

        # 9b. Portfolio Reasoner — agent reviews top candidates for concentration
        #     risk AND writes trading theses (combined into single Opus call)
        portfolio_theses = {}
        try:
            from agents.portfolio_reasoner import PortfolioReasoner
            reasoner = PortfolioReasoner()
            # Give agent top 6 to choose from (select best 3)
            top_candidates = scored[:6] if len(scored) >= 6 else scored
            portfolio_result = reasoner.select(top_candidates, market_summary, macro_edge)
            if portfolio_result.get("reasoning"):
                selected_tickers = set(portfolio_result["selected"])
                mechanical_tickers = [c.get("ticker") for c in top5]
                # Reorder top picks to match agent's recommendation
                from config import PORTFOLIO_SIZE
                agent_top = [c for c in scored if c.get("ticker") in selected_tickers][:PORTFOLIO_SIZE]
                if len(agent_top) == PORTFOLIO_SIZE:
                    agent_tickers = [c.get("ticker") for c in agent_top]
                    override = set(agent_tickers) != set(mechanical_tickers)
                    # Track portfolio selection decision
                    log_decision(
                        agent_name="portfolio_reasoner",
                        ticker="PORTFOLIO",
                        mechanical_signal=",".join(mechanical_tickers),
                        agent_signal=",".join(agent_tickers),
                        override_occurred=override,
                        context={
                            "dropped": portfolio_result.get("dropped", []),
                            "reasoning": portfolio_result.get("reasoning", "")[:300],
                        },
                    )
                    top5 = agent_top
                    logger.info("Portfolio reasoner selected: %s",
                                 ", ".join(t.get("ticker", "?") for t in top5))
                    logger.info("Dropped: %s", ", ".join(portfolio_result.get("dropped", [])))
                portfolio_theses = portfolio_result.get("theses", {})
        except Exception as exc:
            logger.error("Portfolio reasoner failed: %s", exc)

        # 10. For top 3, call find_optimal_strikes with weekly expiry
        picks = []
        for c in top5:
            ticker = c.get("ticker")
            direction = c.get("direction", "call")
            opts = c.get("options", {})

            scanner_direction = "bullish" if direction == "call" else "bearish"

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
            pick["current_price"] = strike_data.get("current_price")
            pick["breakeven"] = strike_data.get("breakeven")
            pick["breakeven_move_pct"] = strike_data.get("breakeven_move_pct")
            pick["expected_daily_move_pct"] = strike_data.get("expected_daily_move_pct")
            pick["estimated_delta"] = strike_data.get("estimated_delta")
            pick["entry"] = strike_data.get("entry", {})
            pick["exit"] = strike_data.get("exit", {})
            # Flag earnings warning if present
            if c.get("earnings_warning"):
                pick["earnings_warning"] = True
            picks.append(pick)

        # 10b. Attach trading theses from Portfolio Reasoner (no extra API call)
        if portfolio_theses:
            for pick in picks:
                thesis = portfolio_theses.get(pick.get("ticker"), "")
                if thesis:
                    pick["thesis"] = thesis

        # 11. Generate report and send to Discord
        report = self._generate_report(picks, market_summary, date_str)

        report_path = config.reports_dir / f"{date_str}_picks.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Report saved to %s", report_path)

        try:
            self.notifier.send_picks(picks, market_summary)
        except Exception as exc:
            logger.error("Failed to send Discord notification: %s", exc)

        # 12. Save picks for tracking
        self.tracker.record_picks(date_str, picks)
        self.narrower.save_stage_results("monday_picks", picks, date_str)

        try:
            from tracking.database import Database
            db = Database()
            db.record_picks(date_str, picks, market_summary)
        except Exception as exc:
            logger.error("Failed to record picks to database: %s", exc)

        logger.info("Monday picks complete: %d picks", len(picks))
        return picks

    # ------------------------------------------------------------------
    #  Monday entry confirmation (10:00 AM ET — 30 min after open)
    # ------------------------------------------------------------------

    def monday_entry_confirmation(self) -> list:
        """Re-validate Monday picks with fresh market-hours data.

        Runs 30 minutes after market open (10:00 AM ET). For each of the
        5 Monday picks, fetches current price via yfinance and checks:

        1. Gap check: did the stock gap > 2% from Friday close? → SKIP
        2. Delta drift: has our strike's estimated delta shifted
           significantly from the 0.35 target? → ADJUST or SKIP
        3. Premium change: is the option now significantly more expensive
           or cheaper than our 8AM estimate? → flag for limit adjustment
        4. Direction validation: does early price action confirm or
           contradict our directional thesis?

        Sends a GO / ADJUST / SKIP Discord message per pick.

        Returns
        -------
        list[dict]
            Updated picks with entry signals.
        """
        date_str = self._current_date_str()
        logger.info("=== MONDAY ENTRY CONFIRMATION — %s ===", date_str)

        # Load this morning's picks
        picks = self._load_previous_stage("monday_picks")
        if not picks:
            logger.warning("No Monday picks found — cannot confirm entries")
            return []

        opts_scanner = self._get_options_scanner()
        confirmations = []

        for pick in picks:
            ticker = pick.get("ticker", "?")
            original_strike = pick.get("strike")
            original_premium = pick.get("premium")
            original_price = pick.get("current_price")
            direction = pick.get("direction", "call")
            original_delta = pick.get("estimated_delta", 0.35)

            logger.info("Confirming entry for %s (%s)...", ticker, direction)

            # If pre-market run failed to get strike data, re-fetch now (market open)
            if not original_strike or not original_premium:
                logger.info(
                    "Strike/premium missing for %s — re-fetching with live market data",
                    ticker,
                )
                if opts_scanner:
                    try:
                        scanner_direction = "bullish" if direction == "call" else "bearish"
                        strike_data = opts_scanner.find_optimal_strikes(ticker, scanner_direction)
                        if strike_data and strike_data.get("strike"):
                            pick["strike"] = strike_data.get("strike")
                            pick["premium"] = strike_data.get("premium")
                            pick["current_price"] = strike_data.get("current_price")
                            pick["breakeven"] = strike_data.get("breakeven")
                            pick["breakeven_move_pct"] = strike_data.get("breakeven_move_pct")
                            pick["expected_daily_move_pct"] = strike_data.get("expected_daily_move_pct")
                            pick["estimated_delta"] = strike_data.get("estimated_delta")
                            pick["iv"] = strike_data.get("iv")
                            pick["expiry"] = strike_data.get("expiry") or pick.get("expiry")
                            pick["entry"] = strike_data.get("entry", {})
                            pick["exit"] = strike_data.get("exit", {})
                            # Update locals for downstream checks
                            original_strike = pick["strike"]
                            original_premium = pick["premium"]
                            original_price = pick.get("current_price")
                            original_delta = pick.get("estimated_delta", 0.35)
                            logger.info(
                                "Re-fetched %s: strike=%.2f premium=%.2f delta=%.3f",
                                ticker, original_strike, original_premium, original_delta,
                            )
                    except Exception as exc:
                        logger.error("Strike re-fetch failed for %s: %s", ticker, exc)

            # Fetch fresh quote
            try:
                import yfinance as yf
                tk = yf.Ticker(ticker)
                hist = tk.history(period="1d", interval="5m")
                if hist.empty:
                    hist = tk.history(period="2d")

                if hist.empty:
                    confirmations.append(self._build_confirmation(
                        pick, "SKIP", "Could not fetch current price data",
                    ))
                    continue

                current_price = float(hist["Close"].iloc[-1])
                open_price = float(hist["Open"].iloc[0])
                day_high = float(hist["High"].max())
                day_low = float(hist["Low"].min())
            except Exception as exc:
                logger.error("Price fetch failed for %s: %s", ticker, exc)
                confirmations.append(self._build_confirmation(
                    pick, "SKIP", f"Price fetch error: {exc}",
                ))
                continue

            # 1. Gap check
            prev_close = original_price or open_price
            gap_pct = ((open_price - prev_close) / prev_close * 100) if prev_close else 0

            if abs(gap_pct) > 2.0:
                confirmations.append(self._build_confirmation(
                    pick, "SKIP",
                    f"Gap {gap_pct:+.1f}% exceeds 2% threshold — move may be priced in",
                    current_price=current_price, gap_pct=gap_pct,
                ))
                continue

            # 2. Delta drift — re-estimate delta at current price
            if original_strike and original_strike > 0:
                from scanners.options import _bsm_greeks, _RISK_FREE_RATE
                expiry_str = pick.get("expiry")
                if expiry_str:
                    try:
                        from datetime import datetime as _dt
                        expiry_dt = _dt.strptime(expiry_str, "%Y-%m-%d").date()
                        T = max((expiry_dt - _dt.now().date()).days, 1) / 365.0
                    except ValueError:
                        T = 5 / 365.0
                else:
                    T = 5 / 365.0

                iv = pick.get("iv") or 0.30
                option_type = "call" if direction == "call" else "put"
                new_greeks = _bsm_greeks(current_price, original_strike, T,
                                          _RISK_FREE_RATE, iv, option_type)
                new_delta = abs(new_greeks["delta"])

                delta_shift = abs(new_delta - abs(original_delta))

                if new_delta < 0.10:
                    confirmations.append(self._build_confirmation(
                        pick, "SKIP",
                        f"Delta collapsed to {new_delta:.2f} — option nearly worthless",
                        current_price=current_price, new_delta=new_delta,
                    ))
                    continue

                if delta_shift > 0.15:
                    # Delta shifted significantly — suggest adjusting strike
                    confirmations.append(self._build_confirmation(
                        pick, "ADJUST",
                        f"Delta shifted from {abs(original_delta):.2f} to {new_delta:.2f} "
                        f"(price moved from ${original_price:.2f} to ${current_price:.2f}). "
                        f"Consider re-selecting strike closer to current price.",
                        current_price=current_price, new_delta=new_delta,
                    ))
                    continue
            else:
                new_delta = original_delta

            # 3. Premium re-estimate
            new_premium = None
            if original_strike and opts_scanner:
                try:
                    from scanners.options import _bsm_price, _RISK_FREE_RATE
                    option_type = "call" if direction == "call" else "put"
                    iv = pick.get("iv") or 0.30
                    new_premium = _bsm_price(current_price, original_strike, T,
                                              _RISK_FREE_RATE, iv, option_type)
                    new_premium = round(new_premium, 2)
                except Exception:
                    pass

            premium_change = ""
            if new_premium and original_premium and original_premium > 0:
                prem_change_pct = ((new_premium - original_premium) / original_premium) * 100
                if abs(prem_change_pct) > 15:
                    premium_change = (
                        f"Premium moved {prem_change_pct:+.0f}% "
                        f"(${original_premium:.2f} → ${new_premium:.2f}). "
                        f"Adjust limit order accordingly."
                    )

            # 4. Direction validation — early price action
            price_move_pct = ((current_price - open_price) / open_price * 100) if open_price else 0
            direction_confirmed = (
                (direction == "call" and price_move_pct > 0)
                or (direction == "put" and price_move_pct < 0)
            )

            if not direction_confirmed and abs(price_move_pct) > 1.0:
                note = (
                    f"Price moving against thesis ({price_move_pct:+.1f}% since open). "
                    f"Proceed with caution or wait for reversal."
                )
            elif direction_confirmed:
                note = f"Early action confirms {direction} thesis ({price_move_pct:+.1f}% since open)."
            else:
                note = f"Flat since open ({price_move_pct:+.1f}%) — no strong confirmation yet."

            detail = note
            if premium_change:
                detail += f" {premium_change}"

            confirmations.append(self._build_confirmation(
                pick, "GO", detail,
                current_price=current_price,
                new_delta=new_delta if isinstance(new_delta, float) else None,
                new_premium=new_premium,
                gap_pct=gap_pct,
            ))

        # Run Pre-Trade Analyst agent — single batched call for all picks.
        # Agent SKIP overrides mechanical GO (agent outputs are actionable).
        try:
            from agents.pre_trade import PreTradeAnalyst
            analyst = PreTradeAnalyst()
            market_summary = None
            try:
                summary_path = config.candidates_dir / date_str / "market_summary.json"
                if summary_path.exists():
                    with open(summary_path) as fh:
                        market_summary = json.load(fh)
            except Exception:
                pass

            if market_summary:
                agent_briefs = analyst.analyze(picks, market_summary, confirmations)
                brief_map = {b["ticker"]: b for b in agent_briefs}
                for conf in confirmations:
                    brief = brief_map.get(conf.get("ticker"))
                    if brief:
                        conf["agent_signal"] = brief["signal"]
                        conf["agent_brief"] = brief["brief"]

                        mechanical = conf["signal"]
                        override = (brief["signal"] == "SKIP" and mechanical == "GO")

                        # Track every pre-trade decision for audit
                        log_decision(
                            agent_name="pre_trade",
                            ticker=conf.get("ticker", "?"),
                            mechanical_signal=mechanical,
                            agent_signal=brief["signal"],
                            override_occurred=override,
                            context={"brief": brief["brief"][:200]},
                        )

                        # Agent SKIP overrides mechanical GO — this is actionable,
                        # not decorative. The agent found a thesis-invalidating
                        # development the mechanical checks couldn't detect.
                        if override:
                            conf["signal"] = "SKIP"
                            conf["detail"] = (
                                f"[AGENT OVERRIDE] {brief['brief']}\n"
                                f"Original mechanical signal: GO — {conf['detail']}"
                            )
                            logger.warning(
                                "Agent SKIP override for %s: %s",
                                conf.get("ticker"), brief["brief"][:80],
                            )
        except Exception as exc:
            logger.error("Pre-trade analyst failed: %s", exc)

        # Re-save monday_picks if any were updated with live strike data
        # (pre-market run may have had missing pricing that was filled in above)
        picks_updated = any(
            p.get("strike") and p.get("premium")
            for p in picks
        )
        if picks_updated:
            self.narrower.save_stage_results("monday_picks", picks, date_str)
            logger.info("Re-saved monday_picks with live strike data")
            # Also update database with pricing
            try:
                from tracking.database import Database
                db = Database()
                db.record_picks(date_str, picks)
            except Exception as exc:
                logger.error("Failed to update picks in database: %s", exc)

        # Send Discord confirmation message
        self._send_entry_confirmations(confirmations)

        # Save confirmations
        conf_path = config.candidates_dir / date_str
        conf_path.mkdir(parents=True, exist_ok=True)
        with open(conf_path / "entry_confirmations.json", "w") as fh:
            json.dump(confirmations, fh, indent=2, default=str)

        go_count = sum(1 for c in confirmations if c["signal"] == "GO")
        skip_count = sum(1 for c in confirmations if c["signal"] == "SKIP")
        adjust_count = sum(1 for c in confirmations if c["signal"] == "ADJUST")
        logger.info(
            "Entry confirmation complete: %d GO, %d ADJUST, %d SKIP",
            go_count, adjust_count, skip_count,
        )
        return confirmations

    @staticmethod
    def _build_confirmation(pick: dict, signal: str, detail: str,
                            current_price: float = None,
                            new_delta: float = None,
                            new_premium: float = None,
                            gap_pct: float = None) -> dict:
        """Build a single entry confirmation dict."""
        return {
            "ticker": pick.get("ticker"),
            "direction": pick.get("direction"),
            "strike": pick.get("strike"),
            "original_premium": pick.get("premium"),
            "original_price": pick.get("current_price"),
            "signal": signal,
            "detail": detail,
            "current_price": round(current_price, 2) if current_price else None,
            "new_delta": round(new_delta, 3) if new_delta else None,
            "new_premium": new_premium,
            "gap_pct": round(gap_pct, 2) if gap_pct else None,
            "timestamp": datetime.now().isoformat(),
        }

    def _send_entry_confirmations(self, confirmations: list) -> None:
        """Send entry confirmation signals to Discord."""
        self.notifier.send_entry_confirmations(confirmations)

    # ------------------------------------------------------------------
    #  Intraday position monitoring (Tue-Fri)
    # ------------------------------------------------------------------

    def position_monitor(self) -> dict:
        """Monitor open positions and generate status alerts.

        Runs at different urgency levels based on day:
        - Monday PM: ROUTINE (just entered, first P&L check)
        - Tue/Wed: ROUTINE (brief status, flag outliers)
        - Thursday: ELEVATED (theta accelerating, aggressive about exits)
        - Friday AM: CRITICAL (everything must close by 2 PM ET)

        Fetches current prices via yfinance, computes estimated option P&L,
        checks stops/targets, and optionally runs the Position Monitor agent
        for qualitative analysis.

        Returns
        -------
        dict
            'positions', 'alerts', 'agent_analysis', 'summary'.
        """
        date_str = self._current_date_str()
        weekday = datetime.now().weekday()

        # Determine urgency
        if weekday == 4:  # Friday
            urgency = "CRITICAL"
        elif weekday == 3:  # Thursday
            urgency = "ELEVATED"
        else:
            urgency = "ROUTINE"

        day_name = datetime.now().strftime("%A")
        logger.info("=== POSITION MONITOR — %s %s [%s] ===", day_name, date_str, urgency)

        # Load this week's picks
        picks = self._load_previous_stage("monday_picks")
        if not picks:
            logger.warning("No Monday picks found — nothing to monitor")
            return {"positions": [], "alerts": [], "agent_analysis": "", "summary": {}}

        # Load market summary if available
        market_summary = None
        try:
            summary_path = config.candidates_dir / date_str / "market_summary.json"
            if summary_path.exists():
                with open(summary_path) as fh:
                    market_summary = json.load(fh)
        except Exception:
            pass

        # Run Position Monitor agent
        try:
            from agents.position_monitor import PositionMonitor
            monitor = PositionMonitor()
            result = monitor.monitor(picks, market_summary=market_summary, urgency=urgency)
        except Exception as exc:
            logger.error("Position monitor failed: %s", exc)
            return {"positions": [], "alerts": [], "agent_analysis": "", "summary": {}}

        # Track position monitor decisions for audit
        for pos in result.get("positions", []):
            if pos.get("status") in ("NO_DATA", "ERROR"):
                continue
            # Mechanical signal is based on thresholds alone
            mech = pos["status"]
            # Agent only runs on ELEVATED/CRITICAL — if it ran, it may have
            # influenced the status interpretation via agent_analysis
            has_agent = bool(result.get("agent_analysis"))
            log_decision(
                agent_name="position_monitor",
                ticker=pos.get("ticker", "?"),
                mechanical_signal=mech,
                agent_signal=mech if not has_agent else f"{mech}+AGENT_REVIEW",
                override_occurred=False,  # Monitor doesn't override yet, it advises
                context={
                    "urgency": urgency,
                    "option_return_pct": pos.get("option_return_pct"),
                    "has_agent_analysis": has_agent,
                },
            )

        # Send Discord notification
        self._send_monitor_update(result, urgency)

        # Save monitor snapshot
        monitor_dir = config.candidates_dir / date_str
        monitor_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M")
        with open(monitor_dir / f"monitor_{timestamp}.json", "w") as fh:
            json.dump(result, fh, indent=2, default=str)

        alerts = result.get("alerts", [])
        if alerts:
            for alert in alerts:
                logger.warning("ALERT: %s", alert)
        else:
            logger.info("No alerts — all positions within bounds")

        return result

    def _send_monitor_update(self, result: dict, urgency: str) -> None:
        """Send position monitoring update to Discord."""
        self.notifier.send_monitor_update(result, urgency)

    # ------------------------------------------------------------------
    #  Friday final exit (1:30 PM ET)
    # ------------------------------------------------------------------

    def final_exit(self) -> dict:
        """Generate mandatory exit instructions for all open positions.

        Runs Friday at 1:30 PM ET — 30 minutes before the 2 PM hard
        deadline for closing weekly options. Every position gets an
        explicit CLOSE instruction regardless of P&L.

        Returns
        -------
        dict
            'positions', 'alerts', 'agent_analysis', 'summary'.
        """
        date_str = self._current_date_str()
        logger.info("=== FINAL EXIT — %s (Friday 1:30 PM ET) ===", date_str)

        picks = self._load_previous_stage("monday_picks")
        if not picks:
            logger.warning("No Monday picks found — nothing to close")
            return {"positions": [], "alerts": [], "agent_analysis": "", "summary": {}}

        # Run position monitor at CRITICAL urgency
        try:
            from agents.position_monitor import PositionMonitor
            monitor = PositionMonitor()
            result = monitor.monitor(picks, urgency="CRITICAL")
        except Exception as exc:
            logger.error("Final exit monitor failed: %s", exc)
            return {"positions": [], "alerts": [], "agent_analysis": "", "summary": {}}

        # Override: ALL positions get CLOSE status
        for p in result.get("positions", []):
            if p.get("status") not in ("NO_DATA", "ERROR"):
                ret = p.get("option_return_pct", 0)
                p["status"] = "CLOSE"
                p["detail"] = (
                    f"MANDATORY CLOSE by 2:00 PM ET. "
                    f"Current P&L: {ret:+.0f}%. "
                    f"Use market order if needed — do not let expire unmanaged."
                )

        # Send urgent Discord notification
        self._send_final_exit(result)

        # Save
        exit_dir = config.candidates_dir / date_str
        exit_dir.mkdir(parents=True, exist_ok=True)
        with open(exit_dir / "final_exit.json", "w") as fh:
            json.dump(result, fh, indent=2, default=str)

        logger.info("Final exit instructions generated for %d positions",
                     len(result.get("positions", [])))
        return result

    def _send_final_exit(self, result: dict) -> None:
        """Send final exit alert to Discord."""
        self.notifier.send_final_exit(result)

    # ------------------------------------------------------------------
    #  Stage dispatch
    # ------------------------------------------------------------------

    def run_stage(self, stage_name: str) -> list:
        """Run a specific stage by name.

        Parameters
        ----------
        stage_name : str
            One of: wednesday, friday, monday, confirm.
        """
        dispatch = {
            "wednesday": self.wednesday_scan,
            "friday": self.friday_refresh,
            "monday": self.monday_picks,
            "confirm": self.monday_entry_confirmation,
            "monitor": self.position_monitor,
            "final_exit": self.final_exit,
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
            Explicit stage name (e.g. 'wednesday', 'friday', 'monday').
            If None, uses today's actual weekday.
        """
        if day is not None:
            return self.run_stage(day)

        weekday_num = datetime.now().weekday()
        # 0=Monday, 2=Wednesday, 4=Friday
        if weekday_num == 0:
            return self.monday_picks()
        elif weekday_num == 2:
            return self.wednesday_scan()
        elif weekday_num == 4:
            return self.friday_refresh()
        else:
            logger.info(
                "Today is weekday %d — no pipeline stage scheduled (Mon=0, Wed=2, Fri=4).",
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
                "earnings_warning": p.get("earnings_warning", False),
                # Per-model scores for post-hoc model comparison
                "ensemble": p.get("ensemble"),
                "scores": p.get("scores"),
                "pattern": p.get("pattern"),
                "active_regime": p.get("active_regime"),
            })

        macro_edge = picks[0].get("macro_edge") if picks else {}

        return {
            "date": date_str,
            "market_context": {
                "vix": vix.get("current"),
                "regime": regime.get("regime"),
                "trend": vix.get("trend_direction"),
                "holding_window": market_summary.get("holding_window", {}),
                "regime_persistence": market_summary.get("regime_persistence", {}),
                "macro_edge": macro_edge,
            },
            "picks": pick_summaries,
            "generated_at": datetime.now().isoformat(),
        }
