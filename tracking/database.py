"""
SQLite database for the Zero-DTE Options Trading Analysis System.

Persistent backend tracking all picks, outcomes, weekly results,
and all-time performance. Single file at data/performance/zero_dte.db.
"""

import sys
import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)
config = Config()

DB_PATH = config.performance_dir / "zero_dte.db"


class Database:
    """SQLite-backed store for all pipeline and scorecard data."""

    def __init__(self, db_path: "str | None" = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS picks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pick_date TEXT NOT NULL,
                expiry TEXT,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                strike REAL,
                entry_premium REAL,
                composite_score REAL,
                confidence REAL,
                sector TEXT,
                earnings_warning INTEGER DEFAULT 0,
                pattern_key TEXT,
                -- Per-model ensemble scores (for model comparison)
                model_linear REAL,
                model_momentum REAL,
                model_reversion REAL,
                model_std REAL,
                -- Outcome fields (filled in after expiry)
                closing_price REAL,
                exit_value REAL,
                pnl REAL,
                pnl_pct REAL,
                result TEXT,
                graded_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS weekly_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pick_date TEXT UNIQUE NOT NULL,
                expiry TEXT,
                total_picks INTEGER,
                wins INTEGER,
                losses INTEGER,
                partials INTEGER,
                total_cost REAL,
                total_exit REAL,
                total_pnl REAL,
                total_return_pct REAL,
                win_rate REAL,
                graded_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                vix REAL,
                vix_regime TEXT,
                vix_trend TEXT,
                breadth_signal TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(pick_date);
            CREATE INDEX IF NOT EXISTS idx_picks_ticker ON picks(ticker);
            CREATE INDEX IF NOT EXISTS idx_weekly_date ON weekly_results(pick_date);
        """)

        # Lightweight migrations for pre-existing DBs (pre-2026-04-15 fork)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(picks)").fetchall()}
        if "earnings_warning" not in cols:
            conn.execute("ALTER TABLE picks ADD COLUMN earnings_warning INTEGER DEFAULT 0")
        if "pattern_key" not in cols:
            conn.execute("ALTER TABLE picks ADD COLUMN pattern_key TEXT")
        if "model_linear" not in cols:
            conn.execute("ALTER TABLE picks ADD COLUMN model_linear REAL")
            conn.execute("ALTER TABLE picks ADD COLUMN model_momentum REAL")
            conn.execute("ALTER TABLE picks ADD COLUMN model_reversion REAL")
            conn.execute("ALTER TABLE picks ADD COLUMN model_std REAL")

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    #  Record picks (Friday pipeline output)
    # ------------------------------------------------------------------

    def record_picks(self, pick_date: str, picks: list, market_summary: dict = None):
        """Store Friday's picks into the database.

        Parameters
        ----------
        pick_date : str
            Date picks were generated (YYYY-MM-DD).
        picks : list[dict]
            The final picks from the pipeline.
        market_summary : dict, optional
            Market context at time of picks.
        """
        conn = self._connect()

        # Clear any existing picks for this date (re-run safe)
        conn.execute("DELETE FROM picks WHERE pick_date = ?", (pick_date,))

        for p in picks:
            pattern_key = (p.get("pattern") or {}).get("pattern_key") or p.get("pattern_key")
            earnings_warning = 1 if p.get("earnings_warning") else 0
            ensemble = p.get("ensemble") or {}
            conn.execute("""
                INSERT INTO picks (pick_date, expiry, ticker, direction, strike,
                                   entry_premium, composite_score, confidence, sector,
                                   earnings_warning, pattern_key,
                                   model_linear, model_momentum, model_reversion, model_std)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pick_date,
                p.get("expiry"),
                p.get("ticker"),
                p.get("direction"),
                p.get("strike"),
                p.get("premium") or p.get("entry_premium"),
                p.get("composite_score"),
                p.get("confidence") or p.get("direction_confidence"),
                p.get("sector"),
                earnings_warning,
                pattern_key,
                ensemble.get("linear"),
                ensemble.get("momentum"),
                ensemble.get("reversion"),
                ensemble.get("model_std"),
            ))

        # Save market snapshot
        if market_summary:
            conn.execute("DELETE FROM market_snapshots WHERE date = ?", (pick_date,))
            vix = market_summary.get("vix", {})
            regime = market_summary.get("vix_regime", {})
            breadth = market_summary.get("breadth", {})
            conn.execute("""
                INSERT INTO market_snapshots (date, vix, vix_regime, vix_trend, breadth_signal)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pick_date,
                vix.get("current"),
                regime.get("regime"),
                vix.get("trend_direction"),
                breadth.get("breadth_signal"),
            ))

        conn.commit()
        conn.close()
        logger.info("Recorded %d picks for %s to database", len(picks), pick_date)

    # ------------------------------------------------------------------
    #  Grade picks (after expiry)
    # ------------------------------------------------------------------

    def grade_picks(self, pick_date: str, graded_picks: list):
        """Update picks with outcome data after grading.

        Parameters
        ----------
        pick_date : str
            The pick date to update.
        graded_picks : list[dict]
            Graded pick results from Scorecard.grade_week().
        """
        conn = self._connect()

        for p in graded_picks:
            conn.execute("""
                UPDATE picks
                SET closing_price = ?,
                    exit_value = ?,
                    pnl = ?,
                    pnl_pct = ?,
                    result = ?,
                    graded_at = ?
                WHERE pick_date = ? AND ticker = ?
            """, (
                p.get("closing_price"),
                p.get("exit_value_total"),
                p.get("pnl"),
                p.get("pnl_pct"),
                p.get("result"),
                datetime.now().isoformat(),
                pick_date,
                p.get("ticker"),
            ))

        conn.commit()
        conn.close()
        logger.info("Graded %d picks for %s in database", len(graded_picks), pick_date)

    def record_weekly_result(self, weekly: dict):
        """Store or update a weekly result summary.

        Parameters
        ----------
        weekly : dict
            Weekly result from Scorecard.grade_week().
        """
        conn = self._connect()

        conn.execute("""
            INSERT OR REPLACE INTO weekly_results
                (pick_date, expiry, total_picks, wins, losses, partials,
                 total_cost, total_exit, total_pnl, total_return_pct, win_rate, graded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            weekly.get("pick_date"),
            weekly.get("expiry"),
            len(weekly.get("picks", [])),
            weekly.get("wins"),
            weekly.get("losses"),
            weekly.get("partials"),
            weekly.get("total_cost"),
            weekly.get("total_exit"),
            weekly.get("total_pnl"),
            weekly.get("total_return_pct"),
            weekly.get("win_rate"),
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    #  Queries
    # ------------------------------------------------------------------

    def get_alltime_stats(self) -> dict:
        """Compute all-time stats from the database."""
        conn = self._connect()

        row = conn.execute("""
            SELECT
                COUNT(*) as total_picks,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result = 'PARTIAL' THEN 1 ELSE 0 END) as partials,
                SUM(entry_premium * 100) as total_invested,
                SUM(exit_value) as total_returned,
                SUM(pnl) as total_pnl,
                MIN(pnl) as worst_pnl,
                MAX(pnl) as best_pnl
            FROM picks
            WHERE result IS NOT NULL
        """).fetchone()

        total_weeks = conn.execute(
            "SELECT COUNT(*) FROM weekly_results"
        ).fetchone()[0]

        # Best and worst individual picks
        best = conn.execute("""
            SELECT ticker, pick_date, pnl, pnl_pct
            FROM picks WHERE result IS NOT NULL
            ORDER BY pnl DESC LIMIT 1
        """).fetchone()

        worst = conn.execute("""
            SELECT ticker, pick_date, pnl, pnl_pct
            FROM picks WHERE result IS NOT NULL
            ORDER BY pnl ASC LIMIT 1
        """).fetchone()

        # Direction breakdown
        direction_stats = {}
        for direction in ("call", "put"):
            d_row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl
                FROM picks
                WHERE direction = ? AND result IS NOT NULL
            """, (direction,)).fetchone()
            direction_stats[direction] = {
                "total": d_row["total"],
                "wins": d_row["wins"],
                "win_rate": round(d_row["wins"] / d_row["total"], 4) if d_row["total"] > 0 else 0,
                "total_pnl": round(d_row["total_pnl"] or 0, 2),
            }

        # Sector breakdown
        sector_stats = {}
        for s_row in conn.execute("""
            SELECT sector,
                   COUNT(*) as total,
                   SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as total_pnl
            FROM picks
            WHERE result IS NOT NULL AND sector IS NOT NULL
            GROUP BY sector
            ORDER BY total DESC
        """).fetchall():
            sector_stats[s_row["sector"]] = {
                "total": s_row["total"],
                "wins": s_row["wins"],
                "win_rate": round(s_row["wins"] / s_row["total"], 4) if s_row["total"] > 0 else 0,
                "total_pnl": round(s_row["total_pnl"] or 0, 2),
            }

        conn.close()

        total_invested = row["total_invested"] or 0
        total_pnl = row["total_pnl"] or 0

        return {
            "total_weeks": total_weeks,
            "total_picks": row["total_picks"],
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "partials": row["partials"] or 0,
            "win_rate": round((row["wins"] or 0) / row["total_picks"], 4) if row["total_picks"] > 0 else 0,
            "total_invested": round(total_invested, 2),
            "total_returned": round(row["total_returned"] or 0, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((total_pnl / total_invested * 100) if total_invested > 0 else 0, 2),
            "best_pick": dict(best) if best else None,
            "worst_pick": dict(worst) if worst else None,
            "by_direction": direction_stats,
            "by_sector": sector_stats,
        }

    def get_weekly_results(self, limit: int = 52) -> list:
        """Get weekly results, newest first."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM weekly_results
            ORDER BY pick_date DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_picks_for_week(self, pick_date: str) -> list:
        """Get all picks for a specific week."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM picks
            WHERE pick_date = ?
            ORDER BY composite_score DESC
        """, (pick_date,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_ticker_history(self, ticker: str) -> list:
        """Get all picks for a specific ticker across all weeks."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM picks
            WHERE ticker = ?
            ORDER BY pick_date DESC
        """, (ticker,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_streak(self) -> dict:
        """Get current win/loss streak."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT pick_date, total_pnl
            FROM weekly_results
            ORDER BY pick_date DESC
        """).fetchall()
        conn.close()

        if not rows:
            return {"type": "none", "count": 0}

        # Check if current streak is winning or losing
        streak_type = "win" if rows[0]["total_pnl"] >= 0 else "loss"
        count = 0
        for r in rows:
            if (r["total_pnl"] >= 0) == (streak_type == "win"):
                count += 1
            else:
                break

        return {"type": streak_type, "count": count}

    def get_monthly_summary(self) -> list:
        """Aggregate results by month."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT
                strftime('%Y-%m', pick_date) as month,
                COUNT(*) as weeks,
                SUM(total_picks) as picks,
                SUM(wins) as wins,
                SUM(losses) as losses,
                SUM(total_cost) as invested,
                SUM(total_pnl) as pnl,
                ROUND(SUM(total_pnl) / SUM(total_cost) * 100, 2) as roi
            FROM weekly_results
            GROUP BY strftime('%Y-%m', pick_date)
            ORDER BY month DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
