"""
Finviz quote-page scraper.

Scrapes free Finviz data for stock tickers — fundamentals, analyst info,
insider activity, short interest, news headlines, and volatility metrics.
Uses only requests + BeautifulSoup (no third-party finviz libraries).
"""

from __future__ import annotations

import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from config import Config

logger = logging.getLogger(__name__)
config = Config()

# Browser-like User-Agent to avoid 403s from Finviz.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

_FINVIZ_QUOTE_URL = "https://finviz.com/quote.ashx"


def _parse_pct(value: str) -> float | None:
    """Parse a percentage string like ``'5.23%'`` into a float.

    Returns ``None`` for N/A, '-', or empty values.
    """
    if not value or value.strip() in ("N/A", "-", ""):
        return None
    try:
        return float(value.strip().replace("%", ""))
    except (ValueError, TypeError):
        return None


def _parse_float(value: str) -> float | None:
    """Parse a generic numeric string into a float.

    Returns ``None`` for N/A, '-', or empty values.
    """
    if not value or value.strip() in ("N/A", "-", ""):
        return None
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return None


class FinvizScanner:
    """Scrape the free Finviz quote page for fundamental / sentiment data."""

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fetch_page(ticker: str) -> BeautifulSoup | None:
        """Download and parse the Finviz quote page for *ticker*.

        Returns ``None`` on any HTTP or parsing failure.
        """
        try:
            resp = requests.get(
                _FINVIZ_QUOTE_URL,
                params={"t": ticker.upper()},
                headers=_HEADERS,
                timeout=15,
            )
            if resp.status_code != 200:
                logger.debug(
                    "Finviz returned %d for %s", resp.status_code, ticker
                )
                return None

            return BeautifulSoup(resp.text, "lxml")

        except Exception as exc:
            logger.debug("Finviz page fetch failed for %s: %s", ticker, exc)
            return None

    @staticmethod
    def _snapshot_table(soup: BeautifulSoup) -> dict[str, str]:
        """Extract the snapshot (fundamentals) key-value table.

        The table lives inside ``<table class="snapshot-table2">`` on the
        Finviz quote page.  Each row alternates label / value cells.
        """
        table: dict[str, str] = {}
        snapshot = soup.find("table", class_="snapshot-table2")
        if not snapshot:
            return table

        for row in snapshot.find_all("tr"):
            cells = row.find_all("td")
            # Cells come in pairs: [label, value, label, value, ...]
            for i in range(0, len(cells) - 1, 2):
                key = cells[i].get_text(strip=True)
                val = cells[i + 1].get_text(strip=True)
                table[key] = val

        return table

    @staticmethod
    def _parse_news(soup: BeautifulSoup) -> list[str]:
        """Extract news headlines from the quote page.

        Headlines live in the ``<table id="news-table">`` element.
        """
        headlines: list[str] = []
        news_table = soup.find("table", id="news-table")
        if not news_table:
            return headlines

        for row in news_table.find_all("tr"):
            link = row.find("a")
            if link:
                text = link.get_text(strip=True)
                if text:
                    headlines.append(text)

        return headlines

    @staticmethod
    def _parse_insider(soup: BeautifulSoup) -> str:
        """Determine recent insider transaction direction.

        Looks at the insider-trading table on the quote page and returns
        ``'Buy'``, ``'Sell'``, or ``'None'`` based on the most recent
        transaction type.
        """
        table = soup.find("table", class_="body-table")
        if not table:
            return "None"

        rows = table.find_all("tr")
        for row in rows[1:]:  # Skip header row
            cells = row.find_all("td")
            if len(cells) >= 5:
                transaction = cells[4].get_text(strip=True)
                if "Buy" in transaction or "Purchase" in transaction:
                    return "Buy"
                if "Sale" in transaction or "Sell" in transaction:
                    return "Sell"

        return "None"

    @staticmethod
    def _parse_pre_market(soup: BeautifulSoup) -> tuple[float | None, float | None]:
        """Extract pre-market / after-hours price and change %.

        Finviz shows these in an element with class ``quote-header_ticker-wrapper_after``
        or similar after-hours / pre-market containers.
        """
        price = None
        change_pct = None

        # Look for the after-hours / pre-market quote section
        after_hours = soup.find("span", class_=re.compile(r"quote-afterhours"))
        if not after_hours:
            after_hours = soup.find("div", class_=re.compile(r"afterhours"))

        if after_hours:
            text = after_hours.get_text(" ", strip=True)
            # Pattern: price followed by (pct%)  e.g. "185.23 (0.45%)"
            m = re.search(r"([\d.]+)", text)
            if m:
                price = _parse_float(m.group(1))

            pct_match = re.search(r"\(([-+]?[\d.]+)%\)", text)
            if pct_match:
                change_pct = _parse_float(pct_match.group(1))

        return price, change_pct

    @staticmethod
    def _insider_direction_from_snap(snap: dict) -> str:
        """Infer insider direction from the snapshot table's ``Insider Trans`` field.

        ``Insider Trans`` is a percentage like ``'-1.39%'``.  Negative means
        net selling, positive means net buying.
        """
        raw = snap.get("Insider Trans", "")
        val = _parse_pct(raw)
        if val is not None:
            if val > 0.5:
                return "Buy"
            elif val < -0.5:
                return "Sell"
        return "None"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def scan_ticker(self, ticker: str) -> dict:
        """Scrape the Finviz quote page for a single ticker.

        Parameters
        ----------
        ticker : str
            Stock symbol (e.g. ``'AAPL'``).

        Returns
        -------
        dict
            Parsed quote data.  Returns an empty dict on any failure.
        """
        soup = self._fetch_page(ticker)
        if soup is None:
            return {}

        try:
            snap = self._snapshot_table(soup)

            pre_price, pre_change = self._parse_pre_market(soup)

            # Analyst rating — Finviz labels it "Recom" (was "Recom." previously)
            # The value is a numeric 1-5 scale; map to text as well.
            recom_raw = snap.get("Recom", "") or snap.get("Recom.", "")
            recom_val = _parse_float(recom_raw)
            if recom_val is not None:
                if recom_val <= 1.5:
                    analyst_rating = "Strong Buy"
                elif recom_val <= 2.5:
                    analyst_rating = "Buy"
                elif recom_val <= 3.5:
                    analyst_rating = "Hold"
                elif recom_val <= 4.5:
                    analyst_rating = "Sell"
                else:
                    analyst_rating = "Strong Sell"
            else:
                analyst_rating = None

            # Current price — Finviz labels it "Price"
            current_price = _parse_float(snap.get("Price", ""))

            return {
                "ticker": ticker.upper(),
                "current_price": current_price,
                "pre_market_price": pre_price,
                "pre_market_change_pct": pre_change,
                "analyst_rating": analyst_rating,
                "analyst_target_price": _parse_float(snap.get("Target Price", "")),
                "insider_transactions": (
                    lambda r: r if r not in (None, "None") else self._insider_direction_from_snap(snap)
                )(self._parse_insider(soup)),
                "insider_own_pct": _parse_pct(snap.get("Insider Own", "")),
                "inst_own_pct": _parse_pct(snap.get("Inst Own", "")),
                "short_float_pct": _parse_pct(snap.get("Short Float", "")),
                "earnings_date": snap.get("Earnings", None),
                "news_headlines": self._parse_news(soup),
                "volatility_week": _parse_pct(
                    snap.get("Volatility", "").split(" ")[0]
                    if snap.get("Volatility") else ""
                ),
                "volatility_month": _parse_pct(
                    snap.get("Volatility", "").split(" ")[1]
                    if snap.get("Volatility") and len(snap.get("Volatility", "").split(" ")) > 1
                    else ""
                ),
                "perf_week": _parse_pct(snap.get("Perf Week", "")),
                "perf_month": _parse_pct(snap.get("Perf Month", "")),
                "rel_volume": _parse_float(snap.get("Rel Volume", "")),
            }

        except Exception as exc:
            logger.debug("Finviz parse failed for %s: %s", ticker, exc)
            return {}

    def scan_batch(
        self, tickers: list[str], rate_limit: float = 0.5
    ) -> dict[str, dict]:
        """Scan multiple tickers with rate limiting.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to scan.
        rate_limit : float
            Seconds to sleep between requests (default 0.5).

        Returns
        -------
        dict[str, dict]
            Mapping of ticker -> scan result.  Tickers that fail are
            included with an empty dict so callers can detect gaps.
        """
        results: dict[str, dict] = {}

        for i, ticker in enumerate(tickers):
            logger.info("Finviz scan %d/%d: %s", i + 1, len(tickers), ticker)
            results[ticker.upper()] = self.scan_ticker(ticker)

            # Rate-limit between requests (skip sleep after the last one)
            if i < len(tickers) - 1:
                time.sleep(rate_limit)

        return results
