"""
SEC EDGAR insider-trading scanner (Form 4 filings).

Fetches insider buy/sell activity from the free SEC EDGAR API and
derives a directional signal (bullish / bearish / neutral) for each
ticker.  No API key is required; SEC only asks for an identifying
User-Agent header.
"""

import logging
import time
from datetime import datetime, timedelta
from xml.etree import ElementTree

import requests

from config import Config

logger = logging.getLogger(__name__)
config = Config()

# SEC EDGAR requires a descriptive User-Agent with contact info.
_SEC_HEADERS = {
    "User-Agent": "ZeroDTE/1.0 (zerodte@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Module-level cache for the ticker -> CIK mapping (fetched once).
_CIK_MAP: dict[str, str] | None = None


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _load_cik_map() -> dict[str, str]:
    """Download the SEC ticker -> CIK mapping file and return a
    dict keyed by uppercase ticker symbol.

    The file is fetched at most once per process and cached in
    ``_CIK_MAP``.
    """
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        mapping: dict[str, str] = {}
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", ""))
            if ticker and cik:
                mapping[ticker] = cik

        _CIK_MAP = mapping
        logger.debug("Loaded SEC CIK map: %d tickers", len(mapping))
        return mapping

    except Exception as exc:
        logger.error("Failed to load SEC CIK map: %s", exc)
        _CIK_MAP = {}
        return {}


def _cik_for_ticker(ticker: str) -> str | None:
    """Return the CIK string for *ticker*, or ``None`` if unknown."""
    cik_map = _load_cik_map()
    return cik_map.get(ticker.upper())


def _pad_cik(cik: str) -> str:
    """Zero-pad a CIK to 10 digits as required by some SEC endpoints."""
    return cik.zfill(10)


def _empty_result() -> dict:
    """Return a neutral / empty result dict."""
    return {
        "total_buys": 0,
        "total_sells": 0,
        "net_shares": 0,
        "net_value": 0.0,
        "largest_transaction": {},
        "insider_signal": "neutral",
        "recent_trades": [],
    }


# ------------------------------------------------------------------ #
#  Form 4 XML parsing
# ------------------------------------------------------------------ #

def _parse_form4_xml(xml_text: str) -> list[dict]:
    """Parse a single Form 4 XML document into a list of transactions.

    Each returned dict contains: ``name``, ``title``, ``type``
    (``"buy"`` or ``"sell"``), ``shares``, ``price``, ``value``,
    ``date``.
    """
    trades: list[dict] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return trades

    # Reporting owner info (name + title)
    owner_name = ""
    owner_title = ""
    for owner in root.iter("reportingOwner"):
        oid = owner.find("reportingOwnerId")
        if oid is not None:
            name_el = oid.find("rptOwnerName")
            if name_el is not None and name_el.text:
                owner_name = name_el.text.strip()
        rel = owner.find("reportingOwnerRelationship")
        if rel is not None:
            for tag in ("officerTitle", "isOfficer", "isDirector"):
                el = rel.find(tag)
                if el is not None and el.text and el.text.strip():
                    if tag == "officerTitle":
                        owner_title = el.text.strip()
                        break
                    elif tag == "isDirector" and el.text.strip() == "1":
                        owner_title = "Director"
                    elif tag == "isOfficer" and el.text.strip() == "1":
                        owner_title = "Officer"

    # Non-derivative transactions (the most common insider buys/sells)
    for txn in root.iter("nonDerivativeTransaction"):
        trade = _extract_transaction(txn, owner_name, owner_title)
        if trade:
            trades.append(trade)

    return trades


def _extract_transaction(txn_el, owner_name: str, owner_title: str) -> dict | None:
    """Pull fields out of a single <nonDerivativeTransaction> element."""
    try:
        coding = txn_el.find("transactionCoding")
        if coding is None:
            return None
        code_el = coding.find("transactionCode")
        if code_el is None or not code_el.text:
            return None
        txn_code = code_el.text.strip()

        # P = open-market purchase, S = open-market sale
        if txn_code == "P":
            txn_type = "buy"
        elif txn_code == "S":
            txn_type = "sell"
        else:
            return None  # Skip grants, exercises, etc.

        # Shares
        amounts = txn_el.find("transactionAmounts")
        if amounts is None:
            return None
        shares_el = amounts.find("transactionShares")
        shares = 0.0
        if shares_el is not None:
            val = shares_el.find("value")
            if val is not None and val.text:
                shares = float(val.text)

        # Price per share
        price_el = amounts.find("transactionPricePerShare")
        price = 0.0
        if price_el is not None:
            val = price_el.find("value")
            if val is not None and val.text:
                price = float(val.text)

        # Date
        date_el = txn_el.find("transactionDate")
        date_str = ""
        if date_el is not None:
            val = date_el.find("value")
            if val is not None and val.text:
                date_str = val.text.strip()

        value = round(shares * price, 2)

        return {
            "name": owner_name,
            "title": owner_title,
            "type": txn_type,
            "shares": shares,
            "price": price,
            "value": value,
            "date": date_str,
        }

    except Exception:
        return None


# ------------------------------------------------------------------ #
#  EdgarScanner class
# ------------------------------------------------------------------ #

class EdgarScanner:
    """Scan SEC EDGAR Form 4 filings for insider-trading signals."""

    # ------------------------------------------------------------------ #
    #  Single ticker
    # ------------------------------------------------------------------ #

    def get_insider_trades(self, ticker: str, days_back: int = 30) -> dict:
        """Fetch recent Form 4 insider transactions for *ticker*.

        Parameters
        ----------
        ticker : str
            Stock symbol (e.g. ``"AAPL"``).
        days_back : int
            How many calendar days of filings to consider.

        Returns
        -------
        dict
            Keys: total_buys, total_sells, net_shares, net_value,
            largest_transaction, insider_signal, recent_trades.
        """
        result = _empty_result()

        try:
            cik = _cik_for_ticker(ticker)
            if not cik:
                logger.debug("No CIK found for ticker %s", ticker)
                return result

            padded_cik = _pad_cik(cik)

            # Fetch recent Form 4 filing index from EDGAR submissions
            filings = self._fetch_form4_filings(padded_cik, days_back)
            if not filings:
                logger.debug("No Form 4 filings found for %s (CIK %s)", ticker, cik)
                return result

            # Parse each filing's XML for transactions
            all_trades: list[dict] = []
            for filing_url in filings:
                time.sleep(0.12)  # Stay well under SEC 10-req/s limit
                trades = self._fetch_and_parse_filing(filing_url)
                all_trades.extend(trades)

            if not all_trades:
                return result

            # Aggregate
            total_buys = sum(1 for t in all_trades if t["type"] == "buy")
            total_sells = sum(1 for t in all_trades if t["type"] == "sell")

            buy_shares = sum(t["shares"] for t in all_trades if t["type"] == "buy")
            sell_shares = sum(t["shares"] for t in all_trades if t["type"] == "sell")
            net_shares = buy_shares - sell_shares

            buy_value = sum(t["value"] for t in all_trades if t["type"] == "buy")
            sell_value = sum(t["value"] for t in all_trades if t["type"] == "sell")
            net_value = round(buy_value - sell_value, 2)

            # Largest transaction by absolute dollar value
            largest = max(all_trades, key=lambda t: abs(t["value"]))

            # Signal
            if net_value > 0:
                signal = "bullish"
            elif net_value < 0:
                signal = "bearish"
            else:
                signal = "neutral"

            # Recent trades (last 5 by date descending)
            sorted_trades = sorted(all_trades, key=lambda t: t.get("date", ""), reverse=True)
            recent = [
                {
                    "name": t["name"],
                    "title": t["title"],
                    "type": t["type"],
                    "shares": t["shares"],
                    "date": t["date"],
                }
                for t in sorted_trades[:5]
            ]

            result = {
                "total_buys": total_buys,
                "total_sells": total_sells,
                "net_shares": net_shares,
                "net_value": net_value,
                "largest_transaction": {
                    "name": largest["name"],
                    "title": largest["title"],
                    "type": largest["type"],
                    "shares": largest["shares"],
                    "value": largest["value"],
                    "date": largest["date"],
                },
                "insider_signal": signal,
                "recent_trades": recent,
            }

        except Exception as exc:
            logger.debug("EDGAR scan failed for %s: %s", ticker, exc)

        return result

    # ------------------------------------------------------------------ #
    #  Batch
    # ------------------------------------------------------------------ #

    def scan_batch(self, tickers: list[str], rate_limit: float = 0.3) -> dict[str, dict]:
        """Scan multiple tickers for insider trading activity.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to scan.
        rate_limit : float
            Seconds to sleep between tickers to respect SEC rate limits.

        Returns
        -------
        dict[str, dict]
            Keyed by ticker, each value is the result from
            ``get_insider_trades()``.
        """
        results: dict[str, dict] = {}

        for ticker in tickers:
            logger.info("Scanning EDGAR insider trades for %s ...", ticker)
            results[ticker] = self.get_insider_trades(ticker)
            time.sleep(rate_limit)

        return results

    # ------------------------------------------------------------------ #
    #  Internal: filing discovery
    # ------------------------------------------------------------------ #

    def _fetch_form4_filings(self, padded_cik: str, days_back: int) -> list[str]:
        """Return a list of Form 4 XML document URLs for the given CIK.

        Uses the EDGAR submissions endpoint to list recent filings, then
        filters down to Form 4s within the date window.  Returns up to
        20 filing URLs.
        """
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        try:
            resp = requests.get(url, headers=_SEC_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("EDGAR submissions request failed for CIK %s: %s", padded_cik, exc)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        filing_urls: list[str] = []
        for i, form_type in enumerate(forms):
            if form_type not in ("4", "4/A"):
                continue
            if i < len(dates) and dates[i] < cutoff:
                continue
            if i >= len(accessions) or i >= len(primary_docs):
                continue

            accession_no = accessions[i].replace("-", "")
            doc = primary_docs[i]
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{padded_cik.lstrip('0')}/{accession_no}/{doc}"
            )
            filing_urls.append(doc_url)

            if len(filing_urls) >= 20:
                break

        return filing_urls

    # ------------------------------------------------------------------ #
    #  Internal: filing parsing
    # ------------------------------------------------------------------ #

    def _fetch_and_parse_filing(self, url: str) -> list[dict]:
        """Download a single Form 4 XML document and extract trades."""
        try:
            resp = requests.get(url, headers=_SEC_HEADERS, timeout=10)
            resp.raise_for_status()
            return _parse_form4_xml(resp.text)
        except Exception as exc:
            logger.debug("Failed to parse Form 4 at %s: %s", url, exc)
            return []
