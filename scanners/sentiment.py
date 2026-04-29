"""
Multi-source sentiment scanner.

Aggregates sentiment signals from Finnhub news, Reddit (via PRAW),
StockTwits, and VADER NLP into a composite score for each ticker.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import Config

logger = logging.getLogger(__name__)
config = Config()

VADER = SentimentIntensityAnalyzer()


# Process-level cache for Finnhub responses within a single pipeline run.
# Keyed by (endpoint, params_tuple) → response data.
# Eliminates duplicate API calls when multiple modules need the same data.
_finnhub_cache: dict[tuple, object] = {}


def _cached_finnhub_get(url: str, params: dict, timeout: int = 10):
    """Fetch from Finnhub with per-process caching.

    Returns parsed JSON. Cache lives for the duration of the pipeline run.
    """
    cache_key = (url, tuple(sorted((k, v) for k, v in params.items() if k != "token")))
    if cache_key in _finnhub_cache:
        return _finnhub_cache[cache_key]

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    _finnhub_cache[cache_key] = data
    return data


class SentimentScanner:
    """Aggregate sentiment from multiple free data sources."""

    def __init__(self):
        self._social_data = None

    def load_social_data(self, tickers: list = None):
        """Pre-crawl social media once per pipeline stage.

        Call this before scan_batch() to avoid launching the browser
        per-ticker. Results are cached in self._social_data.

        Parameters
        ----------
        tickers : list or None
            Target tickers to track (improves loose matching accuracy).
        """
        try:
            from scanners.social_crawler import SocialCrawler
            crawler = SocialCrawler(target_tickers=tickers)
            self._social_data = crawler.crawl_all()
            mention_count = len(self._social_data.get("ticker_sentiment", {}))
            logger.info("Social crawl complete: %d tickers mentioned", mention_count)
        except ImportError:
            logger.warning("Playwright not installed — skipping social crawl. "
                           "Run: pip install playwright && playwright install chromium")
        except Exception as exc:
            logger.error("Social crawl failed: %s", exc)
            self._social_data = None

    def get_social_trending(self) -> list:
        """Return trending tickers from the most recent social crawl.

        Returns
        -------
        list[dict]
            Top trending tickers with mentions, sentiment, consensus.
        """
        if not self._social_data:
            return []
        return self._social_data.get("combined_trending", [])

    # ------------------------------------------------------------------ #
    #  Finnhub News
    # ------------------------------------------------------------------ #

    def get_finnhub_news(self, ticker: str, days_back: int = 3) -> list[dict]:
        """Fetch recent company news from Finnhub.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        days_back : int
            How many days of news to retrieve.

        Returns
        -------
        list[dict]
            Each dict has: headline, summary, source, datetime, url.
        """
        if not config.has_finnhub():
            logger.info("Finnhub API key not configured; skipping news fetch")
            return []

        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": start,
                "to": end,
                "token": config.finnhub_api_key,
            }

            articles = _cached_finnhub_get(url, params)

            results = []
            for art in articles[:50]:  # Cap at 50 articles
                results.append({
                    "headline": art.get("headline", ""),
                    "summary": art.get("summary", ""),
                    "source": art.get("source", ""),
                    "datetime": datetime.fromtimestamp(art.get("datetime", 0)).isoformat(),
                    "url": art.get("url", ""),
                })

            return results

        except Exception as exc:
            logger.error("Finnhub news fetch failed for %s: %s", ticker, exc)
            return []

    # Keyword patterns for headline classification
    _CATALYST_KEYWORDS = {
        "upgrade", "buy rating", "outperform", "overweight", "price target raised",
        "beat", "beats", "exceeded", "surpass", "record revenue", "record earnings",
        "fda approval", "fda approves", "cleared", "partnership", "deal",
        "acquisition", "merger", "buyback", "dividend increase", "split",
        "contract win", "awarded", "product launch", "launches new", "breakthrough",
    }
    _RISK_KEYWORDS = {
        "downgrade", "sell rating", "underperform", "underweight", "price target cut",
        "miss", "misses", "missed", "warns", "warning", "recall", "lawsuit",
        "investigation", "probe", "subpoena", "sec inquiry", "fraud",
        "layoffs", "restructuring", "guidance cut", "guidance lower",
        "tariff", "sanctions", "ban", "restrict", "block",
        "ceo resign", "cfo resign", "departure", "fired",
        "default", "bankruptcy", "delisted",
    }

    @classmethod
    def _classify_headline(cls, headline: str) -> str:
        """Classify a headline as catalyst, risk, or neutral."""
        hl = headline.lower()
        if any(kw in hl for kw in cls._CATALYST_KEYWORDS):
            return "catalyst"
        if any(kw in hl for kw in cls._RISK_KEYWORDS):
            return "risk"
        return "neutral"

    def get_finnhub_sentiment(self, ticker: str) -> dict:
        """Derive sentiment for a ticker using free company-news endpoint + VADER.

        Fetches recent headlines via ``get_finnhub_news()`` and runs VADER
        sentiment analysis on each headline. Headlines are also classified
        as catalyst/risk/neutral so agents can see *what* the news says,
        not just a sentiment float.

        Returns
        -------
        dict
            Keys: score (-1 to 1), article_count, headlines (top 3),
            classified_headlines (list of {headline, sentiment, type, source}),
            catalyst_count, risk_count.
        """
        try:
            articles = self.get_finnhub_news(ticker, days_back=3)
            if not articles:
                return {}

            vader_scores = []
            headlines = []
            classified = []
            catalyst_count = 0
            risk_count = 0

            for article in articles:
                headline = article.get("headline", "")
                if not headline:
                    continue
                compound = VADER.polarity_scores(headline)["compound"]
                vader_scores.append(compound)
                headlines.append(headline)

                htype = self._classify_headline(headline)
                if htype == "catalyst":
                    catalyst_count += 1
                elif htype == "risk":
                    risk_count += 1

                classified.append({
                    "headline": headline,
                    "sentiment": round(compound, 3),
                    "type": htype,
                    "source": article.get("source", ""),
                })

            if not vader_scores:
                return {}

            avg_score = sum(vader_scores) / len(vader_scores)

            # Sort: catalysts and risks first (most actionable), then by sentiment magnitude
            classified.sort(key=lambda x: (
                0 if x["type"] == "catalyst" else 1 if x["type"] == "risk" else 2,
                -abs(x["sentiment"]),
            ))

            return {
                "score": round(avg_score, 4),
                "article_count": len(vader_scores),
                "headlines": headlines[:3],
                "classified_headlines": classified[:6],
                "catalyst_count": catalyst_count,
                "risk_count": risk_count,
            }

        except Exception as exc:
            logger.error("Finnhub sentiment failed for %s: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Reddit
    # ------------------------------------------------------------------ #

    def get_reddit_sentiment(self, tickers: list[str]) -> dict:
        """Check WSB, r/options, and r/stocks for ticker mentions and sentiment.

        Uses PRAW (Reddit API wrapper) and VADER for NLP scoring.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to search for.

        Returns
        -------
        dict
            Keyed by ticker: mentions (int), avg_sentiment (-1 to 1),
            bullish_count, bearish_count, neutral_count.
        """
        # Reddit integration requires PRAW + credentials in env vars.
        # These are optional — skip gracefully if not configured.
        reddit_id = os.getenv("REDDIT_CLIENT_ID", "")
        reddit_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        reddit_ua = os.getenv("REDDIT_USER_AGENT", "")
        if not all([reddit_id, reddit_secret, reddit_ua]):
            logger.info("Reddit credentials not configured; skipping Reddit sentiment")
            return {}

        try:
            import praw

            reddit = praw.Reddit(
                client_id=reddit_id,
                client_secret=reddit_secret,
                user_agent=reddit_ua,
            )

            subreddits = ["wallstreetbets", "options", "stocks"]
            results = {t: {"mentions": 0, "sentiments": [], "bullish": 0, "bearish": 0, "neutral": 0}
                       for t in tickers}

            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    # Search hot + new posts (limited to keep API calls reasonable)
                    posts = list(subreddit.hot(limit=50)) + list(subreddit.new(limit=50))

                    for post in posts:
                        title = post.title.upper()
                        body = (post.selftext or "").upper()
                        full_text = f"{post.title} {post.selftext or ''}"

                        for ticker in tickers:
                            # Look for ticker mentions (with word boundary via $ prefix or spaces)
                            ticker_upper = ticker.upper()
                            if (
                                f"${ticker_upper}" in title
                                or f" {ticker_upper} " in f" {title} "
                                or f"${ticker_upper}" in body
                                or f" {ticker_upper} " in f" {body} "
                            ):
                                results[ticker]["mentions"] += 1

                                vader_score = VADER.polarity_scores(full_text)
                                compound = vader_score["compound"]
                                results[ticker]["sentiments"].append(compound)

                                if compound >= 0.05:
                                    results[ticker]["bullish"] += 1
                                elif compound <= -0.05:
                                    results[ticker]["bearish"] += 1
                                else:
                                    results[ticker]["neutral"] += 1

                    time.sleep(1)  # Rate-limit between subreddits

                except Exception as exc:
                    msg = str(exc)
                    logger.error("Error scanning r/%s: %s", sub_name, msg)
                    # Reddit 401 = app-only OAuth rejected (script app needs
                    # username/password since 2023). Bail early — every
                    # subreddit will return the same error.
                    if "401" in msg:
                        logger.warning(
                            "PRAW returned 401 — falling back silently. "
                            "Social crawler JSON path is the primary Reddit source."
                        )
                        return {}

            # Compute averages
            output = {}
            for ticker in tickers:
                data = results[ticker]
                sents = data["sentiments"]
                avg_sent = sum(sents) / len(sents) if sents else 0.0

                output[ticker] = {
                    "mentions": data["mentions"],
                    "avg_sentiment": round(avg_sent, 4),
                    "bullish_count": data["bullish"],
                    "bearish_count": data["bearish"],
                    "neutral_count": data["neutral"],
                }

            return output

        except ImportError:
            logger.error("praw is not installed; cannot fetch Reddit sentiment")
            return {}
        except Exception as exc:
            logger.error("Reddit sentiment scan failed: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    #  StockTwits
    # ------------------------------------------------------------------ #

    def get_stocktwits_sentiment(self, ticker: str) -> dict:
        """Fetch sentiment from StockTwits (no API key required).

        Returns
        -------
        dict
            Keys: score (-1 to 1), bullish, bearish, available.
            If the API is unavailable (403/failure), returns a neutral
            result with ``available=False`` so the pipeline continues.
        """
        neutral = {"score": 0, "bullish": 0, "bearish": 0, "available": False}

        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            resp = requests.get(url, timeout=10)

            if resp.status_code in (403, 429):
                logger.debug("StockTwits returned %d for %s; using neutral fallback", resp.status_code, ticker)
                return neutral

            resp.raise_for_status()
            data = resp.json()

            messages = data.get("messages", [])
            bullish = 0
            bearish = 0
            total = len(messages)

            for msg in messages:
                sentiment = msg.get("entities", {}).get("sentiment", {})
                if sentiment:
                    basic = sentiment.get("basic")
                    if basic == "Bullish":
                        bullish += 1
                    elif basic == "Bearish":
                        bearish += 1

            # Convert to -1..+1 score
            if total > 0:
                score = (bullish - bearish) / total
            else:
                score = 0.0

            return {
                "score": round(score, 4),
                "bullish": bullish,
                "bearish": bearish,
                "available": True,
            }

        except Exception:
            logger.debug("StockTwits unavailable for %s; using neutral fallback", ticker)
            return neutral

    # ------------------------------------------------------------------ #
    #  Earnings Calendar
    # ------------------------------------------------------------------ #

    def get_earnings_calendar(self, date_from: str, date_to: str) -> list[dict]:
        """Fetch upcoming earnings using Finnhub.

        Parameters
        ----------
        date_from : str
            Start date ``YYYY-MM-DD``.
        date_to : str
            End date ``YYYY-MM-DD``.

        Returns
        -------
        list[dict]
            Each dict: symbol, date, epsEstimate, epsActual, hour.
        """
        if not config.has_finnhub():
            logger.info("Finnhub API key not configured; skipping earnings calendar")
            return []

        try:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {
                "from": date_from,
                "to": date_to,
                "token": config.finnhub_api_key,
            }

            data = _cached_finnhub_get(url, params)
            earnings = data.get("earningsCalendar", [])
            results = []
            for item in earnings:
                results.append({
                    "symbol": item.get("symbol", ""),
                    "date": item.get("date", ""),
                    "eps_estimate": item.get("epsEstimate"),
                    "eps_actual": item.get("epsActual"),
                    "revenue_estimate": item.get("revenueEstimate"),
                    "hour": item.get("hour", ""),  # bmo / amc
                })

            return results

        except Exception as exc:
            logger.error("Earnings calendar fetch failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Analyst Revision Trend
    # ------------------------------------------------------------------ #

    def get_analyst_revision_trend(self, ticker: str) -> dict:
        """Fetch analyst recommendation revision velocity from Finnhub.

        Pre-earnings upgrades/downgrades are one of the strongest public
        predictors of earnings surprises. This compares the most recent
        month's bullish/bearish counts vs the prior month to detect a
        revision cycle.

        Returns
        -------
        dict
            Keys: direction ("upgrade" | "downgrade" | "flat" | "unavailable"),
            delta (int — net change in buy+strongBuy count), latest (dict
            with the current month's raw counts), period (str YYYY-MM-DD of
            latest data point).
        """
        neutral = {"direction": "unavailable", "delta": 0, "latest": {}, "period": ""}

        if not config.has_finnhub():
            return neutral

        try:
            data = _cached_finnhub_get(
                "https://finnhub.io/api/v1/stock/recommendation",
                {"symbol": ticker, "token": config.finnhub_api_key},
            )

            # Finnhub returns array sorted most-recent first
            if not data or len(data) < 2:
                return neutral

            latest = data[0]
            prior = data[1]

            bullish_now = (latest.get("strongBuy", 0) or 0) + (latest.get("buy", 0) or 0)
            bullish_prior = (prior.get("strongBuy", 0) or 0) + (prior.get("buy", 0) or 0)
            bearish_now = (latest.get("strongSell", 0) or 0) + (latest.get("sell", 0) or 0)
            bearish_prior = (prior.get("strongSell", 0) or 0) + (prior.get("sell", 0) or 0)

            bullish_delta = bullish_now - bullish_prior
            bearish_delta = bearish_now - bearish_prior
            net_delta = bullish_delta - bearish_delta

            if net_delta >= 2:
                direction = "upgrade"
            elif net_delta <= -2:
                direction = "downgrade"
            else:
                direction = "flat"

            return {
                "direction": direction,
                "delta": net_delta,
                "bullish_delta": bullish_delta,
                "bearish_delta": bearish_delta,
                "latest": {
                    "strongBuy": latest.get("strongBuy", 0),
                    "buy": latest.get("buy", 0),
                    "hold": latest.get("hold", 0),
                    "sell": latest.get("sell", 0),
                    "strongSell": latest.get("strongSell", 0),
                },
                "period": latest.get("period", ""),
            }

        except Exception as exc:
            logger.debug("Analyst revision fetch failed for %s: %s", ticker, exc)
            return neutral

    # ------------------------------------------------------------------ #
    #  Composite Sentiment
    # ------------------------------------------------------------------ #

    def analyze_ticker_sentiment(self, ticker: str) -> dict:
        """Combine all sentiment sources into a composite score.

        The composite ranges from -1 (very bearish) to +1 (very bullish).

        Primary signal is VADER on Finnhub news headlines.  StockTwits and
        Reddit are blended in when available.

        Returns
        -------
        dict
            Keys: composite_score (float -1 to 1), article_count (int),
            sources (list), headlines (list of top 3 headlines).
        """
        scores = []
        weights = []
        sources = []
        article_count = 0
        top_headlines = []

        # --- Finnhub news + VADER (primary signal) ---
        finnhub = self.get_finnhub_sentiment(ticker)
        if finnhub and finnhub.get("score") is not None:
            scores.append(finnhub["score"])
            weights.append(2)  # Primary signal gets double weight
            sources.append("finnhub")
            article_count = finnhub.get("article_count", 0)
            top_headlines = finnhub.get("headlines", [])[:3]

        time.sleep(0.3)

        # --- StockTwits (blend if available) ---
        stocktwits = self.get_stocktwits_sentiment(ticker)
        if stocktwits.get("available", False):
            scores.append(stocktwits["score"])
            weights.append(1)
            sources.append("stocktwits")

        # --- Social crawler (headless browser — Reddit + Twitter/UW) ---
        # Preferred source: pre-crawled once per stage, covers Reddit + Twitter + UW
        social_has_reddit = False
        social_intel = {}
        if self._social_data:
            social = self._social_data.get("ticker_sentiment", {}).get(ticker.upper(), {})
            if social and social.get("mentions", 0) > 0:
                scores.append(social["avg_sentiment"])
                # Weight by signal_strength (quality-adjusted, not just volume)
                signal = social.get("signal_strength", social["mentions"])
                social_weight = min(signal / 5, 3)  # Cap at 3x
                weights.append(max(social_weight, 0.5))
                sources.append("social_crawler")

                # Flow alignment bonus: if UW flow agrees with sentiment direction
                flow_consensus = social.get("flow_consensus", "neutral")
                if flow_consensus != "neutral":
                    flow_conviction = social.get("flow_conviction", 0)
                    if flow_conviction > 0.5:
                        weights[-1] *= 1.3  # Directional flow adds conviction

                social_has_reddit = any(
                    s in ("reddit", "wallstreetbets", "stocks", "options", "thetagang")
                    for s in social.get("sources", []))

                # Preserve qualitative intelligence for downstream consumption
                social_intel = {
                    "narrative": social.get("narrative", ""),
                    "catalysts": social.get("catalysts", []),
                    "risks": social.get("risks", []),
                    "flow_consensus": flow_consensus,
                    "flow_conviction": social.get("flow_conviction", 0),
                    "flow_signals": social.get("flow_signals", []),
                    "signal_strength": social.get("signal_strength", 0),
                    "top_posts": social.get("top_posts", []),
                    "price_targets": social.get("price_targets", []),
                    "post_types": social.get("post_types", {}),
                }

        # --- Reddit PRAW (fallback when social crawler didn't cover Reddit) ---
        if not social_has_reddit:
            try:
                reddit = self.get_reddit_sentiment([ticker])
                reddit_data = reddit.get(ticker, {})
                if reddit_data and reddit_data.get("mentions", 0) > 0:
                    scores.append(reddit_data["avg_sentiment"])
                    weights.append(1)
                    sources.append("reddit")
            except Exception as exc:
                logger.debug("Reddit PRAW unavailable for %s: %s", ticker, exc)

        # --- Weighted composite ---
        if scores:
            total_weight = sum(weights)
            composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            composite = 0.0

        result = {
            "ticker": ticker,
            "composite_score": round(composite, 4),
            "article_count": article_count,
            "sources": sources,
            "headlines": top_headlines,
        }

        # Attach social intelligence when available
        if social_intel:
            result["social"] = social_intel

        # Attach analyst revision trend (free Finnhub endpoint)
        try:
            analyst = self.get_analyst_revision_trend(ticker)
            if analyst.get("direction") != "unavailable":
                result["analyst_revision"] = analyst
        except Exception as exc:
            logger.debug("Analyst revision attach failed for %s: %s", ticker, exc)

        # Attach recent 8-K filings (SEC EDGAR, free)
        try:
            from scanners.edgar import EdgarScanner
            if not hasattr(self, "_edgar_scanner"):
                self._edgar_scanner = EdgarScanner()
            filings = self._edgar_scanner.get_recent_8k_filings(ticker, days_back=3)
            if filings.get("has_recent_8k"):
                result["sec_8k"] = filings
        except Exception as exc:
            logger.debug("8-K attach failed for %s: %s", ticker, exc)

        return result

    # ------------------------------------------------------------------ #
    #  Batch
    # ------------------------------------------------------------------ #

    def scan_batch(self, tickers: list[str]) -> pd.DataFrame:
        """Analyze sentiment for multiple tickers.

        Returns
        -------
        pd.DataFrame
            One row per ticker with composite score, label, sources used.
        """
        records = []

        for ticker in tickers:
            logger.info("Scanning sentiment for %s ...", ticker)
            result = self.analyze_ticker_sentiment(ticker)
            if result:
                records.append({
                    "ticker": result["ticker"],
                    "composite_score": result["composite_score"],
                    "article_count": result.get("article_count", 0),
                    "sources": result.get("sources", []),
                })
            time.sleep(0.5)  # Rate-limit across sources

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("ticker")
