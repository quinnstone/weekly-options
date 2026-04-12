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

            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json()

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

    def get_finnhub_sentiment(self, ticker: str) -> dict:
        """Derive sentiment for a ticker using free company-news endpoint + VADER.

        Fetches recent headlines via ``get_finnhub_news()`` and runs VADER
        sentiment analysis on each headline to produce an average score.

        Returns
        -------
        dict
            Keys: score (-1 to 1), article_count, headlines (top 3).
        """
        try:
            articles = self.get_finnhub_news(ticker, days_back=3)
            if not articles:
                return {}

            vader_scores = []
            headlines = []
            for article in articles:
                headline = article.get("headline", "")
                if not headline:
                    continue
                compound = VADER.polarity_scores(headline)["compound"]
                vader_scores.append(compound)
                headlines.append(headline)

            if not vader_scores:
                return {}

            avg_score = sum(vader_scores) / len(vader_scores)

            return {
                "score": round(avg_score, 4),
                "article_count": len(vader_scores),
                "headlines": headlines[:3],
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
                    logger.error("Error scanning r/%s: %s", sub_name, exc)

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

            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

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

        # --- Reddit (blend if available — try PRAW first, fall back to crawler) ---
        try:
            reddit = self.get_reddit_sentiment([ticker])
            reddit_data = reddit.get(ticker, {})
            if reddit_data and reddit_data.get("mentions", 0) > 0:
                scores.append(reddit_data["avg_sentiment"])
                weights.append(1)
                sources.append("reddit")
        except Exception as exc:
            logger.debug("Reddit PRAW unavailable for %s: %s", ticker, exc)

        # --- Social crawler (headless browser — Reddit + Twitter/UW) ---
        # Only if we have pre-crawled data (crawl_all is expensive, run once per stage)
        if hasattr(self, "_social_data") and self._social_data:
            social = self._social_data.get("ticker_sentiment", {}).get(ticker.upper(), {})
            if social and social.get("mentions", 0) > 0:
                scores.append(social["avg_sentiment"])
                # Weight by mention count — more mentions = more signal
                social_weight = min(social["mentions"] / 5, 2)  # Cap at 2x
                weights.append(max(social_weight, 0.5))
                sources.append("social_crawler")
                # High engagement bonus
                if social.get("engagement", 0) > 100:
                    weights[-1] *= 1.5  # Viral posts carry more weight

        # --- Weighted composite ---
        if scores:
            total_weight = sum(weights)
            composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            composite = 0.0

        return {
            "ticker": ticker,
            "composite_score": round(composite, 4),
            "article_count": article_count,
            "sources": sources,
            "headlines": top_headlines,
        }

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
