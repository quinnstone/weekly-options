"""
Social Media Crawler — headless browser scraping for financial sentiment.

Crawls public pages from Reddit and finance Twitter/Unusual Whales to capture
retail sentiment, trending tickers, and consensus direction. Uses Playwright
for reliable rendering of dynamic content.

Sources:
    Reddit: r/wallstreetbets, r/stocks, r/options, r/thetagang, r/personalfinance
    Finance: Unusual Whales flow page, StockTwits trending

Rate-limited and respectful: 2-3 second delays between pages, headless only.
"""

import logging
import re
import time
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
VADER = SentimentIntensityAnalyzer()

# Common ticker patterns — avoid matching common English words
# Only match 1-5 char uppercase preceded by $ or in known context
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
# Secondary pattern for standalone tickers (less reliable, needs context)
LOOSE_TICKER = re.compile(r'\b([A-Z]{2,5})\b')

# Words that look like tickers but aren't
TICKER_BLACKLIST = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "ITS", "HIS", "HOW", "MAN", "NEW",
    "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET", "HIM", "LET",
    "SAY", "SHE", "TOO", "USE", "DAD", "MOM", "IMO", "YOLO", "HODL", "CEO",
    "IPO", "ETF", "OTM", "ITM", "ATM", "DTE", "IV", "OI", "RSI", "EPS",
    "GDP", "CPI", "FED", "SEC", "NYSE", "FOMC", "EDIT", "POST", "JUST",
    "LIKE", "WHAT", "WILL", "THIS", "THAT", "WITH", "HAVE", "FROM", "BEEN",
    "SOME", "WHEN", "THEY", "YOUR", "MUCH", "LONG", "VERY", "AFTER", "ALSO",
    "BACK", "BEEN", "CALL", "COME", "EACH", "EVEN", "GOOD", "HELP", "HERE",
    "HIGH", "KEEP", "LAST", "MAKE", "MANY", "MORE", "MOST", "MOVE", "MUCH",
    "MUST", "NAME", "NEXT", "ONLY", "OVER", "PART", "PLAY", "PUTS", "REAL",
    "SELL", "SHOW", "SIDE", "TAKE", "TELL", "THAN", "THEM", "THEN", "TOOK",
    "TURN", "WANT", "WELL", "WENT", "WERE", "WORK", "YEAR", "GAIN", "LOSS",
    "PUMP", "DUMP", "BEAR", "BULL", "LMAO", "FREE", "CASH", "DEBT", "RISK",
    "FUND", "HOLD", "BUY", "LOW", "RUN", "SET", "TRY", "PUT", "TOP", "TWO",
    "ANY", "FEW", "GOT", "MAY", "OWN", "PAY", "LOT", "BIG", "BAD", "ASK",
    "BID", "GAP", "RED", "MAX", "NET",
}

REDDIT_SUBS = [
    "wallstreetbets",
    "stocks",
    "options",
    "thetagang",
    "personalfinance",
]

# Finance Twitter accounts to check via Nitter instances or direct scrape
FINANCE_TWITTER = [
    "unusual_whales",
    "DeItaone",      # Walter Bloomberg — breaking financial news
    "zaboravom",     # Market commentary
]


class SocialCrawler:
    """Crawl Reddit and finance social media for ticker sentiment."""

    def __init__(self, target_tickers: list = None):
        """Initialize crawler.

        Parameters
        ----------
        target_tickers : list or None
            Specific tickers to track. If None, discovers trending tickers.
        """
        self.target_tickers = set(t.upper() for t in (target_tickers or []))
        self._browser = None
        self._context = None

    def _ensure_browser(self):
        """Lazy-initialize Playwright browser."""
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True)
            self._context = self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
            )
        except Exception as exc:
            logger.error("Failed to launch Playwright browser: %s", exc)
            raise

    def close(self):
        """Clean up browser resources."""
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if hasattr(self, "_pw") and self._pw:
            self._pw.stop()
        self._browser = None
        self._context = None

    def _extract_tickers(self, text: str) -> list:
        """Extract ticker symbols from text.

        Uses $TICKER pattern (high confidence) and loose uppercase
        matching (lower confidence, filtered against blacklist).
        """
        tickers = set()

        # High confidence: $AAPL, $TSLA etc
        for match in TICKER_PATTERN.findall(text):
            if match not in TICKER_BLACKLIST:
                tickers.add(match)

        # Lower confidence: standalone uppercase words, only if we have
        # target tickers to match against (avoids false positives)
        if self.target_tickers:
            for match in LOOSE_TICKER.findall(text):
                if match in self.target_tickers:
                    tickers.add(match)

        return list(tickers)

    def _score_text(self, text: str) -> dict:
        """Score text sentiment using VADER.

        Returns
        -------
        dict
            'compound' (-1 to 1), 'positive', 'negative', 'neutral'.
        """
        scores = VADER.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    # ------------------------------------------------------------------
    #  Reddit Crawling
    # ------------------------------------------------------------------

    def crawl_reddit(self) -> dict:
        """Crawl all target subreddits for ticker mentions and sentiment.

        Uses old.reddit.com for simpler HTML parsing.

        Returns
        -------
        dict
            'posts' (list), 'ticker_mentions' (dict), 'trending' (list).
        """
        self._ensure_browser()
        all_posts = []

        for sub in REDDIT_SUBS:
            try:
                posts = self._crawl_subreddit(sub)
                all_posts.extend(posts)
                logger.info("Crawled r/%s: %d posts", sub, len(posts))
            except Exception as exc:
                logger.error("Failed to crawl r/%s: %s", sub, exc)
            time.sleep(2)  # Respectful delay between subs

        # Aggregate ticker mentions
        ticker_data = self._aggregate_mentions(all_posts)

        # Find trending tickers (most mentioned)
        trending = sorted(
            ticker_data.items(),
            key=lambda x: x[1]["mentions"],
            reverse=True,
        )[:20]

        return {
            "posts": all_posts,
            "ticker_mentions": ticker_data,
            "trending": [{"ticker": t, **d} for t, d in trending],
            "source": "reddit",
            "subs_crawled": len(REDDIT_SUBS),
            "total_posts": len(all_posts),
            "timestamp": datetime.now().isoformat(),
        }

    def _crawl_subreddit(self, subreddit: str, limit: int = 50) -> list:
        """Crawl a single subreddit's hot posts.

        Parameters
        ----------
        subreddit : str
            Subreddit name (without r/ prefix).
        limit : int
            Max posts to scrape per page.

        Returns
        -------
        list[dict]
            Each with 'title', 'body', 'score', 'comments', 'tickers',
            'sentiment', 'subreddit', 'url'.
        """
        page = self._context.new_page()
        posts = []

        try:
            url = f"https://old.reddit.com/r/{subreddit}/hot/"
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)

            # Parse post entries from old.reddit.com
            entries = page.query_selector_all("div.thing.link")

            for entry in entries[:limit]:
                try:
                    title_el = entry.query_selector("a.title")
                    title = title_el.inner_text().strip() if title_el else ""

                    score_el = entry.query_selector("div.score.unvoted")
                    score_text = score_el.inner_text().strip() if score_el else "0"
                    try:
                        score = int(score_text.replace(",", ""))
                    except (ValueError, AttributeError):
                        score = 0

                    comments_el = entry.query_selector("a.comments")
                    comments_text = comments_el.inner_text().strip() if comments_el else "0"
                    try:
                        comments = int(re.search(r"(\d+)", comments_text).group(1))
                    except (ValueError, AttributeError):
                        comments = 0

                    # Only process posts with meaningful engagement
                    if score < 5 and comments < 3:
                        continue

                    tickers = self._extract_tickers(title)
                    sentiment = self._score_text(title)

                    post_url_el = entry.query_selector("a.title")
                    post_url = post_url_el.get_attribute("href") if post_url_el else ""

                    posts.append({
                        "title": title,
                        "score": score,
                        "comments": comments,
                        "tickers": tickers,
                        "sentiment": sentiment,
                        "subreddit": subreddit,
                        "url": post_url,
                    })

                except Exception:
                    continue

        except Exception as exc:
            logger.error("Error loading r/%s: %s", subreddit, exc)
        finally:
            page.close()

        return posts

    # ------------------------------------------------------------------
    #  Finance Twitter / Unusual Whales
    # ------------------------------------------------------------------

    def crawl_finance_twitter(self) -> dict:
        """Crawl finance accounts via Nitter instances or direct scrape.

        Falls back gracefully if Nitter instances are down (common).

        Returns
        -------
        dict
            'posts' (list), 'ticker_mentions' (dict), 'trending' (list).
        """
        self._ensure_browser()
        all_posts = []

        # Try Nitter instances (public Twitter mirrors)
        nitter_instances = [
            "nitter.privacydev.net",
            "nitter.poast.org",
            "nitter.1d4.us",
        ]

        for account in FINANCE_TWITTER:
            scraped = False
            for instance in nitter_instances:
                if scraped:
                    break
                try:
                    posts = self._crawl_nitter(instance, account)
                    if posts:
                        all_posts.extend(posts)
                        logger.info("Crawled @%s via %s: %d posts",
                                    account, instance, len(posts))
                        scraped = True
                except Exception as exc:
                    logger.debug("Nitter %s failed for @%s: %s",
                                 instance, account, exc)

            if not scraped:
                logger.info("All Nitter instances failed for @%s, skipping", account)

            time.sleep(2)

        # Also try Unusual Whales flow page directly
        try:
            uw_posts = self._crawl_unusual_whales()
            all_posts.extend(uw_posts)
            logger.info("Crawled Unusual Whales: %d items", len(uw_posts))
        except Exception as exc:
            logger.error("Unusual Whales crawl failed: %s", exc)

        ticker_data = self._aggregate_mentions(all_posts)

        trending = sorted(
            ticker_data.items(),
            key=lambda x: x[1]["mentions"],
            reverse=True,
        )[:20]

        return {
            "posts": all_posts,
            "ticker_mentions": ticker_data,
            "trending": [{"ticker": t, **d} for t, d in trending],
            "source": "twitter",
            "total_posts": len(all_posts),
            "timestamp": datetime.now().isoformat(),
        }

    def _crawl_nitter(self, instance: str, account: str,
                      limit: int = 30) -> list:
        """Crawl a Twitter account via a Nitter instance.

        Parameters
        ----------
        instance : str
            Nitter mirror hostname.
        account : str
            Twitter handle (without @).
        limit : int
            Max tweets to scrape.

        Returns
        -------
        list[dict]
            Each with 'text', 'tickers', 'sentiment', 'source', 'account'.
        """
        page = self._context.new_page()
        posts = []

        try:
            url = f"https://{instance}/{account}"
            page.goto(url, timeout=10000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)

            # Nitter uses .timeline-item for tweets
            items = page.query_selector_all(".timeline-item")

            for item in items[:limit]:
                try:
                    content_el = item.query_selector(".tweet-content")
                    text = content_el.inner_text().strip() if content_el else ""
                    if not text:
                        continue

                    tickers = self._extract_tickers(text)
                    sentiment = self._score_text(text)

                    posts.append({
                        "title": text[:200],
                        "tickers": tickers,
                        "sentiment": sentiment,
                        "source": "twitter",
                        "account": account,
                    })
                except Exception:
                    continue

        except Exception as exc:
            raise exc
        finally:
            page.close()

        return posts

    def _crawl_unusual_whales(self) -> list:
        """Crawl Unusual Whales public flow/trending page.

        Returns
        -------
        list[dict]
            Ticker mentions with flow direction context.
        """
        page = self._context.new_page()
        posts = []

        try:
            page.goto("https://unusualwhales.com/flow",
                       timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

            # UW renders a table of unusual flow — extract ticker + direction
            rows = page.query_selector_all("table tbody tr")

            for row in rows[:50]:
                try:
                    cells = row.query_selector_all("td")
                    if len(cells) < 4:
                        continue

                    text = " ".join(c.inner_text().strip() for c in cells)
                    tickers = self._extract_tickers(text)

                    # Try to detect call/put direction from the row
                    text_lower = text.lower()
                    if "call" in text_lower:
                        direction = "bullish"
                        sentiment = {"compound": 0.3, "positive": 0.3,
                                     "negative": 0.0, "neutral": 0.7}
                    elif "put" in text_lower:
                        direction = "bearish"
                        sentiment = {"compound": -0.3, "positive": 0.0,
                                     "negative": 0.3, "neutral": 0.7}
                    else:
                        direction = "neutral"
                        sentiment = self._score_text(text)

                    if tickers:
                        posts.append({
                            "title": text[:200],
                            "tickers": tickers,
                            "sentiment": sentiment,
                            "source": "unusual_whales",
                            "direction": direction,
                        })
                except Exception:
                    continue

        except Exception as exc:
            logger.error("Unusual Whales page error: %s", exc)
        finally:
            page.close()

        return posts

    # ------------------------------------------------------------------
    #  Aggregation
    # ------------------------------------------------------------------

    def _aggregate_mentions(self, posts: list) -> dict:
        """Aggregate all posts into per-ticker sentiment data.

        Parameters
        ----------
        posts : list
            Raw posts from all sources.

        Returns
        -------
        dict
            Keyed by ticker: mentions, avg_sentiment, bullish/bearish counts,
            engagement_score, sources.
        """
        ticker_data = {}

        for post in posts:
            for ticker in post.get("tickers", []):
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "mentions": 0,
                        "sentiments": [],
                        "bullish": 0,
                        "bearish": 0,
                        "neutral": 0,
                        "engagement": 0,
                        "sources": set(),
                    }

                td = ticker_data[ticker]
                td["mentions"] += 1
                compound = post.get("sentiment", {}).get("compound", 0)
                td["sentiments"].append(compound)

                if compound >= 0.05:
                    td["bullish"] += 1
                elif compound <= -0.05:
                    td["bearish"] += 1
                else:
                    td["neutral"] += 1

                # Engagement = upvotes + comments (Reddit) or just count (Twitter)
                td["engagement"] += post.get("score", 0) + post.get("comments", 0)
                td["sources"].add(post.get("source", post.get("subreddit", "unknown")))

        # Compute averages and clean up
        for ticker, td in ticker_data.items():
            sents = td.pop("sentiments")
            td["avg_sentiment"] = round(sum(sents) / len(sents), 4) if sents else 0
            td["sources"] = sorted(td["sources"])

            # Consensus direction
            total = td["bullish"] + td["bearish"] + td["neutral"]
            if total > 0:
                bull_pct = td["bullish"] / total
                bear_pct = td["bearish"] / total
                if bull_pct > 0.6:
                    td["consensus"] = "bullish"
                elif bear_pct > 0.6:
                    td["consensus"] = "bearish"
                else:
                    td["consensus"] = "mixed"
            else:
                td["consensus"] = "unknown"

        return ticker_data

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def crawl_all(self) -> dict:
        """Run all crawlers and merge results.

        Returns
        -------
        dict
            'reddit' (dict), 'twitter' (dict), 'combined_trending' (list),
            'ticker_sentiment' (dict — merged from all sources).
        """
        try:
            self._ensure_browser()
        except Exception as exc:
            logger.error("Browser init failed: %s — returning empty results", exc)
            return {
                "reddit": {}, "twitter": {},
                "combined_trending": [], "ticker_sentiment": {},
                "error": str(exc),
            }

        reddit_data = {}
        twitter_data = {}

        try:
            reddit_data = self.crawl_reddit()
        except Exception as exc:
            logger.error("Reddit crawl failed: %s", exc)

        try:
            twitter_data = self.crawl_finance_twitter()
        except Exception as exc:
            logger.error("Twitter crawl failed: %s", exc)

        # Merge ticker sentiment from all sources
        merged = {}
        for source_data in [reddit_data, twitter_data]:
            for ticker, data in source_data.get("ticker_mentions", {}).items():
                if ticker not in merged:
                    merged[ticker] = {
                        "mentions": 0, "avg_sentiment": 0, "sentiments": [],
                        "bullish": 0, "bearish": 0, "neutral": 0,
                        "engagement": 0, "sources": [],
                    }
                m = merged[ticker]
                m["mentions"] += data["mentions"]
                m["sentiments"].append(data["avg_sentiment"])
                m["bullish"] += data["bullish"]
                m["bearish"] += data["bearish"]
                m["neutral"] += data["neutral"]
                m["engagement"] += data["engagement"]
                m["sources"].extend(data.get("sources", []))

        # Finalize merged averages
        for ticker, m in merged.items():
            sents = m.pop("sentiments")
            m["avg_sentiment"] = round(sum(sents) / len(sents), 4) if sents else 0
            m["sources"] = sorted(set(m["sources"]))
            total = m["bullish"] + m["bearish"] + m["neutral"]
            if total > 0:
                bull_pct = m["bullish"] / total
                bear_pct = m["bearish"] / total
                m["consensus"] = "bullish" if bull_pct > 0.6 else "bearish" if bear_pct > 0.6 else "mixed"
            else:
                m["consensus"] = "unknown"

        # Combined trending by total mentions
        combined_trending = sorted(
            [{"ticker": t, **d} for t, d in merged.items()],
            key=lambda x: x["mentions"],
            reverse=True,
        )[:25]

        self.close()

        return {
            "reddit": reddit_data,
            "twitter": twitter_data,
            "combined_trending": combined_trending,
            "ticker_sentiment": merged,
            "timestamp": datetime.now().isoformat(),
        }

    def get_ticker_sentiment(self, tickers: list) -> dict:
        """Get sentiment for specific tickers only.

        More efficient than crawl_all when you already know which tickers
        to check — still crawls all sources but only returns matches.

        Parameters
        ----------
        tickers : list
            Tickers to get sentiment for.

        Returns
        -------
        dict
            Keyed by ticker: mentions, avg_sentiment, consensus, sources.
        """
        self.target_tickers = set(t.upper() for t in tickers)
        result = self.crawl_all()
        all_sentiment = result.get("ticker_sentiment", {})

        # Return only requested tickers
        return {
            t.upper(): all_sentiment.get(t.upper(), {
                "mentions": 0, "avg_sentiment": 0,
                "consensus": "unknown", "sources": [],
            })
            for t in tickers
        }
