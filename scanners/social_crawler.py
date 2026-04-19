"""
Social Media Crawler — intelligent scraping for financial signal extraction.

Goes beyond mention counting: classifies post types (DD, catalyst, position,
meme), extracts specific narratives (price targets, catalysts, risks, flow
direction), and weights by post quality and engagement.

Sources:
    Reddit: r/wallstreetbets, r/stocks, r/options, r/thetagang, r/personalfinance
    Finance: Unusual Whales flow page
    Twitter: @unusual_whales, @DeItaone, @zaboravom via Nitter mirrors

Rate-limited and respectful: 2-3 second delays between pages, headless only.
"""

import logging
import re
import time
from datetime import datetime

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
VADER = SentimentIntensityAnalyzer()

# ------------------------------------------------------------------
#  Ticker extraction patterns
# ------------------------------------------------------------------

TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
LOOSE_TICKER = re.compile(r'\b([A-Z]{2,5})\b')

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

# ------------------------------------------------------------------
#  Post classification patterns
# ------------------------------------------------------------------

# Post type detection — ordered by signal quality (DD > catalyst > position > news > meme)
_DD_PATTERNS = re.compile(
    r'\b(DD|due diligence|deep dive|analysis|thesis|bull case|bear case|'
    r'valuation|dcf|price target|fair value|undervalued|overvalued|'
    r'free cash flow|revenue growth|margin expansion|TAM|moat)\b',
    re.IGNORECASE,
)
_CATALYST_PATTERNS = re.compile(
    r'\b(FDA|approval|phase \d|trial|earnings|guidance|buyback|'
    r'dividend|split|acquisition|merger|M&A|partnership|deal|'
    r'contract|launch|release|upgrade|downgrade|recall|investigation|'
    r'lawsuit|settlement|tariff|regulation|ban|shortage|delay)\b',
    re.IGNORECASE,
)
_POSITION_PATTERNS = re.compile(
    r'\b(bought|sold|opened|closed|holding|position|shares|contracts|'
    r'calls?|puts?|spread|straddle|strangle|iron condor|'
    r'wheel|CSP|CC|LEAPS?|FD|weeklies?|monthlies?|'
    r'\d+[cp]\s|\d+\s*strike)\b',
    re.IGNORECASE,
)
_RISK_PATTERNS = re.compile(
    r'\b(warning|risk|danger|overextended|overbought|oversold|'
    r'bubble|crash|correction|pullback|resistance|breakdown|'
    r'dilution|shelf offering|secondary|insider sell|lock.?up|'
    r'short report|citron|hindenburg|muddy waters|grizzly)\b',
    re.IGNORECASE,
)
_PRICE_TARGET_PATTERN = re.compile(
    r'(?:price target|pt|target|fair value)[:\s]*\$?\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
_POSITION_SIZE_PATTERN = re.compile(
    r'(\d+)\s*(?:shares|contracts|calls|puts|options)',
    re.IGNORECASE,
)

# Unusual Whales flow — extract premium size for weighting
_PREMIUM_PATTERN = re.compile(r'\$[\d,.]+[KMB]|\$[\d,]+(?:\.\d+)?')
_EXPIRY_PATTERN = re.compile(r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)')
_STRIKE_PATTERN = re.compile(r'\$?(\d+(?:\.\d+)?)\s*(?:c|p|call|put)', re.IGNORECASE)

REDDIT_SUBS = [
    "wallstreetbets",
    "stocks",
    "options",
    "thetagang",
    "personalfinance",
]

FINANCE_TWITTER = [
    "unusual_whales",
    "DeItaone",
    "zaboravom",
]


def _classify_post(text: str) -> dict:
    """Classify a post's type and extract structured signals.

    Returns
    -------
    dict
        'post_type' (str): dd, catalyst, position, risk_flag, news, meme
        'quality_score' (float): 0-1 quality multiplier for weighting
        'catalysts' (list[str]): specific catalysts mentioned
        'risks' (list[str]): specific risks flagged
        'price_targets' (list[float]): any price targets mentioned
        'positions_disclosed' (list[str]): disclosed positions
        'flow_direction' (str|None): bullish/bearish from options flow
    """
    result = {
        "post_type": "noise",
        "quality_score": 0.1,
        "catalysts": [],
        "risks": [],
        "price_targets": [],
        "positions_disclosed": [],
        "flow_direction": None,
    }

    text_lower = text.lower()

    # Classify post type (highest-signal type wins)
    dd_matches = _DD_PATTERNS.findall(text)
    catalyst_matches = _CATALYST_PATTERNS.findall(text)
    position_matches = _POSITION_PATTERNS.findall(text)
    risk_matches = _RISK_PATTERNS.findall(text)

    if dd_matches:
        result["post_type"] = "dd"
        result["quality_score"] = 0.9
    elif catalyst_matches:
        result["post_type"] = "catalyst"
        result["quality_score"] = 0.8
    elif risk_matches:
        result["post_type"] = "risk_flag"
        result["quality_score"] = 0.7
    elif position_matches:
        result["post_type"] = "position"
        result["quality_score"] = 0.5
    else:
        # Check if it's at least news-like (has numbers, dates, specifics)
        has_numbers = bool(re.search(r'\d+%|\$\d+', text))
        has_length = len(text) > 100
        if has_numbers and has_length:
            result["post_type"] = "news"
            result["quality_score"] = 0.4
        else:
            result["post_type"] = "noise"
            result["quality_score"] = 0.1

    # Extract catalysts
    if catalyst_matches:
        result["catalysts"] = list(set(m.lower() for m in catalyst_matches))

    # Extract risks
    if risk_matches:
        result["risks"] = list(set(m.lower() for m in risk_matches))

    # Extract price targets
    pt_matches = _PRICE_TARGET_PATTERN.findall(text)
    if pt_matches:
        result["price_targets"] = [float(pt) for pt in pt_matches]

    # Detect flow direction from options language
    call_signals = len(re.findall(r'\bbought?\s+calls?\b|\blong\s+calls?\b|\bbullish\s+flow\b', text_lower))
    put_signals = len(re.findall(r'\bbought?\s+puts?\b|\blong\s+puts?\b|\bbearish\s+flow\b', text_lower))
    if call_signals > put_signals:
        result["flow_direction"] = "bullish"
    elif put_signals > call_signals:
        result["flow_direction"] = "bearish"

    # Extract position disclosures
    positions = _POSITION_SIZE_PATTERN.findall(text)
    if positions:
        # Reconstruct position strings from surrounding context
        for m in re.finditer(r'(\d+)\s*(shares|contracts|calls|puts|options)', text, re.IGNORECASE):
            result["positions_disclosed"].append(m.group(0))

    return result


class SocialCrawler:
    """Crawl Reddit and finance social media for intelligent signal extraction."""

    def __init__(self, target_tickers: list = None):
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
        """Extract ticker symbols from text."""
        tickers = set()
        for match in TICKER_PATTERN.findall(text):
            if match not in TICKER_BLACKLIST:
                tickers.add(match)
        if self.target_tickers:
            for match in LOOSE_TICKER.findall(text):
                if match in self.target_tickers:
                    tickers.add(match)
        return list(tickers)

    def _score_text(self, text: str) -> dict:
        """Score text sentiment using VADER."""
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
        """Crawl target subreddits via Reddit's public JSON API.

        Uses reddit.com/r/SUB/hot.json (no OAuth needed). Falls back to
        Playwright scrape of old.reddit.com if JSON returns nothing.
        """
        all_posts = []

        for sub in REDDIT_SUBS:
            try:
                posts = self._crawl_subreddit_json(sub)
                if not posts:
                    # OAuth/JSON failed — try RSS (works from datacenter IPs)
                    posts = self._crawl_subreddit_rss(sub)
                if not posts:
                    # RSS failed — try Playwright as last resort
                    self._ensure_browser()
                    posts = self._crawl_subreddit(sub)
                all_posts.extend(posts)
                logger.info("Crawled r/%s: %d posts", sub, len(posts))
            except Exception as exc:
                logger.error("Failed to crawl r/%s: %s", sub, exc)
            time.sleep(2)  # respect Reddit's 60 req / 10 min unauthenticated cap

        ticker_data = self._aggregate_mentions(all_posts)

        trending = sorted(
            ticker_data.items(),
            key=lambda x: x[1]["signal_strength"],
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

    def _get_reddit_token(self) -> str:
        """Get OAuth token for Reddit's authenticated API.

        Tries multiple grant types to handle different Reddit app configurations:
        1. client_credentials (web/confidential apps)
        2. installed_client (installed/public apps — no username needed)
        3. password (script apps — needs REDDIT_USERNAME + REDDIT_PASSWORD)

        Authenticated API uses oauth.reddit.com which works from datacenter IPs
        (unlike www.reddit.com/r/X.json which returns 403 from GitHub Actions).
        """
        if hasattr(self, "_reddit_token") and self._reddit_token:
            return self._reddit_token

        import os
        client_id = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        if not client_id:
            return ""

        ua = "weekly-options-scanner/1.0 (by /u/weekly-options-bot)"

        # Try 1: installed_client grant (works for any app type, read-only)
        try:
            resp = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(client_id, client_secret or ""),
                data={
                    "grant_type": "https://oauth.reddit.com/grants/installed_client",
                    "device_id": "DO_NOT_TRACK_THIS_DEVICE",
                },
                headers={"User-Agent": ua},
                timeout=10,
            )
            if resp.status_code == 200:
                token = resp.json().get("access_token", "")
                if token:
                    self._reddit_token = token
                    logger.info("Reddit OAuth: installed_client grant succeeded")
                    return token
        except Exception:
            pass

        # Try 2: client_credentials (confidential/web apps)
        if client_secret:
            try:
                resp = requests.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=(client_id, client_secret),
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": ua},
                    timeout=10,
                )
                if resp.status_code == 200:
                    token = resp.json().get("access_token", "")
                    if token:
                        self._reddit_token = token
                        logger.info("Reddit OAuth: client_credentials grant succeeded")
                        return token
            except Exception:
                pass

        # Try 3: password grant (script apps — requires username/password)
        username = os.getenv("REDDIT_USERNAME", "")
        password = os.getenv("REDDIT_PASSWORD", "")
        if username and password and client_secret:
            try:
                resp = requests.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=(client_id, client_secret),
                    data={
                        "grant_type": "password",
                        "username": username,
                        "password": password,
                    },
                    headers={"User-Agent": ua},
                    timeout=10,
                )
                if resp.status_code == 200:
                    token = resp.json().get("access_token", "")
                    if token:
                        self._reddit_token = token
                        logger.info("Reddit OAuth: password grant succeeded")
                        return token
            except Exception:
                pass

        logger.warning("All Reddit OAuth grant types failed")
        return ""

    def _crawl_subreddit_json(self, subreddit: str, limit: int = 50) -> list:
        """Fetch posts via Reddit API with OAuth fallback.

        Tries authenticated OAuth endpoint first (works from datacenter IPs),
        falls back to public JSON (works from residential IPs).
        """
        ua = "weekly-options-scanner/1.0 (signal aggregation; contact via repo)"

        # Try OAuth endpoint first (works from GitHub Actions / datacenter IPs)
        token = self._get_reddit_token()
        if token:
            url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit={limit}"
            headers = {"Authorization": f"Bearer {token}", "User-Agent": ua}
            try:
                resp = requests.get(url, headers=headers, timeout=12)
                if resp.status_code == 200:
                    children = resp.json().get("data", {}).get("children", [])
                    return self._parse_reddit_children(children, subreddit)
                logger.warning("Reddit OAuth r/%s returned %d, trying public JSON",
                               subreddit, resp.status_code)
            except Exception as exc:
                logger.warning("Reddit OAuth failed for r/%s: %s, trying public JSON",
                               subreddit, exc)

        # Fallback: public JSON (works from residential IPs, blocked from datacenters)
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        headers = {"User-Agent": ua, "Accept": "application/json"}
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                logger.warning("Reddit JSON r/%s returned %d", subreddit, resp.status_code)
                return []
            children = resp.json().get("data", {}).get("children", [])
        except Exception as exc:
            logger.warning("Reddit JSON fetch failed for r/%s: %s", subreddit, exc)
            return []

        return self._parse_reddit_children(children, subreddit)

    def _parse_reddit_children(self, children: list, subreddit: str) -> list:
        """Parse Reddit API children into post dicts for aggregation."""
        posts = []
        for child in children:
            d = child.get("data", {})
            title = (d.get("title") or "").strip()
            if not title:
                continue
            score = int(d.get("score") or 0)
            comments = int(d.get("num_comments") or 0)
            if score < 5 and comments < 3:
                continue

            flair = (d.get("link_flair_text") or "").strip()
            tickers = self._extract_tickers(title)
            sentiment = self._score_text(title)
            classification = _classify_post(title)

            flair_lower = flair.lower()
            if flair_lower in ("dd", "due diligence", "research", "analysis"):
                classification["quality_score"] = max(classification["quality_score"], 0.9)
                classification["post_type"] = "dd"
            elif flair_lower in ("catalyst", "news"):
                classification["quality_score"] = max(classification["quality_score"], 0.7)
            elif flair_lower in ("yolo", "gain", "loss"):
                classification["quality_score"] = max(classification["quality_score"], 0.3)
                classification["post_type"] = "position"
            elif flair_lower in ("meme", "shitpost"):
                classification["quality_score"] = 0.05

            posts.append({
                "title": title,
                "flair": flair,
                "score": score,
                "comments": comments,
                "tickers": tickers,
                "sentiment": sentiment,
                "classification": classification,
                "subreddit": subreddit,
                "url": d.get("permalink", ""),
            })
        return posts

    def _crawl_subreddit_rss(self, subreddit: str, limit: int = 50) -> list:
        """Fetch posts via Reddit's Atom RSS feed.

        RSS uses a different endpoint (/hot/.rss) that may not be blocked
        from datacenter IPs. Returns title + content snippet (HTML) but
        NO scores, comment counts, or flair. Quality weighting falls back
        to pattern-based classification only.
        """
        import xml.etree.ElementTree as ET

        url = f"https://www.reddit.com/r/{subreddit}/hot/.rss"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "weekly-options-scanner/1.0"},
                timeout=12,
            )
            if resp.status_code != 200:
                logger.warning("Reddit RSS r/%s returned %d", subreddit, resp.status_code)
                return []
            root = ET.fromstring(resp.text)
        except Exception as exc:
            logger.warning("Reddit RSS failed for r/%s: %s", subreddit, exc)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        posts = []
        for entry in entries[:limit]:
            title_el = entry.find("atom:title", ns)
            content_el = entry.find("atom:content", ns)
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            if not title:
                continue

            # Strip HTML from content for clean text
            raw_content = content_el.text if content_el is not None and content_el.text else ""
            clean_content = re.sub(r"<[^>]+>", " ", raw_content).strip()
            full_text = f"{title} {clean_content}"

            tickers = self._extract_tickers(full_text)
            sentiment = self._score_text(full_text)
            classification = _classify_post(full_text)

            # No flair from RSS — rely entirely on text pattern classification
            posts.append({
                "title": title,
                "flair": "",
                "score": 0,      # Unknown from RSS
                "comments": 0,   # Unknown from RSS
                "tickers": tickers,
                "sentiment": sentiment,
                "classification": classification,
                "subreddit": subreddit,
                "url": "",
                "source_method": "rss",
            })

        if posts:
            logger.info("Reddit RSS r/%s: %d posts (no scores/flair — text-only signal)",
                        subreddit, len(posts))
        return posts

    def _crawl_subreddit(self, subreddit: str, limit: int = 50) -> list:
        """Crawl a subreddit with post classification and narrative extraction."""
        page = self._context.new_page()
        posts = []

        try:
            url = f"https://old.reddit.com/r/{subreddit}/hot/"
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)

            entries = page.query_selector_all("div.thing.link")

            for entry in entries[:limit]:
                try:
                    title_el = entry.query_selector("a.title")
                    title = title_el.inner_text().strip() if title_el else ""

                    # Extract flair (DD, Discussion, YOLO, etc.)
                    flair_el = entry.query_selector("span.linkflairlabel")
                    flair = flair_el.inner_text().strip() if flair_el else ""

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

                    if score < 5 and comments < 3:
                        continue

                    tickers = self._extract_tickers(title)
                    sentiment = self._score_text(title)
                    classification = _classify_post(title)

                    # Flair boosts quality score — Reddit's own categorization
                    flair_lower = flair.lower()
                    if flair_lower in ("dd", "due diligence", "research", "analysis"):
                        classification["quality_score"] = max(classification["quality_score"], 0.9)
                        classification["post_type"] = "dd"
                    elif flair_lower in ("catalyst", "news"):
                        classification["quality_score"] = max(classification["quality_score"], 0.7)
                    elif flair_lower in ("yolo", "gain", "loss"):
                        # YOLO/gain/loss posts reveal positioning, not analysis
                        classification["quality_score"] = max(classification["quality_score"], 0.3)
                        classification["post_type"] = "position"
                    elif flair_lower in ("meme", "shitpost"):
                        classification["quality_score"] = 0.05

                    post_url_el = entry.query_selector("a.title")
                    post_url = post_url_el.get_attribute("href") if post_url_el else ""

                    posts.append({
                        "title": title,
                        "flair": flair,
                        "score": score,
                        "comments": comments,
                        "tickers": tickers,
                        "sentiment": sentiment,
                        "classification": classification,
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
        """Crawl finance Twitter accounts with narrative extraction."""
        self._ensure_browser()
        all_posts = []

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

        # Unusual Whales flow page
        try:
            uw_posts = self._crawl_unusual_whales()
            all_posts.extend(uw_posts)
            logger.info("Crawled Unusual Whales: %d items", len(uw_posts))
        except Exception as exc:
            logger.error("Unusual Whales crawl failed: %s", exc)

        ticker_data = self._aggregate_mentions(all_posts)

        trending = sorted(
            ticker_data.items(),
            key=lambda x: x[1]["signal_strength"],
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
        """Crawl a Twitter account via Nitter with content classification."""
        page = self._context.new_page()
        posts = []

        try:
            url = f"https://{instance}/{account}"
            page.goto(url, timeout=10000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)

            items = page.query_selector_all(".timeline-item")

            for item in items[:limit]:
                try:
                    content_el = item.query_selector(".tweet-content")
                    text = content_el.inner_text().strip() if content_el else ""
                    if not text:
                        continue

                    tickers = self._extract_tickers(text)
                    sentiment = self._score_text(text)
                    classification = _classify_post(text)

                    # Finance Twitter accounts are inherently higher quality
                    # than random Reddit posts — boost quality floor
                    classification["quality_score"] = max(
                        classification["quality_score"], 0.5)

                    # @DeItaone (Walter Bloomberg) is breaking news — treat as catalyst
                    if account == "DeItaone" and tickers:
                        classification["post_type"] = "catalyst"
                        classification["quality_score"] = max(
                            classification["quality_score"], 0.8)

                    posts.append({
                        "title": text[:300],
                        "tickers": tickers,
                        "sentiment": sentiment,
                        "classification": classification,
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
        """Crawl Unusual Whales with structured flow extraction."""
        page = self._context.new_page()
        posts = []

        try:
            page.goto("https://unusualwhales.com/flow",
                       timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

            rows = page.query_selector_all("table tbody tr")

            for row in rows[:50]:
                try:
                    cells = row.query_selector_all("td")
                    if len(cells) < 4:
                        continue

                    cell_texts = [c.inner_text().strip() for c in cells]
                    text = " ".join(cell_texts)
                    tickers = self._extract_tickers(text)

                    text_lower = text.lower()

                    # Structured flow extraction
                    is_call = "call" in text_lower
                    is_put = "put" in text_lower

                    # Extract premium size for institutional signal detection
                    premiums = _PREMIUM_PATTERN.findall(text)
                    premium_str = premiums[0] if premiums else ""

                    # Detect sweep vs block (sweeps = more urgency)
                    is_sweep = "sweep" in text_lower
                    is_block = "block" in text_lower

                    # Extract strike and expiry for context
                    strikes = _STRIKE_PATTERN.findall(text)
                    expiries = _EXPIRY_PATTERN.findall(text)

                    if is_call:
                        direction = "bullish"
                        sentiment = {"compound": 0.4, "positive": 0.4,
                                     "negative": 0.0, "neutral": 0.6}
                    elif is_put:
                        direction = "bearish"
                        sentiment = {"compound": -0.4, "positive": 0.0,
                                     "negative": 0.4, "neutral": 0.6}
                    else:
                        direction = "neutral"
                        sentiment = self._score_text(text)

                    # Institutional flow = high quality signal
                    quality = 0.7
                    if is_sweep:
                        quality = 0.9  # Sweeps indicate urgency
                    if premium_str and any(c in premium_str for c in "MBmb"):
                        quality = 0.95  # Million+ premium = institutional

                    classification = {
                        "post_type": "flow",
                        "quality_score": quality,
                        "catalysts": [],
                        "risks": [],
                        "price_targets": [],
                        "positions_disclosed": [],
                        "flow_direction": direction,
                        "flow_details": {
                            "type": "sweep" if is_sweep else "block" if is_block else "standard",
                            "premium": premium_str,
                            "strike": strikes[0] if strikes else None,
                            "expiry": expiries[0] if expiries else None,
                            "direction": direction,
                        },
                    }

                    if tickers:
                        posts.append({
                            "title": text[:300],
                            "tickers": tickers,
                            "sentiment": sentiment,
                            "classification": classification,
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
    #  Intelligent Aggregation
    # ------------------------------------------------------------------

    def _aggregate_mentions(self, posts: list) -> dict:
        """Aggregate posts into per-ticker intelligence, not just counts.

        Output per ticker includes:
        - Quality-weighted sentiment (DD posts count more than memes)
        - Extracted catalysts and risks
        - Institutional flow signals
        - Top narratives (highest-quality posts summarized)
        - Signal strength (composite of quality, engagement, conviction)
        """
        ticker_data = {}

        for post in posts:
            for ticker in post.get("tickers", []):
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "mentions": 0,
                        "_weighted_sentiments": [],
                        "_raw_sentiments": [],
                        "bullish": 0,
                        "bearish": 0,
                        "neutral": 0,
                        "engagement": 0,
                        "sources": set(),
                        "catalysts": [],
                        "risks": [],
                        "price_targets": [],
                        "flow_signals": [],
                        "top_posts": [],
                        "post_types": {},
                    }

                td = ticker_data[ticker]
                td["mentions"] += 1

                compound = post.get("sentiment", {}).get("compound", 0)
                cls = post.get("classification", {})
                quality = cls.get("quality_score", 0.1)

                td["_weighted_sentiments"].append((compound, quality))
                td["_raw_sentiments"].append(compound)

                if compound >= 0.05:
                    td["bullish"] += 1
                elif compound <= -0.05:
                    td["bearish"] += 1
                else:
                    td["neutral"] += 1

                # Engagement weighted by quality — a DD post with 500 upvotes
                # means more than a meme with 500 upvotes
                raw_engagement = post.get("score", 0) + post.get("comments", 0)
                td["engagement"] += raw_engagement * quality

                td["sources"].add(
                    post.get("source", post.get("subreddit", "unknown")))

                # Accumulate qualitative signals
                post_type = cls.get("post_type", "noise")
                td["post_types"][post_type] = td["post_types"].get(post_type, 0) + 1

                if cls.get("catalysts"):
                    td["catalysts"].extend(cls["catalysts"])
                if cls.get("risks"):
                    td["risks"].extend(cls["risks"])
                if cls.get("price_targets"):
                    td["price_targets"].extend(cls["price_targets"])

                # Flow signals from UW or position disclosures
                flow_dir = cls.get("flow_direction")
                if flow_dir:
                    flow_entry = {"direction": flow_dir, "source": post.get("source", "?")}
                    flow_details = cls.get("flow_details")
                    if flow_details:
                        flow_entry.update(flow_details)
                    td["flow_signals"].append(flow_entry)

                # Track top posts by quality * engagement
                post_signal = quality * max(raw_engagement, 1)
                td["top_posts"].append({
                    "title": post.get("title", "")[:200],
                    "type": post_type,
                    "quality": quality,
                    "engagement": raw_engagement,
                    "signal": post_signal,
                    "source": post.get("source", post.get("subreddit", "?")),
                    "sentiment": compound,
                    "catalysts": cls.get("catalysts", []),
                    "risks": cls.get("risks", []),
                })

        # Finalize each ticker's intelligence
        for ticker, td in ticker_data.items():
            # Quality-weighted sentiment (DD and catalyst posts weigh more)
            weighted = td.pop("_weighted_sentiments")
            raw = td.pop("_raw_sentiments")

            if weighted:
                total_weight = sum(w for _, w in weighted)
                td["avg_sentiment"] = round(
                    sum(s * w for s, w in weighted) / total_weight, 4
                ) if total_weight > 0 else 0
            else:
                td["avg_sentiment"] = 0

            td["raw_avg_sentiment"] = round(
                sum(raw) / len(raw), 4) if raw else 0

            td["sources"] = sorted(td["sources"])

            # Consensus
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

            # Deduplicate catalysts and risks
            td["catalysts"] = sorted(set(td["catalysts"]))
            td["risks"] = sorted(set(td["risks"]))

            # Aggregate flow direction
            flow_bulls = sum(1 for f in td["flow_signals"] if f["direction"] == "bullish")
            flow_bears = sum(1 for f in td["flow_signals"] if f["direction"] == "bearish")
            if flow_bulls + flow_bears > 0:
                td["flow_consensus"] = "bullish" if flow_bulls > flow_bears else "bearish"
                td["flow_conviction"] = abs(flow_bulls - flow_bears) / (flow_bulls + flow_bears)
            else:
                td["flow_consensus"] = "neutral"
                td["flow_conviction"] = 0

            # Keep only top 5 posts by signal strength
            td["top_posts"] = sorted(
                td["top_posts"], key=lambda x: x["signal"], reverse=True
            )[:5]

            # Build narrative summary — the key qualitative output
            td["narrative"] = self._build_narrative(td)

            # Signal strength: composite that accounts for quality, not just volume
            # DD mention + catalyst + flow alignment > 10 meme mentions
            dd_count = td["post_types"].get("dd", 0)
            catalyst_count = td["post_types"].get("catalyst", 0)
            flow_count = td["post_types"].get("flow", 0)
            risk_count = td["post_types"].get("risk_flag", 0)

            quality_mentions = (
                dd_count * 3
                + catalyst_count * 2.5
                + flow_count * 2
                + risk_count * 1.5
                + td["post_types"].get("position", 0) * 0.5
                + td["post_types"].get("news", 0) * 0.5
                + td["post_types"].get("noise", 0) * 0.1
            )
            engagement_factor = min(td["engagement"] / 500, 3)  # Cap at 3x
            td["signal_strength"] = round(quality_mentions * (1 + engagement_factor), 2)

        return ticker_data

    @staticmethod
    def _build_narrative(td: dict) -> str:
        """Build a human-readable narrative summary of what social is saying.

        This is the qualitative output that scoring and Discord can use
        to understand *what* people are talking about, not just how many.
        """
        parts = []

        # Mention volume context
        mentions = td["mentions"]
        if mentions >= 10:
            parts.append(f"High social activity ({mentions} mentions)")
        elif mentions >= 5:
            parts.append(f"Moderate buzz ({mentions} mentions)")

        # What type of discussion
        dd_count = td["post_types"].get("dd", 0)
        if dd_count:
            parts.append(f"{dd_count} DD/analysis post{'s' if dd_count > 1 else ''}")

        # Catalysts being discussed
        catalysts = td.get("catalysts", [])
        if catalysts:
            parts.append(f"Catalysts: {', '.join(catalysts[:4])}")

        # Risks being flagged
        risks = td.get("risks", [])
        if risks:
            parts.append(f"Risks flagged: {', '.join(risks[:3])}")

        # Price targets
        pts = td.get("price_targets", [])
        if pts:
            avg_pt = sum(pts) / len(pts)
            parts.append(f"Avg price target: ${avg_pt:.0f}")

        # Flow signals
        flow_signals = td.get("flow_signals", [])
        sweeps = [f for f in flow_signals if f.get("type") == "sweep"]
        if sweeps:
            direction = sweeps[0].get("direction", "?")
            parts.append(f"{len(sweeps)} {direction} sweep{'s' if len(sweeps) > 1 else ''} on UW")
        elif flow_signals:
            parts.append(f"Options flow: {td.get('flow_consensus', '?')}")

        # Top post snippet
        top = td.get("top_posts", [])
        if top and top[0]["quality"] >= 0.5:
            best = top[0]
            # Truncate title for narrative
            title_short = best["title"][:80]
            if len(best["title"]) > 80:
                title_short += "..."
            parts.append(f"Top signal ({best['type']}): \"{title_short}\"")

        return " | ".join(parts) if parts else "Minimal social signal"

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def crawl_all(self) -> dict:
        """Run all crawlers, merge, and produce intelligence output.

        Returns
        -------
        dict
            'reddit', 'twitter': raw source data
            'combined_trending': top tickers by signal_strength
            'ticker_sentiment': per-ticker intelligence with narratives
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

        # Merge all posts and re-aggregate for combined intelligence
        all_posts = []
        for source in [reddit_data, twitter_data]:
            all_posts.extend(source.get("posts", []))

        merged = self._aggregate_mentions(all_posts)

        # Combined trending by signal strength (not just mention count)
        combined_trending = sorted(
            [{"ticker": t, **d} for t, d in merged.items()],
            key=lambda x: x["signal_strength"],
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
        """Get intelligence for specific tickers only."""
        self.target_tickers = set(t.upper() for t in tickers)
        result = self.crawl_all()
        all_sentiment = result.get("ticker_sentiment", {})

        return {
            t.upper(): all_sentiment.get(t.upper(), {
                "mentions": 0, "avg_sentiment": 0,
                "consensus": "unknown", "sources": [],
                "signal_strength": 0, "narrative": "No social signal",
                "catalysts": [], "risks": [], "flow_signals": [],
            })
            for t in tickers
        }
