"""News & signal aggregator for the AI Market Research assistant.

Pulls from public sources that don't need an API key:
  - Curated RSS feeds (general AI press + corp/research blogs)
  - Hacker News top stories
  - Reddit JSON listings
  - Optional NewsAPI (if NEWSAPI_KEY env var set)
"""
from __future__ import annotations

import os
import time
import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Iterable

import feedparser
import requests


UA = "Mozilla/5.0 (compatible; AI-MarketResearch/1.0)"

# ---- Curated feed registry -------------------------------------------------
# Grouped so the UI can let the user toggle whole categories.
FEEDS: dict[str, dict[str, str]] = {
    "press": {
        "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        "Wired": "https://www.wired.com/feed/rss",
        "IEEE Spectrum AI": "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss",
        "The Register AI": "https://www.theregister.com/software/ai_ml/headlines.atom",
    },
    "corp_research": {
        "OpenAI": "https://openai.com/blog/rss.xml",
        "Google AI": "https://blog.google/technology/ai/rss/",
        "Google DeepMind": "https://deepmind.google/blog/rss.xml",
        "Microsoft AI": "https://blogs.microsoft.com/ai/feed/",
        "Meta AI": "https://ai.meta.com/blog/rss/",
        "NVIDIA Blog": "https://blogs.nvidia.com/feed/",
        "AWS ML Blog": "https://aws.amazon.com/blogs/machine-learning/feed/",
        "Apple ML": "https://machinelearning.apple.com/rss.xml",
        "Amazon Science": "https://www.amazon.science/index.rss",
        "Hugging Face": "https://huggingface.co/blog/feed.xml",
    },
    "business": {
        "WSJ Tech": "https://feeds.a.dj.com/rss/RSSWSJD.xml",
        "FT Tech": "https://www.ft.com/technology?format=rss",
        "Bloomberg Tech": "https://feeds.bloomberg.com/technology/news.rss",
        "Reuters Tech": "https://www.reutersagency.com/feed/?best-topics=tech&post_type=best",
        "CNBC Tech": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        "HBR": "https://hbr.org/feed",
    },
    "research": {
        "arXiv cs.AI": "http://export.arxiv.org/rss/cs.AI",
        "arXiv cs.LG": "http://export.arxiv.org/rss/cs.LG",
        "arXiv cs.CL": "http://export.arxiv.org/rss/cs.CL",
    },
}

# Companies whose presence in a story marks it as "big tech / industry-tone-setting".
BIG_PLAYERS = [
    "Meta", "Apple", "Amazon", "Netflix", "Google", "Alphabet",
    "Microsoft", "OpenAI", "Anthropic", "NVIDIA", "Tesla", "xAI",
    "DeepMind", "Hugging Face", "Mistral", "Cohere", "IBM", "Oracle",
    "Salesforce", "Databricks", "Snowflake", "Palantir", "AMD",
]


@dataclass
class Article:
    title: str
    url: str
    source: str
    published: str  # ISO 8601 string
    summary: str = ""
    category: str = ""
    score: int = 0  # for HN/Reddit ranking signal
    companies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Helpers --------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


def _parse_dt(entry) -> str:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        val = getattr(entry, key, None) or (entry.get(key) if isinstance(entry, dict) else None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()


def _detect_companies(text: str) -> list[str]:
    if not text:
        return []
    found = []
    lower = text.lower()
    for name in BIG_PLAYERS:
        if name.lower() in lower and name not in found:
            found.append(name)
    return found


# ---- Fetchers -------------------------------------------------------------
def fetch_feed(name: str, url: str, category: str, limit: int = 25) -> list[Article]:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception:
        return []

    out: list[Article] = []
    for e in parsed.entries[:limit]:
        title = _strip_html(getattr(e, "title", ""))
        link = getattr(e, "link", "") or ""
        summary = _strip_html(getattr(e, "summary", "") or getattr(e, "description", ""))
        if not title or not link:
            continue
        text = f"{title} {summary}"
        out.append(
            Article(
                title=title,
                url=link,
                source=name,
                published=_parse_dt(e),
                summary=summary[:600],
                category=category,
                companies=_detect_companies(text),
            )
        )
    return out


def fetch_hackernews(limit: int = 30) -> list[Article]:
    try:
        ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            headers={"User-Agent": UA},
            timeout=10,
        ).json()[:limit]
    except Exception:
        return []

    out: list[Article] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(
                requests.get,
                f"https://hacker-news.firebaseio.com/v0/item/{i}.json",
                headers={"User-Agent": UA},
                timeout=10,
            ): i
            for i in ids
        }
        for fut in as_completed(futures):
            try:
                item = fut.result().json()
            except Exception:
                continue
            if not item or item.get("type") != "story":
                continue
            title = item.get("title") or ""
            url = item.get("url") or f"https://news.ycombinator.com/item?id={item.get('id')}"
            text = title.lower()
            # Rough AI relevance filter
            if not any(k in text for k in ("ai", "gpt", "llm", "model", "openai", "anthropic", "claude", "gemini", "meta", "neural", "ml ")):
                continue
            ts = item.get("time", time.time())
            out.append(
                Article(
                    title=title,
                    url=url,
                    source="Hacker News",
                    published=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    summary="",
                    category="community",
                    score=item.get("score", 0),
                    companies=_detect_companies(title),
                )
            )
    return out


def fetch_reddit(subs: Iterable[str] = ("MachineLearning", "artificial", "singularity", "LocalLLaMA"), limit: int = 15) -> list[Article]:
    out: list[Article] = []
    for sub in subs:
        try:
            data = requests.get(
                f"https://www.reddit.com/r/{sub}/top.json?t=day&limit={limit}",
                headers={"User-Agent": UA},
                timeout=10,
            ).json()
        except Exception:
            continue
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            title = d.get("title") or ""
            if not title:
                continue
            url = d.get("url_overridden_by_dest") or f"https://reddit.com{d.get('permalink', '')}"
            ts = d.get("created_utc", time.time())
            out.append(
                Article(
                    title=title,
                    url=url,
                    source=f"r/{sub}",
                    published=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    summary=_strip_html(d.get("selftext", ""))[:500],
                    category="community",
                    score=d.get("score", 0),
                    companies=_detect_companies(title),
                )
            )
    return out


def fetch_newsapi(query: str = "artificial intelligence", page_size: int = 30) -> list[Article]:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": page_size,
                "apiKey": key,
            },
            timeout=15,
        )
        items = resp.json().get("articles", [])
    except Exception:
        return []
    return [
        Article(
            title=a.get("title", ""),
            url=a.get("url", ""),
            source=a.get("source", {}).get("name", "NewsAPI"),
            published=a.get("publishedAt", datetime.now(timezone.utc).isoformat()),
            summary=(a.get("description") or "")[:500],
            category="press",
            companies=_detect_companies(f"{a.get('title','')} {a.get('description','')}"),
        )
        for a in items
        if a.get("title") and a.get("url")
    ]


# ---- Aggregator -----------------------------------------------------------
def fetch_all(
    enabled_categories: Iterable[str] = ("press", "corp_research", "business", "research"),
    include_hn: bool = True,
    include_reddit: bool = True,
    extra_query: str | None = None,
    per_feed_limit: int = 20,
) -> list[Article]:
    """Fetch and dedupe articles from all enabled sources concurrently."""
    jobs: list[tuple] = []
    for cat in enabled_categories:
        for name, url in FEEDS.get(cat, {}).items():
            jobs.append((fetch_feed, (name, url, cat, per_feed_limit)))
    if include_hn:
        jobs.append((fetch_hackernews, (40,)))
    if include_reddit:
        jobs.append((fetch_reddit, ()))
    if extra_query:
        jobs.append((fetch_newsapi, (extra_query,)))

    results: list[Article] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fn, *args) for fn, args in jobs]
        for fut in as_completed(futures):
            try:
                results.extend(fut.result())
            except Exception:
                continue

    # Dedupe by URL, then sort newest-first
    seen = set()
    deduped = []
    for a in results:
        if a.url in seen:
            continue
        seen.add(a.url)
        deduped.append(a)
    deduped.sort(key=lambda a: a.published, reverse=True)
    return deduped


def filter_by_industry(articles: list[Article], industries: list[str]) -> list[Article]:
    """Lightweight keyword match against title + summary."""
    if not industries:
        return articles
    keywords = [i.lower() for i in industries]
    out = []
    for a in articles:
        hay = f"{a.title} {a.summary}".lower()
        if any(k in hay for k in keywords):
            out.append(a)
    return out
