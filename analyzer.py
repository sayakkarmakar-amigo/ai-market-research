"""Claude-powered analysis of the day's AI signal.

Produces a structured JSON briefing the UI can render directly. Uses prompt
caching on the static system prompt so repeated daily runs are cheap.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from anthropic import Anthropic


# Default to Sonnet 4.6 — best balance of cost/quality. Override via env.
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
HAIKU_MODEL = "claude-haiku-4-5-20251001"


SYSTEM_PROMPT = """You are the Market Research Analyst for the user's "Virtual Crew".

Your job: turn raw AI-industry news into a sharp, executive-grade daily briefing.

Be ruthlessly concrete. Distinguish:
- ANNOUNCEMENTS / vapor (press releases, vague roadmap items)
- PILOTS (limited deployments, partnerships, beta access)
- PRODUCTION (live customer-facing, GA, measurable adoption)

Always identify the *industry vertical* a story belongs to (Healthcare, Finance,
Retail, Manufacturing, Defense, Education, Media, Legal, Energy, Logistics,
Public Sector, Developer Tools, Consumer, Enterprise SaaS, Hardware/Chips,
Research/Academia, Other).

When you quote an industry leader, attribute the exact source. Never invent
quotes — if no quote is in the source material, omit the quotes section rather
than fabricate.

Output ONLY valid JSON matching the requested schema. No prose, no markdown
fences, no commentary outside the JSON.
"""


SCHEMA_INSTRUCTIONS = """Return JSON of the form:

{
  "headline": "<one-sentence summary of the day's most important development>",
  "executive_summary": "<3-5 sentences. What a busy CEO needs to know.>",
  "top_stories": [
    {
      "title": "...",
      "url": "...",
      "source": "...",
      "why_it_matters": "<1-2 sentences>",
      "industry": "<vertical>",
      "stage": "announcement|pilot|production",
      "companies": ["..."]
    }
  ],
  "industry_leaders": [
    {
      "person": "<name>",
      "role": "<title @ company>",
      "quote_or_position": "<actual quote OR paraphrased position with source>",
      "url": "..."
    }
  ],
  "big_tech_moves": [
    {
      "company": "Meta|Apple|Amazon|Netflix|Google|Microsoft|OpenAI|Anthropic|NVIDIA|...",
      "move": "<what they did>",
      "signal": "<what it tells the market>",
      "url": "..."
    }
  ],
  "implementations": [
    {
      "company": "<adopter>",
      "what": "<concrete deployment>",
      "industry": "<vertical>",
      "stage": "pilot|production",
      "outcome_or_metric": "<measurable result if cited, else 'not disclosed'>",
      "url": "..."
    }
  ],
  "themes": ["<3-6 short trend phrases>"],
  "watchlist": ["<2-4 things to track over the coming week>"]
}

Include 5-8 top stories, up to 6 leader quotes, up to 8 big-tech moves, and as
many implementations as the source material supports (target 5-15).
"""


def _client() -> Anthropic:
    return Anthropic()


def _articles_to_context(articles: list[dict], max_chars: int = 60_000) -> str:
    """Render the article list as a compact context block. Truncate if huge."""
    lines = []
    for i, a in enumerate(articles, 1):
        block = (
            f"[{i}] {a.get('title','').strip()}\n"
            f"    source: {a.get('source','')} | published: {a.get('published','')}\n"
            f"    url: {a.get('url','')}\n"
            f"    companies: {', '.join(a.get('companies', []))}\n"
            f"    summary: {(a.get('summary') or '').strip()[:500]}\n"
        )
        lines.append(block)
        if sum(len(x) for x in lines) > max_chars:
            break
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction — strip markdown fences, find first {…}."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first balanced brace block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON in model output")


def analyze(
    articles: list[dict],
    industries: list[str] | None = None,
    model: str | None = None,
    max_articles: int = 80,
) -> dict[str, Any]:
    """Run Claude over the article list and return the structured briefing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — add it to your environment or .env file."
        )

    articles = articles[:max_articles]
    context = _articles_to_context(articles)
    industries_clause = (
        f"\n\nThe user is *especially* interested in these industries: "
        f"{', '.join(industries)}. Weight stories from those verticals higher."
        if industries
        else ""
    )

    user_msg = (
        f"TODAY: {datetime.utcnow().date().isoformat()}\n\n"
        f"Here is today's harvested AI news ({len(articles)} items):\n\n"
        f"{context}\n\n"
        f"{SCHEMA_INSTRUCTIONS}{industries_clause}"
    )

    client = _client()
    resp = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": SCHEMA_INSTRUCTIONS,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_msg}],
    )

    text = "".join(block.text for block in resp.content if block.type == "text")
    briefing = _extract_json(text)
    briefing["_meta"] = {
        "generated_at": datetime.utcnow().isoformat(),
        "model": model or DEFAULT_MODEL,
        "article_count": len(articles),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
        "cache_creation_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
    }
    return briefing


def quick_email_summary(briefing: dict) -> str:
    """Plaintext-ish HTML body for the morning email."""
    lines = [
        f"<h2>Daily AI Briefing — {briefing.get('_meta', {}).get('generated_at', '')[:10]}</h2>",
        f"<p><strong>{briefing.get('headline','')}</strong></p>",
        f"<p>{briefing.get('executive_summary','')}</p>",
        "<h3>Top stories</h3><ol>",
    ]
    for s in briefing.get("top_stories", []):
        lines.append(
            f"<li><a href='{s.get('url','')}'>{s.get('title','')}</a> "
            f"<em>({s.get('industry','')}, {s.get('stage','')})</em><br>"
            f"{s.get('why_it_matters','')}</li>"
        )
    lines.append("</ol><h3>Big-tech moves</h3><ul>")
    for m in briefing.get("big_tech_moves", []):
        lines.append(
            f"<li><strong>{m.get('company','')}</strong>: {m.get('move','')} "
            f"— <em>{m.get('signal','')}</em></li>"
        )
    lines.append("</ul><h3>Implementations in the wild</h3><ul>")
    for imp in briefing.get("implementations", []):
        lines.append(
            f"<li><strong>{imp.get('company','')}</strong> ({imp.get('industry','')}, "
            f"{imp.get('stage','')}): {imp.get('what','')} — "
            f"<em>{imp.get('outcome_or_metric','')}</em></li>"
        )
    lines.append("</ul>")
    if briefing.get("themes"):
        lines.append("<h3>Themes</h3><p>" + " · ".join(briefing["themes"]) + "</p>")
    if briefing.get("watchlist"):
        lines.append(
            "<h3>Watchlist</h3><ul>"
            + "".join(f"<li>{w}</li>" for w in briefing["watchlist"])
            + "</ul>"
        )
    return "\n".join(lines)
