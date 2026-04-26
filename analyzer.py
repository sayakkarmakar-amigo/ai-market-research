"""Gemini-powered analysis of the day's AI signal.

Produces a structured JSON briefing the UI can render directly. Uses Gemini
2.5 Flash by default — generous free tier and JSON-mode support.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types as gtypes


# Default to Gemini 2.5 Flash — best free-tier balance. Override via env or UI.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


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


def _client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set — get a free key at "
            "https://aistudio.google.com/apikey and add it to your environment."
        )
    return genai.Client(api_key=key)


def _articles_to_context(articles: list[dict], max_chars: int = 80_000) -> str:
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


def _repair_truncated_json(text: str) -> str:
    """Close any unclosed strings/arrays/objects in a truncated JSON blob.

    Tracks the deepest "safe" prefix — the last point where the parser could
    have stopped cleanly — along with the stack at that moment, so the right
    set of closers can be appended. Used when Gemini cuts off mid-emission.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    last_safe_pos = -1
    last_safe_stack: list[str] | None = None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                # After a balanced close we're at a complete value.
                last_safe_pos = i + 1
                last_safe_stack = stack[:]
        elif ch == "," and stack:
            # Comma inside a structure marks the end of the previous element.
            last_safe_pos = i
            last_safe_stack = stack[:]

    if last_safe_pos > 0:
        prefix = text[:last_safe_pos].rstrip().rstrip(",")
        return prefix + "".join(reversed(last_safe_stack or []))
    # No safe spot found — best-effort close of what we have.
    return text + "".join(reversed(stack))


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction — strip fences, find first {…}, repair if truncated."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    candidate = text[start:]
    # First try: balanced-brace scan
    depth = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[: i + 1])
                except json.JSONDecodeError:
                    break
    # Second try: repair truncated JSON
    repaired = _repair_truncated_json(candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse JSON even after repair ({e}). "
            f"Output started with: {candidate[:200]!r}"
        )


def analyze(
    articles: list[dict],
    industries: list[str] | None = None,
    model: str | None = None,
    max_articles: int = 80,
) -> dict[str, Any]:
    """Run Gemini over the article list and return the structured briefing."""
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
    model_id = model or DEFAULT_MODEL

    # Gemini 2.5 models default to internal "thinking" tokens that count
    # against max_output_tokens — for a JSON-extraction task we don't need
    # them and they cause silent truncation. Disable for 2.5 models.
    cfg_kwargs: dict[str, Any] = dict(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.3,
        max_output_tokens=16384,
    )
    if "2.5" in model_id:
        try:
            cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

    resp = client.models.generate_content(
        model=model_id,
        contents=user_msg,
        config=gtypes.GenerateContentConfig(**cfg_kwargs),
    )

    text = resp.text or ""
    briefing = _extract_json(text)

    usage = getattr(resp, "usage_metadata", None)
    briefing["_meta"] = {
        "generated_at": datetime.utcnow().isoformat(),
        "model": model_id,
        "article_count": len(articles),
        "input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
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
