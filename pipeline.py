"""End-to-end daily run: fetch -> analyze -> persist -> (optional) email."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import sources
import storage
from analyzer import analyze, quick_email_summary
from mailer import send_email


def run_daily(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or storage.load_config()

    articles = sources.fetch_all(
        enabled_categories=cfg.get("enabled_categories", []),
        include_hn=cfg.get("include_hackernews", True),
        include_reddit=cfg.get("include_reddit", True),
        extra_query=cfg.get("newsapi_query") or None,
    )
    industries = cfg.get("industries", [])
    if industries:
        # Soft filter: keep all but reorder so industry-relevant float to top
        focused = sources.filter_by_industry(articles, industries)
        focused_urls = {a.url for a in focused}
        rest = [a for a in articles if a.url not in focused_urls]
        articles = focused + rest

    article_dicts = [a.to_dict() for a in articles]
    storage.cache_articles(article_dicts)

    briefing = analyze(
        articles=article_dicts,
        industries=industries,
        model=cfg.get("model"),
        max_articles=cfg.get("max_articles", 80),
    )
    storage.save_briefing(briefing, industries)

    email_cfg = cfg.get("email", {})
    if email_cfg.get("enabled") and email_cfg.get("to") and email_cfg.get("smtp_user"):
        try:
            send_email(
                subject=f"AI Briefing — {datetime.utcnow().date().isoformat()}",
                html_body=quick_email_summary(briefing),
                to=email_cfg["to"],
                sender=email_cfg.get("from", ""),
                smtp_host=email_cfg["smtp_host"],
                smtp_port=int(email_cfg["smtp_port"]),
                smtp_user=email_cfg["smtp_user"],
                smtp_password=email_cfg["smtp_password"],
            )
            briefing.setdefault("_meta", {})["emailed"] = True
        except Exception as e:
            briefing.setdefault("_meta", {})["email_error"] = str(e)
    return briefing


if __name__ == "__main__":
    out = run_daily()
    print(f"Generated briefing — {out.get('headline','(no headline)')}")
