"""Headless daily run for GitHub Actions / cron.

Reads SMTP config from environment variables (AIMR_*) so secrets stay out of
the repo, runs the pipeline, and emails the briefing.
"""
from __future__ import annotations

import os
import sys

import storage
from pipeline import run_daily


def main() -> int:
    cfg = storage.load_config()

    email_to = os.getenv("AIMR_EMAIL_TO")
    if email_to:
        cfg["email"] = {
            "enabled": True,
            "to": email_to,
            "from": os.getenv("AIMR_EMAIL_FROM", ""),
            "smtp_host": os.getenv("AIMR_SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("AIMR_SMTP_PORT", "587")),
            "smtp_user": os.getenv("AIMR_SMTP_USER", ""),
            "smtp_password": os.getenv("AIMR_SMTP_PASSWORD", ""),
        }

    briefing = run_daily(cfg)
    meta = briefing.get("_meta", {})
    print(f"OK: {briefing.get('headline','(no headline)')}")
    print(f"   model={meta.get('model')} articles={meta.get('article_count')}")
    if meta.get("email_error"):
        print(f"   email_error={meta['email_error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
