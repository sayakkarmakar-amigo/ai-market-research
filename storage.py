"""Persistent state: SQLite cache for briefings + JSON config for user prefs."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "briefings.db"
CONFIG_PATH = DATA_DIR / "config.json"


# ---- Config (JSON) --------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "delivery_time": "07:30",  # 24h, local time
    "timezone": "Asia/Kolkata",
    "industries": [],  # empty = sector-agnostic
    "enabled_categories": ["press", "corp_research", "business", "research"],
    "include_hackernews": True,
    "include_reddit": True,
    "newsapi_query": "",
    "model": "gemini-2.5-flash",
    "max_articles": 80,
    "email": {
        "enabled": False,
        "to": "",
        "from": "",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_password": "",  # use an app password
    },
    "auto_run_in_app": True,
}


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        with CONFIG_PATH.open() as f:
            cfg = json.load(f)
    except json.JSONDecodeError:
        return dict(DEFAULT_CONFIG)
    # Backfill any new keys
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    if "email" in cfg:
        email = dict(DEFAULT_CONFIG["email"])
        email.update(cfg["email"])
        merged["email"] = email
    return merged


def save_config(cfg: dict[str, Any]) -> None:
    with CONFIG_PATH.open("w") as f:
        json.dump(cfg, f, indent=2)


# ---- Briefings DB ---------------------------------------------------------
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS briefings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            industries TEXT,
            payload TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            published TEXT,
            payload TEXT,
            fetched_at TEXT
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_briefings_date ON briefings(date)")
    return conn


def save_briefing(briefing: dict[str, Any], industries: list[str] | None) -> int:
    today = datetime.utcnow().date().isoformat()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO briefings(date, generated_at, industries, payload) VALUES (?, ?, ?, ?)",
            (
                today,
                briefing.get("_meta", {}).get("generated_at", datetime.utcnow().isoformat()),
                json.dumps(industries or []),
                json.dumps(briefing),
            ),
        )
        return cur.lastrowid


def latest_briefing() -> dict[str, Any] | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT payload FROM briefings ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def briefing_for_date(date: str) -> dict[str, Any] | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT payload FROM briefings WHERE date = ? ORDER BY id DESC LIMIT 1",
            (date,),
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def list_dates(limit: int = 30) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM briefings ORDER BY date DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [r["date"] for r in rows]


def cache_articles(articles: list[dict]) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO articles(url, title, source, published, payload, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    a["url"],
                    a.get("title", ""),
                    a.get("source", ""),
                    a.get("published", ""),
                    json.dumps(a),
                    now,
                )
                for a in articles
            ],
        )
