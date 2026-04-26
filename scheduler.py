"""In-process daily scheduler. Started once per Streamlit session via cache_resource."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import storage
from pipeline import run_daily


class DailyScheduler:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_run: datetime | None = None
        self.last_error: str | None = None
        self.next_run: datetime | None = None
        self._lock = threading.Lock()

    def _next_fire(self, hhmm: str) -> datetime:
        h, m = (int(x) for x in hhmm.split(":"))
        now = datetime.now()
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    def _loop(self):
        while not self._stop.is_set():
            cfg = storage.load_config()
            self.next_run = self._next_fire(cfg.get("delivery_time", "07:30"))
            sleep_secs = max(30.0, (self.next_run - datetime.now()).total_seconds())
            # Wake up at most every 5 min to honor config changes
            self._stop.wait(min(sleep_secs, 300))
            if self._stop.is_set():
                return
            if datetime.now() >= self.next_run:
                try:
                    with self._lock:
                        run_daily(cfg)
                        self.last_run = datetime.now()
                        self.last_error = None
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {e}"

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ai-research-daily")
        self._thread.start()

    def stop(self):
        self._stop.set()


def run_now_async() -> threading.Thread:
    """Kick off a one-shot run immediately in the background."""
    t = threading.Thread(target=run_daily, daemon=True, name="ai-research-oneshot")
    t.start()
    return t
