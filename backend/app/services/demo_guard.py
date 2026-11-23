"""Demo protections: a per-IP rate limit and a daily budget cap (docs/SERVING.md, "Demo protections").

Both protect the endpoints that can spend money:

| Endpoint | Over the per-IP limit | Daily budget exhausted |
|---|---|---|
| ``POST /api/profiles/{id}/chat`` | HTTP 429 with ``Retry-After`` | keyword extractor (no LLM) |
| ``GET /api/routes/recommendations`` | served by the no-LLM path | served by the no-LLM path |
| narratives (welcome text, post-route summary) | template text | template text |

The recommendations page therefore never fails because of a limit; the response says which path served
it. Everything is in memory and per process (a restart resets the day's counter; each LLM client keeps
its own per-process hard stop as well). Settings: ``RATE_LIMIT_*`` and ``DAILY_BUDGET_USD``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any, Optional

from fastapi import HTTPException, Request

from app.logger import get_logger
from app.settings import Settings, get_settings

logger = get_logger(__name__)

BUCKETS = ("chat", "recommendations", "narrative")
MAX_KEYS = 20_000  # bound on remembered (bucket, ip) pairs; least recently used are dropped


class RateLimiter:
    """Sliding-window log per ``(bucket, key)``: at most ``limit`` hits in any ``window_s`` seconds."""

    def __init__(self, limits: dict[str, int], window_s: float, *, clock: Callable[[], float] = time.monotonic,
                 enabled: bool = True) -> None:
        self.limits = dict(limits)
        self.window_s = float(window_s)
        self.enabled = enabled
        self._clock = clock
        self._hits: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, bucket: str, key: str) -> tuple[bool, float]:
        """Record one request. Returns ``(allowed, retry_after_s)``; a rejected request is not recorded."""
        if not self.enabled:
            return True, 0.0
        limit = self.limits.get(bucket)
        if limit is None or limit <= 0:
            return True, 0.0
        now = self._clock()
        with self._lock:
            q = self._hits.get((bucket, key))
            if q is None:
                q = deque()
                self._hits[(bucket, key)] = q
            self._hits.move_to_end((bucket, key))
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= limit:
                return False, max(q[0] + self.window_s - now, 0.0)
            q.append(now)
            while len(self._hits) > MAX_KEYS:
                self._hits.popitem(last=False)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def _live_clients() -> list[Any]:
    """Every LLM client of the process that can spend: narratives, chat, serving."""
    import app.chat.llm as chat_llm
    import app.llm.client as llm_client

    out = []
    for c in (getattr(llm_client, "_client", None), getattr(chat_llm, "_client", None)):
        if c is not None:
            out.append(c)
    from app.services import recsys_service

    c = recsys_service.serving_client_if_built()
    if c is not None:
        out.append(c)
    return out


class DailyBudget:
    """Real USD spent today (UTC) by the process' LLM clients, against ``cap_usd`` (0 = no cap).

    Spend is read from the clients' own running totals; at the first check of a new UTC day the current
    total becomes the day's baseline. ``extra_usd`` lets the caller account for spend outside the clients.
    """

    def __init__(self, cap_usd: float, *, clients: Callable[[], Iterable[Any]] = _live_clients,
                 today: Callable[[], str] = lambda: datetime.now(UTC).date().isoformat()) -> None:
        self.cap_usd = float(cap_usd)
        self._clients = clients
        self._today = today
        self._day: Optional[str] = None
        self._baseline = 0.0
        self._seen: dict[int, float] = {}
        self._lock = threading.Lock()

    def _total(self) -> float:
        # A client replaced at runtime (tests, set_client) restarts at 0: count spend per client object
        total = 0.0
        for c in self._clients():
            self._seen[id(c)] = max(self._seen.get(id(c), 0.0), float(getattr(c, "spent_usd", 0.0)))
        for v in self._seen.values():
            total += v
        return total

    def spent_today(self) -> float:
        with self._lock:
            total = self._total()
            day = self._today()
            if day != self._day:
                self._day, self._baseline = day, total
            return max(total - self._baseline, 0.0)

    def exhausted(self) -> bool:
        if self.cap_usd <= 0:
            return False
        spent = self.spent_today()
        if spent >= self.cap_usd:
            logger.warning("daily budget exhausted: %.4f of %.2f USD; serving no-LLM paths", spent, self.cap_usd)
            return True
        return False

    def status(self) -> dict[str, Any]:
        spent = self.spent_today()
        return {"cap_usd": self.cap_usd, "spent_today_usd": round(spent, 6),
                "exhausted": self.cap_usd > 0 and spent >= self.cap_usd, "day": self._day}


class DemoGuard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.limiter = RateLimiter(
            {
                "chat": settings.rate_limit_chat,
                "recommendations": settings.rate_limit_recommendations,
                "narrative": settings.rate_limit_narrative,
            },
            settings.rate_limit_window_s,
            enabled=settings.rate_limit_enabled,
        )
        self.budget = DailyBudget(settings.daily_budget_usd)

    def client_ip(self, request: Optional[Request]) -> str:
        if request is None:
            return "unknown"
        if self.settings.rate_limit_trust_forwarded:
            fwd = request.headers.get("x-forwarded-for")
            if fwd:
                return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def hit(self, request: Optional[Request], bucket: str) -> tuple[bool, float]:
        ok, retry = self.limiter.hit(bucket, self.client_ip(request))
        if not ok:
            logger.warning("rate limit: bucket=%s ip=%s retry_after=%.1fs", bucket, self.client_ip(request), retry)
        return ok, retry


_guard: Optional[DemoGuard] = None
_guard_lock = threading.Lock()


def get_guard() -> DemoGuard:
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = DemoGuard(get_settings())
        return _guard


def set_guard(guard: Optional[DemoGuard]) -> None:
    """Install a guard (tests) or drop the current one (None: rebuilt from settings on next use)."""
    global _guard
    with _guard_lock:
        _guard = guard


def enforce_chat_limit(request: Request) -> None:
    """FastAPI dependency for the chat turn: 429 over the per-IP limit."""
    ok, retry = get_guard().hit(request, "chat")
    if not ok:
        raise HTTPException(
            status_code=429,
            detail="Too many chat messages; please wait a moment.",
            headers={"Retry-After": str(max(1, math.ceil(retry)))},
        )


def llm_narrative_allowed(request: Optional[Request]) -> bool:
    """False if this narrative should use its template text (per-IP limit or daily budget)."""
    g = get_guard()
    if g.budget.exhausted():
        return False
    ok, _ = g.hit(request, "narrative")
    return ok


def chat_llm_allowed() -> bool:
    """False once the daily budget is spent: the chat then uses the keyword extractor."""
    return not get_guard().budget.exhausted()
