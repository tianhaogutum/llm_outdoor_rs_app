"""Serving the recommender pipeline behind ``GET /api/routes/recommendations`` (docs/SERVING.md).

- **Resources** (:func:`load_resources`, called from the app's lifespan): catalogue, BM25 index, document
  embedding matrix, learned-ranker trees and the pipelines are built once per process and shared by every
  request (a :class:`~app.recsys.pipeline.Pipeline` holds no per-request state).
- **Query** (:func:`build_user_query`): the stored profile becomes a :class:`~app.recsys.query.UserQuery`:
  the questionnaire vector (never the feedback-rewritten ``user_vector_json``) plus the chat's slots,
  the category tab, home / radius / distance cap, the stored intent, and the feedback history
  (negative feedback rows and completions with their real ages).
- **Paths** (:func:`recommend`): ``pipeline`` (the configured system), ``no_llm`` (same filters, BM25
  candidates, no embedding call) when the provider fails, times out, the budget is spent or the per-IP
  limit is reached; ``baseline`` (``v0_rules``) when configured or when the DB and the catalogue disagree;
  ``random`` for guests. The response says which path served it and why.
- **Explanations** (:func:`route_explanation`, :func:`reason_sentence`): per-stage contributions from the
  trace plus a one-line reason built only from those facts and the catalogue record (no LLM).
"""

from __future__ import annotations

import asyncio
import json
import resource
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.logger import get_logger
from app.models.entities import Breakpoint, DemoProfile, ProfileFeedback, Route, Souvenir
from app.settings import Settings, get_settings

logger = get_logger(__name__)

BASELINE = "v0_rules"
NO_LLM_SYSTEM = "v2_bm25"  # filters + BM25F ranked by BM25; without an intent: filters + rule scorer
TABS = ("hiking", "cycling", "running")
NEGATIVE_REASONS = ("too-hard", "too-easy", "too-far", "not-interested")
# Interim serving rule until the pipeline's feedback stage exists (docs/SERVING.md "Feedback"): a route
# the user rejected or completed is hidden from the list for this many days (two 30-day half-lives of the
# baseline's decay, i.e. while the feedback still weighs at least a quarter).
HIDE_WINDOW_DAYS = 60.0
CHAT_VECTOR_SLOTS = ("activity", "difficulty", "distance_band", "duration_hours", "distance_cap_km",
                     "tags_preferred", "tags_avoided")


# ------------------------------------------------------------------------------------- resources
def _rss_mb() -> float:
    """Peak resident set size of the process in MB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


@dataclass
class ServingResources:
    configured: str
    catalog: Any = None
    pipeline: Any = None  # the configured system (None for v0_rules or after a load failure)
    no_llm: Any = None  # the no-LLM path
    guest: Any = None  # random over the tab, seeded per request
    error: Optional[str] = None
    startup: dict[str, Any] = field(default_factory=dict)
    db_checks: dict[str, dict[str, Any]] = field(default_factory=dict)  # DB url -> consistency check

    @property
    def ok(self) -> bool:
        return self.pipeline is not None and self.error is None


_serving_client = None
_resources: dict[tuple, ServingResources] = {}
_lock = threading.Lock()


def serving_client_if_built():
    return _serving_client


def serving_client(settings: Optional[Settings] = None):
    """LLM client for serving-time embedding / rank calls: own budget, few retries, short timeout."""
    global _serving_client
    with _lock:
        if _serving_client is None:
            from app.llm import LLMClient, LLMConfig

            s = settings or get_settings()
            cfg = LLMConfig.from_settings(s).with_(
                budget_usd=s.recsys_budget_usd,
                max_retries=max(0, s.recsys_embed_retries),
                retry_base_delay_s=0.2,
                retry_max_delay_s=1.0,
                timeout_s=max(0.5, s.recsys_timeout_s),
            )
            _serving_client = LLMClient(cfg)
        return _serving_client


def set_serving_client(client) -> None:
    """Install a serving client (tests). Drops cached resources so pipelines are rebuilt with it."""
    global _serving_client
    with _lock:
        _serving_client = client
        _resources.clear()


def reset_resources() -> None:
    with _lock:
        _resources.clear()


def _key(s: Settings) -> tuple:
    return (s.recsys_system, str(s.recsys_catalog_dir or ""), s.llm_provider, s.resolved_embed_provider,
            s.resolved_rank_provider, id(_serving_client))


def load_resources(settings: Optional[Settings] = None) -> ServingResources:
    """Build (once per process and configuration) everything the recommendations endpoint needs."""
    s = settings or get_settings()
    client = serving_client(s)
    key = _key(s)
    with _lock:
        if key in _resources:
            return _resources[key]
    res = ServingResources(configured=s.recsys_system)
    t_all = time.perf_counter()
    rss0 = _rss_mb()
    timings: dict[str, float] = {}
    try:
        from app.recsys import build_system, get_catalog, list_systems

        t = time.perf_counter()
        res.catalog = get_catalog(s.recsys_catalog_dir)
        timings["catalogue_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        res.guest = build_system("random", res.catalog)
        res.no_llm = build_system(NO_LLM_SYSTEM, res.catalog, llm=client)  # BM25 index
        timings["bm25_index_ms"] = (time.perf_counter() - t) * 1000
        if s.recsys_system != BASELINE:
            if s.recsys_system not in list_systems() or s.recsys_system == "random":
                raise ValueError(f"RECSYS_SYSTEM={s.recsys_system!r} is not a servable system")
            t = time.perf_counter()
            res.pipeline = build_system(s.recsys_system, res.catalog, llm=client)  # ranker trees
            timings["pipeline_ms"] = (time.perf_counter() - t) * 1000
            if res.pipeline.needs_embeddings and res.pipeline._provider("embed") != "mock":
                t = time.perf_counter()
                res.pipeline.doc_embeddings()  # loads the committed matrix; fails loudly on a model mismatch
                timings["doc_embeddings_ms"] = (time.perf_counter() - t) * 1000
    except Exception as e:  # never fail start-up: the baseline still serves
        res.error = f"{type(e).__name__}: {e}"
        res.pipeline = None
        logger.error("recsys: could not load %s (%s); serving the baseline v0_rules", s.recsys_system, res.error)
    timings["total_ms"] = (time.perf_counter() - t_all) * 1000
    res.startup = {
        **{k: round(v, 1) for k, v in timings.items()},
        "peak_rss_mb_before": round(rss0, 1),
        "peak_rss_mb_after": round(_rss_mb(), 1),
        "catalogue_routes": len(res.catalog) if res.catalog is not None else 0,
        "embed_provider": s.resolved_embed_provider,
    }
    with _lock:
        _resources[key] = res
    return res


async def warm_up(res: ServingResources) -> None:
    """One intent-free request per pipeline (no provider call): builds lazy structures (mock document
    matrix in tests, tab memos) before the first user request. Timed into ``res.startup``."""
    from app.recsys import UserQuery

    if res.catalog is None:
        return
    t = time.perf_counter()
    q = UserQuery("warm-up", {}, "hiking")
    for pipe in (res.pipeline, res.no_llm):
        if pipe is None:
            continue
        try:
            await pipe.arecommend(q, 5)
        except Exception as e:  # pragma: no cover - logged, the request path has its own fallbacks
            logger.warning("recsys warm-up of %s failed: %s", getattr(pipe, "name", "?"), e)
    if res.pipeline is not None and res.pipeline.needs_embeddings and res.pipeline._provider("embed") == "mock":
        await res.pipeline._adoc_embeddings()
    res.startup["warm_up_ms"] = round((time.perf_counter() - t) * 1000, 1)
    res.startup["peak_rss_mb_after"] = round(_rss_mb(), 1)


async def check_db(db: AsyncSession, res: ServingResources) -> dict[str, Any]:
    """Does the DB hold exactly the catalogue's route ids? Cached per database URL."""
    url = str(db.get_bind().url)
    if url in res.db_checks:
        return res.db_checks[url]
    ids = set((await db.execute(select(Route.id))).scalars().all())
    cat_ids = {int(i) for i in res.catalog.ids} if res.catalog is not None else set()
    missing_in_db = len(cat_ids - ids)
    extra_in_db = len(ids - cat_ids)
    check = {"ok": bool(ids) and not missing_in_db and not extra_in_db, "db_routes": len(ids),
             "catalogue_routes": len(cat_ids), "missing_in_db": missing_in_db, "extra_in_db": extra_in_db}
    if not check["ok"]:
        logger.warning(
            "recsys: DB routes do not match the in-memory catalogue (db=%d, catalogue=%d, missing in DB=%d, "
            "not in catalogue=%d); serving the baseline v0_rules. Reseed with `make seed`.",
            len(ids), len(cat_ids), missing_in_db, extra_in_db)
    else:
        logger.info("recsys: DB matches the catalogue (%d routes)", len(ids))
    res.db_checks[url] = check
    return check


# ----------------------------------------------------------------------------------------- query
def _loads(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    try:
        v = json.loads(text)
    except (TypeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def pipeline_vector(profile: DemoProfile) -> tuple[dict[str, Any], str]:
    """The user vector the pipeline starts from, and where it came from.

    The questionnaire vector as created (``questionnaire_vector_json``; never rewritten by feedback) with
    the chat's vector slots (difficulty, distance band / duration / cap, wanted and avoided tags) projected
    onto it by the chat's own rules. Feedback is *not* in it: it travels as events in ``UserQuery.feedback``.
    """
    from app.chat.projection import project_vector
    from app.chat.schema import ChatSlots, SlotChange
    from app.services.chat_service import _state

    base = _loads(profile.questionnaire_vector_json)
    source = "questionnaire"
    if base is None:
        base = _loads(profile.user_vector_json)  # rows without the column: as stored (may carry feedback)
        source = "user_vector" if base is not None else "none"
        base = base or {}
    elif base.pop("_reconstructed", False):
        source = "reconstructed"
    state = _state(profile)
    chat_set = [s for s in CHAT_VECTOR_SLOTS if s in state.sources]
    if not chat_set:
        return base, source
    activity = profile.activity if profile.activity in ("hiking", "cycling", "mtb", "running") else None
    new = ChatSlots(activity=activity, difficulty=state.difficulty, distance_band=state.distance_band,
                    duration_hours=state.duration_hours, distance_cap_km=profile.distance_cap_km,
                    tags_preferred=list(state.tags_preferred), tags_avoided=list(state.tags_avoided))
    changes = [SlotChange(slot=s, new=getattr(new, s)) for s in chat_set]
    return project_vector(base, base, ChatSlots(), new, changes), f"{source}+chat"


def resolve_category(category: Optional[str], profile: Optional[DemoProfile]) -> str:
    """The tab for the hard filter: the requested tab; else the profile's activity; else every category."""
    if category in TABS:
        return category
    from app.chat.vocab import ACTIVITY_TAB

    if profile is not None and profile.activity in ACTIVITY_TAB:
        return ACTIVITY_TAB[profile.activity]
    return ""


def _age_days(moment: Optional[datetime], now: datetime) -> float:
    from app.services.recommendation_service import feedback_age_days

    return max(0.0, round(feedback_age_days(moment, now), 4))


async def feedback_events(db: AsyncSession, profile_id: int, now: Optional[datetime] = None) -> tuple:
    """Negative feedback rows and completions (souvenirs) as pipeline feedback events with real ages,
    oldest first. Pass them as ``UserQuery.feedback``."""
    from app.recsys import FeedbackEvent

    now = now or datetime.now(UTC)
    rows = (await db.execute(
        select(ProfileFeedback.route_id, ProfileFeedback.reason, ProfileFeedback.created_at)
        .where(ProfileFeedback.demo_profile_id == profile_id)
    )).all()
    done = (await db.execute(
        select(Souvenir.route_id, Souvenir.completed_at).where(Souvenir.demo_profile_id == profile_id)
    )).all()
    events = [FeedbackEvent(int(r), str(reason), _age_days(at, now)) for r, reason, at in rows]
    events += [FeedbackEvent(int(r), "completed", _age_days(at, now)) for r, at in done]
    events.sort(key=lambda e: -e.age_days)
    return tuple(events)


def build_user_query(profile: DemoProfile, *, category: Optional[str], feedback: tuple = (),
                     request_id: Optional[str] = None):
    """The ``UserQuery`` for a stored profile. Missing home -> no radius filter; missing intent -> the
    pipeline's no-intent path (filters + rule scorer + MMR)."""
    from app.recsys import UserQuery

    vector, source = pipeline_vector(profile)
    home = None
    if profile.home_lat is not None and profile.home_lon is not None:
        home = (float(profile.home_lat), float(profile.home_lon))
    q = UserQuery(
        request_id=request_id or f"p{profile.id}-{uuid.uuid4().hex[:12]}",
        user_vector=vector,
        category=resolve_category(category, profile),
        home=home,
        max_travel_km=float(profile.travel_radius_km) if (home and profile.travel_radius_km) else None,
        max_distance_km=float(profile.distance_cap_km) if profile.distance_cap_km else None,
        intent=(profile.intent_text or "").strip() or None,
        feedback=tuple(feedback),
    )
    return q, source


def hidden_route_ids(events: tuple, window_days: float = HIDE_WINDOW_DAYS) -> set[int]:
    return {e.route_id for e in events
            if (e.reason in NEGATIVE_REASONS or e.reason == "completed") and e.age_days < window_days}


# ---------------------------------------------------------------------------------- explanations
def _known_stage_fields() -> set[str]:
    from app.recsys.explain import StageContributions

    return set(StageContributions.model_fields)


def route_explanation(rec, rank: int, route_id: int, score: float, *, events: tuple, hidden: int,
                      n_candidates: Optional[int], n_eligible: Optional[int]) -> dict[str, Any]:
    """The typed per-route explanation (``RouteExplanationResponse``) as a plain dict."""
    contribs = rec.contributions(route_id)
    known = _known_stage_fields()
    stages = {k: v for k, v in contribs.items() if k in known}
    extra = {k: v for k, v in contribs.items() if k not in known}
    # Phase 5 stages (feedback re-scoring, exploration slot): part of the schema once the pipeline has them
    explore = contribs.get("explore") or contribs.get("exploration")
    fb_stage = contribs.get("feedback")
    rule = stages.get("v0_rules") or {}
    on_route = [{"reason": e.reason, "age_days": round(e.age_days, 2)} for e in events if e.route_id == route_id]
    return {
        "schema_version": "recsys.explanation/v1",
        "rank": rank,
        "score": float(score),
        "stages": stages,
        "feedback": {
            "events_considered": len(events),
            "on_this_route": on_route,
            "rule_penalty": rule.get("feedback_penalty"),
            "hidden_routes": hidden,
            "window_days": HIDE_WINDOW_DAYS,
            "stage": fb_stage if isinstance(fb_stage, dict) else None,
        },
        "other_stages": extra or None,
        "exploration": {
            "is_exploration": bool(explore),
            "detail": explore if isinstance(explore, dict) else None,
        },
        "n_candidates": n_candidates,
        "n_eligible": n_eligible,
    }


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _km(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def display_term(term: str, intent: Optional[str]) -> str:
    """A matched BM25 term as the user wrote it: stems ("hik") map back to the intent's word ("hiking");
    expansion terms not in the intent stay as they are."""
    import re

    for word in re.findall(r"[^\W_]+", (intent or "").lower()):
        if word == term:
            return term
    for word in re.findall(r"[^\W_]+", (intent or "").lower()):
        if word.startswith(term):
            return word
    return term


def reason_sentence(route: Any, expl: dict[str, Any], home_name: Optional[str],
                    intent: Optional[str] = None) -> str:
    """One grounded line per route: every clause is read from the explanation or the catalogue record."""
    st = expl.get("stages") or {}
    parts: list[str] = []
    length = (route.length_meters or 0) / 1000.0
    shape = "loop" if getattr(route, "is_loop", None) else "route"
    head = f"{_km(length)} km {shape}"
    flt = st.get("filter") or {}
    rad = flt.get("radius") or {}
    if rad.get("applied") and rad.get("distance_km") is not None:
        where = f" from {home_name}" if home_name else " from home"
        if rad.get("ok"):
            head += f" starting {_km(rad['distance_km'])} km{where}, inside your {_km(rad['limit_km'])} km radius"
        elif rad.get("relaxed"):
            head += (f" starting {_km(rad['distance_km'])} km{where}, beyond your {_km(rad['limit_km'])} km radius "
                     f"(too few routes closer)")
    elif rad.get("distance_km") is not None and home_name:
        head += f" starting {_km(rad['distance_km'])} km from {home_name}"
    parts.append(head)
    cap = flt.get("distance_cap") or {}
    if cap.get("applied") and cap.get("ok") and cap.get("limit_km"):
        parts.append(f"within your {_km(cap['limit_km'])} km limit")
    terms = list(dict.fromkeys(display_term(t["term"], intent)
                               for t in ((st.get("bm25") or {}).get("matched_terms") or [])))[:3]
    if terms:
        parts.append("matches " + ", ".join(terms))
    if expl.get("exploration", {}).get("is_exploration"):
        parts.append("an exploration pick, outside your usual matches")
    else:
        ltr = st.get("ltr") or st.get("rank_api")
        if ltr and expl.get("n_candidates"):
            parts.append(f"ranked {_ordinal(int(ltr['rank']))} of {expl['n_candidates']} candidates")
        elif st.get("v0_rules"):
            r = st["v0_rules"]  # weighted sub-scores: 0.4 / 0.3 = a full match on that component
            fits = [name for name, key, full in (("difficulty", "difficulty", 0.4), ("length", "distance", 0.3))
                    if abs(float(r.get(key, 0.0)) - full) < 1e-6]
            if fits:
                parts.append("fits your " + " and ".join(fits) + " preference")
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


def baseline_reason(route: Any, breakdown: Optional[dict]) -> Optional[str]:
    """One grounded line for the rule baseline, from its score breakdown."""
    if not breakdown:
        return None
    parts = []
    d = breakdown.get("difficulty") or {}
    if d.get("score") == 1.0:
        parts.append("difficulty within your range")
    dist = breakdown.get("distance") or {}
    if dist.get("score") == 1.0 and dist.get("user_range"):
        lo, hi = dist["user_range"]
        parts.append(f"{_km(dist.get('route_value', 0))} km is within your {_km(lo)}-{_km(hi)} km")
    tags = breakdown.get("tags") or {}
    user_tags = {t.lower() for t in tags.get("user_tags") or []}
    shared = sorted(user_tags & {t.lower() for t in tags.get("route_tags") or []})
    if shared:
        parts.append("shares " + ", ".join(shared[:3]))
    if breakdown.get("feedback_penalty") is not None:
        parts.append("pushed down by your feedback")
    if not parts:
        return None
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


# ----------------------------------------------------------------------------------------- serve
@dataclass
class Served:
    routes: list[Route]
    extras: dict[int, dict[str, Any]]  # route id -> {explanation, reason, is_exploration, score, breakdown}
    served_by: dict[str, Any]
    is_personalized: bool


async def load_routes(db: AsyncSession, ids: list[int]) -> list[Route]:
    """ORM routes (with breakpoints and quests) in the given order; ids not in the DB are skipped."""
    if not ids:
        return []
    rows = (await db.execute(
        select(Route).where(Route.id.in_(ids))
        .options(selectinload(Route.breakpoints).selectinload(Breakpoint.mini_quests))
    )).scalars().all()
    by_id = {r.id: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def _baseline(db, profile_id, category, limit, *, configured: str, reason: str,
                    degraded: bool) -> Served:
    from app.services.recommendation_service import get_recommended_routes

    t = time.perf_counter()
    routes = await get_recommended_routes(db=db, profile_id=profile_id, category=category, limit=limit)
    extras = {}
    for r in routes:
        bd = getattr(r, "recommendation_score_breakdown", None)
        extras[r.id] = {"reason": baseline_reason(r, bd)}
    return Served(routes, extras, {
        "configured_system": configured, "system": BASELINE if profile_id is not None else "random_db",
        "path": "baseline" if profile_id is not None else "random", "reason": reason, "degraded": degraded,
        "latency_ms": round((time.perf_counter() - t) * 1000, 1),
    }, profile_id is not None)


def _stage_summaries(rec) -> list[dict[str, Any]]:
    return [{"stage": t.stage, "n_in": t.n_in, "n_out": t.n_out, "local_ms": round(t.local_ms, 3),
             "upstream_latency_ms": round(t.upstream_latency_ms, 3),
             "upstream_cost_usd": round(t.upstream_cost_usd, 9), "info": t.info} for t in rec.trace]


async def recommend(db: AsyncSession, *, res: Optional[ServingResources], profile: Optional[DemoProfile],
                    category: Optional[str], limit: int, request: Any = None,
                    settings: Optional[Settings] = None) -> Served:
    """Serve one recommendations request (see the module docstring for the paths)."""
    from app.llm import LLMError
    from app.services.demo_guard import get_guard

    s = settings or get_settings()
    configured = s.recsys_system
    pid = profile.id if profile is not None else None

    if configured == BASELINE:
        return await _baseline(db, pid, category, limit, configured=configured, reason="configured",
                               degraded=False)
    if res is None or res.catalog is None:
        return await _baseline(db, pid, category, limit, configured=configured, reason="pipeline_not_loaded",
                               degraded=True)
    check = await check_db(db, res)
    if not check["ok"]:
        return await _baseline(db, pid, category, limit, configured=configured, reason="catalogue_mismatch",
                               degraded=True)

    t0 = time.perf_counter()
    if profile is None:  # guest: uniform random over the tab (seeded per request)
        from app.recsys import UserQuery

        q = UserQuery(f"guest-{uuid.uuid4().hex[:12]}", {}, category if category in TABS else "")
        rec = res.guest.recommend(q, limit)
        routes = await load_routes(db, rec.route_ids)
        return Served(routes, {}, {
            "configured_system": configured, "system": "random", "path": "random", "reason": "guest",
            "degraded": False, "request_id": q.request_id,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }, False)

    if not res.ok:
        return await _baseline(db, pid, category, limit, configured=configured,
                               reason="pipeline_not_loaded", degraded=True)

    events = await feedback_events(db, profile.id)
    query, vector_source = build_user_query(profile, category=category, feedback=events)
    # Feedback: if the pipeline has its own feedback stage it applies the route effects itself; otherwise
    # the interim serving rule hides recently rejected or completed routes (docs/SERVING.md "Feedback").
    pipeline_feedback = getattr(res.pipeline.cfg, "feedback", None) is not None
    hidden = set() if pipeline_feedback else hidden_route_ids(events)
    k = limit + len(hidden)

    pipe, path, reason, degraded = res.pipeline, "pipeline", None, False
    if not query.intent:
        reason = "no_intent"  # the pipeline's own rule path (filters + v0 + MMR): no provider call
    elif res.pipeline.needs_embeddings or res.pipeline.uses_rank_api:
        guard = get_guard()
        if guard.budget.exhausted():
            pipe, path, reason, degraded = res.no_llm, "no_llm", "daily_budget", True
        else:
            ok, _ = guard.hit(request, "recommendations")
            if not ok:
                pipe, path, reason, degraded = res.no_llm, "no_llm", "rate_limited", True
    try:
        rec = await asyncio.wait_for(pipe.arecommend(query, k), timeout=s.recsys_timeout_s)
    except (TimeoutError, LLMError, ValueError) as e:
        kind = type(e).__name__
        reason = {"TimeoutError": "timeout", "BudgetExceeded": "serving_budget"}.get(kind, "provider_unavailable")
        logger.warning("recsys: %s failed for profile %s (%s); serving the no-LLM path", pipe.name, pid, kind)
        pipe, path, degraded = res.no_llm, "no_llm", True
        rec = await res.no_llm.arecommend(query, k)

    ids = [i for i in rec.route_ids if i not in hidden][:limit]
    routes = await load_routes(db, ids)
    if len(routes) != len(ids):  # the DB changed under us: never return ids it does not have
        res.db_checks.pop(str(db.get_bind().url), None)
        logger.warning("recsys: %d recommended ids missing in the DB; serving the baseline",
                       len(ids) - len(routes))
        return await _baseline(db, pid, category, limit, configured=configured, reason="catalogue_mismatch",
                               degraded=True)

    by_stage = {t.stage: t for t in rec.trace}
    if "feedback" in by_stage:  # the pipeline's own route effects
        ex = (by_stage["feedback"].info or {}).get("excluded")
        n_hidden = len(ex) if isinstance(ex, (dict, list, tuple, set)) else int(ex or 0)
    else:
        n_hidden = len(hidden)
    n_eligible = by_stage["filter"].n_out if "filter" in by_stage else None
    ranker = by_stage.get("ltr") or by_stage.get("rank_api") or by_stage.get("v0_rules")
    n_candidates = ranker.n_in if ranker is not None else None
    score_of = dict(zip(rec.route_ids, rec.scores, strict=True))
    extras: dict[int, dict[str, Any]] = {}
    for rank, r in enumerate(routes, start=1):
        expl = route_explanation(rec, rank, r.id, score_of[r.id], events=events, hidden=n_hidden,
                                 n_candidates=n_candidates, n_eligible=n_eligible)
        extras[r.id] = {
            "explanation": expl,
            "reason": reason_sentence(r, expl, profile.home_name, query.intent),
            "is_exploration": expl["exploration"]["is_exploration"],
        }
    fallback = (by_stage.get("filter").info or {}).get("fallback") if "filter" in by_stage else None
    served_by = {
        "configured_system": configured,
        "system": rec.system,
        "path": path,
        "reason": reason or fallback,
        "degraded": degraded,
        "request_id": query.request_id,
        "explanation_schema": "recsys.explanation/v1",
        "query": {
            "category": query.category,
            "home_name": profile.home_name if query.home else None,
            "home": list(query.home) if query.home else None,
            "travel_radius_km": query.max_travel_km,
            "distance_cap_km": query.max_distance_km,
            "intent": query.intent,
            "vector_source": vector_source,
            "feedback_events": len(events),
            "suppressed_route_ids": sorted(hidden),
            "feedback_handled_by": "pipeline" if pipeline_feedback else "serving_interim_rule",
        },
        "stages": _stage_summaries(rec),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "upstream_cost_usd": round(rec.upstream_cost_usd, 9),
    }
    logger.info("recsys: profile=%s system=%s path=%s reason=%s n=%d %.0f ms", pid, rec.system, path,
                served_by["reason"], len(routes), served_by["latency_ms"])
    return Served(routes, extras, served_by, True)


def startup_report(res: Optional[ServingResources]) -> dict[str, Any]:
    if res is None:
        return {}
    # DB checks by file name only (the status endpoint is public; no local paths)
    checks = {url.rsplit("/", 1)[-1]: c for url, c in res.db_checks.items()}
    return {"configured": res.configured, "ok": res.ok, "error": res.error, **res.startup, "db_checks": checks}


__all__ = [
    "Served", "ServingResources", "build_user_query", "check_db", "feedback_events", "hidden_route_ids",
    "load_resources", "pipeline_vector", "reason_sentence", "recommend", "route_explanation",
    "serving_client", "set_serving_client", "startup_report", "warm_up",
]

