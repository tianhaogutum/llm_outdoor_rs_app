"""Serving the recommender (docs/SERVING.md): system selection, paths and fallbacks, catalogue/DB check,
explanations, feedback events, the original vector, rate limit and daily budget, static frontend.

Offline: LLM_PROVIDER=mock (tests/conftest.py); the serving client is a mock client installed per test.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.api.schemas import RouteExplanationResponse
from app.llm import LLMClient, LLMConfig
from app.recsys import get_catalog, haversine_km
from app.services import demo_guard, recsys_service
from app.services.demo_guard import DailyBudget, RateLimiter
from app.settings import get_settings
from tests.conftest import run_alembic

ANSWERS = {"fitness": "intermediate", "type": ["natural-scenery"], "narrative": "mystery"}
HOME = {"name": "Grainau"}
REC = "/api/routes/recommendations"


@pytest.fixture
def serving(monkeypatch, make_client, full_osm_db, tmp_path):
    """``serving(**env)`` -> TestClient on a copy of the full OSM DB with the given settings."""

    def configure(db: Path | None = None, client: LLMClient | None = None, **env):
        for k, v in env.items():
            monkeypatch.setenv(k.upper(), str(v))
        get_settings.cache_clear()
        demo_guard.set_guard(None)
        recsys_service.set_serving_client(client or LLMClient(LLMConfig(max_retries=0)))
        if db is None:
            db = tmp_path / f"serve_{len(list(tmp_path.iterdir()))}.db"
            shutil.copy(full_osm_db, db)
        api = make_client(db)
        api.db_path = db
        return api

    yield configure
    monkeypatch.undo()
    get_settings.cache_clear()
    demo_guard.set_guard(None)
    recsys_service.set_serving_client(None)


def new_profile(api, **slots) -> int:
    resp = api.post("/api/profiles", json={**ANSWERS, **slots})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def recs(api, pid=None, **params):
    resp = api.get(REC, params={**({"profile_id": pid} if pid else {}), "limit": 10, **params})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------------------------- contract
def test_default_system_serves_the_pipeline_with_explanations(serving):
    api = serving()
    pid = new_profile(api, home=HOME, travel_radius_km=40, distance_cap_km=15, intent="easy lake loop with a hut")
    body = recs(api, pid, category="hiking")
    sb = body["served_by"]
    assert sb["configured_system"] == sb["system"] == "recsys_serve"
    assert sb["path"] == "pipeline" and sb["degraded"] is False and sb["reason"] is None
    assert sb["explanation_schema"] == "recsys.explanation/v1.1"
    assert sb["query"]["home_name"] == "Grainau" and sb["query"]["intent"] == "easy lake loop with a hut"
    assert sb["query"]["vector_source"] == "questionnaire"
    assert [s["stage"] for s in sb["stages"]][:5] == ["filter", "bm25", "dense", "fusion", "ltr"]
    assert body["total"] == 10
    cat = get_catalog()
    home = sb["query"]["home"]
    for rank, route in enumerate(body["routes"], start=1):
        expl = RouteExplanationResponse.model_validate(route["explanation"])  # the typed schema
        st = expl.stages
        assert expl.rank == rank and route["recommendation_score"] is None and route["reason"]
        assert {"bm25", "dense", "fusion", "ltr", "mmr"} <= {k for k, v in st if v is not None}
        # the filter's numbers are the real ones
        rec = cat.get(route["id"])
        assert st.filter.radius.applied and st.filter.radius.limit_km == 40.0
        assert st.filter.radius.distance_km == pytest.approx(
            haversine_km(home[0], home[1], rec.start_lat, rec.start_lon), abs=0.01)
        assert st.filter.distance_cap.length_km == pytest.approx(rec.length_km, abs=0.01)
        assert st.filter.distance_cap.length_km <= 15.0 and st.filter.distance_cap.ok
        assert route["category_name"] in ("Hiking trail", "Theme trail")
        # LTR contributions sum to its score (Saabas)
        assert st.ltr.bias + sum(f.contribution for f in st.ltr.features) == pytest.approx(st.ltr.score, abs=1e-3)
        assert expl.n_candidates == st.mmr.rank_before or expl.n_candidates >= st.mmr.rank_before
        # the reason sentence quotes the explanation's facts
        assert "Grainau" in route["reason"] and "inside your 40 km radius" in route["reason"]
        if st.explore is not None:
            # the served default reserves rank 10 for an exploration pick, and says so
            assert rank == 10 and "exploration pick" in route["reason"]
        else:
            assert f"ranked {_ordinal(st.ltr.rank)} of {expl.n_candidates} candidates" in route["reason"]
    # MMR lambda is serialised under its alias
    assert body["routes"][0]["explanation"]["stages"]["mmr"]["lambda"] == 0.8


def _ordinal(n):
    return recsys_service._ordinal(n)


def test_v0_rules_setting_serves_the_unchanged_baseline(serving):
    api = serving(recsys_system="v0_rules")
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    body = recs(api, pid, category="hiking")
    sb = body["served_by"]
    assert sb["system"] == "v0_rules" and sb["path"] == "baseline" and sb["reason"] == "configured"
    for route in body["routes"]:
        assert route["explanation"] is None
        assert route["recommendation_score_breakdown"]["total"] == route["recommendation_score"]
    assert any(r["reason"] for r in body["routes"])  # grounded baseline line from the breakdown


def test_other_registered_pipeline_is_selectable(serving):
    api = serving(recsys_system="v7_mmr")  # the variant that calls the ranker API (mock here)
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="summit with a view")
    body = recs(api, pid, category="hiking")
    assert body["served_by"]["system"] == "v7_mmr" and body["served_by"]["path"] == "pipeline"
    assert body["routes"][0]["explanation"]["stages"]["rank_api"]["model"]


def test_unknown_system_setting_falls_back_to_the_baseline(serving):
    api = serving(recsys_system="no_such_system")
    pid = new_profile(api)
    sb = recs(api, pid)["served_by"]
    assert sb["path"] == "baseline" and sb["reason"] == "pipeline_not_loaded" and sb["degraded"]


# ------------------------------------------------------------------------------------ fallbacks
def test_profile_without_home_has_no_radius_filter(serving):
    api = serving()
    pid = new_profile(api, intent="castle and a beer garden")
    body = recs(api, pid, category="cycling")
    for route in body["routes"]:
        rad = route["explanation"]["stages"]["filter"]["radius"]
        assert rad["applied"] is False and rad["distance_km"] is None  # never claims a check that did not run
        assert "radius" not in route["reason"]
        assert route["category_name"] in ("Cycling", "Mountainbiking", "Long distance cycling")


def test_profile_without_intent_uses_the_rule_path_without_a_provider_call(serving):
    client = LLMClient(LLMConfig(max_retries=0))
    client.mock.fail("embed", times=100, status_code=503)  # would break any embedding call
    api = serving(client=client)
    pid = new_profile(api, home=HOME, travel_radius_km=30)
    body = recs(api, pid, category="hiking")
    sb = body["served_by"]
    assert sb["path"] == "pipeline" and sb["reason"] == "no_intent" and not sb["degraded"]
    for route in body["routes"]:
        st = route["explanation"]["stages"]
        assert st["v0_rules"] is not None and st["bm25"] is None and st["ltr"] is None
        assert st["filter"]["radius"]["ok"]


def test_guest_gets_uniform_random_over_the_tab(serving):
    api = serving()
    a = recs(api, category="hiking", limit=20)
    b = recs(api, category="hiking", limit=20)
    assert a["served_by"]["path"] == "random" and a["is_personalized"] is False
    ids_a = [r["id"] for r in a["routes"]]
    assert ids_a != [r["id"] for r in b["routes"]]  # seeded per request
    hiking = sorted(int(r.id) for r in get_catalog() if r.category_name in ("Hiking trail", "Theme trail"))
    assert max(ids_a) > hiking[60]  # not the lowest ids of the tab (DECISIONS.md D7.3)


def test_embedding_provider_failure_serves_the_no_llm_path(serving):
    client = LLMClient(LLMConfig(max_retries=0))
    client.mock.fail("embed", times=100, status_code=503)
    api = serving(client=client)
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="waterfall gorge walk")
    body = recs(api, pid, category="hiking")
    sb = body["served_by"]
    assert sb["path"] == "no_llm" and sb["system"] == "v2_bm25" and sb["degraded"]
    assert sb["reason"] == "provider_unavailable" and body["total"] == 10
    st = body["routes"][0]["explanation"]["stages"]
    assert st["bm25"]["matched_terms"] and st["dense"] is None


def test_serving_budget_exhausted_serves_the_no_llm_path(serving):
    client = LLMClient(LLMConfig(max_retries=0, budget_usd=0.0, mock_cost_usd_per_call=0.01))
    api = serving(client=client)
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    sb = recs(api, pid, category="hiking")["served_by"]
    assert sb["path"] == "no_llm" and sb["reason"] == "serving_budget"


def test_pipeline_timeout_serves_the_no_llm_path(serving, monkeypatch):
    api = serving(recsys_timeout_s=0.2)
    res = api.app.state.recsys

    async def slow(query, k=10):
        await asyncio.sleep(5)

    monkeypatch.setattr(res.pipeline, "arecommend", slow)
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    sb = recs(api, pid, category="hiking")["served_by"]
    assert sb["path"] == "no_llm" and sb["reason"] == "timeout"


# ------------------------------------------------------------------------- catalogue / DB check
def test_db_with_another_catalogue_falls_back_to_the_baseline(serving, seeded_db_copy, monkeypatch):
    api = serving(db=seeded_db_copy)  # the legacy 111-route seed, not the OSM catalogue
    warnings = []
    monkeypatch.setattr(recsys_service.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    pid = new_profile(api, intent="lake loop")
    body = recs(api, pid)
    sb = body["served_by"]
    assert sb["path"] == "baseline" and sb["reason"] == "catalogue_mismatch" and sb["degraded"]
    assert any("do not match the in-memory catalogue" in w and "serving the baseline" in w for w in warnings)
    checks = api.get("/api/recsys/status").json()["recsys"]["db_checks"]
    legacy = checks[seeded_db_copy.name]
    assert legacy["ok"] is False and legacy["db_routes"] == 111 and legacy["extra_in_db"] == 111
    legacy_ids = {r[0] for r in sqlite3.connect(seeded_db_copy).execute("SELECT id FROM routes")}
    assert {r["id"] for r in body["routes"]} <= legacy_ids
    guest = recs(api)
    assert guest["served_by"]["reason"] == "catalogue_mismatch"


def test_ids_missing_from_the_db_at_request_time_fall_back(serving, tmp_path, full_osm_db):
    db = tmp_path / "shrinking.db"
    shutil.copy(full_osm_db, db)
    api = serving(db=db)
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    first = recs(api, pid, category="hiking")
    assert first["served_by"]["path"] == "pipeline"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DELETE FROM routes WHERE id = ?", (first["routes"][0]["id"],))
    conn.commit()
    conn.close()
    again = recs(api, pid, category="hiking")
    assert again["served_by"]["path"] == "baseline" and again["served_by"]["reason"] == "catalogue_mismatch"


# ----------------------------------------------------------------------------- feedback events
def test_feedback_events_carry_real_ages_and_completions(serving, tmp_path, full_osm_db):
    db = tmp_path / "fb.db"
    shutil.copy(full_osm_db, db)
    # recsys_v1 has no feedback stage, so this also covers the serving layer's interim hiding rule; the
    # served default handles feedback inside the pipeline (see the recsys_v1_fb_explore test below)
    api = serving(db=db, recsys_system="recsys_v1")
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    ids = [r["id"] for r in recs(api, pid, category="hiking")["routes"]]
    now = datetime.now(UTC)
    conn = sqlite3.connect(db)
    for rid, reason, days in ((ids[0], "too-hard", 10), (ids[1], "not-interested", 90)):
        conn.execute("INSERT INTO profile_feedback (demo_profile_id, route_id, reason, created_at) VALUES (?,?,?,?)",
                     (pid, rid, reason, (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")))
    conn.execute("INSERT INTO souvenirs (demo_profile_id, route_id, completed_at, total_xp_gained) VALUES (?,?,?,0)",
                 (pid, ids[2], (now - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    async def events():
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
        async with async_sessionmaker(engine, class_=AsyncSession)() as s:
            out = await recsys_service.feedback_events(s, pid, now=now)
        await engine.dispose()
        return out

    ev = {(e.route_id, e.reason): e.age_days for e in asyncio.run(events())}
    assert ev[(ids[0], "too-hard")] == pytest.approx(10, abs=0.01)
    assert ev[(ids[1], "not-interested")] == pytest.approx(90, abs=0.01)
    assert ev[(ids[2], "completed")] == pytest.approx(3, abs=0.01)

    body = recs(api, pid, category="hiking")
    q = body["served_by"]["query"]
    assert q["feedback_events"] == 3
    shown = {r["id"] for r in body["routes"]}
    # recent rejection and completion are hidden; the 90-day-old one is past the window and may return
    assert ids[0] not in shown and ids[2] not in shown
    assert set(q["suppressed_route_ids"]) == {ids[0], ids[2]}
    assert body["routes"][0]["explanation"]["feedback"]["events_considered"] == 3


def test_feedback_and_completion_through_the_api_change_the_next_list(serving):
    api = serving()
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    before = [r["id"] for r in recs(api, pid, category="hiking")["routes"]]
    resp = api.post(f"/api/profiles/{pid}/feedback", json={"route_id": before[0], "reason": "too-far"})
    assert resp.status_code == 201
    mid = [r["id"] for r in recs(api, pid, category="hiking")["routes"]]
    assert before[0] not in mid and mid[0] == before[1]
    done = api.post(f"/api/profiles/{pid}/souvenirs", json={"route_id": mid[0], "completed_quest_ids": []})
    assert done.status_code == 200, done.text
    after = [r["id"] for r in recs(api, pid, category="hiking")["routes"]]
    assert mid[0] not in after and len(after) == 10


# --------------------------------------------------------------------------- original vector
def test_pipeline_starts_from_the_questionnaire_vector_not_the_rewritten_one(serving):
    api = serving()
    pid = new_profile(api, home=HOME, travel_radius_km=40)
    created = json.loads(api.get(f"/api/profiles/{pid}").json()["user_vector_json"])
    ids = [r["id"] for r in recs(api, pid, category="hiking", limit=5)["routes"]]
    for rid in ids[:3]:
        api.post(f"/api/profiles/{pid}/feedback", json={"route_id": rid, "reason": "too-hard"})
    rewritten = json.loads(api.get(f"/api/profiles/{pid}").json()["user_vector_json"])
    assert rewritten["difficulty_range"] != created["difficulty_range"]  # the baseline's compounding (D7.1)

    async def vector():
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from app.models.entities import DemoProfile

        engine = create_async_engine(f"sqlite+aiosqlite:///{api.db_path}")
        async with async_sessionmaker(engine, class_=AsyncSession)() as s:
            prof = await s.get(DemoProfile, pid)
            out = recsys_service.pipeline_vector(prof)
        await engine.dispose()
        return out

    vec, source = asyncio.run(vector())
    assert source == "questionnaire" and vec == created
    # the chat's slots are projected onto the questionnaire vector, feedback still not in it
    api.patch(f"/api/profiles/{pid}/chat/slots", json={"difficulty": "hard", "tags_add": ["Lake"]})
    vec, source = asyncio.run(vector())
    assert source == "questionnaire+chat"
    assert vec["difficulty_range"] == [2, 3] and "Lake" in vec["preferred_tags"]
    assert vec["min_distance_km"] == created["min_distance_km"]


def test_migration_backfills_the_questionnaire_vector(tmp_path):
    db = tmp_path / "m.db"
    run_alembic(db, "upgrade", "a7c3e9d2b5f4")
    conn = sqlite3.connect(db)
    original = {"difficulty_range": [1, 2], "max_distance_km": 20.0, "min_distance_km": 5.0,
                "preferred_tags": ["flora", "scenic"], "fitness_level": "intermediate"}
    compounded = {**original, "difficulty_range": [0, 1.0], "max_distance_km": 14.0, "preferred_tags": ["scenic"]}
    conn.execute("INSERT INTO routes (id, title, xp_required, base_xp_reward) VALUES (1, 'r', 0, 0)")
    conn.execute("INSERT INTO demo_profiles (id, user_vector_json) VALUES (1, ?)", (json.dumps(original),))
    conn.execute("INSERT INTO demo_profiles (id, user_vector_json) VALUES (2, ?)", (json.dumps(compounded),))
    conn.execute("INSERT INTO demo_profiles (id, user_vector_json, chat_state_json) VALUES (3, ?, ?)",
                 (json.dumps(compounded), json.dumps({"base_vector": original})))
    conn.execute("INSERT INTO demo_profiles (id) VALUES (4)")
    conn.execute("INSERT INTO profile_feedback (demo_profile_id, route_id, reason) VALUES (2, 1, 'too-hard')")
    conn.commit()
    conn.close()
    run_alembic(db, "upgrade", "head")
    conn = sqlite3.connect(db)
    got = {pid: (json.loads(v) if v else None)
           for pid, v in conn.execute("SELECT id, questionnaire_vector_json FROM demo_profiles")}
    conn.close()
    assert got[1] == original  # no feedback: as stored
    assert got[2]["difficulty_range"] == [1, 2] and got[2]["max_distance_km"] == 20.0  # rebuilt from fitness
    assert got[2]["preferred_tags"] == ["scenic"] and got[2]["_reconstructed"] is True  # tags: as stored
    assert got[3] == original  # chat's pre-chat vector
    assert got[4] is None
    run_alembic(db, "downgrade", "a7c3e9d2b5f4")


# ---------------------------------------------------------------------- rate limit and budget
def test_rate_limiter_sliding_window():
    now = [0.0]
    rl = RateLimiter({"chat": 2}, 10.0, clock=lambda: now[0])
    assert rl.hit("chat", "a")[0] and rl.hit("chat", "a")[0]
    ok, retry = rl.hit("chat", "a")
    assert not ok and retry == pytest.approx(10.0)
    assert rl.hit("chat", "b")[0]  # per key
    now[0] = 10.5
    assert rl.hit("chat", "a")[0]
    assert RateLimiter({"chat": 1}, 10.0, enabled=False).hit("chat", "a") == (True, 0.0)


def test_daily_budget_counts_spend_per_utc_day():
    class C:
        spent_usd = 0.0

    c, day = C(), ["2026-11-23"]
    b = DailyBudget(0.05, clients=lambda: [c], today=lambda: day[0])
    assert b.spent_today() == 0 and not b.exhausted()
    c.spent_usd = 0.06
    assert b.exhausted()
    day[0] = "2026-11-24"  # new day: what was spent before is the baseline
    assert not b.exhausted() and b.spent_today() == 0
    assert not DailyBudget(0, clients=lambda: [c]).exhausted()  # 0 = off


def test_chat_rate_limit_returns_429_with_retry_after(serving):
    api = serving(rate_limit_enabled="true", rate_limit_chat=2)
    pid = new_profile(api)
    for _ in range(2):
        assert api.post(f"/api/profiles/{pid}/chat", json={"message": "hiking near Grainau"}).status_code == 200
    resp = api.post(f"/api/profiles/{pid}/chat", json={"message": "with a lake"})
    assert resp.status_code == 429 and int(resp.headers["Retry-After"]) >= 1


def test_recommendation_rate_limit_degrades_instead_of_failing(serving):
    api = serving(rate_limit_enabled="true", rate_limit_recommendations=1)
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    assert recs(api, pid, category="hiking")["served_by"]["path"] == "pipeline"
    sb = recs(api, pid, category="hiking")["served_by"]
    assert sb["path"] == "no_llm" and sb["reason"] == "rate_limited"
    nointent = new_profile(api, home=HOME, travel_radius_km=40)  # no provider call: not counted, not limited
    assert recs(api, nointent, category="hiking")["served_by"]["path"] == "pipeline"


def test_narrative_rate_limit_uses_the_template(serving, monkeypatch):
    from app.api.v1 import profiles

    calls = []

    async def welcome(questionnaire):
        calls.append(1)
        return "LLM welcome"

    api = serving(rate_limit_enabled="true", rate_limit_narrative=1)
    monkeypatch.setattr(profiles, "generate_welcome_summary", welcome)
    first = api.post("/api/profiles", json=ANSWERS).json()
    second = api.post("/api/profiles", json=ANSWERS).json()
    assert first["welcome_summary"] == "LLM welcome"
    assert second["welcome_summary"] != "LLM welcome" and "Welcome" in second["welcome_summary"]
    assert len(calls) == 1


def test_daily_budget_exhausted_degrades_recommendations_and_chat(serving):
    spender = LLMClient(LLMConfig(max_retries=0, mock_cost_usd_per_call=0.02))
    api = serving(client=spender, daily_budget_usd=0.03)
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop")
    assert recs(api, pid, category="hiking")["served_by"]["path"] == "pipeline"
    api.patch(f"/api/profiles/{pid}", json={"intent": "castle ruin walk"})  # a new intent: a new embed call
    assert recs(api, pid, category="hiking")["served_by"]["path"] == "pipeline"
    sb = recs(api, pid, category="hiking")["served_by"]
    assert sb["path"] == "no_llm" and sb["reason"] == "daily_budget"
    turn = api.post(f"/api/profiles/{pid}/chat", json={"message": "cycling with a lake please"})
    assert turn.status_code == 200 and turn.json()["model"] == "keyword-rules"
    assert turn.json()["slots"]["activity"] == "cycling"
    status = api.get("/api/recsys/status").json()
    assert status["daily_budget"]["exhausted"] is True and status["recsys"]["ok"] is True


# --------------------------------------------------------------------------------- start-up
def test_status_reports_start_up_and_the_frontend_is_still_served(serving, monkeypatch, tmp_path):
    dist = tmp_path / "out"
    dist.mkdir()
    (dist / "index.html").write_text("<html>trailsaga</html>", encoding="utf-8")
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(dist))
    api = serving()
    assert "trailsaga" in api.get("/").text
    assert api.get("/api/does-not-exist").status_code == 404
    st = api.get("/api/recsys/status").json()["recsys"]
    assert st["configured"] == "recsys_serve" and st["ok"] and st["catalogue_routes"] == len(get_catalog())
    assert st["total_ms"] >= 0 and st["peak_rss_mb_after"] > 0


def test_reason_shows_the_users_words_not_stems():
    assert recsys_service.display_term("hik", "an easy hiking route") == "hiking"
    assert recsys_service.display_term("lake", "lake walk") == "lake"
    assert recsys_service.display_term("circular", "a loop") == "circular"  # expansion term: as is


def test_pipeline_with_its_own_feedback_and_exploration_stages(serving):
    api = serving(recsys_system="recsys_v1_fb_explore")
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="lake loop with a hut")
    before = recs(api, pid, category="hiking")
    assert before["served_by"]["query"]["feedback_handled_by"] == "pipeline"
    rejected = before["routes"][0]["id"]
    api.post(f"/api/profiles/{pid}/feedback", json={"route_id": rejected, "reason": "not-interested"})
    after = recs(api, pid, category="hiking")
    assert rejected not in {r["id"] for r in after["routes"]}  # the pipeline's own route effect
    assert after["served_by"]["query"]["suppressed_route_ids"] == []  # interim rule off
    assert sum(r["is_exploration"] for r in after["routes"]) == 1  # one exploration slot in the top 10
    for route in after["routes"]:
        RouteExplanationResponse.model_validate(route["explanation"])
        if route["is_exploration"]:
            assert route["explanation"]["stages"]["explore"] and "exploration pick" in route["reason"]


# ------------------------------------------------------------------------------- stated needs (iteration 2)
def test_served_default_has_no_needs_stage_and_v1_fields_are_unchanged(serving):
    api = serving()
    pid = new_profile(api, home=HOME, travel_radius_km=40, intent="an easy walk with our stroller, around 6 km")
    body = recs(api, pid, category="hiking")
    assert "needs" not in [s["stage"] for s in body["served_by"]["stages"]]
    for route in body["routes"]:
        e = route["explanation"]
        assert e["schema_version"] == "recsys.explanation/v1.1" and e["stages"]["needs"] is None
        assert "you asked for" not in route["reason"]


def test_needs_system_checks_and_explains_what_the_user_asked_for(serving):
    api = serving(recsys_system="recsys_serve_needs")
    pid = new_profile(api, home={"name": "Erlangen"}, travel_radius_km=40,
                      intent="An easy walk with our stroller, around 6 km")
    body = recs(api, pid, category="hiking")
    sb = body["served_by"]
    assert sb["system"] == "recsys_serve_needs" and sb["path"] == "pipeline"
    stage = next(s for s in sb["stages"] if s["stage"] == "needs")
    info = stage["info"]
    assert {n["need"] for n in info["extracted"]["needs"]} == {"stroller", "length"}
    assert stage["n_out"] >= min(20, stage["n_in"])  # never below the floor
    cat = get_catalog()
    for route in body["routes"]:
        expl = RouteExplanationResponse.model_validate(route["explanation"])
        checks = {c.need: c for c in expl.stages.needs.checks}
        assert set(checks) == {"stroller", "length"}
        st = checks["stroller"]
        rec = cat.get(route["id"])
        steps = rec.features["profile"].get("steps_m", 0) or 0
        assert st.values["steps_m"] == steps  # the real catalogue number
        assert ("no steps" in st.fact) == (steps <= 0)
        if expl.stages.explore is None and st.applied:
            assert st.ok and st.met_level is not None and st.met_level <= st.applied_level
        if st.ok:
            assert "you asked for stroller-friendly: " in route["reason"]
    # n_eligible counts the routes past the hard filter and the needs
    assert body["routes"][0]["explanation"]["n_eligible"] == stage["n_out"]


def test_chat_slots_feed_the_needs_stage(serving):
    api = serving(recsys_system="recsys_serve_needs")
    pid = new_profile(api, home={"name": "Erlangen"}, travel_radius_km=40, intent="a quiet walk")
    resp = api.patch(f"/api/profiles/{pid}/chat/slots", json={"tags_add": ["Train access", "Round trip"]})
    assert resp.status_code == 200, resp.text
    body = recs(api, pid, category="hiking")
    assert body["served_by"]["query"]["slot_needs"] == ["train", "loop"]
    stage = next(s for s in body["served_by"]["stages"] if s["stage"] == "needs")
    assert {n["need"]: n["source"] for n in stage["info"]["extracted"]["needs"]} == {"train": "slot", "loop": "slot"}
    for route in body["routes"]:
        for c in route["explanation"]["stages"]["needs"]["checks"]:
            assert c["phrase"].startswith("slot: ")


def test_needs_from_slots_maps_only_verifiable_tags():
    from app.chat.projection import needs_from_slots

    assert needs_from_slots("mtb", ["Lake", "Train access"], 2.5) == ("mtb", "train", "duration_h:2.5")
    assert needs_from_slots("hiking", ["suitable for strollers", "Multi-stage route"]) == ("stroller", "days:2")
    assert needs_from_slots(None, []) == ()


def test_reason_names_met_and_unmet_needs():
    class R:
        length_meters, is_loop = 6000, True

    expl = {"stages": {"needs": {"checks": [
        {"need": "stroller", "label": "stroller-friendly", "ok": True, "applied": True,
         "fact": "no steps, 94% of the route on stroller-suitable surface, 120 m of climbing"},
        {"need": "train", "label": "reachable by train", "ok": False, "applied": False,
         "fact": "no train station or stop mapped near the start"},
    ]}}}
    line = recsys_service.reason_sentence(R(), expl, None)
    assert "you asked for stroller-friendly: no steps, 94% of the route on stroller-suitable surface" in line
    assert "not reachable by train: no train station or stop mapped near the start (too few routes met it)" in line

