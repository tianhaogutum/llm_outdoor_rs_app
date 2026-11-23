"""Pre-generated narratives (docs/DEPLOY.md): the committed store, the order live LLM -> pre-generated ->
template on the two endpoints that show LLM text, provenance, grounding checks, and the offline chat fallback.

Offline: LLM_PROVIDER=mock (tests/conftest.py); stores for the endpoint tests are written to tmp dirs.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.schemas import ProfileCreate
from app.services import demo_guard
from app.services import narratives as nv
from app.services.user_profile_service import generate_fallback_welcome
from app.settings import Settings, get_settings
from tests.app.conftest import CATALOG_DIR, KNOWN_ROUTES

ANSWERS = {"fitness": "intermediate", "type": ["natural-scenery", "history-culture"], "narrative": "mystery"}
STORED_ROUTE = KNOWN_ROUTES[0]  # in the test subset DB
OTHER_ROUTE = KNOWN_ROUTES[1]


@pytest.fixture(scope="module")
def catalogue() -> dict[int, dict]:
    with gzip.open(CATALOG_DIR / "routes.jsonl.gz", "rt", encoding="utf-8") as fh:
        return {int(r["id"]): r for r in (json.loads(line) for line in fh if line.strip())}


def write_store(directory: Path, *, welcome: dict[str, str] | None = None,
                routes: dict[int, str] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"llm_generated": True, "model": "test-model", "prompt_version": "test/v1", "generated_at": "2026-11-23"}
    if welcome is not None:
        (directory / nv.WELCOME_FILE).write_text(json.dumps({
            "schema": nv.STORE_SCHEMA, "kind": "welcome", "provenance": {"model": "test-model"},
            "items": [{"key": k, "text": t, **meta} for k, t in welcome.items()]}), encoding="utf-8")
    if routes is not None:
        (directory / nv.ROUTES_FILE).write_text(json.dumps({
            "schema": nv.STORE_SCHEMA, "kind": "routes", "provenance": {"model": "test-model"},
            "items": [{"route_id": r, "text": t, **meta} for r, t in routes.items()]}), encoding="utf-8")
    return directory


@pytest.fixture
def api(monkeypatch, make_client, osm_db_copy):
    """``api(**env)`` -> TestClient on a copy of the subset DB with the given settings."""
    from app.api.v1 import profiles

    calls: list[str] = []

    async def live_welcome(questionnaire):
        calls.append("welcome")
        return "LIVE welcome"

    def configure(**env):
        for k, v in env.items():
            monkeypatch.setenv(k.upper(), str(v))
        get_settings.cache_clear()
        demo_guard.set_guard(None)
        nv.reset_store()
        client = make_client(osm_db_copy)
        monkeypatch.setattr(profiles, "generate_welcome_summary", live_welcome)  # after make_client's own stub
        client.calls = calls
        return client

    yield configure
    monkeypatch.undo()
    get_settings.cache_clear()
    demo_guard.set_guard(None)
    nv.reset_store()


# ------------------------------------------------------------------------------- committed store
def test_committed_store_matches_the_catalogue_and_carries_provenance(catalogue):
    store = nv.load_store(nv.DEFAULT_NARRATIVES_DIR)
    assert store.routes, "backend/data/narratives/routes.json is missing or empty"
    missing = sorted(set(store.routes) - set(catalogue))
    assert not missing, f"pre-generated texts for routes not in the catalogue: {missing[:10]}"
    expected = {nv.welcome_key(f, ts, n) for f, ts, n in nv.all_welcome_combinations()}
    assert len(expected) == 72 and set(store.welcome) == expected
    for header in (store.welcome_header, store.routes_header):
        assert header["schema"] == nv.STORE_SCHEMA and header["llm_generated"] is True
        assert {"provider", "model", "prompt_version", "params"} <= set(header["provenance"])
        assert "LLM-generated" in header["notice"]
    assert store.routes_header["catalogue_sha256"]
    for e in [*store.welcome.values(), *store.routes.values()]:
        assert e.meta["llm_generated"] is True
        for f in ("model", "prompt_version", "generated_at", "cache_key"):
            assert e.meta.get(f), f"entry without {f}"
    assert {e.meta["prompt_version"] for e in store.routes.values()} == {nv.ROUTE_PROMPT_VERSION}
    assert {e.meta["prompt_version"] for e in store.welcome.values()} == {nv.WELCOME_PROMPT_VERSION}


def test_committed_texts_pass_the_grounding_checks(catalogue):
    store = nv.load_store(nv.DEFAULT_NARRATIVES_DIR)
    for rid, e in store.routes.items():
        checks = nv.check_route_text(e.text, nv.route_facts(catalogue[rid]))
        assert checks["ok"], (rid, e.text, checks)
    for key, e in store.welcome.items():
        assert nv.check_welcome_text(e.text)["ok"], (key, e.text)


def test_demo_route_set_and_store_agree(catalogue):
    demo = json.loads((nv.DEFAULT_NARRATIVES_DIR / nv.DEMO_ROUTES_FILE).read_text(encoding="utf-8"))
    ids = {r["route_id"] for r in demo["routes"]}
    assert ids <= set(catalogue)
    assert set(nv.load_store(nv.DEFAULT_NARRATIVES_DIR).routes) <= ids


# ------------------------------------------------------------------------------- grounding checks
def test_route_check_flags_invented_numbers_names_and_game_terms():
    facts = "Name: Eibsee Rundwanderweg\nLength: 7.0 km\nClimbing: 113 m\nDescription: passes the lake Eibsee."
    good = "You completed the Eibsee Rundwanderweg! You walked 7.0 km, climbed 113 m and passed the lake Eibsee. " \
           "Well done on this lovely loop, a fine way to spend the day outside."
    assert nv.check_route_text(good, facts)["ok"]
    assert nv.check_route_text(good.replace("7.0 km", "7 km"), facts)["ok"]  # rounding a fact is fine
    bad = nv.check_route_text(good.replace("113 m", "450 m"), facts)
    assert not bad["ok"] and bad["unsupported_numbers"] == ["450"]
    bad = nv.check_route_text(good + " Then you stopped at the Kreuzeck Hut.", facts)
    assert not bad["ok"] and set(bad["unknown_names"]) == {"Kreuzeck", "Hut"}
    bad = nv.check_route_text(good + " That earns you XP and completes a quest.", facts)
    assert not bad["ok"] and set(bad["banned_terms"]) == {"xp", "quest"}


def test_welcome_key_is_canonical():
    assert nv.welcome_key("advanced", ["family-fun", "history-culture", "family-fun"], "playful") == \
        "advanced|history-culture+family-fun|playful"
    assert nv.welcome_key("beginner", [], "mystery") == "beginner|-|mystery"
    assert nv.welcome_key("beginner", ["skiing"], "mystery") is None
    assert nv.welcome_key("expert", [], "mystery") is None


# ------------------------------------------------------------------------------- welcome endpoint
def test_pregenerated_welcome_is_served_when_the_llm_is_off(api, tmp_path):
    key = nv.welcome_key(ANSWERS["fitness"], ANSWERS["type"], ANSWERS["narrative"])
    store = write_store(tmp_path / "store", welcome={key: "Welcome, Pre-generated Seeker!"})
    client = api(narratives_live="false", narratives_dir=store)
    body = client.post("/api/profiles", json=ANSWERS).json()
    assert body["welcome_summary"] == "Welcome, Pre-generated Seeker!"
    assert body["welcome_summary_source"] == nv.SOURCE_PREGENERATED
    assert client.calls == []  # the LLM was not asked
    assert client.get(f"/api/profiles/{body['id']}").json()["genai_welcome_summary"] == body["welcome_summary"]


def test_template_welcome_when_nothing_is_pregenerated(api, tmp_path):
    client = api(narratives_live="false", narratives_dir=write_store(tmp_path / "empty"))
    body = client.post("/api/profiles", json=ANSWERS).json()
    assert body["welcome_summary_source"] == nv.SOURCE_TEMPLATE
    assert body["welcome_summary"] == generate_fallback_welcome(ProfileCreate(**ANSWERS))


def test_live_llm_first_then_pregenerated_when_the_guard_says_no(api, tmp_path):
    key = nv.welcome_key(ANSWERS["fitness"], ANSWERS["type"], ANSWERS["narrative"])
    store = write_store(tmp_path / "store", welcome={key: "Welcome, Pre-generated Seeker!"})
    client = api(narratives_dir=store, rate_limit_enabled="true", rate_limit_narrative=1)
    first = client.post("/api/profiles", json=ANSWERS).json()
    second = client.post("/api/profiles", json=ANSWERS).json()
    assert (first["welcome_summary"], first["welcome_summary_source"]) == ("LIVE welcome", nv.SOURCE_LLM)
    assert (second["welcome_summary"], second["welcome_summary_source"]) == (
        "Welcome, Pre-generated Seeker!", nv.SOURCE_PREGENERATED)
    assert client.calls == ["welcome"]


# ------------------------------------------------------------------------------- post-route summary
def test_post_route_summary_pregenerated_then_template(api, tmp_path):
    store = write_store(tmp_path / "store", routes={STORED_ROUTE: "You completed the Eibsee loop."})
    client = api(narratives_live="false", narratives_dir=store)
    pid = client.post("/api/profiles", json=ANSWERS).json()["id"]
    done = client.post(f"/api/profiles/{pid}/souvenirs", json={"route_id": STORED_ROUTE, "completed_quest_ids": []})
    assert done.status_code == 200, done.text
    body = done.json()
    xp = body["total_xp_gained"]
    assert body["summary_source"] == nv.SOURCE_PREGENERATED
    assert body["souvenir"]["genai_summary"] == f"You completed the Eibsee loop. You earned {xp} XP."
    other = client.post(f"/api/profiles/{pid}/souvenirs", json={"route_id": OTHER_ROUTE, "completed_quest_ids": []})
    body = other.json()
    assert body["summary_source"] == nv.SOURCE_TEMPLATE
    title = body["souvenir"]["route"]["title"]
    assert body["souvenir"]["genai_summary"] == nv.template_post_run(title, body["total_xp_gained"])


def test_offline_post_run_mentions_quests_only_when_there_are_some(tmp_path):
    settings = SimpleNamespace(narratives_dir=write_store(tmp_path / "s", routes={7: "Route text."}))
    nv.reset_store()
    assert nv.offline_post_run(7, "T", 120, 2, settings) == (
        "Route text. You completed 2 quests and earned 120 XP.", nv.SOURCE_PREGENERATED)
    assert nv.offline_post_run(7, "T", 120, 1, settings)[0].endswith("You completed 1 quest and earned 120 XP.")
    assert nv.offline_post_run(8, "T", 120, 0, settings)[1] == nv.SOURCE_TEMPLATE
    nv.reset_store()


def test_broken_store_falls_back_to_templates(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / nv.WELCOME_FILE).write_text("{not json", encoding="utf-8")
    nv.reset_store()
    settings = SimpleNamespace(narratives_dir=d)
    text, source = nv.offline_welcome(ProfileCreate(**ANSWERS), settings)
    assert source == nv.SOURCE_TEMPLATE and text
    nv.reset_store()


# ------------------------------------------------------------------------------- no credentials
def test_provider_checks_without_credentials():
    from app.services.provider_status import provider_configured

    bare = Settings(_env_file=None, llm_provider="vertex", google_cloud_project=None, google_cloud_location=None,
                    openrouter_api_key=None)
    assert not provider_configured("vertex", bare)
    assert not provider_configured("openrouter", bare)
    assert provider_configured("mock", bare) and provider_configured("ollama", bare)
    assert not nv.live_narratives_enabled(bare)
    with_vertex = Settings(_env_file=None, llm_provider="vertex", google_cloud_project="p",
                           google_cloud_location="europe-west4")
    assert nv.live_narratives_enabled(with_vertex)
    assert not nv.live_narratives_enabled(Settings(_env_file=None, llm_provider="mock", narratives_live=False))


async def test_chat_uses_keyword_rules_without_a_configured_provider(monkeypatch):
    from app.chat.schema import ChatSlots
    from app.services import chat_service, provider_status

    monkeypatch.setattr(provider_status, "chat_configured", lambda settings=None: False)
    ext = chat_service.llm_or_rules_extractor()
    res = await ext("easy hike with a lake", ChatSlots(), None)
    assert res.model == "keyword-rules" and res.update.activity == "hiking"


async def test_chat_falls_back_to_keyword_rules_on_a_configuration_error(monkeypatch):
    import app.chat.engine as engine
    from app.chat.schema import ChatSlots
    from app.llm import ConfigError
    from app.services import chat_service, provider_status

    def broken_extractor(client=None):
        async def run(message, current, pending):
            raise ConfigError("cannot create Vertex client: DefaultCredentialsError")

        return run

    monkeypatch.setattr(provider_status, "chat_configured", lambda settings=None: True)
    monkeypatch.setattr(engine, "default_extractor", broken_extractor)
    res = await chat_service.llm_or_rules_extractor()("cycling along a river", ChatSlots(), None)
    assert res.model == "keyword-rules" and res.update.activity == "cycling"
