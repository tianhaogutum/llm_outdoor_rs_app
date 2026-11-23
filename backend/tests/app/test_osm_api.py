"""The API on a DB seeded from the OSM catalogue: route payloads, geometry, tabs, flows, static site."""

from __future__ import annotations

import json

import pytest

from app.services.recommendation_service import CATEGORY_MAPPING

ANSWERS = {"fitness": "beginner", "type": ["natural-scenery"], "narrative": "mystery"}
CATALOG_KEYS = {"source", "source_url", "activity", "start_lat", "start_lon", "is_loop", "ascent_m", "descent_m",
                "locality", "admin_district", "admin_region", "facts", "attribution"}


@pytest.fixture
def client(make_client, osm_db_copy):
    return make_client(osm_db_copy)


@pytest.fixture
def full_client(make_client, full_osm_db, tmp_path):
    import shutil

    db = tmp_path / "full_copy.db"  # flows write feedback/souvenirs: keep the session DB pristine
    shutil.copy(full_osm_db, db)
    return make_client(db)


def assert_osm_route_payload(route: dict) -> None:
    assert CATALOG_KEYS <= route.keys()
    assert route["source"] == "osm"
    assert route["source_url"] == f"https://www.openstreetmap.org/relation/{route['id']}"
    assert route["attribution"].startswith("© OpenStreetMap contributors")
    assert route["ascent_m"] == route["elevation"]
    assert isinstance(route["facts"], dict) and isinstance(route["is_loop"], bool)
    assert route["locality"] and route["start_lat"] and route["start_lon"]


def test_recommendations_expose_catalogue_facts_for_every_tab(full_client):
    pid = full_client.post("/api/profiles", json=ANSWERS).json()["id"]
    for category in (None, "running", "hiking", "cycling"):
        params = {"profile_id": pid, "limit": 20, **({"category": category} if category else {})}
        resp = full_client.get("/api/routes/recommendations", params=params)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["is_personalized"] is True and body["total"] == 20
        # the DB holds the catalogue, so the default pipeline serves (no intent: its rule path)
        assert body["served_by"]["system"] == "recsys_serve" and body["served_by"]["reason"] == "no_intent"
        for route in body["routes"]:
            assert_osm_route_payload(route)
            if category:
                assert route["category_name"] in CATEGORY_MAPPING[category]
            bps = route["breakpoints"]
            assert bps[0]["poi_type"] == "start" and bps[-1]["poi_type"] == "finish"
            assert route["explanation"]["stages"]["filter"]["verdict"] == "pass" and route["reason"]
            assert route["recommendation_score"] is None  # the baseline's fields stay baseline-only
    # guest (random) recommendations work too
    guest = full_client.get("/api/routes/recommendations", params={"limit": 10}).json()
    assert guest["is_personalized"] is False and guest["total"] == 10


def test_recommendation_limit_is_bounded(client):
    assert client.get("/api/routes/recommendations", params={"limit": 0}).status_code == 422
    assert client.get("/api/routes/recommendations", params={"limit": 1001}).status_code == 422


def test_profile_without_vector_falls_back_without_500(client):
    pid = client.post("/api/profiles", json=ANSWERS).json()["id"]
    client.patch(f"/api/profiles/{pid}", json={"user_vector_json": None})
    resp = client.get("/api/routes/recommendations", params={"profile_id": pid, "limit": 5})
    assert resp.status_code == 200, resp.text
    assert all(r["breakpoints"] for r in resp.json()["routes"])


def test_route_list_is_paged_and_light(client):
    body = client.get("/api/routes", params={"limit": 7, "offset": 3}).json()
    assert body["limit"] == 7 and body["offset"] == 3 and len(body["routes"]) == 7
    ids = [r["id"] for r in body["routes"]]
    assert ids == sorted(ids)
    everything = client.get("/api/routes", params={"limit": 1000}).json()
    assert body["total"] == everything["total"] == len(everything["routes"])
    assert ids == [r["id"] for r in everything["routes"][3:10]]
    for r in body["routes"]:
        assert "breakpoints" not in r and "story_prologue_body" not in r and "gpx_data_raw" not in r
        assert_osm_route_payload(r)
    hiking = client.get("/api/routes", params={"category": "hiking", "limit": 1000}).json()
    assert 0 < hiking["total"] < everything["total"]
    assert {r["category_name"] for r in hiking["routes"]} <= set(CATEGORY_MAPPING["hiking"])


def test_route_detail(client):
    route = client.get("/api/routes/1356555").json()
    assert_osm_route_payload(route)
    assert route["title"] == "Eibsee Rundwanderweg" and route["is_loop"] is True
    assert (route["ascent_m"], route["descent_m"]) == (113, 116)
    assert route["facts"]["access"]["train_station"]["name"] == "Eibsee"
    assert [b["order_index"] for b in route["breakpoints"]] == list(range(len(route["breakpoints"])))
    assert [b["at_m"] for b in route["breakpoints"]] == sorted(b["at_m"] for b in route["breakpoints"])
    assert client.get("/api/routes/999999999").status_code == 404


def test_route_geometry(client):
    geom = client.get("/api/routes/1356555/geometry").json()
    assert geom["route_id"] == 1356555 and len(geom["coords"]) > 10
    lons, lats = zip(*geom["coords"], strict=True)
    assert geom["bbox"] == [min(lons), min(lats), max(lons), max(lats)]
    assert 47 < min(lats) and max(lats) < 48 and 10 < min(lons) and max(lons) < 12  # Eibsee, Bavaria
    assert geom["ele_step_m"] == 200.0 and len(geom["ele_profile_m"]) > 10
    assert geom["length_m"] == 6962.0 and geom["attribution"].startswith("© OpenStreetMap")
    # breakpoints lie within the line's bounding box
    for bp in client.get("/api/routes/1356555").json()["breakpoints"]:
        assert geom["bbox"][0] <= bp["longitude"] <= geom["bbox"][2]
        assert geom["bbox"][1] <= bp["latitude"] <= geom["bbox"][3]
    resp = client.get("/api/routes/999999999/geometry")
    assert resp.status_code == 404 and "not found" in resp.json()["detail"]


def test_geometry_404_for_legacy_routes(make_client, seeded_db_copy):
    client = make_client(seeded_db_copy)
    resp = client.get("/api/routes/1362328/geometry")
    assert resp.status_code == 404 and "no stored geometry" in resp.json()["detail"]
    legacy = client.get("/api/routes/1362328").json()
    assert legacy["source"] is None and legacy["facts"] is None and legacy["attribution"] is None
    assert legacy["ascent_m"] == legacy["elevation"] == 60


def test_feedback_souvenir_statistics_flow(full_client):
    pid = full_client.post("/api/profiles", json={**ANSWERS, "home": {"name": "Grainau"}, "travel_radius_km": 50,
                                                  "distance_cap_km": 15, "intent": "lake loop"}).json()["id"]
    rec = full_client.get("/api/routes/recommendations", params={"profile_id": pid, "category": "hiking"}).json()
    top = rec["routes"][0]

    resp = full_client.post(f"/api/profiles/{pid}/feedback", json={"route_id": top["id"], "reason": "too-hard"})
    assert resp.status_code == 201, resp.text
    after = full_client.get("/api/routes/recommendations",
                            params={"profile_id": pid, "category": "hiking", "limit": 1000}).json()
    penalised = next((r for r in after["routes"] if r["id"] == top["id"]), None)
    if penalised is not None:  # it may drop out of the top 1000 entirely
        assert penalised["recommendation_score"] < top["recommendation_score"]
        assert penalised["recommendation_score_breakdown"]["feedback_penalty"] == 0.5  # extra keys survive

    target = rec["routes"][1]
    resp = full_client.post(f"/api/profiles/{pid}/souvenirs",
                            json={"route_id": target["id"], "completed_quest_ids": []})
    assert resp.status_code == 200, resp.text
    done = resp.json()
    assert done["total_xp_gained"] > 0
    assert done["souvenir"]["route"]["source"] == "osm" and done["souvenir"]["route"]["attribution"]
    assert {a["achievement_key"] for a in done["newly_unlocked_achievements"]} >= {"first-steps", "hiker"}

    stats = full_client.get(f"/api/profiles/{pid}/statistics").json()
    assert stats["routes_completed"] == 1 and stats["activity_breakdown"]["hiking"] == 1
    assert stats["total_elevation_m"] == target["ascent_m"]
    assert stats["total_distance_km"] == round(target["length_meters"] / 1000, 1)
    gallery = full_client.get(f"/api/profiles/{pid}/souvenirs").json()
    assert gallery["total"] == 1 and gallery["souvenirs"][0]["route"]["locality"] == target["locality"]
    one = full_client.get(f"/api/profiles/{pid}/souvenirs/{gallery['souvenirs'][0]['id']}").json()
    assert one["route"]["facts"] == target["facts"]
    achievements = full_client.get(f"/api/achievements/profiles/{pid}").json()
    assert sum(a["unlocked"] for a in achievements) == 2


def test_story_generation_uses_real_breakpoints(client):
    resp = client.post("/api/routes/1356555/generate-story", json={"narrative_style": "adventure"})
    assert resp.status_code == 200, resp.text
    story = resp.json()
    route = client.get("/api/routes/1356555").json()
    assert len(story["breakpoints"]) == len(route["breakpoints"])
    assert all(b["main_quest_snippet"] for b in route["breakpoints"])  # saved for the simulator
    assert client.get("/api/routes/1356555/story").status_code == 200


def test_dashboard_counts_full_catalogue(full_client):
    resp = full_client.get("/dashboard")
    assert resp.status_code == 200 and "9347" in resp.text


# --- static frontend ---------------------------------------------------------------------------------


def make_dist(tmp_path):
    dist = tmp_path / "out"
    (dist / "_next" / "static").mkdir(parents=True)
    (dist / "index.html").write_text("<!DOCTYPE html><title>TrailSaga test build</title>", encoding="utf-8")
    (dist / "404.html").write_text("<!DOCTYPE html><title>not here</title>", encoding="utf-8")
    (dist / "_next" / "static" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    return dist


def test_static_frontend_is_served_without_shadowing_the_api(make_client, osm_db_copy, tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(make_dist(tmp_path)))
    client = make_client(osm_db_copy)
    root = client.get("/")
    assert root.status_code == 200 and "TrailSaga test build" in root.text
    assert client.get("/_next/static/app.js").text == "console.log('ok')"
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200 and "swagger" in client.get("/docs").text.lower()
    assert client.get("/openapi.json").json()["info"]["title"]
    assert client.get("/dashboard").status_code == 200
    assert client.get("/api/routes/1356555").json()["id"] == 1356555
    missing_api = client.get("/api/does-not-exist")
    assert missing_api.status_code == 404 and missing_api.json() == {"detail": "Not Found"}
    page = client.get("/some/page")
    assert page.status_code == 404 and "not here" in page.text


def test_without_a_build_root_is_the_dashboard(make_client, osm_db_copy, tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(tmp_path / "missing"))
    client = make_client(osm_db_copy)
    root = client.get("/")
    assert root.status_code == 200 and "/docs" in root.text
    assert json.loads(client.get("/openapi.json").text)["paths"].get("/") is None  # not in the API schema
