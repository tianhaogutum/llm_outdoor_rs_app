"""Place lookup (GET /api/places) and the optional profile slots (home, radius, distance cap, intent)."""

from __future__ import annotations

import json

import pytest

from app.models.entities import Place
from app.services.place_service import name_key, rank_places
from scripts.osm_catalog import build_places

ANSWERS = {"fitness": "intermediate", "type": ["natural-scenery"], "narrative": "mystery"}


@pytest.mark.parametrize(("raw", "key"), [
    ("Garmisch-Partenkirchen", "garmisch partenkirchen"),
    ("München", "munchen"),
    ("Muenchen", "munchen"),
    ("  Straßlach ", "strasslach"),
    ("Nußdorf am Inn", "nussdorf am inn"),
    ("St. Englmar", "st englmar"),
])
def test_name_key(raw, key):
    assert name_key(raw) == key


def place(pid, name, district="LK", routes=1):
    return Place(id=pid, name=name, name_key=name_key(name), admin_district=district, admin_region="R",
                 lat=48.0, lon=11.0, accuracy_m=100, route_count=routes)


PLACES = [
    place(1, "Garmisch-Partenkirchen", routes=6),
    place(2, "Partenstein", routes=11),
    place(3, "Au", "Landkreis Freising", routes=2),
    place(4, "Au", "Landkreis Rosenheim", routes=5),
    place(5, "Aurach", routes=3),
    place(6, "München", "München", routes=22),
    place(7, "Münchberg", routes=3),
]


def test_rank_exact_before_prefix_and_more_routes_first():
    out = rank_places(PLACES, "au", limit=5)
    assert [(m.place.id, m.match) for m in out] == [(4, "exact"), (3, "exact"), (5, "prefix")]


def test_rank_prefix_umlaut_spelling_and_word_prefix():
    assert [m.place.name for m in rank_places(PLACES, "muench")] == ["München", "Münchberg"]
    out = rank_places(PLACES, "partenk")
    assert (out[0].place.name, out[0].match) == ("Garmisch-Partenkirchen", "word")


def test_rank_fuzzy_for_typos_only_when_needed():
    out = rank_places(PLACES, "Garmish")
    assert (out[0].place.name, out[0].match) == ("Garmisch-Partenkirchen", "fuzzy")
    assert 0.8 <= out[0].score < 1.0
    assert rank_places(PLACES, "zz") == [] and rank_places(PLACES, "   ") == []
    assert rank_places(PLACES, "xyzxyz") == []


def test_build_places_uses_the_start_closest_to_the_locality():
    records = [
        {"id": 2, "locality": "Au", "admin_district": "A", "admin_region": "R", "locality_dist_m": 900,
         "start_lat": 1.0, "start_lon": 1.0},
        {"id": 3, "locality": "Au", "admin_district": "A", "admin_region": "R", "locality_dist_m": 100,
         "start_lat": 2.0, "start_lon": 2.0},
        {"id": 1, "locality": "Au", "admin_district": "B", "admin_region": "R", "locality_dist_m": 50,
         "start_lat": 3.0, "start_lon": 3.0},
    ]
    places = build_places(records)
    assert [(p["admin_district"], p["lat"], p["accuracy_m"], p["route_count"]) for p in places] == [
        ("A", 2.0, 100, 2), ("B", 3.0, 50, 1),
    ]


# --- API -------------------------------------------------------------------------------------------


@pytest.fixture
def osm_client(make_client, osm_db_copy):
    return make_client(osm_db_copy)


def test_places_endpoint(osm_client):
    body = osm_client.get("/api/places", params={"q": "grain"}).json()
    assert body["total"] >= 1
    top = body["places"][0]
    assert top["name"] == "Grainau" and top["match"] == "prefix"
    assert top["admin_district"] == "Landkreis Garmisch-Partenkirchen"
    assert 47 < top["lat"] < 48 and 10 < top["lon"] < 12 and top["route_count"] >= 1
    assert osm_client.get("/api/places", params={"q": "Grainau"}).json()["places"][0]["match"] == "exact"
    assert osm_client.get("/api/places", params={"q": "qqqqqq"}).json() == {"places": [], "total": 0}
    assert osm_client.get("/api/places", params={"q": ""}).status_code == 422
    assert osm_client.get("/api/places", params={"q": "a", "limit": 51}).status_code == 422


def test_places_on_full_catalogue_find_big_towns(make_client, full_osm_db):
    client = make_client(full_osm_db)
    for typed, expected in [("Muenchen", "München"), ("garmisch", "Garmisch-Partenkirchen"),
                            ("Tegernsee", "Tegernsee"), ("wurzburg", "Würzburg")]:
        assert client.get("/api/places", params={"q": typed}).json()["places"][0]["name"] == expected


def test_places_empty_for_legacy_catalogue(make_client, seeded_db_copy):
    client = make_client(seeded_db_copy)
    assert client.get("/api/places", params={"q": "Bamberg"}).json()["total"] == 0


def test_profile_without_slots_is_unchanged(osm_client):
    resp = osm_client.post("/api/profiles", json=ANSWERS)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["profile"]["home"] is None and body["profile"]["intent"] is None
    assert body["profile"]["travel_radius_km"] is None and body["profile"]["distance_cap_km"] is None
    # the user vector does not depend on the slots
    with_slots = osm_client.post("/api/profiles", json={**ANSWERS, "intent": "lake", "distance_cap_km": 9}).json()
    assert with_slots["user_vector"] == body["user_vector"]
    assert json.loads(with_slots["profile"]["user_vector_json"]) == body["user_vector"]


def test_profile_slots_round_trip(osm_client):
    payload = {**ANSWERS, "home": {"name": "Grainau", "lat": 47.476, "lon": 10.99}, "travel_radius_km": 30,
               "distance_cap_km": 12.5, "intent": "  quiet lake loop\n   with a hut  "}
    resp = osm_client.post("/api/profiles", json=payload)
    assert resp.status_code == 201, resp.text
    pid = resp.json()["id"]
    got = osm_client.get(f"/api/profiles/{pid}").json()
    assert got["home"] == {"name": "Grainau", "lat": 47.476, "lon": 10.99}
    assert (got["travel_radius_km"], got["distance_cap_km"]) == (30.0, 12.5)
    assert got["intent"] == "quiet lake loop with a hut"  # whitespace normalised

    # partial update: only the given slot changes
    got = osm_client.patch(f"/api/profiles/{pid}", json={"distance_cap_km": 20}).json()
    assert got["distance_cap_km"] == 20.0 and got["travel_radius_km"] == 30.0 and got["home"]["name"] == "Grainau"
    # explicit nulls clear
    got = osm_client.patch(f"/api/profiles/{pid}", json={"home": None, "intent": None}).json()
    assert got["home"] is None and got["intent"] is None and got["distance_cap_km"] == 20.0
    # blank intent is stored as no intent
    assert osm_client.patch(f"/api/profiles/{pid}", json={"intent": "   "}).json()["intent"] is None
    # legacy fields still update
    assert osm_client.patch(f"/api/profiles/{pid}", json={"total_xp": 5}).json()["total_xp"] == 5


def test_home_by_name_is_resolved_from_the_catalogue(osm_client):
    resp = osm_client.post("/api/profiles", json={**ANSWERS, "home": {"name": "grainau"}})
    assert resp.status_code == 201, resp.text
    home = resp.json()["profile"]["home"]
    assert home["name"] == "Grainau" and 47 < home["lat"] < 48

    resp = osm_client.post("/api/profiles", json={**ANSWERS, "home": {"name": "Atlantis"}})
    assert resp.status_code == 422 and "Atlantis" in resp.json()["detail"]
    pid = osm_client.post("/api/profiles", json=ANSWERS).json()["id"]
    resp = osm_client.patch(f"/api/profiles/{pid}", json={"home": {"name": "Atlantis"}, "total_xp": 9})
    assert resp.status_code == 422
    assert osm_client.get(f"/api/profiles/{pid}").json()["total_xp"] == 0  # nothing applied


@pytest.mark.parametrize("bad", [
    {"travel_radius_km": 0},
    {"distance_cap_km": -3},
    {"home": {"name": "X", "lat": 48.0}},
    {"home": {"name": "X", "lat": 95, "lon": 11}},
    {"home": {"name": ""}},
    {"intent": "x" * 1001},
])
def test_profile_slot_validation(osm_client, bad):
    assert osm_client.post("/api/profiles", json={**ANSWERS, **bad}).status_code == 422
