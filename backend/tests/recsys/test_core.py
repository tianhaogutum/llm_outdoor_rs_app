"""app.recsys: catalogue, hard constraints, registry, random and v0_rules systems (synthetic data)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from app.recsys import (
    CATEGORY_TABS,
    Catalog,
    HardConstraints,
    Recommendation,
    StageTrace,
    UserQuery,
    build_system,
    haversine_km,
    list_systems,
    register_system,
)
from app.services.recommendation_service import CATEGORY_MAPPING, rank_routes_v0
from tests.recsys.factory import route_record, synthetic_catalog

UV = {"difficulty_range": [0, 1], "max_distance_km": 8.0, "min_distance_km": 0.0, "preferred_tags": ["scenic"]}


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return synthetic_catalog()


def test_category_tabs_match_v0_mapping():
    assert {k: list(v) for k, v in CATEGORY_TABS.items()} == CATEGORY_MAPPING


def test_haversine_known_distances():
    assert haversine_km(48.0, 11.0, 48.0, 11.0) == 0.0
    # one degree of latitude on the mean-radius sphere
    assert haversine_km(48.0, 11.0, 49.0, 11.0) == pytest.approx(111.195, abs=0.01)
    # Munich Marienplatz -> Nuremberg Hauptmarkt, about 150 km
    assert haversine_km(48.1374, 11.5755, 49.4540, 11.0775) == pytest.approx(150.3, abs=1.0)
    v = haversine_km(48.0, 11.0, np.array([48.0, 49.0]), np.array([11.0, 11.0]))
    assert v.shape == (2,) and v[1] == pytest.approx(111.195, abs=0.01)


def test_catalog_arrays_aligned_and_sorted(cat):
    assert list(cat.ids) == sorted(cat.ids)
    r = cat.get(int(cat.ids[5]))
    i = cat.index_of(r.id)
    assert cat.length_km[i] == pytest.approx(r.length_km)
    assert cat.category_name[i] == r.category_name
    assert r.v0_vector()["tags"] == [t.lower() for t in r.tags]
    with pytest.raises(ValueError):
        Catalog.from_records([route_record(1), route_record(1)])


def test_hard_constraints_single_and_vectorised_agree(cat):
    hc = HardConstraints(category="hiking", home=(48.0, 11.0), max_travel_km=15.0, max_distance_km=10.0)
    mask = hc.mask(cat)
    for i, r in enumerate(cat.routes):
        assert mask[i] == hc.ok(r)
    assert 0 < mask.sum() < len(cat)
    near_long = route_record(1, lat=48.0, lon=11.0, length_m=12000)
    far = route_record(2, lat=49.0, lon=11.0, length_m=5000, category="Cycling", activity="cycling")
    c = Catalog.from_records([near_long, far])
    assert hc.violations(c.get(1)) == ["distance_cap"]
    assert hc.violations(c.get(2)) == ["category", "radius"]
    assert HardConstraints().violations(c.get(2)) == []  # unset slots are not applied


def test_hard_constraints_from_query():
    q = UserQuery("r", UV, "cycling", home=(48.1, 11.2), max_travel_km=30, max_distance_km=50)
    hc = HardConstraints.from_query(q)
    assert (hc.category, hc.home, hc.max_travel_km, hc.max_distance_km) == ("cycling", (48.1, 11.2), 30, 50)


def test_user_query_has_no_latent_fields():
    names = {f.name for f in dataclasses.fields(UserQuery)}
    assert names == {
        "request_id",
        "user_vector",
        "category",
        "home",
        "max_travel_km",
        "max_distance_km",
        "intent",
        "feedback",
        "needs",
    }


def test_registry():
    assert {"random", "v0_rules"} <= set(list_systems())
    with pytest.raises(KeyError):
        build_system("no-such-system", synthetic_catalog(10))
    with pytest.raises(ValueError):
        register_system("random")(lambda catalog: None)


def test_random_is_seeded_per_request_and_stays_in_tab(cat):
    s = build_system("random", cat)
    q1 = UserQuery("P001", UV, "hiking")
    a, b = s.recommend(q1, 10), s.recommend(q1, 10)
    assert a.route_ids == b.route_ids and len(a.route_ids) == 10
    assert s.recommend(UserQuery("P002", UV, "hiking"), 10).route_ids != a.route_ids
    assert all(cat.get(r).category_name in CATEGORY_TABS["hiking"] for r in a.route_ids)
    assert build_system("random", cat, seed=1).recommend(q1, 10).route_ids != a.route_ids


def test_v0_rules_equals_rank_routes_v0_and_ignores_slots(cat):
    s = build_system("v0_rules", cat)
    q = UserQuery("x", UV, "hiking")
    rec = s.recommend(q, 20)
    tab = [r for r in cat.routes if r.category_name in CATEGORY_TABS["hiking"]]
    expected = rank_routes_v0(UV, [{"id": r.id, **r.v0_vector()} for r in tab], limit=20)
    assert rec.route_ids == [e.route_id for e in expected]
    assert rec.scores == [round(e.score, 6) for e in expected]
    q2 = UserQuery("y", UV, "hiking", home=(49.0, 12.0), max_travel_km=5, max_distance_km=3, intent="lake")
    assert s.recommend(q2, 20).route_ids == rec.route_ids  # v0 sees only the vector and the tab
    first = rec.route_ids[0]
    assert set(rec.contributions(first)["v0_rules"]) == {"difficulty", "distance", "tags"}


def test_recommendation_validation_and_trace_summary():
    with pytest.raises(ValueError):
        Recommendation("r", "s", [1, 1], [0.5, 0.4])
    with pytest.raises(ValueError):
        Recommendation("r", "s", [1], [])
    t = StageTrace("stage", n_in=5, n_out=2, local_ms=3.0, upstream_latency_ms=10, upstream_cost_usd=0.001)
    rec = Recommendation("r", "s", [1, 2], [0.9, 0.1], [t, StageTrace("b", upstream_cost_usd=0.002)])
    assert "local_ms" not in t.summary()
    assert rec.upstream_cost_usd == pytest.approx(0.003) and rec.local_ms == 3.0
