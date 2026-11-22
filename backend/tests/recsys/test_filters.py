"""Phase 3a hard filter: the same predicates as the metric, missing slots, tiny eligible sets, verdicts."""

from __future__ import annotations

import numpy as np
import pytest

from app.recsys import Catalog, HardConstraints, UserQuery, build_system, haversine_km
from app.recsys.filters import FilterConfig, apply_filters
from tests.recsys.factory import route_record, synthetic_catalog


@pytest.fixture(scope="module")
def cat():
    return synthetic_catalog(300, seed=3)


def _q(**kw) -> UserQuery:
    base = dict(
        request_id="r1",
        user_vector={"difficulty_range": [0, 2], "min_distance_km": 3, "max_distance_km": 15, "preferred_tags": []},
        category="hiking",
        home=(48.0, 11.0),
        max_travel_km=25.0,
        max_distance_km=12.0,
    )
    base.update(kw)
    return UserQuery(**base)


def test_filter_equals_constraint_mask(cat):
    q = _q()
    hc = HardConstraints.from_query(q)
    fr = apply_filters(cat, hc, FilterConfig(min_eligible=0))
    assert np.array_equal(fr.indices, np.flatnonzero(hc.mask(cat)))
    for i in fr.indices:
        assert hc.ok(cat.routes[i])
    assert fr.info["n_eligible"] == len(fr.indices) and fr.info["relax_steps"] == 0


def test_missing_slots_are_not_applied(cat):
    q = _q(home=None, max_distance_km=None)
    fr = apply_filters(cat, HardConstraints.from_query(q))
    assert fr.info["applied"] == {"category": True, "radius": False, "distance_cap": False}
    hiking = {"Hiking trail", "Theme trail"}
    assert {cat.routes[i].category_name for i in fr.indices} <= hiking
    assert len(fr.indices) == sum(r.category_name in hiking for r in cat)
    v = fr.verdict(cat, HardConstraints.from_query(q), int(fr.indices[0]))
    assert v["radius"]["applied"] is False and v["radius"]["distance_km"] is None
    assert v["distance_cap"]["applied"] is False and v["verdict"] == "pass"
    # no tab at all: every category
    fr2 = apply_filters(cat, HardConstraints.from_query(_q(category=None, home=None, max_distance_km=None)))
    assert len(fr2.indices) == len(cat)


def test_unknown_tab_gives_empty_set(cat):
    fr = apply_filters(cat, HardConstraints.from_query(_q(category="skiing")))
    assert len(fr.indices) == 0 and fr.info["short"]


def _ring_catalog() -> Catalog:
    # hiking routes at 5, 15, 25, ... km north of home; one long route near home; one cycling route at home
    recs = []
    for j in range(12):
        recs.append(route_record(100 + j, lat=48.0 + (5 + 10 * j) / 111.2, lon=11.0, length_m=6000))
    recs.append(route_record(200, lat=48.0, lon=11.0, length_m=40000))
    recs.append(route_record(201, lat=48.0, lon=11.0, category="Cycling", activity="cycling"))
    return Catalog.from_records(recs)


def test_tiny_set_relaxes_only_the_radius():
    cat = _ring_catalog()
    q = _q(max_travel_km=20.0, max_distance_km=10.0)
    hc = HardConstraints.from_query(q)
    fr = apply_filters(cat, hc, FilterConfig(min_eligible=3, relax_factor=1.25, relax_max_factor=2.0))
    ids = [int(cat.ids[i]) for i in fr.indices]
    # 20 km: 2 routes; 25 km: 3 routes -> one step
    assert fr.info["relax_steps"] == 1 and fr.radius_used_km == pytest.approx(25.0)
    assert ids == [100, 101, 102]
    assert 200 not in ids and 201 not in ids  # cap and category are never relaxed
    v = fr.verdict(cat, hc, cat.index_of(102))
    assert v["verdict"] == "pass_relaxed" and v["radius"]["relaxed"] and not v["radius"]["ok"]
    assert v["radius"]["limit_km"] == 20.0 and v["radius"]["limit_used_km"] == pytest.approx(25.0)
    d = haversine_km(48.0, 11.0, cat.get(102).start_lat, cat.get(102).start_lon)
    assert v["radius"]["distance_km"] == pytest.approx(d, abs=0.01)
    assert fr.verdict(cat, hc, cat.index_of(100))["verdict"] == "pass"


def test_relaxation_stops_at_the_maximum():
    cat = _ring_catalog()
    q = _q(max_travel_km=10.0, max_distance_km=10.0)
    fr = apply_filters(cat, HardConstraints.from_query(q), FilterConfig(min_eligible=8, relax_max_factor=2.0))
    assert fr.radius_used_km <= 20.0 + 1e-6
    assert len(fr.indices) == 2 and fr.info["short"]  # honest short list, never padded


def test_v1_filters_system_returns_only_feasible_routes(cat):
    q = _q()
    sysm = build_system("v1_filters", cat)
    rec = sysm.recommend(q, 10)
    hc = HardConstraints.from_query(q)
    assert rec.route_ids and all(hc.ok(cat.get(r)) for r in rec.route_ids)
    for rid in rec.route_ids:
        c = rec.contributions(rid)
        assert c["filter"]["verdict"] == "pass"
        assert c["filter"]["distance_cap"]["length_km"] <= 12.0
        assert set(c["v0_rules"]) >= {"difficulty", "distance", "tags", "total"}
    # same order as v0_rules restricted to the eligible set
    v0 = build_system("v0_rules", cat).recommend(q, len(cat))
    expected = [r for r in v0.route_ids if hc.ok(cat.get(r))][:10]
    assert rec.route_ids == expected
