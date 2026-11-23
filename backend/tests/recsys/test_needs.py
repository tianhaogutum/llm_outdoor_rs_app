"""Stated needs (iteration 2): extraction (positive, negated, German, mixed), verification, relaxation, the
explanation and the pipeline modes, on hand-built catalogues with the mock provider (no network). No
assertion depends on a request id or a seed: lists are checked by the needs they meet, not by their order."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.llm import LLMClient, LLMConfig
from app.recsys import Catalog, UserQuery, build_system
from app.recsys.explain import RecommendationExplanation, explain
from app.recsys.needs import (
    NeedsConfig,
    apply_needs,
    check_route,
    demote_key,
    extract_needs,
    parse_slot_need,
)
from tests.recsys.factory import route_record


def codes(text, category="hiking", slots=()):
    return extract_needs(text, category, slots).codes()


def need(text, code, category="hiking"):
    return next(n for n in extract_needs(text, category).needs if n.code == code)


# ------------------------------------------------------------------------------------------ extraction
@pytest.mark.parametrize(
    "text,expected",
    [
        ("An easy stroller-friendly walk with a playground", ["stroller"]),
        ("We'll have a pram with us", ["stroller"]),
        ("A walk without any steps, please", ["no_steps"]),
        ("Something step-free for my wheelchair-using father", ["no_steps"]),
        ("A quiet forest walk reachable by train from Schweinfurt", ["train"]),
        ("We travel by public transport", ["public_transport"]),
        ("We don't have a car", ["public_transport"]),
        ("A loop around the lake", ["loop"]),
        ("A one-way hike from village to village", ["one_way"]),
        ("A relaxed four-day hike", ["days"]),
        ("A multi-day tour", ["days"]),
        ("An easy walk of around 8 km", ["length"]),
        ("No more than about 10 km, please", ["max_length"]),
        ("About two hours of walking", ["duration"]),
        ("A lake and a castle", []),
    ],
)
def test_english_needs(text, expected):
    assert codes(text) == expected


def test_mtb_only_on_the_cycling_tab():
    assert codes("A challenging mountain bike ride on singletrails", "cycling") == ["mtb"]
    sn = extract_needs("A mountain bike ride", "hiking")
    assert sn.codes() == [] and sn.ignored[0]["why"] == "not the cycling tab"


@pytest.mark.parametrize(
    "text,absent",
    [
        ("No stroller needed, the kids walk", "stroller"),
        ("We'll bring a baby carrier instead of a pram", "stroller"),
        ("It doesn't have to be a loop", "loop"),
        ("A loop is not necessary", "loop"),
        ("Not by train, we drive", "train"),
        ("ohne Kinderwagen", "stroller"),
        ("Es muss kein Rundweg sein", "loop"),
        ("A day hike, not a multi-day trip", "days"),
    ],
)
def test_negated_mentions_are_not_needs(text, absent):
    sn = extract_needs(text, "hiking")
    assert absent not in sn.codes()
    assert any(i["need"] == absent for i in sn.ignored)


def test_negation_scope_stops_at_a_new_phrase():
    # "no" negates "steep climbs", not the train further on
    assert codes("A walk with no steep climbs reachable by train") == ["train"]
    assert codes("No busy roads, and a loop please") == ["loop"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Eine Wanderung mit Kinderwagen", ["stroller"]),
        ("Barrierefrei, keine Treppen", ["no_steps"]),
        ("Mit dem Zug erreichbar, eine Rundwanderung", ["train", "loop"]),
        ("Eine dreitägige Wanderung", ["days"]),
        ("Eine drei-tägige Wanderung", ["days"]),
        ("Mehrtägige Radtour", ["days"]),
        ("Eine Streckenwanderung von etwa 12 km", ["one_way", "length"]),
        ("Höchstens 15 km", ["max_length"]),
    ],
)
def test_german_needs(text, expected):
    assert codes(text) == expected


def test_mixed_language_and_several_needs():
    sn = extract_needs("A Rundweg with the Kinderwagen, reachable by train, around 6 km", "hiking")
    assert sn.codes() == ["stroller", "train", "loop", "length"]


def test_lengths_and_days_are_read():
    n = need("An easy walk of around 8 km", "length")
    assert (n.p("lo"), n.p("hi")) == (4.0, 12.0)
    n = need("About 30–70 km on forest trails", "length", "cycling")
    assert (n.p("lo"), n.p("hi")) == (24.0, 84.0)
    n = need("no more than about 10 km", "max_length")
    assert n.p("hi") == pytest.approx(11.0)
    assert need("A four-day one-way cycling tour", "days", "cycling").p("days") == 4
    assert need("Eine dreitägige Wanderung", "days").p("days") == 3
    assert need("around 10 km of walking", "length").p("stated") == 10
    assert need("a long weekend of hiking", "days").p("days") == 3
    # distances to the start are not route lengths
    assert codes("Within about 40 km of Neustadt") == []
    assert codes("A short drive from Munich") == []
    assert need("Easy short walk by a lake", "length").p("hi") == 8.0
    assert need("A half-day ride", "duration", "cycling").p("hi_h") == 5.0


def test_loop_and_one_way_together_cancel():
    sn = extract_needs("A loop or a one-way route, either is fine", "hiking")
    assert "loop" not in sn.codes() and "one_way" not in sn.codes()


def test_slot_needs_feed_the_same_stage():
    assert parse_slot_need("stroller").source == "slot"
    assert parse_slot_need("days:3").p("days") == 3
    assert parse_slot_need("duration_h:2").p("hi_h") == 3.0
    assert parse_slot_need("nonsense") is None and parse_slot_need("days:x") is None
    assert codes(None, "cycling", ("mtb", "train")) == ["train", "mtb"]
    assert codes("A lake walk", "hiking", ("mtb",)) == []  # an MTB slot means nothing on the hiking tab


def test_extraction_is_fast():
    import time

    text = "An easy stroller-friendly loop of around 8 km reachable by train, no more than 3 hours " * 3
    t0 = time.perf_counter()
    for _ in range(200):
        extract_needs(text, "hiking")
    assert (time.perf_counter() - t0) / 200 < 0.005


# ------------------------------------------------------------------------------------------ verification
def _route(i, *, steps=0.0, ok_share=0.9, ascent=100, loop=True, length_m=6000, activity="hiking",
           category="Hiking trail", tags=("Lake",), paved=0.8, station=None):  # fmt: skip
    r = route_record(i, category=category, activity=activity, ascent_m=ascent, is_loop=loop, length_m=length_m,
                     tags=tags)  # fmt: skip
    r["features"]["profile"].update(steps_m=steps, stroller_ok_share=ok_share, paved_share=paved)
    if station:
        r["features"]["near_start"] = {"train_station": [{"name": station, "dist_m": 400}]}
    return r


@pytest.fixture(scope="module")
def small():
    recs = [
        _route(1, steps=0, ok_share=0.94, ascent=120, tags=("Lake", "Train access"), station="Forchheim"),
        _route(2, steps=40, ok_share=0.9, ascent=120),  # steps
        _route(3, steps=0, ok_share=0.3, ascent=250),  # rough surface, more climbing: relaxed level only
        _route(4, steps=0, ok_share=0.2, ascent=600),  # no steps only
        _route(5, steps=0, ok_share=0.8, ascent=150, loop=False),
    ]
    return Catalog.from_records(recs)


def test_stroller_levels_and_the_explanation_numbers(small):
    sn = extract_needs("We have a stroller", "hiking")
    n = sn.needs[0]
    got = {int(small.ids[i]): check_route(small, i, n, "hiking", 0) for i in range(len(small))}
    assert got[1]["met_level"] == 0 and got[1]["ok"]
    assert "no steps" in got[1]["fact"] and "94% of the route on stroller-suitable surface" in got[1]["fact"]
    assert got[2]["met_level"] is None and "40 m of steps" in got[2]["fact"]
    assert got[3]["met_level"] == 1 and not got[3]["ok"]
    assert got[4]["met_level"] == 2
    assert check_route(small, 0, need("reachable by train", "train"), "hiking", 0)["fact"] == (
        "train station Forchheim 400 m from the start"
    )


def test_relaxation_keeps_the_floor_and_reports_why(small):
    sn = extract_needs("A loop with a stroller", "hiking")
    elig = np.arange(len(small))
    res = apply_needs(small, elig, sn, "hiking", NeedsConfig(floor=1))
    assert {int(small.ids[i]) for i in res.indices} == {1}
    assert res.info["applied"] == ["stroller", "loop"] and not res.info["relax_steps"]
    # floor 3: the loop goes first (it is relaxed before the stroller), then the stroller is relaxed step by step
    res = apply_needs(small, elig, sn, "hiking", NeedsConfig(floor=3))
    assert len(res.indices) >= 3
    assert [s["need"] for s in res.info["relax_steps"]][0] == "loop"
    assert all("floor 3" in s["why"] for s in res.info["relax_steps"])
    assert res.levels["loop"] is None and res.levels["stroller"] in (1, 2)
    assert {int(small.ids[i]) for i in res.strict} == {1}
    # never below the floor, even when every need has to go
    res = apply_needs(small, elig, sn, "hiking", NeedsConfig(floor=5))
    assert len(res.indices) == 5 and res.info["dropped"]


def test_demotion_key_orders_by_how_badly_needs_are_missed(small):
    sn = extract_needs("A loop with a stroller", "hiking")
    keys = {int(small.ids[i]): demote_key(small, i, sn, "hiking") for i in range(len(small))}
    assert keys[1] == 0.0 and keys[2] == 1.0 and keys[3] < keys[4] < keys[2]


# ------------------------------------------------------------------------------------------ pipeline modes
def _catalog(n=60):
    rng = np.random.default_rng(3)
    recs = []
    for i in range(n):
        stroller_ok = i % 3 == 0
        recs.append(
            _route(
                100 + i,
                steps=0 if stroller_ok else 30,
                ok_share=0.9 if stroller_ok else 0.2,
                ascent=int(rng.uniform(50, 150)),
                loop=bool(i % 2),
                length_m=int(rng.uniform(3, 12) * 1000),
                tags=("Lake", "Playground") if i % 4 else ("Lake",),
            )
        )
    return Catalog.from_records(recs)


@pytest.fixture(scope="module")
def cat60():
    return _catalog()


@pytest.fixture
def llm(tmp_path):
    return LLMClient(LLMConfig(cache_dir=tmp_path / "cache"))


def _q(intent, **kw):
    base = dict(request_id="needs-test", user_vector={"difficulty_range": [0, 2], "min_distance_km": 2,
                "max_distance_km": 15, "preferred_tags": []}, category="hiking", home=(48.0, 11.0),
                max_travel_km=60.0, max_distance_km=14.0, intent=intent)  # fmt: skip
    base.update(kw)
    return UserQuery(**base)


@pytest.mark.parametrize("name", ["recsys_serve_needs", "recsys_serve_needs_promote", "recsys_serve_needs_demote"])
def test_modes_return_stroller_routes_and_explain_them(name, cat60, llm):
    rec = build_system(name, cat60, llm=llm).recommend(_q("A lake walk with our stroller and a playground"), 10)
    ex = explain(rec)
    assert isinstance(ex, RecommendationExplanation) and ex.schema_version == "recsys.explanation/v1.1"
    doc = json.loads(ex.model_dump_json(by_alias=True))
    non_explore = [r for r in doc["routes"] if r["stages"].get("explore") is None]
    checks = [r["stages"]["needs"]["checks"][0] for r in non_explore]
    assert all(c["need"] == "stroller" and c["fact"] for c in checks)
    # 20 of the 60 routes meet the need strictly, more than the floor: every non-exploration pick meets it
    assert all(c["met_level"] == 0 for c in checks)
    stage = next(t for t in rec.trace if t.stage == "needs")
    if name == "recsys_serve_needs_demote":
        assert stage.n_in == stage.n_out and stage.info["mode"] == "demote"
    else:
        assert stage.n_out == 20 and stage.info["applied"] == ["stroller"]


def test_filter_relaxes_instead_of_returning_a_handful(cat60, llm):
    # a loop + stroller + "at least 11 km" leaves almost nothing: the length goes first, then the loop
    rec = build_system("recsys_serve_needs", cat60, llm=llm).recommend(
        _q("A loop with the stroller, at least 12 km"), 10
    )
    stage = next(t for t in rec.trace if t.stage == "needs")
    assert stage.n_out >= 20 and len(rec.route_ids) == 10
    relaxed_first = stage.info["relax_steps"][0]["need"]
    assert relaxed_first == "length"
    assert "stroller" in stage.info["applied"]


def test_no_stated_need_means_no_stage_and_the_served_list(cat60, llm):
    q = _q("A lake walk with a playground")
    a = build_system("recsys_serve", cat60, llm=llm).recommend(q, 10)
    for name in ("recsys_serve_needs", "recsys_serve_needs_promote", "recsys_serve_needs_demote"):
        b = build_system(name, cat60, llm=llm).recommend(q, 10)
        assert "needs" not in {t.stage for t in b.trace} and b.route_ids == a.route_ids


def test_slot_needs_work_without_an_intent(cat60, llm):
    rec = build_system("recsys_serve_needs", cat60, llm=llm).recommend(_q(None, needs=("stroller",)), 10)
    stage = next(t for t in rec.trace if t.stage == "needs")
    assert stage.n_out == 20 and rec.trace[0].info.get("fallback") == "no_intent"
    for rid in rec.route_ids:
        c = rec.contributions(rid)["needs"]["checks"][0]
        assert c["source"] == "slot" and c["met_level"] == 0


def test_config_of_earlier_systems_is_unchanged(cat60, llm):
    assert "needs" not in build_system("recsys_serve", cat60, llm=llm).config()["pipeline"]
    cfg = build_system("recsys_serve_needs", cat60, llm=llm).config()["pipeline"]["needs"]
    assert cfg == {"mode": "filter", "floor": 20, "version": "needs/v1"}
