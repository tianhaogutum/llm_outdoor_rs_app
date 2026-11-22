"""Unit tests for the v0_rules scoring functions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services import recommendation_service as rs


def approx(x: float):
    return pytest.approx(x, rel=1e-12, abs=1e-12)


# --- difficulty -------------------------------------------------------------------------------------


def test_difficulty_scale_constants():
    assert (rs.DIFFICULTY_MIN, rs.DIFFICULTY_MAX) == (0, 3)
    assert rs.default_difficulty_range() == [0, 3]
    # a fresh list every time, so callers can mutate it safely
    assert rs.default_difficulty_range() is not rs.default_difficulty_range()


@pytest.mark.parametrize("route_difficulty", [0, 1, 2])
def test_difficulty_inside_range_scores_one(route_difficulty):
    assert rs.calculate_difficulty_score([0, 2], route_difficulty) == 1.0


def test_difficulty_range_bounds_are_inclusive():
    assert rs.calculate_difficulty_score([1, 2], 1) == 1.0
    assert rs.calculate_difficulty_score([1, 2], 2) == 1.0


@pytest.mark.parametrize(
    ("user_range", "route_difficulty", "expected"),
    [
        ([1, 2], 0, 0.5),  # one below
        ([2, 3], 0, 0.25),  # two below
        ([0, 1], 2, 0.5),  # one above
        ([0, 1], 3, 0.25),  # two above
        ([0.5, 1.5], 2, 0.5**0.5),  # fractional range after feedback
        ([0.5, 1.5], 0, 0.5**0.5),
    ],
)
def test_difficulty_outside_range_halves_per_level(user_range, route_difficulty, expected):
    assert rs.calculate_difficulty_score(user_range, route_difficulty) == approx(expected)


@pytest.mark.parametrize("user_range", [None, [], [1]])
def test_difficulty_without_preference_is_neutral(user_range):
    assert rs.calculate_difficulty_score(user_range, 2) == 0.5


# --- distance ---------------------------------------------------------------------------------------


def test_distance_inside_range_scores_one():
    assert rs.calculate_distance_score(5.0, 20.0, 5.0) == 1.0
    assert rs.calculate_distance_score(5.0, 20.0, 12.3) == 1.0
    assert rs.calculate_distance_score(5.0, 20.0, 20.0) == 1.0


def test_distance_missing_length_scores_low():
    assert rs.calculate_distance_score(0.0, 8.0, 0.0) == 0.3


def test_distance_below_range_decays_relative_to_max():
    # (5 - 1) / 20 = 0.2 -> 0.7 ** 0.2
    assert rs.calculate_distance_score(5.0, 20.0, 1.0) == approx(0.7**0.2)


def test_distance_above_range_decays_relative_to_max():
    # (30 - 20) / 20 = 0.5 -> 0.7 ** 0.5
    assert rs.calculate_distance_score(5.0, 20.0, 30.0) == approx(0.7**0.5)
    # twice the max -> 0.7
    assert rs.calculate_distance_score(0.0, 8.0, 16.0) == approx(0.7)


def test_distance_score_decreases_with_distance_from_range():
    scores = [rs.calculate_distance_score(0.0, 10.0, km) for km in (10.0, 15.0, 20.0, 40.0, 100.0)]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


# --- tags -------------------------------------------------------------------------------------------


def test_tags_jaccard():
    assert rs.calculate_tag_score(["scenic", "flora"], ["scenic", "round trip"]) == approx(1 / 3)
    assert rs.calculate_tag_score(["scenic"], ["scenic"]) == 1.0
    assert rs.calculate_tag_score(["scenic"], ["flora"]) == 0.0


def test_tags_are_case_insensitive_and_deduplicated():
    assert rs.calculate_tag_score(["Scenic", "scenic"], ["SCENIC"]) == 1.0


def test_tags_empty_cases():
    assert rs.calculate_tag_score([], []) == 0.5
    assert rs.calculate_tag_score(["scenic"], []) == 0.2
    assert rs.calculate_tag_score([], ["scenic"]) == 0.2


# --- combined score ---------------------------------------------------------------------------------

USER = {
    "difficulty_range": [1, 2],
    "min_distance_km": 5.0,
    "max_distance_km": 20.0,
    "preferred_tags": ["scenic", "flora"],
}


def test_score_weights_sum_to_one():
    assert sum(rs.SCORE_WEIGHTS.values()) == approx(1.0)
    assert rs.SCORE_WEIGHTS == {"difficulty": 0.4, "distance": 0.3, "tags": 0.3}


def test_cbf_score_is_weighted_sum_and_breakdown_is_consistent():
    route = {"difficulty": 0, "length_km": 30.0, "tags": ["scenic", "round trip"]}
    score, breakdown = rs.calculate_cbf_score(USER, route)

    d, k, t = 0.5, 0.7**0.5, 1 / 3
    assert breakdown["difficulty"]["score"] == approx(d)
    assert breakdown["distance"]["score"] == approx(k)
    assert breakdown["tags"]["score"] == approx(t)
    assert score == approx(0.4 * d + 0.3 * k + 0.3 * t)
    assert breakdown["total"] == score
    for name in ("difficulty", "distance", "tags"):
        part = breakdown[name]
        assert part["weight"] == rs.SCORE_WEIGHTS[name]
        assert part["weighted_score"] == approx(part["weight"] * part["score"])
    assert sum(breakdown[n]["weighted_score"] for n in ("difficulty", "distance", "tags")) == approx(score)
    assert breakdown["difficulty"]["user_range"] == [1, 2]
    assert breakdown["difficulty"]["route_value"] == 0
    assert breakdown["distance"]["user_range"] == [5.0, 20.0]
    assert breakdown["distance"]["route_value"] == 30.0
    assert breakdown["tags"]["user_tags"] == ["scenic", "flora"]
    assert breakdown["tags"]["route_tags"] == ["scenic", "round trip"]


def test_cbf_perfect_match_scores_one():
    score, _ = rs.calculate_cbf_score(USER, {"difficulty": 1, "length_km": 10.0, "tags": ["scenic", "flora"]})
    assert score == approx(1.0)


def test_cbf_defaults_for_empty_user_vector():
    score, breakdown = rs.calculate_cbf_score({}, {"difficulty": 3, "length_km": 150.0, "tags": []})
    assert breakdown["difficulty"]["user_range"] == [0, 3]
    assert breakdown["difficulty"]["score"] == 1.0
    assert breakdown["distance"]["user_range"] == [0.0, 100.0]
    assert breakdown["distance"]["score"] == approx(0.7**0.5)
    assert breakdown["tags"]["score"] == 0.5
    assert score == approx(0.4 + 0.3 * 0.7**0.5 + 0.15)


# --- route vectors ----------------------------------------------------------------------------------


def _orm_like(**kw):
    base = {"difficulty": 2, "length_meters": 12345.0, "tags_json": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_extract_route_vector_basic():
    vec = rs.extract_route_vector(_orm_like(tags_json=json.dumps(["Round trip", "Scenic"])))
    assert vec == {"difficulty": 2, "length_km": 12.345, "tags": ["round trip", "scenic"]}


def test_extract_route_vector_tag_dicts_and_junk():
    tags = json.dumps(["Flora", {"name": "Fauna"}, {"id": 3}, 7])
    assert rs.extract_route_vector(_orm_like(tags_json=tags))["tags"] == ["flora", "fauna"]


@pytest.mark.parametrize("tags_json", [None, "", "not json", "[]"])
def test_extract_route_vector_missing_or_bad_tags(tags_json):
    assert rs.extract_route_vector(_orm_like(tags_json=tags_json))["tags"] == []


def test_extract_route_vector_missing_values():
    vec = rs.extract_route_vector(_orm_like(difficulty=None, length_meters=None))
    assert vec["difficulty"] == rs.DIFFICULTY_MIN
    assert vec["length_km"] == 0.0
    assert rs.extract_route_vector(_orm_like(length_meters=0))["length_km"] == 0.0


def test_route_vector_from_record_matches_orm_path(seed_records):
    for record in seed_records:
        orm = SimpleNamespace(**{k: record.get(k) for k in ("difficulty", "length_meters", "tags_json")})
        assert rs.route_vector_from_record(record) == rs.extract_route_vector(orm)


def test_route_vector_from_record_accepts_tag_lists():
    assert rs.route_vector_from_record({"tags": ["Scenic"], "length_meters": 1000})["tags"] == ["scenic"]
    assert rs.route_vector_from_record({"tags_json": ["Scenic"]})["tags"] == ["scenic"]


def test_routes_in_category_mirrors_sql_filter(seed_records):
    assert rs.routes_in_category(seed_records, None) == seed_records
    assert rs.routes_in_category(seed_records, "unknown") == seed_records
    counts = {c: len(rs.routes_in_category(seed_records, c)) for c in rs.CATEGORY_MAPPING}
    assert counts == {"running": 40, "hiking": 34, "cycling": 37}
    assert sum(counts.values()) == len(seed_records) == 111
