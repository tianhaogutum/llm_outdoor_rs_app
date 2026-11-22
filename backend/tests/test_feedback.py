"""Feedback adjustment, penalty schedule, filter threshold and time decay of v0_rules."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.models.entities import ProfileFeedback
from app.services import recommendation_service as rs

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
ROUTE = {"difficulty": 1, "length_km": 10.0, "tags": ["scenic", "flora"]}
VECTORS = {1: ROUTE, 2: {"difficulty": 2, "length_km": 30.0, "tags": ["culture"]}}


def fb(reason: str, route_id: int = 1, days_ago: float | None = 0.0) -> dict:
    created_at = None if days_ago is None else NOW - timedelta(days=days_ago)
    return {"route_id": route_id, "reason": reason, "created_at": created_at}


def adjust(user_vector: dict, *entries, now: datetime = NOW) -> dict:
    return rs.adjust_user_vector_with_feedback(user_vector, list(entries), VECTORS, now=now)


def approx(x):
    return pytest.approx(x, rel=1e-12, abs=1e-12)


# --- time decay ---------------------------------------------------------------------------------------


def test_decay_zero_age_is_one():
    assert rs.calculate_time_decay_weight(0.0) == 1.0


def test_decay_is_a_true_half_life():
    hl = rs.TIME_DECAY_HALF_LIFE_DAYS
    assert rs.calculate_time_decay_weight(hl) == approx(0.5)
    assert rs.calculate_time_decay_weight(2 * hl) == approx(0.25)
    assert rs.calculate_time_decay_weight(10.0, half_life_days=10.0) == approx(0.5)


@pytest.mark.parametrize("days", [-0.001, -1.0, -365.0])
def test_decay_negative_age_is_one(days):
    assert rs.calculate_time_decay_weight(days) == 1.0


def test_decay_is_monotone():
    weights = [rs.calculate_time_decay_weight(d) for d in (0, 1, 10, 30, 90, 365)]
    assert weights == sorted(weights, reverse=True)
    assert all(0.0 < w <= 1.0 for w in weights)


def test_feedback_age_naive_is_treated_as_utc():
    naive = datetime(2026, 5, 2, 12, 0)  # what SQLite hands back
    assert rs.feedback_age_days(naive, NOW) == approx(30.0)
    assert rs.feedback_age_days(naive, NOW.replace(tzinfo=None)) == approx(30.0)


def test_feedback_age_aware_other_timezone():
    cest = timezone(timedelta(hours=2))
    created = datetime(2026, 5, 2, 14, 0, tzinfo=cest)  # == 12:00 UTC
    assert rs.feedback_age_days(created, NOW) == approx(30.0)


def test_feedback_age_future_and_missing():
    assert rs.feedback_age_days(NOW + timedelta(days=2), NOW) == approx(-2.0)
    assert rs.feedback_age_days(None, NOW) == 0.0


def test_feedback_age_defaults_to_current_time():
    just_now = datetime.now(UTC) - timedelta(seconds=1)
    assert 0.0 < rs.feedback_age_days(just_now) < 1.0


# --- adjustment: the four reasons ---------------------------------------------------------------------


def test_too_hard_shifts_range_down_by_half_a_level():
    out = adjust({"difficulty_range": [1, 2]}, fb("too-hard"))
    assert out["difficulty_range"] == [0.5, 1.5]


def test_too_hard_clamps_at_bottom_and_keeps_width_one():
    assert adjust({"difficulty_range": [0, 1]}, fb("too-hard"))["difficulty_range"] == [0, 1.0]
    assert adjust({"difficulty_range": [0.25, 1.0]}, fb("too-hard"))["difficulty_range"] == [0, 1.0]
    # repeated feedback cannot push below the scale
    out = adjust({"difficulty_range": [1, 2]}, *[fb("too-hard") for _ in range(6)])
    assert out["difficulty_range"] == [rs.DIFFICULTY_MIN, rs.DIFFICULTY_MIN + 1.0]


def test_too_easy_shifts_range_up_by_half_a_level():
    out = adjust({"difficulty_range": [1, 2]}, fb("too-easy"))
    assert out["difficulty_range"] == [1.5, 2.5]


def test_too_easy_clamps_at_top_and_keeps_width_one():
    assert adjust({"difficulty_range": [2, 3]}, fb("too-easy"))["difficulty_range"] == [2.0, 3]
    assert adjust({"difficulty_range": [2.2, 2.8]}, fb("too-easy"))["difficulty_range"] == [2.0, 3]
    out = adjust({"difficulty_range": [0, 1]}, *[fb("too-easy") for _ in range(8)])
    assert out["difficulty_range"] == [rs.DIFFICULTY_MAX - 1.0, rs.DIFFICULTY_MAX]


def test_too_far_shrinks_max_distance_by_ten_percent():
    out = adjust({"min_distance_km": 5.0, "max_distance_km": 20.0}, fb("too-far"))
    assert out["max_distance_km"] == approx(18.0)


def test_too_far_floor_is_min_plus_two_km():
    out = adjust({"min_distance_km": 5.0, "max_distance_km": 7.5}, fb("too-far"))
    assert out["max_distance_km"] == 7.0
    out = adjust({"min_distance_km": 10.0, "max_distance_km": 50.0}, *[fb("too-far") for _ in range(60)])
    assert out["max_distance_km"] == 12.0
    # without min_distance_km the floor is 0 + 2
    out = adjust({"max_distance_km": 2.1}, fb("too-far"))
    assert out["max_distance_km"] == 2.0


def test_not_interested_removes_route_tags_case_insensitively():
    out = adjust({"preferred_tags": ["Scenic", "culture", "flora"]}, fb("not-interested"))
    assert out["preferred_tags"] == ["culture"]


def test_unknown_reason_changes_nothing():
    vec = {"difficulty_range": [1, 2], "max_distance_km": 20.0, "preferred_tags": ["scenic"]}
    assert adjust(vec, fb("meh")) == vec


def test_feedback_for_unknown_route_is_ignored():
    out = adjust({"difficulty_range": [1, 2]}, fb("too-hard", route_id=999))
    assert out["difficulty_range"] == [1, 2]


def test_adjust_fills_defaults_and_does_not_mutate_input():
    vec = {"difficulty_range": [1, 2], "preferred_tags": ["scenic"]}
    original = copy.deepcopy(vec)
    out = adjust(vec, fb("too-hard"))
    assert vec == original
    assert out["max_distance_km"] == rs.DEFAULT_MAX_DISTANCE_KM
    assert adjust({}, fb("too-far"))["difficulty_range"] == [0, 3]


def test_invalid_range_falls_back_to_original_vector():
    vec = {"difficulty_range": [3, 1]}
    assert adjust(vec, fb("too-far"))["difficulty_range"] == [3, 1]


# --- adjustment with decay ----------------------------------------------------------------------------


def test_adjustment_uses_real_feedback_age():
    hl = rs.TIME_DECAY_HALF_LIFE_DAYS
    out = adjust({"difficulty_range": [1, 2]}, fb("too-hard", days_ago=hl))
    assert out["difficulty_range"] == [approx(0.75), approx(1.75)]
    out = adjust({"min_distance_km": 0.0, "max_distance_km": 20.0}, fb("too-far", days_ago=2 * hl))
    assert out["max_distance_km"] == approx(20.0 * (1 - 0.1 * 0.25))


def test_adjustment_without_timestamp_counts_as_fresh():
    out = adjust({"difficulty_range": [1, 2]}, fb("too-hard", days_ago=None))
    assert out["difficulty_range"] == [0.5, 1.5]


def test_adjustment_accepts_orm_feedback_with_naive_timestamp():
    entry = ProfileFeedback(route_id=1, reason="too-easy", demo_profile_id=1, created_at=datetime(2026, 5, 2, 12, 0))
    out = rs.adjust_user_vector_with_feedback({"difficulty_range": [1, 2]}, [entry], VECTORS, now=NOW)
    assert out["difficulty_range"] == [approx(1.25), approx(2.25)]


# --- penalty schedule and filter ----------------------------------------------------------------------


@pytest.mark.parametrize(("count", "expected"), [(0, 1.0), (1, 0.5), (2, 0.1), (3, 0.01), (4, 0.01), (7, 0.01)])
def test_feedback_penalty_schedule(count, expected):
    entries = [fb("too-hard", route_id=1) for _ in range(count)] + [fb("too-hard", route_id=2)]
    assert rs.calculate_feedback_penalty(1, entries) == expected


def test_feedback_penalty_counts_all_reasons_and_ignores_age():
    entries = [fb("too-hard", days_ago=0), fb("not-interested", days_ago=400)]
    assert rs.calculate_feedback_penalty(1, entries) == 0.1


def _ranked_ids(entries):
    routes = [{"id": rid, **vec} for rid, vec in VECTORS.items()]
    return {r.route_id: r for r in rs.rank_routes_v0({"difficulty_range": [1, 2]}, routes, entries, now=NOW)}


def test_filter_threshold():
    assert rs.FEEDBACK_FILTER_THRESHOLD == 4
    three = _ranked_ids([fb("meh") for _ in range(3)])
    assert 1 in three
    assert three[1].breakdown["feedback_penalty"] == 0.01
    assert three[1].breakdown["feedback_count"] == 3
    assert three[1].score == approx(three[1].breakdown["base_score"] * 0.01)
    four = _ranked_ids([fb("meh") for _ in range(4)])
    assert 1 not in four and 2 in four


def test_rank_routes_v0_marks_unpenalised_routes():
    ranked = _ranked_ids([fb("meh", route_id=2)])
    assert "feedback_penalty" not in ranked[1].breakdown
    assert ranked[1].breakdown["final_score"] == ranked[1].breakdown["base_score"] == ranked[1].score
    assert ranked[2].breakdown["feedback_penalty"] == 0.5


def test_rank_routes_v0_feedback_reorders():
    routes = [{"id": rid, **vec} for rid, vec in VECTORS.items()]
    user = {"difficulty_range": [2, 3], "min_distance_km": 10.0, "max_distance_km": 50.0, "preferred_tags": []}
    before = [r.route_id for r in rs.rank_routes_v0(user, routes, [], now=NOW)]
    after = [r.route_id for r in rs.rank_routes_v0(user, routes, [fb("too-hard", route_id=2)], now=NOW)]
    assert before == [2, 1]
    assert after == [1, 2]
