"""Merge rules (app/chat/merge.py): deterministic, explicit, one test per rule."""

from __future__ import annotations

import pytest

from app.chat.merge import DEFAULT_RADIUS_KM, apply_update
from app.chat.schema import ChatSlots, PlaceRef, SlotUpdate
from app.chat.vocab import band_for_km, band_for_range, distance_range

MUC = PlaceRef(name="München", lat=48.137, lon=11.575, admin_district="München")
GAP = PlaceRef(name="Garmisch-Partenkirchen", lat=47.49, lon=11.096)


def merge(slots=None, place=None, **update):
    return apply_update(slots or ChatSlots(), SlotUpdate(**update), place=place)


def test_null_fields_change_nothing():
    start = ChatSlots(activity="hiking", difficulty="easy", tags_preferred=["Lake"], intent="x")
    new, changes = merge(start)
    assert new == start and changes == []


def test_scalar_replaces_and_reports_old_to_new():
    new, changes = merge(ChatSlots(activity="hiking"), activity="cycling")
    assert new.activity == "cycling"
    assert [(c.slot, c.old, c.new, c.source) for c in changes] == [("activity", "hiking", "cycling", "chat")]


@pytest.mark.parametrize(("current", "change", "expected"), [
    ("moderate", "easier", "easy"), ("easy", "easier", "easy"), ("moderate", "harder", "hard"),
    ("hard", "harder", "hard"), (None, "easier", "easy"), (None, "harder", "hard"),
])
def test_difficulty_change_moves_one_band_and_clamps(current, change, expected):
    new, _ = merge(ChatSlots(difficulty=current), difficulty_change=change)
    assert new.difficulty == expected


def test_absolute_band_wins_over_change_in_the_same_update():
    new, _ = merge(ChatSlots(difficulty="moderate"), difficulty="hard", difficulty_change="easier")
    assert new.difficulty == "hard"


@pytest.mark.parametrize(("current", "change", "expected"), [
    ("long", "shorter", "medium"), ("short", "shorter", "short"), (None, "shorter", "short"),
    (None, "longer", "long"), ("medium", "longer", "long"),
])
def test_distance_change_moves_one_band(current, change, expected):
    new, _ = merge(ChatSlots(distance_band=current), distance_change=change)
    assert new.distance_band == expected


def test_duration_maps_to_band_by_activity_and_is_stored():
    hike, _ = merge(ChatSlots(activity="hiking"), duration_hours=2)
    ride, _ = merge(ChatSlots(activity="cycling"), duration_hours=7)
    assert (hike.distance_band, hike.duration_hours) == ("short", 2)
    assert (ride.distance_band, ride.duration_hours) == ("long", 7)
    half_day, _ = merge(ChatSlots(activity="hiking"), duration_hours=4)
    assert half_day.distance_band == "medium"


def test_target_beats_duration_for_the_band_but_duration_is_kept():
    new, _ = merge(ChatSlots(activity="hiking"), duration_hours=7, distance_target_km=20)
    assert (new.distance_band, new.duration_hours) == ("medium", 7)


def test_a_length_statement_without_duration_clears_an_old_duration():
    start = ChatSlots(activity="hiking", duration_hours=4, distance_band="medium")
    new, _ = merge(start, distance_change="shorter")
    assert (new.distance_band, new.duration_hours) == ("short", None)


def test_cap_replaces_and_clear_empties():
    new, _ = merge(ChatSlots(distance_cap_km=20), distance_cap_km=12)
    assert new.distance_cap_km == 12
    new, changes = merge(new, clear=["distance_cap_km"])
    assert new.distance_cap_km is None and changes[0].new is None


def test_new_place_gets_default_radius_marked_as_default():
    new, changes = merge(ChatSlots(), place=GAP)
    assert new.place == GAP and new.travel_radius_km == DEFAULT_RADIUS_KM
    sources = {c.slot: c.source for c in changes}
    assert sources == {"place": "chat", "travel_radius_km": "default"}


def test_stated_radius_wins_over_default_and_existing_radius_is_kept_on_move():
    new, changes = merge(ChatSlots(), place=GAP, travel_radius_km=20)
    assert new.travel_radius_km == 20 and {c.source for c in changes} == {"chat"}
    moved, _ = merge(ChatSlots(place=MUC, travel_radius_km=30), place=GAP)
    assert moved.travel_radius_km == 30


def test_clearing_place_clears_radius():
    new, _ = merge(ChatSlots(place=MUC, travel_radius_km=30), clear=["place"])
    assert new.place is None and new.travel_radius_km is None


def test_unresolved_place_changes_nothing():
    new, changes = merge(ChatSlots(place=MUC), place_name="Atlantis")  # resolver returned None
    assert new.place == MUC and changes == []


def test_tags_three_states():
    s = ChatSlots(tags_preferred=["Lake", "Hut"], tags_avoided=["Summit route"])
    new, _ = merge(s, tags_remove=["Lake"])  # "no lakes after all": neutral
    assert (new.tags_preferred, new.tags_avoided) == (["Hut"], ["Summit route"])
    new, _ = merge(new, tags_avoid=["Hut"])  # wanted -> avoided
    assert (new.tags_preferred, new.tags_avoided) == ([], ["Summit route", "Hut"])
    new, _ = merge(new, tags_add=["Summit route", "Beer garden"])  # avoided -> wanted
    assert (new.tags_preferred, new.tags_avoided) == (["Summit route", "Beer garden"], ["Hut"])
    new, _ = merge(new, tags_add=["Beer garden"])  # idempotent
    assert new.tags_preferred == ["Summit route", "Beer garden"]
    new, _ = merge(new, clear=["tags"])
    assert (new.tags_preferred, new.tags_avoided) == ([], [])


def test_intent_replaced_only_by_non_empty_restatement():
    new, _ = merge(ChatSlots(intent="quiet lake walk"), intent="quiet lake walk without the summit")
    assert new.intent == "quiet lake walk without the summit"
    same, changes = merge(new)
    assert same.intent == new.intent and changes == []


def test_merge_is_deterministic_and_does_not_mutate_input():
    start = ChatSlots(tags_preferred=["Lake"])
    upd = SlotUpdate(tags_add=["Hut"], difficulty="easy")
    a = apply_update(start, upd)
    b = apply_update(start, upd)
    assert a == b and start.tags_preferred == ["Lake"]


def test_two_turn_correction_sequence():
    """'easy hike with a lake, about 10 km' then 'actually make it shorter, no lakes after all'."""
    s, _ = merge(ChatSlots(), activity="hiking", difficulty="easy", distance_target_km=10, tags_add=["Lake"])
    assert s.distance_band == "medium"
    s, changes = merge(s, distance_change="shorter", tags_remove=["Lake"])
    assert (s.distance_band, s.tags_preferred) == ("short", [])
    assert [(c.slot, c.old, c.new) for c in changes] == [("distance_band", "medium", "short"),
                                                       ("tags_preferred", ["Lake"], [])]


def test_band_tables():
    assert distance_range("short", "hiking") == (0.0, 8.0)  # = questionnaire "beginner"
    assert distance_range("medium", "cycling") == (20.0, 80.0)
    assert band_for_range(5.0, 20.0, None) == "medium" and band_for_range(1, 2, None) is None
    assert band_for_km(8, "hiking") == "short" and band_for_km(30, "hiking") == "long"
