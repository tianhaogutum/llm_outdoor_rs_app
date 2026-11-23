"""Deterministic merge of a validated update into the current slots. The model proposes, this code decides.

Rules (docs/CHAT.md section 4), applied in this order:

1. ``clear``: each named slot becomes empty. Clearing ``place`` also clears ``travel_radius_km`` (a
   radius means nothing without a centre); clearing ``tags`` empties both tag lists.
2. ``activity``: a stated activity replaces the current one.
3. difficulty: an absolute band replaces the current one; otherwise ``difficulty_change`` moves one band
   (easy < moderate < hard, clamped). With no current band, "easier" gives easy and "harder" gives hard.
4. length band, first rule that applies: an absolute ``distance_band``; a ``distance_target_km`` (mapped
   to the band for the activity); a ``duration_hours`` (hours x planning speed of the activity, then
   mapped); a ``distance_change`` (one band, clamped; no current band: shorter = short, longer = long).
   A stated duration is always stored; a length statement without a duration clears an older duration
   (it no longer describes the wish).
5. ``distance_cap_km`` replaces the cap.
6. ``place`` (already resolved by the place lookup) replaces the place. If the place changed, no radius
   is set and the message gave none, the radius gets the documented default (source ``default``).
7. ``travel_radius_km`` replaces the radius.
8. tags, three states per tag (wanted / neutral / avoided): ``tags_remove`` makes a tag neutral,
   ``tags_avoid`` makes it avoided (and not wanted), ``tags_add`` makes it wanted (and not avoided).
   The validator already removed tags that one message both adds and avoids.
9. ``intent``: a non-empty restatement replaces the stored one (the model sees the stored intent and is
   asked to carry it forward); an empty one keeps it.

Null fields never change anything. The function is pure: same inputs, same outputs.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from .schema import ChatSlots, PlaceRef, SlotChange, SlotUpdate
from .vocab import DIFFICULTY_BANDS, DISTANCE_BANDS, PLANNING_SPEED_KMH, band_for_km

DEFAULT_RADIUS_KM = 50.0
SLOT_ORDER = ("activity", "difficulty", "distance_band", "duration_hours", "distance_cap_km", "place",
              "travel_radius_km", "tags_preferred", "tags_avoided", "intent")

Source = Literal["chat", "edit", "default"]


def _shift(order: tuple[str, ...], current: Optional[str], step: int) -> str:
    if current is None:
        return order[0] if step < 0 else order[-1]
    i = order.index(current) + step
    return order[max(0, min(len(order) - 1, i))]


def _view(slot: str, value: Any) -> Any:
    if isinstance(value, PlaceRef):
        return value.name
    return value


def diff(old: ChatSlots, new: ChatSlots, source: Source, defaulted: frozenset[str] = frozenset()) -> list[SlotChange]:
    out = []
    for slot in SLOT_ORDER:
        a, b = getattr(old, slot), getattr(new, slot)
        if a != b:
            out.append(SlotChange(slot=slot, old=_view(slot, a), new=_view(slot, b),
                                  source="default" if slot in defaulted else source))
    return out


def apply_update(
    slots: ChatSlots,
    update: SlotUpdate,
    *,
    place: Optional[PlaceRef] = None,
    source: Source = "chat",
    default_radius_km: Optional[float] = DEFAULT_RADIUS_KM,
) -> tuple[ChatSlots, list[SlotChange]]:
    """Merge ``update`` into ``slots``. ``place`` is the resolved ``update.place_name`` (None: unresolved
    or not given). Returns the new slots and the slot-by-slot changes (old -> new)."""
    s = slots.model_copy(deep=True)
    defaulted: set[str] = set()

    # 1. clear
    for name in update.clear:
        if name == "tags":
            s.tags_preferred, s.tags_avoided = [], []
        elif name == "place":
            s.place, s.travel_radius_km = None, None
        else:
            setattr(s, name, None)

    # 2. activity
    if update.activity:
        s.activity = update.activity

    # 3. difficulty
    if update.difficulty:
        s.difficulty = update.difficulty
    elif update.difficulty_change:
        s.difficulty = _shift(DIFFICULTY_BANDS, s.difficulty, -1 if update.difficulty_change == "easier" else 1)

    # 4. length
    if update.distance_band:
        s.distance_band = update.distance_band
    elif update.distance_target_km is not None:
        s.distance_band = band_for_km(update.distance_target_km, s.activity)
    elif update.duration_hours is not None:
        speed = PLANNING_SPEED_KMH.get(s.activity or "hiking", PLANNING_SPEED_KMH["hiking"])
        s.distance_band = band_for_km(update.duration_hours * speed, s.activity)
    elif update.distance_change:
        s.distance_band = _shift(DISTANCE_BANDS, s.distance_band, -1 if update.distance_change == "shorter" else 1)
    if update.duration_hours is not None:
        s.duration_hours = update.duration_hours
    elif update.distance_band or update.distance_target_km is not None or update.distance_change:
        s.duration_hours = None

    # 5. cap
    if update.distance_cap_km is not None:
        s.distance_cap_km = update.distance_cap_km

    # 6. place (+ default radius) and 7. radius
    if place is not None:
        moved = s.place is None or (s.place.name, s.place.lat, s.place.lon) != (place.name, place.lat, place.lon)
        s.place = place
        if moved and s.travel_radius_km is None and update.travel_radius_km is None and default_radius_km:
            s.travel_radius_km = default_radius_km
            defaulted.add("travel_radius_km")
    if update.travel_radius_km is not None:
        s.travel_radius_km = update.travel_radius_km
        defaulted.discard("travel_radius_km")

    # 8. tags
    pref, avoid = list(s.tags_preferred), list(s.tags_avoided)
    for t in update.tags_remove:
        pref = [x for x in pref if x != t]
        avoid = [x for x in avoid if x != t]
    for t in update.tags_avoid:
        pref = [x for x in pref if x != t]
        if t not in avoid:
            avoid.append(t)
    for t in update.tags_add:
        avoid = [x for x in avoid if x != t]
        if t not in pref:
            pref.append(t)
    s.tags_preferred, s.tags_avoided = pref, avoid

    # 9. intent
    if update.intent:
        s.intent = update.intent

    return s, diff(slots, s, source, frozenset(defaulted))
