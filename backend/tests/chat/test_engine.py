"""Turn policy, session state and the harness entry points (no database)."""

from __future__ import annotations

from app.chat import (
    ChatSession,
    ChatSlots,
    ChatState,
    extract_slots,
    run_conversation,
    user_query_from_slots,
)
from app.chat.engine import MAX_CLARIFY_TURNS, default_extractor
from app.chat.extract import EXTRACT_TAG
from app.chat.projection import project_vector
from app.chat.schema import SlotChange
from app.services.user_profile_service import translate_questionnaire_to_vector


async def test_bare_utterance_extraction_needs_no_database(chat_client):
    ext = await extract_slots("Gemütliche Radtour ab Tegernsee, höchstens 30 km", client=chat_client)
    assert ext.update.activity == "cycling" and ext.update.distance_cap_km == 30
    assert ext.update.place_name == "Tegernsee"  # a name; resolution is the engine's job
    assert ext.provider == "mock" and ext.schema_attempts == 1


async def test_required_slots_and_clarify_limit(chat_client, resolver):
    session, results = await run_conversation(["hello", "hmm", "anything"], resolver=resolver, client=chat_client)
    assert [r.status for r in results] == ["clarify", "clarify", "ready"]
    assert [r.question.slot for r in results[:2]] == ["activity", "activity"]
    assert session.state.clarify_turns == MAX_CLARIFY_TURNS
    assert session.state.defaults_applied == ["activity", "place"]
    assert session.slots.activity == "hiking" and session.state.sources["activity"] == "default"
    more = await session.send("still nothing")
    assert more.status == "ready" and more.question is None


async def test_asks_for_place_after_activity_and_accepts_the_answer(chat_client, resolver):
    session, results = await run_conversation(["I'd like an easy hike", "around Füssen"], resolver=resolver,
                                              client=chat_client)
    assert (results[0].status, results[0].question.slot) == ("clarify", "place")
    assert results[1].status == "ready" and session.slots.place.name == "Füssen"
    assert session.state.pending_slot is None


async def test_no_question_when_required_slots_are_known(chat_client, resolver):
    start = ChatSlots(activity="hiking", place=await resolver("München"))
    session, results = await run_conversation(["something with a lake"], resolver=resolver, start=start,
                                              client=chat_client)
    assert results[0].status == "ready" and session.slots.tags_preferred == ["Lake"]


async def test_german_message_gets_a_german_question(chat_client, resolver):
    _, results = await run_conversation(["Ich suche eine Tour mit Hütte bitte"], resolver=resolver,
                                        client=chat_client)
    assert results[0].question.text.startswith("Was möchtest du machen")


async def test_extractor_failure_leaves_the_session_unchanged(chat_client, resolver):
    session = ChatSession(ChatState(), ChatSlots(), {}, resolver=resolver, extractor=default_extractor(chat_client))
    chat_client.mock.register("{not json", tag=EXTRACT_TAG, times=3)
    import pytest

    from app.llm import SchemaValidationError

    with pytest.raises(SchemaValidationError):
        await session.send("easy hike")
    assert session.state.turns == [] and session.slots == ChatSlots()


async def test_session_user_query_and_vector_projection(chat_client, resolver):
    from app.api.schemas import ProfileCreate

    vector = translate_questionnaire_to_vector(ProfileCreate(fitness="beginner", type=["family-fun"],
                                                             narrative="playful"))
    session, _ = await run_conversation(
        ["bike ride near Tegernsee within 25 km, max 40 km, with a beer garden, no summit"],
        resolver=resolver, vector=vector, client=chat_client)
    q = session.user_query(request_id="p1")
    assert q.category == "cycling" and q.home == (session.slots.place.lat, session.slots.place.lon)
    assert (q.max_travel_km, q.max_distance_km) == (25, 40)
    assert q.user_vector["preferred_tags"] == vector["preferred_tags"] + ["Beer garden"]
    assert q.user_vector["avoided_tags"] == ["Summit route"]
    assert q.user_vector["difficulty_range"] == vector["difficulty_range"]  # not mentioned: untouched
    # The pure helper gives the same query from slots + vector.
    assert user_query_from_slots(session.slots, session.vector, request_id="p1") == q


def test_projection_touches_only_changed_keys():
    vector = {"difficulty_range": [0.5, 1.5], "min_distance_km": 0.0, "max_distance_km": 7.2,
              "preferred_tags": ["culture", "Lake"], "fitness_level": "beginner"}
    old = ChatSlots(tags_preferred=["Lake"])
    new = ChatSlots(tags_preferred=[], tags_avoided=["Lake"], distance_cap_km=5)
    changes = [SlotChange(slot="tags_preferred", old=["Lake"], new=[]),
               SlotChange(slot="tags_avoided", old=[], new=["Lake"]),
               SlotChange(slot="distance_cap_km", old=None, new=5)]
    v = project_vector(vector, dict(vector), old, new, changes)
    assert v["difficulty_range"] == [0.5, 1.5]  # feedback-adjusted range survives
    assert v["preferred_tags"] == ["culture"] and v["avoided_tags"] == ["Lake"]
    assert (v["min_distance_km"], v["max_distance_km"]) == (0.0, 5.0)
    assert vector["preferred_tags"] == ["culture", "Lake"]  # input not mutated


def test_harness_query_defaults_to_hiking_without_activity():
    q = user_query_from_slots(ChatSlots(travel_radius_km=30), {}, request_id="x")
    assert q.category == "hiking" and q.home is None and q.max_travel_km is None
