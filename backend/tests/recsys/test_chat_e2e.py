"""Phase 4 end-to-end variants: questionnaire only, and questionnaire + the intent sent as a chat message."""

from __future__ import annotations

import pytest

from app.chat import IndexPlaceResolver
from app.chat.llm import CHAT_DEFAULTS, build_chat_client
from app.llm import LLMClient, LLMConfig
from app.recsys import build_system
from app.recsys.constraints import HardConstraints
from app.recsys.query import UserQuery

from .factory import synthetic_catalog


@pytest.fixture(scope="module")
def cat():
    return synthetic_catalog(400, seed=5)


@pytest.fixture
def llm(tmp_path):
    return LLMClient(LLMConfig(cache_dir=tmp_path / "cache"))


@pytest.fixture
def chat_llm(tmp_path):
    return build_chat_client(CHAT_DEFAULTS["mock"], config=LLMConfig(cache_dir=tmp_path / "chat-cache"))


def _q(**kw) -> UserQuery:
    base = dict(
        request_id="p1",
        user_vector={"difficulty_range": [1, 2], "min_distance_km": 5.0, "max_distance_km": 20.0,
                     "preferred_tags": []},
        category="hiking",
        home=(48.0, 11.0),
        max_travel_km=40.0,
        max_distance_km=14.0,
        intent="An easy lake loop with a beer garden, about 6 km",
    )
    base.update(kw)
    return UserQuery(**base)


def test_questionnaire_only_drops_the_intent(cat, llm):
    rec = build_system("recsys_v1_qonly", cat, llm=llm).recommend(_q(), 10)
    assert rec.system == "recsys_v1_qonly" and rec.trace[0].info["fallback"] == "no_intent"
    assert rec.route_ids == build_system("recsys_v1", cat, llm=llm).recommend(_q(intent=None), 10).route_ids


def test_chat_variant_merges_slots_and_uses_the_restated_intent(cat, llm, chat_llm):
    sysm = build_system("recsys_v1_chat", cat, llm=llm, chat_llm=chat_llm, resolver=IndexPlaceResolver([]))
    q = _q()
    slots = sysm.start_slots(q)
    assert (slots.activity, slots.difficulty, slots.distance_band) == ("hiking", "moderate", "medium")
    assert (slots.place.lat, slots.place.lon, slots.travel_radius_km, slots.distance_cap_km) == (48.0, 11.0, 40.0, 14.0)
    cq, info, _, _ = _chat(sysm, q)
    assert cq.category == "hiking" and cq.home == (48.0, 11.0) and cq.max_travel_km == 40.0
    assert cq.user_vector["difficulty_range"] == [0, 1]  # "easy" from the message, merged into the vector
    assert cq.intent and info["restated_intent"] == cq.intent
    rec = sysm.recommend(q, 10)
    assert rec.trace[0].stage == "chat" and rec.system == "recsys_v1_chat"
    hc = HardConstraints.from_query(cq)
    assert all(hc.ok(cat.get(r)) for r in rec.route_ids)
    cfg = sysm.config()
    assert cfg["chat"]["model"] == "mock-llm-1" and cfg["system"] == "recsys_v1_chat"
    # no message: the form's slots alone (same as questionnaire only)
    q0 = _q(intent=None)
    assert sysm.recommend(q0, 10).route_ids == build_system("recsys_v1_qonly", cat, llm=llm).recommend(q0, 10).route_ids


def _chat(sysm, q):
    from app.llm import run_sync

    return run_sync(sysm.chat_query(q))
