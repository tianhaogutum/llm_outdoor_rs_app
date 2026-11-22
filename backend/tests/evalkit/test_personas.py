"""Persona generator: determinism, realistic correlations, and what systems may see."""

from __future__ import annotations

import dataclasses

import pytest

from app.api.schemas import ProfileCreate
from app.recsys import UserQuery
from app.services.user_profile_service import translate_questionnaire_to_vector
from evalkit import intents
from evalkit.personas import PARAMS, Persona, generate, load_personas
from tests.recsys.factory import TOWNS


def test_generator_is_deterministic_and_seeded(cat):
    a = [p.to_dict() for p in generate(cat, TOWNS, n=30, seed=5)]
    b = [p.to_dict() for p in generate(cat, TOWNS, n=30, seed=5)]
    c = [p.to_dict() for p in generate(cat, TOWNS, n=30, seed=6)]
    assert a == b and a != c


def test_latent_correlations_and_questionnaire(cat):
    ps = generate(cat, TOWNS, n=120, seed=3)
    assert sum(p.split == "test" for p in ps) == round(120 * PARAMS["n_test"] / PARAMS["n_personas"])
    for p in ps:
        x = p.latent
        kind = x["party"]["kind"]
        if kind in PARAMS["ascent_cap"] and not x["multi_day"]:
            assert x["max_ascent_m"] <= PARAMS["ascent_cap"][kind]
        if x["party"]["stroller"]:
            assert x["surface"] == "paved" and x["style"] == "hiking"
        if kind in ("family_young", "stroller", "older"):
            assert x["technical_max"] == 0 and not x["multi_day"]
        assert x["distance_band_km"][0] < x["distance_band_km"][1] <= x["max_distance_km"] + 1e-6
        assert p.eligible["n"] >= PARAMS["min_eligible"] or x["max_travel_km"] == PARAMS["radius_max_km"]
        q = p.questionnaire
        assert p.user_vector == translate_questionnaire_to_vector(
            ProfileCreate(fitness=q["fitness"], type=q["type"], narrative=q["narrative"])
        )


def test_query_exposes_only_visible_fields(cat):
    p = generate(cat, TOWNS, n=3, seed=1)[0]
    q = p.to_query()
    assert isinstance(q, UserQuery)
    assert q.user_vector is p.user_vector and q.category == p.latent["category"]
    assert q.home == tuple(p.profile["home"]) and q.max_travel_km == p.profile["max_travel_km"]
    assert q.max_distance_km == p.profile["max_distance_km"]
    blob = q.fingerprint()
    for latent_key in ("desires", "party", "fitness", "max_ascent_m", "technical_max", "dislikes"):
        assert f'"{latent_key}":' not in blob
    # changing the latent profile does not change what the system sees
    p2 = Persona.from_dict({**p.to_dict(), "latent": {**p.latent, "desires": {"lake": 1.0}, "fitness": 0.0}})
    assert dataclasses.asdict(p2.to_query()) == dataclasses.asdict(q)


def test_intents_with_mock_writer(cat, mock_client):
    ps = generate(cat, TOWNS, n=4, seed=2)
    intents.add_intents(ps, mock_client)
    first = [p.intent for p in ps]
    assert all(first)
    ps2 = generate(cat, TOWNS, n=4, seed=2)
    intents.add_intents(ps2, mock_client)
    assert [p.intent for p in ps2] == first
    assert mock_client.usage_report()["totals"]["generate"]["cache_hits"] == 4
    prompt = intents.intent_prompt(ps[0])
    assert "<untrusted_data" in prompt and "SAC" in intents.SYSTEM


# --------------------------------------------------------------- committed persona file
@pytest.fixture(scope="module")
def committed():
    return load_personas()


def test_committed_personas(committed):
    assert len(committed) == PARAMS["n_personas"]
    assert sum(p.split == "test" for p in committed) == PARAMS["n_test"]
    assert len({p.persona_id for p in committed}) == len(committed)
    for p in committed:
        q = p.questionnaire
        # pins the product's questionnaire translation: if it changes, regenerate the personas
        assert p.user_vector == translate_questionnaire_to_vector(
            ProfileCreate(fitness=q["fitness"], type=q["type"], narrative=q["narrative"])
        )
        assert p.intent and p.eligible["n"] >= PARAMS["min_eligible"]
        assert p.profile["max_travel_km"] == p.latent["max_travel_km"]
        assert p.profile["home"] == [p.latent["home"]["lat"], p.latent["home"]["lon"]]
        assert p.latent["style"] in ("hiking", "touring", "mtb")
