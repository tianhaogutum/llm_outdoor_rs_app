"""Questionnaire -> user vector."""

from __future__ import annotations

import pytest

from app.api.schemas import ProfileCreate
from app.services import recommendation_service as rs
from app.services.user_profile_service import FITNESS_MAPPING, translate_questionnaire_to_vector
from tests.golden.make_golden_v0 import golden_user_vector


@pytest.mark.parametrize("fitness", ["beginner", "intermediate", "advanced"])
@pytest.mark.parametrize("adventure", ["history-culture", "natural-scenery", "family-fun"])
def test_questionnaire_vector_scoring_fields_match_golden_inputs(fitness, adventure):
    vec = translate_questionnaire_to_vector(ProfileCreate(fitness=fitness, type=[adventure], narrative="adventure"))
    expected = golden_user_vector(fitness, adventure)
    assert {k: vec[k] for k in expected} == expected
    assert vec["narrative_prompt_style"]


def test_fitness_ranges_are_on_the_difficulty_scale():
    for cfg in FITNESS_MAPPING.values():
        lo, hi = cfg["difficulty_range"]
        assert rs.DIFFICULTY_MIN <= lo <= hi <= rs.DIFFICULTY_MAX


def test_unknown_fitness_defaults_to_beginner_and_tags_deduplicate():
    vec = translate_questionnaire_to_vector(
        ProfileCreate(fitness="godlike", type=["natural-scenery", "natural-scenery"], narrative="x")
    )
    assert vec["difficulty_range"] == FITNESS_MAPPING["beginner"]["difficulty_range"]
    assert vec["preferred_tags"] == ["flora", "fauna", "panorama", "scenic", "geology"]
