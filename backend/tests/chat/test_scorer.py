"""The scripted-eval scorer (evalkit/chat/scoring.py) on hand-made outcomes, and the runner on a tiny fixture."""

from __future__ import annotations

import json

import pytest

from evalkit.chat.scoring import aggregate, percentile, score_case, value_matches


def test_number_tolerances():
    assert value_matches("duration_hours", 2.4, 2) and not value_matches("duration_hours", 2.6, 2)
    assert value_matches("duration_hours", 0.5, 0.75)  # absolute 0.5 h beats 25 % for short durations
    assert value_matches("distance_cap_km", 15.9, 15) and not value_matches("distance_cap_km", 17, 15)
    assert value_matches("travel_radius_km", 70, 60) and not value_matches("travel_radius_km", 80, 60)
    assert value_matches("activity", "hiking", "hiking") and not value_matches("activity", "mtb", "cycling")


def test_scalar_counting_rules():
    case = {"expect": {"activity": "hiking", "difficulty": "easy", "distance_cap_km": 10},
            "optional": {"distance_band": "short"}, "clarify": None}
    out = {"changes": {"activity": "hiking", "difficulty": "hard", "distance_band": "short", "place": "X|D"},
           "clarify": None}
    c = score_case(case, out)["counts"]
    assert c["activity"] == {"tp": 1, "fp": 0, "fn": 0}
    assert c["difficulty"] == {"tp": 0, "fp": 1, "fn": 1}  # wrong value: both
    assert c["distance_cap_km"] == {"tp": 0, "fp": 0, "fn": 1}  # missed
    assert c["distance_band"] == {"tp": 0, "fp": 0, "fn": 0}  # optional value: ignored
    assert c["place"] == {"tp": 0, "fp": 1, "fn": 0}  # not expected


def test_wrong_value_that_is_optional_is_only_a_miss():
    case = {"expect": {"difficulty": "easy"}, "optional": {"difficulty": "moderate"}}
    assert score_case(case, {"changes": {"difficulty": "moderate"}})["counts"]["difficulty"] == \
        {"tp": 0, "fp": 0, "fn": 1}


def test_tag_set_counting_and_clarify_and_intent():
    case = {"expect": {"tags_preferred": ["Lake", "Hut"], "tags_avoided": ["Summit route"]},
            "optional": {"tags_preferred": ["Scenic"]}, "clarify": "activity",
            "intent_mentions": [["bakery", "bäckerei"], ["quiet"]]}
    out = {"changes": {"tags_preferred": ["Lake", "Scenic", "Castle"]}, "clarify": "place",
           "intent": "A quiet walk to a Bäckerei"}
    sc = score_case(case, out)
    assert sc["counts"]["tags_preferred"] == {"tp": 1, "fp": 1, "fn": 1}
    assert sc["counts"]["tags_avoided"] == {"tp": 0, "fp": 0, "fn": 1}
    assert sc["clarify"] == {"expected": "activity", "predicted": "place", "decision_ok": True, "slot_ok": False}
    assert sc["intent_hit"] is True


def test_aggregate_micro_averages_and_confusion():
    a = score_case({"expect": {"activity": "hiking"}, "clarify": None}, {"changes": {"activity": "hiking"}})
    b = score_case({"expect": {"activity": "cycling"}, "clarify": "place"},
                   {"changes": {"activity": "hiking", "tags_preferred": ["Lake"]}, "clarify": None})
    m = aggregate([a, b], ["g1", "g2"])
    assert m["per_slot"]["activity"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert m["per_type"]["tags"]["precision"] == 0.0 and m["per_type"]["tags"]["recall"] is None
    assert m["overall"]["tp"] == 1 and m["overall"]["fp"] == 2 and m["overall"]["fn"] == 1
    assert m["clarify"]["decision_accuracy"] == 0.5 and m["clarify"]["confusion"] == {"tp": 0, "fp": 0, "fn": 1,
                                                                                       "tn": 1}
    assert m["by_group"]["g1"]["precision"] == 1.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5 and percentile([], 0.5) is None


@pytest.mark.parametrize("cache_only", [False, True])
def test_runner_on_a_tiny_fixture_is_deterministic(tmp_path, resolver, cache_only):
    """The rules candidate (offline) end to end on three cases; replay mode never blocks the mock."""
    import asyncio

    from evalkit.chat.run import CANDIDATES, run_candidate

    cases = [
        {"id": "t1", "group": "single_slot", "lang": "en", "start": {"place": "München"},
         "turns": ["easy hike with a lake"], "expect": {"activity": "hiking", "difficulty": "easy",
                                                         "tags_preferred": ["Lake"]}, "clarify": None},
        {"id": "t2", "group": "vague", "lang": "en", "start": {}, "turns": ["hello"], "expect": {},
         "clarify": "activity"},
        {"id": "t3", "group": "place_typo", "lang": "en", "start": {}, "turns": ["cycling near Garmish"],
         "expect": {"activity": "cycling", "place": "Garmisch-Partenkirchen"}, "clarify": None},
    ]
    runs = [asyncio.run(run_candidate(CANDIDATES["rules"], cases, resolver, cache_only=cache_only, freeze=False))
            for _ in range(2)]
    for r in runs:
        r.pop("volatile")
    assert runs[0] == runs[1]
    m = runs[0]["metrics"]
    assert m["overall"]["precision"] == 1.0 and m["overall"]["recall"] == 1.0
    assert m["clarify"]["slot_accuracy"] == 1.0 and m["schema"]["violation_rate"] == 0.0
    assert runs[0]["cases"][2]["changes"]["place"] == "Garmisch-Partenkirchen|Landkreis Garmisch-Partenkirchen"
    json.dumps(runs[0])


def test_scripted_set_is_well_formed():
    from evalkit.chat.run import DATA, load_cases
    from evalkit.chat.scoring import ALL_SLOTS

    cases = load_cases(DATA)
    assert len(cases) >= 60 and len({c["id"] for c in cases}) == len(cases)
    groups = {c["group"] for c in cases}
    assert {"single_slot", "multi_slot", "vague", "correction", "negation", "numbers", "family_dog",
            "place_typo", "out_of_vocab", "injection"} <= groups
    assert {c["lang"] for c in cases} == {"en", "de"}
    from app.chat.vocab import TAGS

    for c in cases:
        assert set(c["expect"]) <= set(ALL_SLOTS), c["id"]
        for block in (c["expect"], c.get("optional") or {}):
            for slot in ("tags_preferred", "tags_avoided"):
                assert set(block.get(slot, [])) <= set(TAGS), c["id"]
        assert c["clarify"] in (None, "activity", "place")
