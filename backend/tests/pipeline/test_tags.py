from pipeline.tags import TAG_NAMES, TAG_RULES, derive_tags

from .factory import feat, record


def test_vocabulary_unique_and_documented():
    assert len(TAG_NAMES) == len(set(TAG_NAMES))
    assert all(rule for _, _, rule in TAG_RULES)
    legacy = {t for t, leg, _ in TAG_RULES if leg}
    for t in (
        "Round trip",
        "Linear route",
        "Out and back",
        "Scenic",
        "Refreshment stops available",
        "Family-friendly",
        "Cultural/historical interest",
        "Geological highlights",
        "Flora",
        "Fauna",
        "Summit route",
        "Dog-friendly",
        "Suitable for strollers",
        "Multi-stage route",
    ):
        assert t in legacy


def test_shape_tags():
    assert "Round trip" in derive_tags(record(is_loop=True))
    r = record(is_loop=True)
    r["_internal"]["retrace_share"] = 0.9
    t = derive_tags(r)
    assert "Out and back" in t and "Round trip" not in t
    assert derive_tags(record(is_loop=False))[:1] == ["Linear route"]


def test_no_tag_without_fact():
    t = derive_tags(record())
    # a plain record (no features) only gets shape and surface tags backed by numbers;
    # Family-friendly needs an amenity on the route
    assert set(t) <= {"Round trip", "Mostly unpaved"}


def test_feature_tags():
    r = record()
    r["features"]["on_route"] = {
        "peak": [feat("Herzogstand", ele_m=1731)],
        "lake": [feat("Walchensee")],
        "alpine_hut": [feat("Herzogstandhaus")],
        "castle": [feat("Schloss X")],
        "gorge": [feat("Partnachklamm")],
    }
    t = set(derive_tags(r))
    assert {
        "Summit route",
        "Lake",
        "Hut",
        "Refreshment stops available",
        "Scenic",
        "Castle",
        "Cultural/historical interest",
        "Gorge",
        "Geological highlights",
    } <= t


def test_access_tags():
    r = record()
    r["features"]["near_start"] = {"train_station": [feat("Kochel", dist=400)], "parking": [feat(None)]}
    t = set(derive_tags(r))
    assert {"Train access", "Public transport access", "Parking at start"} <= t
    assert "Bus access" not in t


def test_family_and_stroller_rules():
    r = record(length_m=5000, ascent_m=100, difficulty=0)
    r["features"]["profile"].update(stroller_ok_share=0.95)
    t = set(derive_tags(r))
    assert "Family-friendly" not in t and "Suitable for strollers" in t  # no amenity yet
    r["features"]["on_route"] = {"playground": [feat(None)]}
    assert "Family-friendly" in derive_tags(r)
    # T3 terrain removes family-friendly
    r = record(length_m=5000, ascent_m=100, difficulty=0, max_ele_m=1200)
    r["features"]["on_route"] = {"playground": [feat(None)]}
    r["features"]["profile"]["sac_m"] = {"T3": 300}
    t = set(derive_tags(r))
    assert "Family-friendly" not in t and "Alpine terrain" in t
    r["max_ele_m"] = 600  # T3 rock step in low hills is not alpine
    assert "Alpine terrain" not in derive_tags(r)
    # too long
    assert "Family-friendly" not in derive_tags(record(length_m=12000))
    # steps rule out strollers
    r = record(length_m=5000, ascent_m=100)
    r["features"]["profile"].update(stroller_ok_share=1.0, steps_m=20)
    assert "Suitable for strollers" not in derive_tags(r)
    # never for MTB
    assert "Family-friendly" not in derive_tags(
        record(activity="mtb", category_name="Mountainbiking", length_m=5000, ascent_m=50)
    )


def test_surface_and_climb_tags():
    r = record(ascent_m=1200, max_ele_m=1700)
    r["features"]["profile"].update(paved_share=0.8, unpaved_share=0.1)
    t = set(derive_tags(r))
    assert {"Big climb", "Mostly paved", "Alpine terrain"} <= t and "Mostly unpaved" not in t
    t = set(derive_tags(record(ascent_m=40, descent_m=40, length_m=8000)))
    assert "Flat" in t
    # summit-down linear route: no ascent but 1500 m descent is neither flat nor small
    t = set(derive_tags(record(ascent_m=0, descent_m=1500, length_m=5000, is_loop=False)))
    assert "Flat" not in t and "Big climb" in t


def test_dog_rule_requires_positive_fact():
    assert "Dog-friendly" not in derive_tags(record())
    r = record(osm_tags={"name": "x", "dog": "leashed"})
    assert "Dog-friendly" in derive_tags(r)
    r["features"]["profile"]["dog_no_share"] = 0.05
    assert "Dog-friendly" not in derive_tags(r)


def test_flora_fauna_and_theme():
    r = record(title="Waldlehrpfad Moosach", category_name="Theme trail")
    assert "Flora" in derive_tags(r)
    r = record(title="Vogellehrpfad", category_name="Theme trail")
    assert "Fauna" in derive_tags(r)
    r = record(title="Waldweg")  # not a theme trail: name alone does not count
    assert "Flora" not in derive_tags(r)


def test_multistage_and_output_order():
    r = record(stages=[1, 2, 3])
    t = derive_tags(r)
    assert "Multi-stage route" in t
    assert t == [x for x in TAG_NAMES if x in t]
