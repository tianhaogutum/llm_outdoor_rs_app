import pytest

from pipeline import attributes as A


def prof(**kw):
    base = {"sac_m": {}, "mtb_scale_m": {}}
    base.update(kw)
    return base


def test_din33466_examples():
    # 12 km, 600 m up, 600 m down: th = 3 h, tv = 2 + 1.2 = 3.2 h -> 3.2 + 1.5 = 4.7 h
    assert A.duration_min("hiking", 12000, 600, 600) == 282
    # flat 8 km: 2 h
    assert A.duration_min("hiking", 8000, 0, 0) == 120
    # steep: 4 km, 1200 up, 0 down: th 1 h, tv 4 h -> 4.5 h
    assert A.duration_min("hiking", 4000, 1200, 0) == 270


def test_speed_models():
    assert A.duration_min("cycling", 36000, 0, 0) == 120
    assert A.duration_min("cycling", 36000, 600, 600) == 180
    assert A.duration_min("mtb", 24000, 400, 400) == 180
    assert A.duration_min("running", 9000, 0, 0) == 60


def test_effort_levels():
    assert A.effort_level("hiking", 5000, 100) == 0  # 6 Lkm
    assert A.effort_level("hiking", 10000, 300) == 1  # 13
    assert A.effort_level("hiking", 15000, 1000) == 2  # 25
    assert A.effort_level("hiking", 20000, 1500) == 3  # 35
    assert A.effort_level("cycling", 30000, 0) == 0
    assert A.effort_level("cycling", 120000, 0) == 3


def test_difficulty_from_sac_scale():
    d = A.difficulty("hiking", 6000, 300, prof(sac_m={"T1": 3000, "T3": 400}))
    assert d["difficulty"] == 2 and d["difficulty_source"] == "osm:sac_scale" and d["max_grade"] == "T3"
    # grade covering < 100 m is ignored
    d = A.difficulty("hiking", 6000, 300, prof(sac_m={"T1": 3000, "T4": 50}))
    assert d["difficulty"] == 0 and d["max_grade"] == "T1"
    # T5 -> 3
    assert A.difficulty("hiking", 3000, 100, prof(sac_m={"T5": 500}))["difficulty"] == 3


def test_difficulty_effort_dominates_and_untagged():
    d = A.difficulty("hiking", 30000, 1200, prof(sac_m={"T1": 20000}))
    assert d["difficulty"] == 3 and d["difficulty_source"].startswith("derived:effort")
    d = A.difficulty("hiking", 5000, 50, prof())
    assert d["difficulty"] == 0 and d["difficulty_source"] == "derived:effort" and d["difficulty_technical"] is None


def test_difficulty_mtb_scale():
    d = A.difficulty("mtb", 10000, 200, prof(mtb_scale_m={"S2": 800}))
    assert d["difficulty"] == 2 and d["difficulty_source"] == "osm:mtb:scale"


def test_mtb_level_parsing():
    assert A.mtb_level({"mtb:scale": "2+"}) == 2
    assert A.mtb_level({"mtb:scale": "x"}) is None


def test_surface_classification():
    assert A.surface_class({"surface": "asphalt"}) == ("paved", True)
    assert A.surface_class({"surface": "gravel", "highway": "residential"}) == ("unpaved", True)
    assert A.surface_class({"highway": "residential"}) == ("paved", False)
    assert A.surface_class({"highway": "track", "tracktype": "grade1"}) == ("paved", False)
    assert A.surface_class({"highway": "track"}) == ("unpaved", False)
    assert A.surface_class({"highway": "footway"}) == ("unknown", False)


def test_way_profile_weights_by_length():
    p = A.way_profile(
        [
            {"highway": "path", "sac_scale": "mountain_hiking"},
            {"highway": "residential", "surface": "asphalt"},
            {"highway": "steps"},
        ],
        [300.0, 600.0, 100.0],
    )
    assert p["paved_share"] == pytest.approx(0.6)
    assert p["unpaved_share"] == pytest.approx(0.3)
    assert p["surface_tagged_share"] == pytest.approx(0.6)
    assert p["sac_m"] == {"T2": 300}
    assert p["steps_m"] == 100
    assert p["path_share"] == pytest.approx(0.4)


def test_category_mapping():
    assert A.category("hiking", "hiking", "Waldlehrpfad Moosach", {}, 3000, 0.5, 50) == "Theme trail"
    assert A.category("hiking", "hiking", "Herzogstand Rundweg", {}, 9000, 0.5, 700) == "Hiking trail"
    assert A.category("cycling", "bicycle", "Isarradweg", {}, 290000, 0.2, 900) == "Long distance cycling"
    assert A.category("cycling", "bicycle", "Rundtour 3", {"network": "lcn"}, 30000, 0.2, 100) == "Cycling"
    assert A.category("mtb", "mtb", "X", {}, 30000, 0.5, 900) == "Mountainbiking"
    assert A.category("running", "running", "Laufstrecke", {}, 5000, 0.9, 20) == "Trail running"
    assert A.category("running", "fitness_trail", "Trimm-dich-Pfad", {}, 2000, 0.3, 20) == "Jogging"


def test_parse_osm_numbers():
    assert A.parse_length_tag("12.5") == 12500
    assert A.parse_length_tag("12,5 km") == 12500
    assert A.parse_length_tag("800 m") == 800
    assert A.parse_length_tag("abc") is None
    assert A.parse_ele_tag("850") == 850
    assert A.parse_ele_tag("850 m") == 850
    assert A.parse_ele_tag(None) is None


def test_est_days():
    assert A.est_days(480, "hiking") == 1
    assert A.est_days(481, "hiking") == 2


def test_difficulty_uses_descent_for_summit_down_routes():
    up = A.difficulty("hiking", 5000, 0, prof(), descent_m=1500)
    down = A.difficulty("hiking", 5000, 1500, prof(), descent_m=0)
    assert up["difficulty"] == down["difficulty"] == 1  # 5 km + 1500/100 = 20 Lkm
