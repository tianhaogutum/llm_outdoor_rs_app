from pipeline.describe import compose, generate
from pipeline.tags import derive_tags

from .factory import feat, record


def full_record(rid=1):
    r = record(id=rid)
    r["features"]["on_route"] = {
        "peak": [feat("Jochberg", ele_m=1565)],
        "lake": [feat("Kochelsee")],
        "restaurant": [feat("Café Seeblick")],
    }
    r["features"]["near_start"] = {"bus_stop": [feat("Kochel, Bahnhof", dist=120)]}
    r["tags"] = derive_tags(r)
    return r


def test_deterministic():
    assert generate(full_record()) == generate(full_record())
    assert compose(full_record()) == compose(full_record())


def test_mentions_facts_only_from_record():
    r = full_record()
    text = generate(r)
    for fact in ("Jochberg", "Kochelsee", "Café Seeblick", "Kochel am See", "7.0 km", "Kochel, Bahnhof"):
        assert fact in text
    assert "station" not in text.replace("Bahnhof", "")  # no train station in the record


def test_templates_vary_with_id():
    texts = {generate(full_record(rid)) for rid in range(1, 40)}
    assert len(texts) >= 10
    openings = {t.split(".")[0].split(" ")[1] for t in texts}
    assert len(openings) >= 2


def test_osm_description_is_kept_and_marked():
    r = full_record()
    r["osm_tags"]["description"] = "Schöner Rundweg um den See"
    desc, src = compose(r)
    assert src == "osm+generated" and desc.startswith("Schöner Rundweg um den See.")
    r["osm_tags"].pop("description")
    assert compose(r)[1] == "generated"
