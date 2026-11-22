"""Synthetic route records for rule tests (shape of routes.jsonl records)."""

import copy

BASE_PROFILE = {
    "paved_share": 0.2,
    "unpaved_share": 0.7,
    "unknown_surface_share": 0.1,
    "surface_tagged_share": 0.5,
    "stroller_ok_share": 0.3,
    "road_share": 0.1,
    "busy_road_share": 0.0,
    "path_share": 0.5,
    "track_share": 0.3,
    "cycleway_share": 0.0,
    "steps_m": 0,
    "via_ferrata_m": 0,
    "sac_tagged_share": 0.0,
    "sac_m": {},
    "mtb_tagged_share": 0.0,
    "mtb_scale_m": {},
    "dog_no_share": 0.0,
    "dog_yes_share": 0.0,
    "lit_share": 0.0,
}


def record(**over):
    r = {
        "id": 12345,
        "title": "Testweg",
        "activity": "hiking",
        "category_name": "Hiking trail",
        "tags": [],
        "difficulty": 1,
        "difficulty_technical": None,
        "max_grade": None,
        "length_m": 7000,
        "ascent_m": 250,
        "descent_m": 250,
        "min_ele_m": 600,
        "max_ele_m": 850,
        "duration_min": 135,
        "est_days": 1,
        "is_loop": True,
        "locality": "Kochel am See",
        "admin_district": "Landkreis Bad Tölz-Wolfratshausen",
        "stages": [],
        "osm_tags": {"name": "Testweg"},
        "features": {"on_route": {}, "near_start": {}, "protected_areas": [], "profile": copy.deepcopy(BASE_PROFILE)},
        "_internal": {"retrace_share": 0.0},
    }
    for k, v in over.items():
        r[k] = v
    return r


def feat(name, dist=10, at=1000, **kw):
    d = {"name": name, "dist_m": dist, "at_m": at, "osm": "n1"}
    d.update(kw)
    return d
