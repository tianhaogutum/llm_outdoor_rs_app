"""Small synthetic catalogues for recsys / evalkit tests (no files, no network)."""

from __future__ import annotations

import numpy as np

from app.recsys import Catalog

TAG_POOL = [
    "Lake",
    "Viewpoint",
    "Summit route",
    "Castle",
    "Beer garden",
    "Refreshment stops available",
    "River",
    "Playground",
    "Hut",
    "Train access",
    "Mostly paved",
    "Mostly unpaved",
    "Family-friendly",
    "Scenic",
    "Flora",
    "Waterfall",
    "Cave",
    "Museum",
    "Swimming spot",
    "Suitable for strollers",
]
CATEGORIES = [
    ("Hiking trail", "hiking"),
    ("Theme trail", "hiking"),
    ("Cycling", "cycling"),
    ("Mountainbiking", "mtb"),
    ("Long distance cycling", "cycling"),
    ("Jogging", "running"),
]
TOWNS = [
    {
        "name": "Alpha",
        "osm_node": 1,
        "place": "town",
        "population": 20000,
        "lat": 48.0,
        "lon": 11.0,
        "region": "Oberbayern",
    },
    {
        "name": "Beta",
        "osm_node": 2,
        "place": "city",
        "population": 150000,
        "lat": 49.0,
        "lon": 11.0,
        "region": "Mittelfranken",
    },
    {
        "name": "Gamma",
        "osm_node": 3,
        "place": "town",
        "population": 8000,
        "lat": 48.5,
        "lon": 12.0,
        "region": "Niederbayern",
    },
]


def route_record(
    i: int,
    *,
    category: str = "Hiking trail",
    activity: str = "hiking",
    lat: float = 48.0,
    lon: float = 11.0,
    length_m: int = 8000,
    ascent_m: int = 200,
    difficulty: int = 0,
    tags=("Lake",),
    is_loop: bool = True,
) -> dict:
    return {
        "id": i,
        "title": f"Route {i}",
        "category_name": category,
        "activity": activity,
        "tags": list(tags),
        "difficulty": difficulty,
        "difficulty_technical": None,
        "max_grade": None,
        "length_m": length_m,
        "ascent_m": ascent_m,
        "descent_m": ascent_m,
        "max_ele_m": 600,
        "duration_min": 120,
        "est_days": 1,
        "start_lat": lat,
        "start_lon": lon,
        "is_loop": is_loop,
        "locality": "Alpha",
        "end_locality": None,
        "admin_district": "Landkreis Test",
        "admin_region": "Oberbayern",
        "description": f"A {length_m / 1000:.0f} km route with " + ", ".join(tags).lower() + ".",
        "description_source": "generated",
        "max_grade_200m_pct": 8.0,
        "features": {
            "profile": {"paved_share": 0.3, "unpaved_share": 0.7, "road_share": 0.1},
            "on_route": {},
            "on_route_counts": {},
            "near_start": {},
        },
        "osm_tags": {},
    }


def synthetic_catalog(n: int = 240, seed: int = 0) -> Catalog:
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        cat, act = CATEGORIES[int(rng.choice(len(CATEGORIES), p=[0.5, 0.1, 0.2, 0.08, 0.07, 0.05]))]
        town = TOWNS[i % len(TOWNS)]
        length = int(rng.uniform(2, 15) * 1000) if act in ("hiking", "running") else int(rng.uniform(10, 80) * 1000)
        tags = [TAG_POOL[int(j)] for j in rng.choice(len(TAG_POOL), size=int(rng.integers(1, 6)), replace=False)]
        recs.append(
            route_record(
                1000 + i,
                category=cat,
                activity=act,
                lat=town["lat"] + rng.normal(0, 0.15),
                lon=town["lon"] + rng.normal(0, 0.2),
                length_m=length,
                ascent_m=int(rng.uniform(0, 900)),
                difficulty=int(rng.integers(0, 4)),
                tags=tags,
                is_loop=bool(rng.random() < 0.5),
            )
        )
    return Catalog.from_records(recs)
