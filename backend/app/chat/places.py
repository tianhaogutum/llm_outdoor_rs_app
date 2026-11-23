"""Place resolvers: a name the model extracted -> a catalogue locality with coordinates.

Coordinates never come from the model. Both resolvers use the app's place ranking
(``app.services.place_service.rank_places``: exact, prefix, word prefix, fuzzy for typos):

- :class:`DbPlaceResolver` reads the ``places`` table (the API);
- :class:`IndexPlaceResolver` builds the same places from the committed catalogue file with
  ``scripts.osm_catalog.build_places`` (the offline harness; no database needed).

A short exonym table maps English names of large Bavarian cities to their German names first, because
fuzzy matching cannot bridge "Munich" and "München".
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from app.models.entities import Place
from app.services.place_service import name_key, rank_places

from .schema import PlaceRef

PlaceResolver = Callable[[str], Awaitable[Optional[PlaceRef]]]

EXONYMS: dict[str, str] = {
    "munich": "München", "nuremberg": "Nürnberg", "wurzburg": "Würzburg", "ratisbon": "Regensburg",
    "fussen": "Füssen", "garmisch": "Garmisch-Partenkirchen", "lake tegernsee": "Tegernsee",
    "lindau": "Lindau (Bodensee)", "kempten": "Kempten (Allgäu)",
}


def canonical_name(name: str) -> str:
    return EXONYMS.get(name_key(name), name)


def to_ref(place: Place) -> PlaceRef:
    return PlaceRef(name=place.name, lat=place.lat, lon=place.lon, admin_district=place.admin_district)


class DbPlaceResolver:
    def __init__(self, db: Any) -> None:
        self.db = db

    async def __call__(self, name: str) -> Optional[PlaceRef]:
        from app.services.place_service import resolve_place

        place = await resolve_place(self.db, canonical_name(name))
        return to_ref(place) if place is not None else None


class IndexPlaceResolver:
    """In-memory resolver over a list of places (no database)."""

    def __init__(self, places: Iterable[Place]) -> None:
        self.places = list(places)

    @classmethod
    def from_records(cls, rows: Iterable[Mapping[str, Any]]) -> IndexPlaceResolver:
        return cls(Place(id=i + 1, **{k: r[k] for k in ("name", "name_key", "admin_district", "admin_region",
                                                          "lat", "lon", "accuracy_m", "route_count")})
                   for i, r in enumerate(rows))

    @classmethod
    def from_catalog(cls, catalog_dir: Path) -> IndexPlaceResolver:
        from scripts.osm_catalog import build_places

        with gzip.open(Path(catalog_dir) / "routes.jsonl.gz", "rt", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        return cls.from_records(build_places(records))

    def lookup(self, name: str) -> Optional[Place]:
        matches = rank_places(self.places, canonical_name(name), limit=1)
        return matches[0].place if matches else None

    async def __call__(self, name: str) -> Optional[PlaceRef]:
        place = self.lookup(name)
        return to_ref(place) if place is not None else None
