"""
Resolve a typed place name to coordinates from the catalogue's own localities (``places`` table).

No external geocoder: the ``places`` table is built at seed time from the ``locality`` of every
route start (see ``scripts/osm_catalog.py:build_places``), so only places that some route starts
near can be found. With the legacy catalogue the table is empty and every lookup returns nothing.

Matching works on a normalised key (:func:`name_key`): case, accents, ``ß`` and the German
``ae/oe/ue`` spellings are folded, punctuation becomes spaces. Results are ranked

1. ``exact``  – the key equals the query,
2. ``prefix`` – the key starts with the query (autocomplete),
3. ``word``   – a later word of the key starts with the query ("partenk" → Garmisch-Partenkirchen),
4. ``fuzzy``  – only when 1-3 give fewer than ``limit`` results and the query has ≥ 3 characters:
   similarity ≥ :data:`FUZZY_MIN_RATIO` against the whole key or its first ``len(query)`` characters
   (typos such as "Garmish").

Within a group, places with more routes come first, then alphabetical.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Place

FUZZY_MIN_RATIO = 0.8
MATCH_RANK = {"exact": 0, "prefix": 1, "word": 2, "fuzzy": 3}

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_UMLAUT_SPELLINGS = (("ae", "a"), ("oe", "o"), ("ue", "u"))


def name_key(text: str) -> str:
    """Normalised form of a place name for matching: ``"Garmisch-Partenkirchen"`` → ``"garmisch partenkirchen"``."""
    s = unicodedata.normalize("NFKD", text.casefold().replace("ß", "ss"))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    for spelled, folded in _UMLAUT_SPELLINGS:  # "Muenchen" and "München" both become "munchen"
        s = s.replace(spelled, folded)
    return _NON_ALNUM.sub(" ", s).strip()


@dataclass(frozen=True)
class PlaceMatch:
    place: Place
    match: str  # exact | prefix | word | fuzzy
    score: float  # 1.0 except for fuzzy matches


def _classify(key: str, query: str) -> Optional[tuple[str, float]]:
    if key == query:
        return "exact", 1.0
    if key.startswith(query):
        return "prefix", 1.0
    if any(word.startswith(query) for word in key.split(" ")[1:]):
        return "word", 1.0
    return None


def _fuzzy_score(key: str, query: str) -> float:
    whole = SequenceMatcher(None, query, key).ratio()
    head = SequenceMatcher(None, query, key[: len(query)]).ratio()
    return max(whole, head)


def rank_places(places: list[Place], query: str, limit: int = 10) -> list[PlaceMatch]:
    """Pure ranking over already loaded places (see the module docstring)."""
    q = name_key(query)
    if not q or limit <= 0:
        return []
    matches: list[PlaceMatch] = []
    rest: list[Place] = []
    for place in places:
        found = _classify(place.name_key, q)
        if found:
            matches.append(PlaceMatch(place, *found))
        else:
            rest.append(place)
    if len(matches) < limit and len(q) >= 3:
        for place in rest:
            score = _fuzzy_score(place.name_key, q)
            if score >= FUZZY_MIN_RATIO:
                matches.append(PlaceMatch(place, "fuzzy", round(score, 3)))
    matches.sort(key=lambda m: (MATCH_RANK[m.match], -m.score, -(m.place.route_count or 0),
                                m.place.name_key, m.place.admin_district or "", m.place.id or 0))
    return matches[:limit]


async def search_places(db: AsyncSession, query: str, limit: int = 10) -> list[PlaceMatch]:
    """Rank the ``places`` table for ``query``. The table has a few thousand rows; a scan is ~ms."""
    if not name_key(query):
        return []
    result = await db.execute(select(Place))
    return rank_places(list(result.scalars().all()), query, limit)


async def resolve_place(db: AsyncSession, name: str) -> Optional[Place]:
    """Best match for ``name`` (exact, then prefix, word, fuzzy), or None."""
    matches = await search_places(db, name, limit=1)
    return matches[0].place if matches else None
