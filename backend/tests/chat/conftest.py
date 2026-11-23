"""Chat test helpers: an offline chat client and a tiny in-memory place index."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.chat.llm import ChatModel, build_chat_client, set_chat_client
from app.chat.places import IndexPlaceResolver
from app.llm import LLMConfig
from app.services.place_service import name_key

PLACES = [
    {"name": "München", "admin_district": "München", "lat": 48.137, "lon": 11.575, "route_count": 22},
    {"name": "Garmisch-Partenkirchen", "admin_district": "Landkreis Garmisch-Partenkirchen", "lat": 47.49,
     "lon": 11.096, "route_count": 6},
    {"name": "Tegernsee", "admin_district": "Landkreis Miesbach", "lat": 47.71, "lon": 11.758, "route_count": 5},
    {"name": "Füssen", "admin_district": "Landkreis Ostallgäu", "lat": 47.57, "lon": 10.70, "route_count": 3},
]


@pytest.fixture
def resolver() -> IndexPlaceResolver:
    return IndexPlaceResolver.from_records(
        [{**p, "name_key": name_key(p["name"]), "admin_region": "R", "accuracy_m": 0} for p in PLACES])


@pytest.fixture
def chat_client(tmp_path) -> Iterator:
    """A mock chat client (keyword extractor installed, no cache) installed process-wide."""
    client = build_chat_client(ChatModel("mock", "mock-llm-1", thinking=None),
                               config=LLMConfig(cache_dir=tmp_path / "llm", cache_enabled=False), budget_usd=0.05)
    set_chat_client(client)
    yield client
    set_chat_client(None)
