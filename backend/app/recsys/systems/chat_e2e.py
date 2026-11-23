"""Phase 4 end-to-end systems: what the default pipeline does with less or different free text.

- ``recsys_v1_qonly``: questionnaire only. The request's intent is dropped; the user vector and the form's
  profile slots (home, radius, cap) remain, so the pipeline takes its no-intent path (filter -> rule
  scorer -> MMR; docs/RECSYS.md section 6).
- ``recsys_v1`` itself is "questionnaire + raw intent text" (the intent typed into a free-text box).
- ``recsys_v1_chat``: questionnaire + chat. The chat session starts from the form's slots (activity from
  the tab, home as the place, radius, cap, difficulty and distance bands read back from the vector), the
  intent is sent as **one chat message** through the real turn logic (extractor, validation, merge rules,
  place lookup), and the pipeline gets ``ChatSession.user_query``: the merged slots, the incrementally
  projected vector and the extractor's restated intent (no intent if the extractor gave none).

The chat call is an upstream call of the system (cost and latency in the trace stage ``chat``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from ..catalog import Catalog
from ..query import UserQuery
from ..result import Recommendation, StageTrace
from .base import register_system


class QuestionnaireOnly:
    def __init__(self, catalog: Catalog, *, base: str = "recsys_v1", **kw: Any) -> None:
        from . import build_system

        self.name = "recsys_v1_qonly"
        self.pipe = build_system(base, catalog, **kw)
        self.base = base

    def config(self) -> dict[str, Any]:
        return {**self.pipe.config(), "system": self.name, "wraps": self.base, "input": "questionnaire only"}

    async def arecommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        rec = await self.pipe.arecommend(replace(query, intent=None), k)
        rec.system = self.name
        return rec

    def recommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        from app.llm import run_sync

        return run_sync(self.arecommend(query, k))


class WithChat:
    def __init__(self, catalog: Catalog, *, base: str = "recsys_v1", chat_llm=None, resolver=None, **kw: Any) -> None:
        from . import build_system

        self.name = "recsys_v1_chat"
        self.catalog = catalog
        self.pipe = build_system(base, catalog, **kw)
        self.base = base
        self._chat_llm = chat_llm
        self._resolver = resolver

    @property
    def chat_llm(self):
        if self._chat_llm is None:
            from app.chat.llm import get_chat_client

            self._chat_llm = get_chat_client()
        return self._chat_llm

    @property
    def resolver(self):
        if self._resolver is None:
            from app.chat import IndexPlaceResolver
            from app.recsys.catalog import DEFAULT_CATALOG_DIR

            self._resolver = self.catalog.memo(
                "chat:index-place-resolver", lambda: IndexPlaceResolver.from_catalog(DEFAULT_CATALOG_DIR)
            )
        return self._resolver

    def config(self) -> dict[str, Any]:
        cm = getattr(self.chat_llm, "_chat_model", None)
        chat = {"provider": cm.provider, "model": cm.model, "thinking": cm.thinking} if cm else {}
        from app.chat.extract import PROMPT_VERSION

        c = self.pipe.config()
        return {
            **c,
            "system": self.name,
            "wraps": self.base,
            "input": "questionnaire + the intent sent as one chat message",
            "chat": {**chat, "prompt_version": PROMPT_VERSION},
            "models": [*c.get("models", []), f"chat:{chat.get('model')}"],
        }

    def start_slots(self, query: UserQuery):
        from app.chat import ChatSlots, PlaceRef
        from app.chat.vocab import band_for_range, difficulty_for_range

        v = query.user_vector or {}
        activity = query.category if query.category in ("hiking", "cycling", "running") else None
        return ChatSlots(
            activity=activity,
            difficulty=difficulty_for_range(v.get("difficulty_range")),
            distance_band=band_for_range(v.get("min_distance_km"), v.get("max_distance_km"), activity),
            distance_cap_km=query.max_distance_km,
            place=PlaceRef(name="home", lat=query.home[0], lon=query.home[1]) if query.home else None,
            travel_radius_km=query.max_travel_km if query.home else None,
        )

    async def chat_query(self, query: UserQuery) -> tuple[UserQuery, dict[str, Any], float, float]:
        from app.chat import ChatSession, ChatState
        from app.chat.engine import default_extractor

        session = ChatSession(ChatState(), self.start_slots(query), dict(query.user_vector or {}),
                              resolver=self.resolver, extractor=default_extractor(self.chat_llm))
        info: dict[str, Any] = {"message_chars": len(query.intent or "")}
        lat = cost = 0.0
        if query.intent:
            res = await session.send(query.intent)
            ext = res.extraction
            lat, cost = float(ext.nominal_latency_ms), float(ext.nominal_cost_usd)
            info.update(
                status=res.status,
                changes=[{"slot": c.slot, "new": _plain(c.new), "source": c.source} for c in res.changes],
                dropped=[d.field for d in res.dropped],
                restated_intent=session.slots.intent,
                model=ext.model,
            )
        q = session.user_query(request_id=query.request_id, category=query.category)
        return replace(q, feedback=query.feedback), info, lat, cost

    async def arecommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        q, info, lat, cost = await self.chat_query(query)
        rec = await self.pipe.arecommend(q, k)
        rec.trace.insert(0, StageTrace("chat", 1, 1, 0.0, lat, cost, info=info))
        rec.system = self.name
        return rec

    def recommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        from app.llm import run_sync

        return run_sync(self.arecommend(query, k))


def _plain(x: Any) -> Any:
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return x


@register_system("recsys_v1_qonly", description="recsys_v1 on the questionnaire only (intent dropped)", uses_llm=True)
def _make_qonly(catalog: Catalog, **cfg: Any) -> QuestionnaireOnly:
    return QuestionnaireOnly(catalog, **cfg)


@register_system(
    "recsys_v1_chat",
    description="recsys_v1 after the intent is sent as one chat message (slots merged, restated intent)",
    uses_llm=True,
    uses_chat=True,
)
def _make_chat(catalog: Catalog, *, chat_llm: Optional[Any] = None, **cfg: Any) -> WithChat:
    return WithChat(catalog, chat_llm=chat_llm, **cfg)
