"""
Chat preference elicitation on a profile (BUILD_SPEC Phase 4, docs/CHAT.md).

- POST   /api/profiles/{id}/chat         one user message in, one assistant turn out
- GET    /api/profiles/{id}/chat         current conversation, slots and intent
- PATCH  /api/profiles/{id}/chat/slots   structured edit from the UI chips (no LLM)
- DELETE /api/profiles/{id}/chat         start a new conversation (slots are kept)

The conversation state lives on the server (``demo_profiles.chat_state_json``); the slots are written to
the profile (user vector and slot columns), so the recommendations endpoint uses them on the next call.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schema import ChatRequest, ChatResponse, SlotEditRequest
from app.database import get_db
from app.llm import BudgetExceeded, CacheMiss, LLMError, ProviderError, SchemaValidationError
from app.logger import get_logger
from app.models.entities import DemoProfile
from app.services.chat_service import load_session, store_session

logger = get_logger(__name__)

router = APIRouter(prefix="/profiles", tags=["chat"])


async def _profile(db: AsyncSession, profile_id: int) -> DemoProfile:
    profile = await db.get(DemoProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Profile with id {profile_id} not found")
    return profile


@router.post("/{profile_id}/chat", response_model=ChatResponse)
async def chat_turn(
    profile_id: int, body: ChatRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> ChatResponse:
    profile = await _profile(db, profile_id)
    session = load_session(profile, db)
    try:
        result = await session.send(body.message)
    except BudgetExceeded as e:
        logger.warning("chat budget exhausted: %s", e)
        raise HTTPException(status_code=503, detail="The chat is not available right now (budget).") from e
    except SchemaValidationError as e:
        logger.warning("chat extraction violated the schema: %s", e)
        raise HTTPException(status_code=502, detail="The language model returned an invalid answer; try again.") from e
    except (ProviderError, CacheMiss, LLMError) as e:
        logger.warning("chat extraction failed: %s", type(e).__name__)
        code = 504 if isinstance(e, ProviderError) and e.timeout else 503
        raise HTTPException(status_code=code, detail="The language model is not reachable right now.") from e
    store_session(profile, session)
    await db.commit()
    logger.info("chat turn profile=%s status=%s changes=%s dropped=%s", profile_id, result.status,
                [c.slot for c in result.changes], [d.field for d in result.dropped])
    return session.response(profile_id, result)


@router.get("/{profile_id}/chat", response_model=ChatResponse)
async def chat_state(profile_id: int, db: Annotated[AsyncSession, Depends(get_db)]) -> ChatResponse:
    profile = await _profile(db, profile_id)
    return load_session(profile, db).response(profile_id)


@router.patch("/{profile_id}/chat/slots", response_model=ChatResponse)
async def edit_slots(
    profile_id: int, body: SlotEditRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> ChatResponse:
    profile = await _profile(db, profile_id)
    session = load_session(profile, db)
    result = await session.edit(body)
    store_session(profile, session)
    await db.commit()
    return session.response(profile_id, result)


@router.delete("/{profile_id}/chat", response_model=ChatResponse)
async def reset_chat(profile_id: int, db: Annotated[AsyncSession, Depends(get_db)]) -> ChatResponse:
    profile = await _profile(db, profile_id)
    session = load_session(profile, db)
    session.reset()
    store_session(profile, session)
    await db.commit()
    return session.response(profile_id)
