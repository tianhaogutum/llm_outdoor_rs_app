"""
GenAI service for generating personalized narratives and summaries.

This service encapsulates the narrative LLM interactions:
- Welcome summaries (US-04)
- Post-run summaries (US-13)
- Pixel art SVG souvenirs (template-based, no LLM)

Every LLM call goes through the single client in ``app.llm`` (provider chosen
by ``LLM_PROVIDER``: vertex, openrouter, ollama or mock). Prompts are kept as
plain system text + few-shot (user, assistant) examples + the user turn; the
provider layer applies the model-specific format (Llama 3.1 chat template for
Ollama, multi-turn contents for Vertex, chat messages for OpenRouter).

Prompt engineering:
- Few-shot examples for consistency
- Clear system instructions
- Explicit length constraints
"""
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Optional

from fastapi import HTTPException

from app.api.schemas import ProfileCreate
from app.llm import BudgetExceeded, ProviderError, get_client
from app.logger import get_logger, log_api_call

logger = get_logger(__name__)


# Narrative style → LLM prompt style descriptors
# These detailed instructions ensure the LLM produces consistent, style-appropriate output
# Used across all GenAI services (welcome summaries, story generation, etc.)
NARRATIVE_STYLE_PROMPTS = {
    "adventure": (
        "Write in an epic, cinematic tone. Emphasize a sense of grand journey, "
        "meaningful challenges, and heroic progression. Use language that makes the "
        "user feel like the protagonist of a larger saga, with each route as a new "
        "chapter in their legend. Keep the tone inspiring, confident, and slightly "
        "dramatic, but still clear and easy to read."
    ),
    "mystery": (
        "Write in a mysterious, slightly suspenseful tone. Emphasize hidden corners, "
        "quiet details, and secrets waiting to be uncovered. Use language that suggests "
        "clues, layers, and discoveries rather than stating everything directly. "
        "The tone should be intriguing and immersive, inviting the user to look closer "
        "and follow the trail of hints."
    ),
    "playful": (
        "Write in a light-hearted, playful tone. Emphasize fun, curiosity, and small "
        "joyful moments rather than serious challenges. Use friendly, energetic language "
        "and a touch of humor, as if you are a cheerful guide talking to a friend. "
        "The tone should feel casual, encouraging, and welcoming, suitable for users "
        "who want relaxed adventures."
    ),
}


def _current_model() -> str:
    client = get_client()
    return f"{client.config.llm_provider}:{client.config.default_model('generate')}"


async def call_llm(
    prompt: str,
    *,
    system: Optional[str] = None,
    examples: Sequence[tuple[str, str]] = (),
    max_tokens: int = 300,
    temperature: float = 0.8,
    tag: str = "narrative",
) -> str:
    """
    Generate narrative text through the LLM client, with HTTP error mapping.

    Narrative calls are creative (temperature > 0) and user-facing, so they
    bypass the cache (``cache=False``) and retry a transient error once only,
    so the API can fall back quickly. They still count against the budget.

    Raises
    ------
    HTTPException
        upstream HTTP status for provider HTTP errors, 504 on timeout,
        503 when the LLM budget is exhausted, 500 for anything else
        (including an empty response).
    """
    client = get_client()
    provider = client.config.llm_provider
    model = client.config.default_model("generate")
    start_time = time.time()
    logger.debug(
        f"🤖 Calling LLM: provider={provider}, model={model}, max_tokens={max_tokens}, temperature={temperature}"
    )

    def _log(success: bool, **kw) -> None:
        log_api_call(
            logger,
            "LLM",
            f"{provider}:{model}",
            method="POST",
            duration_ms=(time.time() - start_time) * 1000,
            success=success,
            **kw,
        )

    try:
        result = await client.generate(
            prompt,
            system=system,
            examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
            tag=tag,
            cache=False,
            retries=1,
        )
    except ProviderError as e:
        if e.timeout:
            logger.error(f"❌ LLM timeout: provider={provider}")
            _log(False, error="timeout")
            raise HTTPException(status_code=504, detail=f"LLM timeout ({provider})") from e
        if e.status_code is not None and 400 <= e.status_code < 600:
            logger.error(f"❌ LLM HTTP error: status={e.status_code}, detail={e}")
            _log(False, status_code=e.status_code)
            raise HTTPException(status_code=e.status_code, detail=f"LLM provider error: {e}") from e
        logger.error(f"❌ LLM call failed: {e}")
        _log(False, error="provider_error")
        raise HTTPException(status_code=500, detail=f"Story generation failed: {e}") from e
    except BudgetExceeded as e:
        logger.error(f"❌ LLM budget exhausted: {e}")
        _log(False, error="budget_exceeded")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"❌ LLM call failed: {type(e).__name__}: {str(e)}", exc_info=True)
        _log(False, error=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}") from e

    text = result.text.strip()
    if not text:
        logger.error("❌ LLM returned empty response")
        _log(False, error="empty_response")
        raise HTTPException(status_code=500, detail="Story generation failed: Empty response from LLM")
    logger.debug(
        f"✅ LLM call succeeded: response_length={len(text)} chars, cost={result.cost_usd:.6f} USD"
    )
    _log(True, model=result.model, response_length=len(text))
    return text


async def call_ollama(
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.8
) -> str:
    """
    Backwards-compatible wrapper: send ``prompt`` as a single user turn to the
    configured provider (no longer necessarily Ollama). New code should call
    :func:`call_llm` with ``system`` and ``examples`` instead of embedding a
    chat template in the prompt.
    """
    return await call_llm(prompt, max_tokens=max_tokens, temperature=temperature, tag="narrative.raw")


# ---------------------------------------------------------------------------
# Welcome summary (US-04)
# ---------------------------------------------------------------------------
WELCOME_SYSTEM_TEMPLATE = (
    'You are the TrailSaga – Hogwarts Expedition Series AI Guide. Generate a personalized, '
    'engaging welcome message for a new user.\n'
    '\n'
    'Guidelines:\n'
    '- Length: 80-100 words\n'
    '- Tone: Warm, encouraging, and adventurous\n'
    '- Focus on: fitness level, adventure preferences, narrative style\n'
    '- Do NOT: invent character names, describe physical appearance, use generic phrases\n'
    '- Start with a creative explorer title that matches their profile\n'
    '\n'
    'The narrative style should follow this description: "{narrative_hint}"\n'
    '\n'
    'Output only the welcome message, no additional text.'
)

WELCOME_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        (
            'Profile:\n'
            '- Fitness: Beginner\n'
            '- Type: family-fun\n'
            '- Narrative: playful'
        ),
        (
            'Welcome, Joyful Explorer! It is wonderful to have you here. Since you are just starting '
            'your journey and enjoy family-friendly fun, TrailSaga – Hogwarts Expedition Series has '
            'prepared a delightful collection of easygoing and playful adventures just for you. Expect '
            'to find charming theme trails where learning and games go hand in hand. We will keep the '
            "pace relaxed so you can fully enjoy the smiles of your loved ones. Let's turn every small "
            'step into a happy memory!'
        ),
    ),
    (
        (
            'Profile:\n'
            '- Fitness: Advanced\n'
            '- Type: hiking, natural-scenery\n'
            '- Narrative: mystery'
        ),
        (
            'Greetings, Seeker of the Unknown! Your impressive fitness level tells us you are ready to '
            'conquer steep paths and deep forests. Because you are drawn to natural scenery and '
            'mystery, we have curated routes that lead not just to breathtaking views, but to ancient '
            'secrets hidden in the landscape. Prepare for challenging hikes where every turn might '
            'reveal a forgotten legend or a hidden ruin. The wild is calling, and it has a puzzle '
            'waiting for you to solve.'
        ),
    ),
    (
        (
            'Profile:\n'
            '- Fitness: Intermediate\n'
            '- Type: history-culture, urban-exploration\n'
            '- Narrative: adventure'
        ),
        (
            'Welcome, Urban Adventurer! Your solid fitness foundation makes you perfectly suited for '
            "exploring the stories hidden within city streets and historical landmarks. We've selected "
            'routes that blend physical challenge with cultural discovery, taking you through '
            'architectural marvels, forgotten alleyways, and vibrant neighborhoods. Each path is a '
            'chapter in a larger adventure, where past and present intertwine. Get ready to uncover '
            'tales that most visitors never see, one stride at a time.'
        ),
    ),
)


def build_welcome_prompt(questionnaire: ProfileCreate) -> tuple[str, tuple[tuple[str, str], ...], str]:
    """Return (system, few-shot examples, user prompt) for the welcome summary."""
    adventure_types_str = ", ".join(questionnaire.type) if questionnaire.type else "exploration"
    narrative_hint = NARRATIVE_STYLE_PROMPTS.get(
        questionnaire.narrative,
        "Use a neutral, friendly narrative tone."
    )
    system = WELCOME_SYSTEM_TEMPLATE.format(narrative_hint=narrative_hint)
    user = (
        f"Profile:\n- Fitness: {questionnaire.fitness}\n- Type: {adventure_types_str}\n"
        f"- Narrative: {questionnaire.narrative}"
    )
    return system, WELCOME_EXAMPLES, user


async def generate_welcome_summary(questionnaire: ProfileCreate) -> str:
    """
    Generate a personalized welcome summary (US-04).

    This function creates an engaging explorer identity summary based on
    the user's questionnaire answers, using few-shot prompting.

    Parameters
    ----------
    questionnaire : ProfileCreate
        User's questionnaire answers (fitness, type, narrative)

    Returns
    -------
    str
        A personalized welcome message (80-100 words)

    Raises
    ------
    HTTPException
        If the LLM call fails (see :func:`call_llm` for the status mapping)
    """
    system, examples, user = build_welcome_prompt(questionnaire)

    response = await call_llm(
        user,
        system=system,
        examples=examples,
        max_tokens=170,
        temperature=0.6,  # Balanced creativity and consistency
        tag="narrative.welcome",
    )

    # Clean up response
    generated_text = response.strip()

    # Remove any markdown or extra formatting
    if generated_text.startswith("**") or generated_text.startswith("#"):
        lines = generated_text.split('\n')
        generated_text = ' '.join(line.strip('*#').strip() for line in lines if line.strip())

    if not generated_text:
        raise ValueError("Empty response from LLM")

    model = _current_model()
    # Print LLM generated content to console
    print("\n" + "="*80)
    print("✨ LLM GENERATED: Welcome Summary")
    print("="*80)
    print(
        f"📋 Profile: Fitness={questionnaire.fitness}, Type={questionnaire.type}, "
        f"Narrative={questionnaire.narrative}"
    )
    print(f"🤖 Model: {model}")
    print("-"*80)
    print(generated_text)
    print("="*80 + "\n")

    # Log to UI
    from app.llm_logger import log_llm_output
    log_llm_output(
        message_type="welcome",
        title="✨ Welcome Summary Generated",
        content=generated_text,
        metadata={
            "fitness": questionnaire.fitness,
            "type": questionnaire.type,
            "narrative": questionnaire.narrative,
            "model": model,
        }
    )

    return generated_text


# ---------------------------------------------------------------------------
# Post-run summary (US-13)
# ---------------------------------------------------------------------------
POST_RUN_SYSTEM = (
    'You are the TrailSaga – Hogwarts Expedition Series Achievement Generator. Create a '
    'motivating summary for a completed adventure.\n'
    '\n'
    'Guidelines:\n'
    '- Language: English only (do not use any other language)\n'
    '- Length: 60-80 words\n'
    '- Tone: Encouraging, celebratory, forward-looking\n'
    '- Include: specific achievements, recognition of effort, next challenge suggestion\n'
    '- Be specific to their performance and level\n'
    '\n'
    'Output only the summary, no additional text.'
)

POST_RUN_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        (
            'Route: Mountain Vista Trail\n'
            'Distance: 12.5 km\n'
            'Quests: 3/4 (75%)\n'
            'Level: 5'
        ),
        (
            'Congratulations on conquering the Mountain Vista Trail! You pushed through 12.5 kilometers'
            ' of challenging terrain and completed most of your quests. Your determination is truly '
            "impressive. You're ready for even greater challenges now. Why not try a route with more "
            'elevation gain next time?'
        ),
    ),
    (
        (
            'Route: River Loop Discovery\n'
            'Distance: 5.2 km\n'
            'Quests: 5/5 (100%)\n'
            'Level: 3'
        ),
        (
            'Perfect completion of the River Loop Discovery! You completed all five quests and covered '
            "every meter with focus and enthusiasm. This flawless performance shows you're mastering "
            "the fundamentals beautifully. Level 3 suits you well, but don't be surprised if you're "
            'ready for intermediate trails soon. Consider exploring urban heritage routes next.'
        ),
    ),
)


def build_post_run_prompt(
    route_title: str,
    route_length_km: float,
    quests_completed: int,
    total_quests: int,
    user_level: int,
) -> tuple[str, tuple[tuple[str, str], ...], str]:
    """Return (system, few-shot examples, user prompt) for the post-run summary."""
    quest_completion_rate = (quests_completed / total_quests * 100) if total_quests > 0 else 0
    user = (
        f"Route: {route_title}\nDistance: {route_length_km} km\n"
        f"Quests: {quests_completed}/{total_quests} ({quest_completion_rate:.0f}%)\nLevel: {user_level}"
    )
    return POST_RUN_SYSTEM, POST_RUN_EXAMPLES, user


async def generate_post_run_summary(
    route_title: str,
    route_length_km: float,
    quests_completed: int,
    total_quests: int,
    user_level: int,
) -> str:
    """
    Generate a post-run summary and next challenge suggestion (US-13).

    Creates motivating, personalized feedback based on user's performance
    with actionable suggestions for next adventures.

    Parameters
    ----------
    route_title : str
        Completed route name
    route_length_km : float
        Route length in kilometers
    quests_completed : int
        Number of quests completed
    total_quests : int
        Total number of quests
    user_level : int
        User's current level

    Returns
    -------
    str
        Personalized summary and suggestions (60-80 words)

    Raises
    ------
    HTTPException
        If the LLM call fails (see :func:`call_llm` for the status mapping)
    """
    quest_completion_rate = (quests_completed / total_quests * 100) if total_quests > 0 else 0
    system, examples, user = build_post_run_prompt(
        route_title, route_length_km, quests_completed, total_quests, user_level
    )

    response = await call_llm(
        user,
        system=system,
        examples=examples,
        max_tokens=150,
        temperature=0.75,  # Balanced for consistency and variety
        tag="narrative.post_run",
    )

    generated_text = response.strip()
    if not generated_text:
        raise ValueError("Empty response from LLM")

    # Print LLM generated content to console
    print("\n" + "="*80)
    print("✨ LLM GENERATED: Post-Run Summary")
    print("="*80)
    print(f"🗺️ Route: {route_title}")
    print(f"📏 Distance: {route_length_km} km")
    print(f"🏆 Quests: {quests_completed}/{total_quests} ({quest_completion_rate:.0f}%)")
    print(f"📈 Level: {user_level}")
    print("-"*80)
    print(generated_text)
    print("="*80 + "\n")

    # Log to UI
    from app.llm_logger import log_llm_output
    log_llm_output(
        message_type="post_run",
        title=f"🏁 Post-Run Summary: {route_title}",
        content=generated_text,
        metadata={
            "route_title": route_title,
            "distance_km": route_length_km,
            "quests_completed": quests_completed,
            "total_quests": total_quests,
            "user_level": user_level,
            "model": _current_model(),
        }
    )

    return generated_text


async def generate_pixel_art_svg(
    route_title: str,
    route_location: Optional[str],
    completed_at: datetime,
    xp_gained: int,
    distance_km: float,
    difficulty: Optional[int] = None,
) -> str:
    """
    Generate a pixel art style SVG image using pre-designed templates.

    Uses a template-based system with 10 different Harry Potter themed designs.
    Randomly selects a template and fills it with route completion data.

    Parameters
    ----------
    route_title : str
        Name of the completed route
    route_location : Optional[str]
        Location of the route
    completed_at : datetime
        Completion timestamp
    xp_gained : int
        Total XP earned
    distance_km : float
        Route distance in kilometers
    difficulty : Optional[int]
        Route difficulty (0-3 scale)

    Returns
    -------
    str
        SVG code as a string (pixel art style)
    """
    # Use template-based system instead of LLM generation
    from app.services.svg_templates import generate_souvenir_svg

    # Map difficulty
    difficulty_map = {0: "Easy", 1: "Moderate", 2: "Difficult", 3: "Expert"}
    difficulty_str = difficulty_map.get(difficulty, "Unknown") if difficulty is not None else "Unknown"

    # Generate SVG using template system
    generated_svg = generate_souvenir_svg(
        route_title=route_title,
        route_location=route_location,
        completed_at=completed_at,
        xp_gained=xp_gained,
        distance_km=distance_km,
        difficulty=difficulty_str
    )

    # Format for logging
    date_str = completed_at.strftime("%Y-%m-%d")
    time_str = completed_at.strftime("%H:%M")
    location_str = route_location if route_location else "Unknown Location"

    # Print generated content to console
    print("\n" + "="*80)
    print("🎨 TEMPLATE GENERATED: Pixel Art SVG")
    print("="*80)
    print(f"🗺️ Route: {route_title}")
    print(f"📍 Location: {location_str}")
    print(f"📅 Completed: {date_str} at {time_str}")
    print(f"⭐ XP: {xp_gained}")
    print(f"📏 Distance: {distance_km:.1f} km")
    print(f"🎯 Difficulty: {difficulty_str}")
    print("-"*80)
    print(f"SVG Length: {len(generated_svg)} characters")
    print("="*80 + "\n")

    # Log to UI
    from app.llm_logger import log_llm_output
    log_llm_output(
        message_type="pixel_art",
        title=f"🎨 Pixel Art Generated: {route_title}",
        content=f"Generated {len(generated_svg)} character SVG (Template-based)",
        metadata={
            "route_title": route_title,
            "location": location_str,
            "completed_at": date_str,
            "xp_gained": xp_gained,
            "distance_km": distance_km,
            "difficulty": difficulty_str,
            "method": "template"
        }
    )

    return generated_svg
