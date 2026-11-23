"""Pre-generated narrative content (docs/DEPLOY.md, "Pre-generated narratives").

Two user-visible texts come from an LLM when one is configured (``genai_service``):

- the **welcome summary** after the questionnaire (``POST /api/profiles``);
- the **post-route summary** when a route is completed (``POST /api/profiles/{id}/souvenirs``).

Everything else the user reads is built from templates or catalogue facts: route stories and quests
(``story_generator``), the souvenir picture (``svg_templates``), the recommendation reason line and the
explanation modal (``recsys_service``), the chat replies (``app.chat``), the route description (generated
from OpenStreetMap facts by the data pipeline).

So the demo does not depend on a live LLM, both LLM texts are generated once, offline, for a bounded
scope and committed under ``backend/data/narratives/`` as readable JSON with provenance:

- ``welcome.json``: one text per questionnaire combination (3 fitness levels x 8 adventure-type sets x 3
  narrative styles = 72);
- ``routes.json``: one route-level completion text per route in the demo set (``demo_routes.json``: the
  routes the served pipeline puts in the top 10 for a grid of demo profiles, see the script).

Order of sources at request time: live LLM (only if ``NARRATIVES_LIVE`` is on, a provider is configured and
the demo guard allows it) -> pre-generated text -> template. The response says which one was used
(``welcome_summary_source`` / ``summary_source``: ``llm`` | ``pregenerated`` | ``template``).

Regenerate with ``backend/scripts/narratives.py`` (replays from the committed cache without credentials).
The prompts, the route facts and the grounding checks live here so the script, the tests and the server
agree on them.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

from app.llm import UNTRUSTED_DATA_RULE, wrap_untrusted
from app.settings import PROJECT_ROOT

STORE_SCHEMA = "narratives/v1"
DEFAULT_NARRATIVES_DIR = PROJECT_ROOT / "data" / "narratives"
WELCOME_FILE = "welcome.json"
ROUTES_FILE = "routes.json"
DEMO_ROUTES_FILE = "demo_routes.json"
CACHE_SUBDIR = "cache"

SOURCE_LLM = "llm"
SOURCE_PREGENERATED = "pregenerated"
SOURCE_TEMPLATE = "template"

# The questionnaire (frontend/lib/mock-data.ts, questionnaireQuestions). TYPES is the canonical order.
FITNESS_LEVELS = ("beginner", "intermediate", "advanced")
ADVENTURE_TYPES = ("history-culture", "natural-scenery", "family-fun")
NARRATIVE_STYLES = ("adventure", "mystery", "playful")


# --------------------------------------------------------------------------------------- welcome keys
def canonical_types(types: Iterable[str]) -> Optional[tuple[str, ...]]:
    """Adventure types in canonical order without duplicates; None if any value is not a form option."""
    chosen = set(types)
    if not chosen <= set(ADVENTURE_TYPES):
        return None
    return tuple(t for t in ADVENTURE_TYPES if t in chosen)


def welcome_key(fitness: str, types: Iterable[str], narrative: str) -> Optional[str]:
    """``"<fitness>|<type+type>|<narrative>"`` (empty type set: ``-``), or None outside the form's options."""
    ct = canonical_types(types)
    if fitness not in FITNESS_LEVELS or narrative not in NARRATIVE_STYLES or ct is None:
        return None
    return f"{fitness}|{'+'.join(ct) or '-'}|{narrative}"


def all_welcome_combinations() -> list[tuple[str, tuple[str, ...], str]]:
    """Every (fitness, types, narrative) the form can send: 3 x 8 x 3 = 72 (the empty type set included)."""
    type_sets = [c for n in range(len(ADVENTURE_TYPES) + 1) for c in combinations(ADVENTURE_TYPES, n)]
    return [(f, ts, n) for f in FITNESS_LEVELS for ts in type_sets for n in NARRATIVE_STYLES]


# --------------------------------------------------------------------------------------------- prompts
WELCOME_PROMPT_VERSION = "welcome/pregen-v1"
ROUTE_PROMPT_VERSION = "route-complete/pregen-v1"

TYPE_PHRASES = {
    "history-culture": "history and culture",
    "natural-scenery": "natural scenery",
    "family-fun": "family fun",
}

WELCOME_SYSTEM = (
    "You write the welcome message a new user sees after answering a three-question form in an outdoor "
    "route app. The app recommends hiking, cycling and running routes in Bavaria, built from OpenStreetMap "
    "data.\n"
    "\n"
    "Rules:\n"
    "- Length: 60-90 words, English, second person.\n"
    "- Start with 'Welcome, <a short, creative explorer title that fits the answers>!'.\n"
    "- Talk only about the answers: fitness level, the kinds of adventure chosen, the story style.\n"
    "- Do not name or promise any specific place, route, landmark, lake, castle, legend, event or person, "
    "and do not use numbers.\n"
    "- Do not invent features of the app (no rewards, maps, badges, guides or levels).\n"
    "- Tone: {style}\n"
    "\n"
    "Output only the message."
)

ROUTE_SYSTEM = (
    "You write the short congratulation a user sees after marking an outdoor route as completed.\n"
    "\n"
    "Rules:\n"
    "- Length: 40-70 words, English, second person, past tense, warm and specific.\n"
    "- Use only facts stated in the route facts: the route name, its length, climbing, shape, difficulty, "
    "the place it starts near, and named lakes, huts, viewpoints, castles or other features listed there.\n"
    "- Do not invent anything: no names, numbers, weather, season, wildlife, history, people, food or "
    "feelings about specific moments that the facts do not state. Copy any number exactly as written in "
    "the facts, with its unit.\n"
    "- Do not mention quests, XP, points, levels, speed or time taken, and do not suggest another route.\n"
    "- Mention at most three named features.\n"
    "\n"
    "Output only the text. " + UNTRUSTED_DATA_RULE
)

# Style descriptors for the pre-generated welcome (shorter than genai_service.NARRATIVE_STYLE_PROMPTS, without
# the "saga / secrets / clues" vocabulary that invites invented content).
WELCOME_STYLES = {
    "adventure": "epic and inspiring, the user as the protagonist of a journey; confident, still easy to read.",
    "mystery": "quietly mysterious and curious, inviting the user to look closer; never spooky.",
    "playful": "light-hearted, friendly and a little funny, like a cheerful guide talking to a friend.",
}


def build_welcome_pregen_prompt(fitness: str, types: Sequence[str], narrative: str) -> tuple[str, str]:
    """(system, user) for one pre-generated welcome text."""
    system = WELCOME_SYSTEM.format(style=WELCOME_STYLES[narrative])
    kinds = ", ".join(TYPE_PHRASES[t] for t in types) if types else "no preference (open to anything)"
    user = f"Answers:\n- Fitness: {fitness}\n- Kinds of adventure: {kinds}\n- Story style: {narrative}"
    return system, user


DIFFICULTY_WORDS = {0: "easy", 1: "moderate", 2: "difficult", 3: "very demanding"}
ACTIVITY_WORDS = {"hiking": "hiking", "cycling": "cycling", "mtb": "mountain biking", "running": "running"}


def _km(m: Optional[float]) -> Optional[str]:
    return None if m is None else f"{m / 1000:.1f} km"


def _int_m(m: Optional[float]) -> Optional[str]:
    return None if m is None else f"{int(round(m)):,} m"


def _duration(minutes: Optional[float]) -> Optional[str]:
    if not minutes:
        return None
    h, mins = divmod(int(round(minutes)), 60)
    return f"{h} h {mins:02d} min" if h else f"{mins} min"


def route_facts(record: dict[str, Any]) -> str:
    """The only facts a route text may use: the catalogue record's own fields, as plain lines.

    ``record`` is a catalogue record (``backend/data/catalog/routes.jsonl.gz``). Its ``description`` is itself
    generated deterministically from OpenStreetMap facts by the data pipeline (or is OSM's own text)."""
    lines = [f"Name: {record.get('title')}"]
    act = ACTIVITY_WORDS.get(record.get("activity") or "", record.get("activity"))
    if act or record.get("category_name"):
        lines.append(f"Activity: {act} ({record.get('category_name')})")
    lines.append("Shape: circular (ends where it starts)" if record.get("is_loop") else "Shape: one way (A to B)")
    for label, value in (
        ("Length", _km(record.get("length_m"))),
        ("Climbing", _int_m(record.get("ascent_m"))),
        ("Highest point", _int_m(record.get("max_ele_m"))),
        ("Estimated time", _duration(record.get("duration_min"))),
    ):
        if value:
            lines.append(f"{label}: {value}")
    diff = DIFFICULTY_WORDS.get(record.get("difficulty"))  # type: ignore[arg-type]
    if diff:
        lines.append(f"Difficulty: {diff}")
    place = record.get("locality")
    district = record.get("admin_district")
    if place:
        lines.append(f"Starts near: {place}" + (f" ({district})" if district else ""))
    if record.get("admin_region"):
        lines.append(f"Region: {record['admin_region']}, Bavaria")
    tags = record.get("tags") or []
    if tags:
        lines.append("Tags: " + ", ".join(tags))
    desc = (record.get("description") or "").strip()
    if desc:
        lines.append(f"Description: {desc}")
    return "\n".join(lines)


def build_route_pregen_prompt(record: dict[str, Any]) -> tuple[str, str]:
    """(system, user) for one pre-generated route completion text."""
    return ROUTE_SYSTEM, "Route facts:\n" + wrap_untrusted(route_facts(record), "route")


# ------------------------------------------------------------------------------------- grounding checks
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[A-Za-zÄÖÜäöüß][\w'’\-]*")
BANNED_ROUTE_TERMS = ("quest", "xp", "level", "points", "hogwarts", "wizard", "magic")
# The welcome may mention the user's fitness level and quests (the app has both); it may not drift into the
# legacy wizard theme.
BANNED_WELCOME_TERMS = ("hogwarts", "wizard", "magic", "xp")
# Capitalised words that are not names: sentence starts and the like.
COMMON_CAPITALISED = frozenset(
    """a after again along amid and as at before between beyond both but by congratulations every finally
    for from great here how i in it its just like nicely no not now of on once or out over so starting
    that the then there these this through to together today up was way welcome well what when where
    whether while with you your you've you're bravo brilliant fantastic wonderful hooray onward bavaria
    bavarian openstreetmap""".split()
)


def _numbers(text: str) -> list[float]:
    out = []
    for tok in _NUMBER.findall(text):
        t = tok.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", tok) else tok.replace(",", ".")
        try:
            out.append(float(t))
        except ValueError:  # pragma: no cover - e.g. "1.2.3"
            continue
    return out


def _supported_number(n: float, facts: Sequence[float]) -> bool:
    for f in facts:
        if abs(n - f) < 1e-9 or abs(n - round(f)) < 1e-9 or abs(n - round(f, 1)) < 1e-9:
            return True
    return False


def check_route_text(text: str, facts: str) -> dict[str, Any]:
    """Grounding checks for one route text against its facts block.

    - ``unsupported_numbers``: numbers in the text that are not in the facts (allowing rounding of a fact);
    - ``unknown_names``: capitalised words that occur nowhere in the facts and are not common words;
    - ``banned_terms``: quests, XP, levels and the like (those are filled in by a template at request time);
    - ``words``: length. ``ok`` is True when the first three are empty and the length is 20-100 words.
    """
    fact_numbers = _numbers(facts)
    unsupported = sorted({tok for tok in _NUMBER.findall(text)
                          if not _supported_number(_numbers(tok)[0], fact_numbers)})
    facts_lower = facts.lower()
    unknown = []
    for w in _WORD.findall(text):
        if not w[0].isupper():
            continue
        base = w.strip("'’-")
        if base.lower() in COMMON_CAPITALISED:
            continue
        if base.endswith(("'s", "’s")):
            base = base[:-2]
        if base.lower() in facts_lower:
            continue
        unknown.append(base)
    lower = text.lower()
    banned = [t for t in BANNED_ROUTE_TERMS
              if re.search(rf"\b{re.escape(t)}s?\b", lower) and not re.search(rf"\b{re.escape(t)}", facts_lower)]
    words = len(text.split())
    return {
        "ok": not unsupported and not unknown and not banned and 20 <= words <= 100,
        "unsupported_numbers": unsupported,
        "unknown_names": sorted(set(unknown)),
        "banned_terms": banned,
        "words": words,
    }


def check_welcome_text(text: str) -> dict[str, Any]:
    """The welcome text may not contain numbers or the banned terms; 40-120 words."""
    lower = text.lower()
    banned = [t for t in BANNED_WELCOME_TERMS if re.search(rf"\b{re.escape(t)}s?\b", lower)]
    numbers = _NUMBER.findall(text)
    words = len(text.split())
    return {"ok": not banned and not numbers and 40 <= words <= 120, "numbers": numbers, "banned_terms": banned,
            "words": words}


def clean_generated(text: str) -> str:
    """Strip markdown emphasis / headings and surrounding quotes; one paragraph."""
    t = text.strip().strip('"“”').strip()
    t = re.sub(r"[*_#]+", "", t)
    return " ".join(t.split())


# ------------------------------------------------------------------------------------------------ store
@dataclass
class NarrativeEntry:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeStore:
    welcome: dict[str, NarrativeEntry] = field(default_factory=dict)
    routes: dict[int, NarrativeEntry] = field(default_factory=dict)
    welcome_header: dict[str, Any] = field(default_factory=dict)
    routes_header: dict[str, Any] = field(default_factory=dict)

    def welcome_for(self, fitness: str, types: Iterable[str], narrative: str) -> Optional[NarrativeEntry]:
        key = welcome_key(fitness, types, narrative)
        return self.welcome.get(key) if key else None

    def route_for(self, route_id: int) -> Optional[NarrativeEntry]:
        return self.routes.get(int(route_id))

    def summary(self) -> dict[str, Any]:
        return {"welcome": len(self.welcome), "routes": len(self.routes),
                "welcome_model": self.welcome_header.get("provenance", {}).get("model"),
                "routes_model": self.routes_header.get("provenance", {}).get("model")}


def _read(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != STORE_SCHEMA:
        raise ValueError(f"{path.name}: schema {data.get('schema')!r}, expected {STORE_SCHEMA!r}")
    return data


def load_store(directory: Optional[Path] = None) -> NarrativeStore:
    """Read ``welcome.json`` and ``routes.json`` (missing files: empty store)."""
    d = Path(directory) if directory else DEFAULT_NARRATIVES_DIR
    store = NarrativeStore()
    w = _read(d / WELCOME_FILE)
    if w:
        store.welcome_header = {k: v for k, v in w.items() if k != "items"}
        for it in w.get("items", []):
            store.welcome[it["key"]] = NarrativeEntry(it["text"], {k: v for k, v in it.items() if k != "text"})
    r = _read(d / ROUTES_FILE)
    if r:
        store.routes_header = {k: v for k, v in r.items() if k != "items"}
        for it in r.get("items", []):
            store.routes[int(it["route_id"])] = NarrativeEntry(
                it["text"], {k: v for k, v in it.items() if k != "text"}
            )
    return store


_store: Optional[tuple[str, NarrativeStore]] = None
_store_lock = threading.Lock()


def narratives_dir(settings: Any = None) -> Path:
    if settings is None:
        from app.settings import get_settings

        settings = get_settings()
    d = getattr(settings, "narratives_dir", None)
    return Path(d) if d else DEFAULT_NARRATIVES_DIR


def get_store(settings: Any = None) -> NarrativeStore:
    """The process-wide store for the configured directory (read once; an unreadable store is empty)."""
    global _store
    d = narratives_dir(settings)
    with _store_lock:
        if _store is not None and _store[0] == str(d):
            return _store[1]
    try:
        store = load_store(d)
    except Exception as e:  # a broken file must not break onboarding: templates still answer
        from app.logger import get_logger

        get_logger(__name__).error("narratives: cannot read the pre-generated store in %s: %s", d.name, e)
        store = NarrativeStore()
    with _store_lock:
        _store = (str(d), store)
    return store


def reset_store() -> None:
    global _store
    with _store_lock:
        _store = None


# ------------------------------------------------------------------------------------- request helpers
def live_narratives_enabled(settings: Any = None) -> bool:
    """``NARRATIVES_LIVE`` is on and the narrative provider can attempt a call."""
    if settings is None:
        from app.settings import get_settings

        settings = get_settings()
    if not getattr(settings, "narratives_live", True):
        return False
    from app.services.provider_status import generation_configured

    return generation_configured(settings)


def offline_welcome(questionnaire: Any, settings: Any = None) -> tuple[str, str]:
    """(text, source) without an LLM: the pre-generated text for the answers, else the template."""
    entry = get_store(settings).welcome_for(questionnaire.fitness, questionnaire.type or [], questionnaire.narrative)
    if entry is not None:
        return entry.text, SOURCE_PREGENERATED
    from app.services.user_profile_service import generate_fallback_welcome

    return generate_fallback_welcome(questionnaire), SOURCE_TEMPLATE


def template_post_run(route_title: str, xp_total: int) -> str:
    """The original fallback text (souvenirs endpoint before pre-generation)."""
    return (
        f"Congratulations on completing {route_title}! "
        f"You earned {xp_total} XP. "
        f"Keep exploring to discover more adventures!"
    )


def offline_post_run(route_id: int, route_title: str, xp_total: int, quests_completed: int,
                     settings: Any = None) -> tuple[str, str]:
    """(text, source) without an LLM: the route's pre-generated text plus the user's own numbers (template
    sentence), else the template text."""
    entry = get_store(settings).route_for(route_id)
    if entry is None:
        return template_post_run(route_title, xp_total), SOURCE_TEMPLATE
    if quests_completed > 0:
        tail = f"You completed {quests_completed} quest{'s' if quests_completed != 1 else ''} and earned {xp_total} XP."
    else:
        tail = f"You earned {xp_total} XP."
    return f"{entry.text} {tail}", SOURCE_PREGENERATED
