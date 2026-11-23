"""Keyword extractor: the offline stand-in for the LLM.

Registered as the mock provider's responder for the chat tag, so with ``LLM_PROVIDER=mock`` (tests, CI,
a demo without credentials) the chat still understands simple messages. It also runs in the scripted
evaluation as a non-LLM baseline row. It fills the same wire schema (:class:`ChatExtraction`) and goes
through the same validator as the LLM, so nothing it produces bypasses the safety checks.

Deliberately simple: regular expressions over English and German keywords, a three-word window for
negations. It is not meant to be good; it is meant to be deterministic and honest about that.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .schema import MAX_INTENT_CHARS

_F = re.IGNORECASE | re.UNICODE

_ACTIVITY = (
    ("mtb", re.compile(r"\bmtb\b|mountain ?bik\w*|mountainbik\w*|singletrail\w*", _F)),
    ("cycling", re.compile(r"\bradtour\w*|\bfahrrad\w*|\brad(fahren|eln|weg|runde)?\b|\bbik(e|es|ing)\b|"
                           r"\bcycl\w*|\bride\b|\brennrad\w*|\be-?bike\w*", _F)),
    ("running", re.compile(r"\brun(s|ning)?\b|\bjog\w*|\blauf(en|runde|strecke)?\b|\btrail ?run\w*", _F)),
    ("hiking", re.compile(r"\bhik\w*|\bwalk\w*|\bwander\w*|\bspazier\w*|\bstroll\w*|\btrek\w*|\bbergtour\w*", _F)),
)
_EASY = re.compile(r"\b(easy|leicht\w*|einfach\w*|gentle|gemütlich\w*|beginners?|anfänger\w*|relaxed|entspannt\w*)\b|"
                   r"not too (steep|hard|difficult)|nicht zu (steil|schwer|anstrengend)", _F)
_HARD = re.compile(r"\b(challenging|anspruchsvoll\w*|hard|schwer|schwierig\w*|difficult|sporty|sportlich\w*|"
                   r"demanding|strenuous|fordernd\w*)\b", _F)
_MODERATE = re.compile(r"\b(moderate|mittelschwer\w*|medium difficulty|nicht zu leicht)\b", _F)
_EASIER = re.compile(r"\b(easier|leichter|einfacher|less (steep|hard))\b", _F)
_HARDER = re.compile(r"\b(harder|schwerer|anspruchsvoller|more challenging|tougher)\b", _F)
_SHORTER = re.compile(r"\b(shorter|kürzer\w*)\b", _F)
_LONGER = re.compile(r"\b(longer|länger\w*)\b", _F)
_SHORT = re.compile(r"\b(short|kurz\w*|quick|little)\b", _F)
_LONG = re.compile(r"\b(long|lang\w*|big tour)\b", _F)

_NUM = r"(\d+(?:[.,]\d+)?)"
_CAP = re.compile(r"(max(?:imal|imum)?\.?|höchstens|at most|no more than|not more than|under|unter|"
                  r"bis zu|up to|weniger als|less than)\s*" + _NUM + r"\s*(km|k|kilomet\w*)\b", _F)
_RADIUS = re.compile(r"(within|im umkreis von|umkreis von|radius (?:of|von)|nicht weiter als)\s*" + _NUM
                     + r"\s*(km|kilomet\w*)", _F)
_TARGET = re.compile(r"(about|around|circa|ca\.?|etwa|ungefähr|rund|so)\s*" + _NUM + r"\s*(km|k)\b", _F)
_HOURS = re.compile(_NUM + r"\s*(h|hrs?|hours?|stunden?|std\.?)\b", _F)
_DURATION_WORDS = (
    (re.compile(r"couple of hours|paar stunden|zwei stunden|two hours", _F), 2.0),
    (re.compile(r"few hours|einige stunden|drei stunden|three hours", _F), 3.0),
    (re.compile(r"half[- ]?(a )?day|halbe?n? tag\w*|halbtag\w*|vormittag|nachmittag|afternoon|morning", _F), 4.0),
    (re.compile(r"full day|whole day|ganzen tag|tagestour|all day", _F), 7.0),
    (re.compile(r"\ban hour\b|eine stunde|one hour", _F), 1.0),
)
_PLACE = re.compile(
    r"\b(?:near|around|in|from|bei|um|ab|von|nahe|nähe von|starting in|start in|close to)\s+"
    r"([A-ZÄÖÜ][\w\-äöüß.]+(?:\s+(?:am|an der|im|in|ob der|bei|a\.d\.)\s+[A-ZÄÖÜ][\w\-äöüß]+)?)",
    re.UNICODE,
)
_NOT_PLACES = {"i", "the", "a", "bavaria", "bayern", "germany", "deutschland", "allgäu", "summer", "winter",
               "autumn", "spring", "herbst", "sommer", "der", "die", "das", "mai", "june", "july"}
_NEGATION = re.compile(r"\b(no|not|kein\w*|ohne|without|avoid|nicht|never|nie)\b", _F)
_TAKE_BACK = re.compile(r"after all|doch nicht|doch kein\w*|lieber doch|nevermind|never mind|forget the", _F)
_CLEAR_CAP = re.compile(r"(forget|drop|no) (the )?(distance )?limit|egal wie lang|length doesn.t matter", _F)

_TAG_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("Lake", r"\blakes?\b|\bam see\b|\bseeufer|\bbadesee|\bseen\b"),
    ("Hut", r"\bhuts?\b|hütte\w*|\balm\b|\balmen\b"),
    ("Beer garden", r"beer ?garden\w*|biergarten\w*"),
    ("Castle", r"\bcastles?\b|\bschl(o|ö)ss\w*|\bburg(en)?\b|\bpalace\b"),
    ("Waterfall", r"waterfalls?|wasserf(a|ä)ll\w*"),
    ("Gorge", r"\bgorges?\b|\bklamm\w*|schlucht\w*"),
    ("Summit route", r"\bsummits?\b|\bpeaks?\b|\bgipfel\w*"),
    ("Viewpoint", r"\bviews?\b|aussicht\w*|panorama\w*|\bviewpoint\w*"),
    ("River", r"\brivers?\b|\bfluss\w*|\bflüsse\w*"),
    ("Family-friendly", r"\bkids?\b|\bchildren\b|\bfamil\w*|\bkinder\b|\bkindern\b"),
    ("Suitable for strollers", r"stroller\w*|\bpram\b|\bbuggy\b|kinderwagen\w*"),
    ("Dog-friendly", r"\bdogs?\b|\bhunde?\b"),
    ("Public transport access", r"\btrain\b|\bbahn\b|\bzug\b|public transport|öffentlich\w*|\böpnv\b|\bs-bahn\b"),
    ("Round trip", r"\bloops?\b|rundweg\w*|rundtour\w*|round ?trip|circular|\brunde\b"),
    ("Swimming spot", r"\bswim\w*|\bbaden\b|schwimm\w*|\bbathing\b"),
    ("Monastery", r"monaster\w*|kloster\w*|\babbey\b"),
    ("Ruins", r"\bruins?\b|\bruine\w*"),
    ("Museum", r"museum\w*|museen"),
    ("Church/chapel", r"\bchurch\w*|\bkirche\w*|kapelle\w*|\bchapel\w*"),
    ("Cave", r"\bcaves?\b|höhlen?\b"),
    ("Playground", r"playground\w*|spielplatz\w*"),
    ("Picnic area", r"picnic\w*|picknick\w*"),
    ("Flat", r"\bflat\b|\bflach\w*|\beben\b"),
    ("Big climb", r"big climb\w*|lots of climbing|viele höhenmeter|\bclimbs?\b"),
    ("Via ferrata", r"via ferrata|klettersteig\w*"),
    ("Mostly paved", r"\bpaved\b|asphalt\w*"),
    ("Mostly unpaved", r"\bgravel\b|schotter\w*|unpaved"),
    ("Busy road sections", r"busy roads?|\btraffic\b|\bverkehr\w*|stra(ss|ß)en\b"),
    ("Accessibility", r"wheelchair\w*|rollstuhl\w*|barrierefrei\w*"),
    ("Parking at start", r"\bparking\b|parkplatz\w*"),
    ("Nature reserve", r"nature reserve\w*|naturschutz\w*"),
    ("National park", r"national ?park\w*"),
    ("Pilgrimage route", r"pilgrim\w*|jakobsweg\w*"),
    ("Refreshment stops available", r"refreshment\w*|einkehr\w*|\binn\b|gasthaus\w*|gasthof\w*|\bcaf(e|é)\b"),
)
_TAG_RES = tuple((tag, re.compile(p, _F)) for tag, p in _TAG_KEYWORDS)
_GERMAN = re.compile(r"\b(und|mit|ich|eine?n?|nicht|für|bitte|wir|der|die|das|kein\w*|ohne|möchte|gerne?|was|"
                     r"zum|zur|im|auch|mal|schön\w*|lieber|doch)\b", _F)
_USER_MESSAGE = re.compile(
    r'<untrusted_data label="user_message" id="([0-9a-f]+)">\n(.*?)\n</untrusted_data id="\1">', re.DOTALL)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


_CLAUSE = re.compile(r"[,.;!?]|\b(aber|but|mit|with|and|und)\b", _F)


def _negated(text: str, start: int) -> bool:
    """A negation word among the three words before ``start``, within the same clause."""
    before = text[:start]
    cut = [m.end() for m in _CLAUSE.finditer(before)]
    clause = before[cut[-1]:] if cut else before
    return bool(_NEGATION.search(" ".join(clause.split()[-3:])))


def rule_extract(message: str) -> dict[str, Any]:
    """A :class:`ChatExtraction`-shaped dict for ``message``."""
    text = " ".join(message.split())
    out: dict[str, Any] = {"tags_add": [], "tags_avoid": [], "tags_remove": [], "clear": []}
    out["language"] = "de" if len(_GERMAN.findall(text)) >= 2 else "en"
    for activity, rx in _ACTIVITY:
        if rx.search(text):
            out["activity"] = activity
            break
    if _EASIER.search(text):
        out["difficulty_change"] = "easier"
    elif _HARDER.search(text):
        out["difficulty_change"] = "harder"
    elif _EASY.search(text):
        out["difficulty"] = "easy"
    elif _MODERATE.search(text):
        out["difficulty"] = "moderate"
    elif _HARD.search(text):
        out["difficulty"] = "hard"
    if _SHORTER.search(text):
        out["distance_change"] = "shorter"
    elif _LONGER.search(text):
        out["distance_change"] = "longer"

    used: list[tuple[int, int]] = []
    if m := _CAP.search(text):
        out["distance_cap_km"] = _num(m.group(2))
        used.append(m.span())
    if m := _RADIUS.search(text):
        out["travel_radius_km"] = _num(m.group(2))
        used.append(m.span())
    if m := _TARGET.search(text):
        if not any(a <= m.start() < b for a, b in used):
            out["distance_target_km"] = _num(m.group(2))
    if m := _HOURS.search(text):
        out["duration_hours"] = _num(m.group(1))
    else:
        for rx, hours in _DURATION_WORDS:
            if rx.search(text):
                out["duration_hours"] = hours
                break
    if "distance_target_km" not in out and "duration_hours" not in out and "distance_change" not in out:
        if _SHORT.search(text):
            out["distance_band"] = "short"
        elif _LONG.search(text) and not _LONGER.search(text):
            out["distance_band"] = "long"

    for m in _PLACE.finditer(text):
        name = m.group(1).rstrip(".")
        if name.casefold() not in _NOT_PLACES:
            out["place_name"] = name
            break

    take_back = bool(_TAKE_BACK.search(text))
    for tag, rx in _TAG_RES:
        m = rx.search(text)
        if not m:
            continue
        if _negated(text, m.start()):
            out["tags_remove" if take_back else "tags_avoid"].append(tag)
        elif tag == "Busy road sections":
            continue  # roads are only ever mentioned to avoid them
        else:
            out["tags_add"].append(tag)
    if _CLEAR_CAP.search(text):
        out["clear"].append("distance_cap_km")
    out["intent"] = text[:MAX_INTENT_CHARS] or None
    return out


def user_message_from_prompt(prompt: str) -> Optional[str]:
    m = _USER_MESSAGE.search(prompt)
    return m.group(2) if m else None


def mock_responder(req: Any) -> dict[str, Any]:
    """Mock-provider responder: rule extraction of the wrapped user message in the prompt."""
    msg = user_message_from_prompt(getattr(req, "prompt", "") or "")
    return rule_extract(msg or "")
