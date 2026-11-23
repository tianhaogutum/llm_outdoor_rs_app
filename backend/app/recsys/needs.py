"""Stated needs (iteration 2): needs the user states explicitly and the catalogue can verify.

Similarity proposes, verifiable constraints decide eligibility. The hard filter (``filters.py``) verifies
the three profile slots (tab, radius, distance cap). This module adds the needs a user *states* in the
free text (or sets as a chat / profile slot) for which the catalogue has a field that settles them:

==================  ======================================================  ==============================
need                verified by (catalogue field, docs/DATA.md)             levels (strict -> loosest)
==================  ======================================================  ==============================
``stroller``        ``profile.steps_m`` (O), ``profile.stroller_ok_share``  no steps, >= 50 % stroller-
                    (O), ascent (D)                                         suitable surface, <= 200 m up;
                                                                            no steps, <= 300 m up; no steps
``no_steps``        ``profile.steps_m`` (O)                                 no steps
``train``           ``near_start.train_station`` <= 1 km (tag ``Train       station at the start; station at
                    access``), ``near_end.train_station`` (O)               start or end (one-way); public
                                                                            transport at the start
``public_transport`` tag ``Public transport access`` (O)                    train/tram/bus at the start
``mtb``             ``activity`` (O), ``profile.paved_share`` (O/D)         an MTB relation; MTB relation
                                                                            or <= 50 % paved
``loop`` / ``one_way`` ``is_loop`` (D: start and end within 300 m)          the shape
``days``            ``length_m`` (D)                                        >= N x a short day (hike 5 km,
                                                                            bike 20 km, MTB 15 km); x 0.75
``length``          ``length_m`` (D)                                        the stated window; widened
``max_length``      ``length_m`` (D)                                        stated upper bound (+5 %/+10 %
                                                                            for "about"); +25 %
``duration``        ``duration_min`` (D, DIN 33466 / planning speeds)       the stated window; widened
==================  ======================================================  ==============================

Extraction (:func:`extract_needs`) is deterministic: regular expressions over the intent, English and
German, with clause-local negation ("no stroller", "ohne Kinderwagen", "doesn't have to be a loop" are not
needs), plus the needs carried by the request's structured slots (``UserQuery.needs``, from the chat or the
profile). No model call; it runs in well under a millisecond.

Three ways to use the needs (``NeedsConfig.mode``):

- ``filter``: in front of candidate generation, the eligible set keeps only routes that meet every need at
  its current level. If fewer than ``floor`` routes are left, needs are relaxed one at a time in
  :data:`RELAX_ORDER` (each through its levels, then dropped) until the floor is met; a need is never
  relaxed while the floor is met. What was applied, relaxed and dropped, and why, is in the trace.
- ``filter_promote``: the filter as above, and when relaxation was needed, the few routes that meet every need
  strictly are added to the candidates if retrieval missed them, and after MMR every list is stably re-ordered
  by :func:`demote_key` (strict matches first). The floor then only fills the list; it never pushes a route
  that meets everything below one that does not.
- ``demote``: nothing is removed; after the ranker and MMR, routes are stably re-ordered by how badly they
  miss the needs (:func:`demote_key`: a need met only at a relaxed level counts as part of a miss).

Every returned route carries a per-need check with the real numbers (``contrib`` of stage ``needs``).
Design and the dev measurements: docs/RECSYS.md section 13.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import Catalog

NEEDS_VERSION = "needs/v1"
NEED_CODES = (
    "stroller",
    "no_steps",
    "train",
    "public_transport",
    "mtb",
    "loop",
    "one_way",
    "days",
    "length",
    "max_length",
    "duration",
)
# Relaxed first -> relaxed last. Preferences about the shape of the outing go first; needs that decide whether
# the party can do the route at all (steps with a stroller) go last.
RELAX_ORDER = (
    "length",
    "duration",
    "loop",
    "one_way",
    "days",
    "mtb",
    "public_transport",
    "train",
    "max_length",
    "no_steps",
    "stroller",
)
# Lower ends of a short day stage, km (a deliberately low floor: N days must be longer than N short days).
DAY_KM = {"hiking": 5.0, "running": 5.0, "cycling": 20.0, "mtb": 15.0}
# "short" / "long" in the chat's own bands (app.chat.vocab: hiking short 0-8 km, long from 10 km, scaled)
SHORT_KM = {"hiking": 8.0, "running": 8.0, "cycling": 32.0, "mtb": 20.0}
LONG_KM = {"hiking": 10.0, "running": 10.0, "cycling": 40.0, "mtb": 25.0}
LABEL = {
    "stroller": "stroller-friendly",
    "no_steps": "no steps",
    "train": "reachable by train",
    "public_transport": "reachable by public transport",
    "mtb": "a mountain-bike route",
    "loop": "a loop",
    "one_way": "a one-way route",
    "days": "a multi-day route",
    "length": "a length",
    "max_length": "a maximum length",
    "duration": "a duration",
}


@dataclass(frozen=True)
class NeedsConfig:
    mode: Literal["filter", "filter_promote", "demote"] = "filter"
    floor: int = 20  # fewest eligible routes the filter may leave (2 x the list length)

    def __post_init__(self) -> None:
        if self.mode not in ("filter", "filter_promote", "demote"):
            raise ValueError("needs mode must be 'filter', 'filter_promote' or 'demote'")
        if self.floor < 1:
            raise ValueError("needs floor must be >= 1")

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "floor": self.floor, "version": NEEDS_VERSION}


@dataclass(frozen=True)
class Need:
    code: str
    phrase: str  # the words that stated it (or "slot: ..." for a structured slot)
    source: Literal["intent", "slot"] = "intent"
    params: tuple[tuple[str, float], ...] = ()  # e.g. (("days", 4),), (("lo", 6.0), ("hi", 12.0))

    def p(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return dict(self.params).get(key, default)

    def describe(self) -> dict[str, Any]:
        return {"need": self.code, "phrase": self.phrase, "source": self.source, **dict(self.params)}


@dataclass(frozen=True)
class StatedNeeds:
    needs: tuple[Need, ...] = ()
    ignored: tuple[dict[str, str], ...] = ()  # negated or conflicting mentions, for the trace

    def __bool__(self) -> bool:
        return bool(self.needs)

    def codes(self) -> list[str]:
        return [n.code for n in self.needs]

    def describe(self) -> dict[str, Any]:
        return {"needs": [n.describe() for n in self.needs], "ignored": list(self.ignored)}


# --------------------------------------------------------------------------------------- extraction
_NUMWORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1, "half": 0.5, "a couple of": 2, "couple of": 2, "a few": 3, "several": 2,
    "ein": 1, "eine": 1, "einen": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
    "acht": 8, "neun": 9, "zehn": 10, "halb": 0.5, "einhalb": 1.5, "anderthalb": 1.5,
}  # fmt: skip
_NUMWORD_RE = (
    r"(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|zwei|drei|vier|fünf|sechs|sieben|acht"
    r"|neun|zehn|ein|eine|anderthalb)"
)
_NEGATORS = re.compile(
    r"\b(?:no|not|without|avoid|avoiding|never|nothing|none|neither|nor|instead of|rather than|won't|wont|"
    r"don't|doesn't|isn't|aren't|needn't|shouldn't|wouldn't|can't|cannot|kein|keine|keinen|keiner|keinem|"
    r"ohne|nicht|statt|anstatt|anstelle)\b"
)
_POST_NEG = re.compile(
    r"^[^,.;:!?]{0,40}?\b(?:not (?:needed|necessary|required|important|essential|a must)|isn't (?:needed|necessary|"
    r"required|important)|is optional|optional|doesn't matter|don't mind|no need|nicht (?:nötig|notwendig|wichtig)"
    r"|egal)\b"
)
# clause boundaries for negation scope
_BOUNDARY = re.compile(r"[,.;:!?()]|\b(?:but|and|or|aber|und|oder|sondern|though|although|while)\b")


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-").replace("‑", "-")
    t = t.replace("n’t", "n't")
    return re.sub(r"\s+", " ", t)


def _clause_before(t: str, start: int) -> str:
    last = 0
    for m in _BOUNDARY.finditer(t, 0, start):
        last = m.end()
    return t[last:start]


_SCOPE_BREAK = re.compile(
    r"\b(?:with|plus|ideally|please|maybe|preferably|that|which|because|including|reachable|accessible|mit|und)\b"
)
NEG_SCOPE_WORDS = 5  # a negator reaches at most this many words ahead ("doesn't have to be a loop")


def _negated(t: str, m: re.Match, *, own_negation: bool = False) -> bool:
    """A mention is negated if a negator precedes it closely in the same clause (at most NEG_SCOPE_WORDS words,
    no scope-breaking word such as "with" or "reachable" in between), or a "not needed" follows it."""
    before = _clause_before(t, m.start())
    if not own_negation:
        negs = list(_NEGATORS.finditer(before))
        if negs:
            gap = before[negs[-1].end() :]
            if len(gap.split()) <= NEG_SCOPE_WORDS and not _SCOPE_BREAK.search(gap):
                return True
    return bool(_POST_NEG.search(t[m.end() :]))


def _num(s: str) -> Optional[float]:
    s = s.strip()
    if s in _NUMWORDS:
        return float(_NUMWORDS[s])
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


# need -> (pattern, own_negation): own_negation = the pattern itself contains a negator ("no steps")
_PATTERNS: dict[str, list[tuple[re.Pattern, bool]]] = {
    "stroller": [
        (
            re.compile(
                r"\b(?:strollers?|pushchairs?|prams?|baby ?(?:buggy|jogger|carriage)|kinderwagen\w*|buggys?)\b"
                r"(?! ?(?:race|rally))"
            ),
            False,
        ),
    ],
    "no_steps": [
        (
            re.compile(
                r"\b(?:no|without|avoid(?:ing)?|free of)\s+(?:any\s+)?(?:steps|stairs|staircases?|stairways?)\b"
            ),
            True,
        ),
        (
            re.compile(
                r"\b(?:step|stair)[- ]free\b|\bbarrier[- ]free\b|\bwheelchair\w*|\bbarrierefrei\w*|\bstufenlos\w*"
                r"|\brollstuhl\w*|\b(?:keine|ohne)\s+(?:treppen|stufen)\b"
            ),
            False,
        ),
    ],
    "train": [
        (
            re.compile(
                r"\btrains?\b(?! ?(?:ing|ee))|\brailway\b|\bby rail\b|\btrain station\b|\bs-bahn\b|\bbahnhof\w*"
                r"|\bmit (?:dem zug|der bahn)\b|\bregionalbahn\w*|\bzug\b"
            ),
            False,
        ),
    ],
    "public_transport": [
        (
            re.compile(
                r"\bpublic (?:transport|transit)\b|\bby bus\b|\bbus (?:stop|connection|route)\b|\böpnv\b"
                r"|\böffentlich\w* verkehr\w*|\bmit dem bus\b"
            ),
            False,
        ),
        (re.compile(r"\b(?:without|no) (?:a )?car\b|\bcar[- ]free\b|\bdon't (?:have|own) a car\b|\bohne auto\b"), True),
    ],
    "mtb": [
        (
            re.compile(
                r"\bmountain[- ]?bik\w*|\bmtb\w*|\bsingle[- ]?trails?\b|\bmountainbik\w*|\benduro\b"
                r"|\btrail ?rid(?:e|es|ing)\b"
            ),
            False,
        ),
    ],
    "loop": [
        (
            re.compile(
                r"\bloops?\b|\bcircular\b|\bround[- ]?trip\b|\bcircuit\b|\bround walk\b"
                r"|\bback (?:to|at) (?:the |my |our )?(?:car|start|starting point)\b"
                r"|\b(?:ends?|finish(?:es)?) where it (?:starts|started|begins)\b"
                r"|\brund(?:weg|wanderweg|wanderung|tour|kurs|e)\w*\b"
            ),
            False,
        ),
    ],
    "one_way": [
        (
            re.compile(
                r"\bone[- ]way\b|\blinear\b|\bpoint[- ]to[- ]point\b|\b(?:from )?a to b\b|\bstrecken(?:wanderung|tour)"
                r"\w*|\bend (?:somewhere else|elsewhere|at (?:a|another) (?:different )?(?:place|station|town))\b"
            ),
            False,
        ),
        (re.compile(r"\bnot (?:a |as a )?(?:loop|circular|round trip)\b"), True),
    ],
}

_DAYS = re.compile(
    rf"\b({_NUMWORD_RE})[- ]?(?:days?|tage?|tägig\w*|tages\w*)\b(?! (?:a week|per week|ago|in advance|later|before))"
    r"|\b(multi[- ]?day|several days|(?:a )?few days|(?:a )?couple of days|mehrtägig\w*|mehrtages\w*|overnight"
    r"|hut[- ]to[- ]hut|hüttentour\w*"
    r"|(?:long )?weekend|(?:a|one) week(?:'s|-long)?|week[- ]?long|wochenende\w*|eine woche)\b"
)
_NOT_MULTIDAY = re.compile(
    r"\b(?:one|1|a|single)[- ]day\b(?! (?:or|to) )|(?<![-\w])day[- ](?:trip|hike|ride|tour|outing)\b|\beintägig\w*"
)
_DIST = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:(?:-|to|or|bis|und)\s*(\d+(?:[.,]\d+)?)\s*)?(km|kilomet\w*|miles?|mi\b)")
_UPPER = re.compile(
    r"(?:no more than|not more than|at most|up to|max(?:imum|\.)?|under|less than|below|no longer than|"
    r"not longer than|nothing (?:longer|more) than|shorter than|within|keep it (?:to )?(?:under|below|within)|"
    r"höchstens|maximal|bis zu|bis|unter|weniger als|nicht mehr als|nicht länger als|kürzer als)\s*"
    r"(?:about |around |roughly |approx\w*\.? |ca\.? |etwa |ungefähr )?$"
)
_LOWER = re.compile(
    r"(?:at least|more than|over|minimum|min\.|longer than|mindestens|mehr als|über|länger als)\s*"
    r"(?:about |around |roughly |approx\w*\.? |ca\.? |etwa |ungefähr )?$"
)
_ABOUT = re.compile(r"(?:about|around|roughly|approx\w*\.?|ca\.?|circa|etwa|ungefähr|rund|some|~)\s*$")
# a distance followed by these is how far away the start is, not how long the route is
_RADIUS_AFTER = re.compile(
    r"^\s*(?:(?:of|from)\b(?!\s+(?:\w+ing|easy|gentle|flat|trails?|paths?|tracks?|forest|woodland|riverside|lakeside)\b)"
    r"|away|radius|drive|by car|von\b|vom\b|entfernt|um\b|rund um|around (?:[a-zäöüß]+|home|here))"
)
_HOURS = re.compile(
    rf"\b({_NUMWORD_RE}|a couple of|a few|half an?)\s*(?:(?:-|to|or|bis)\s*({_NUMWORD_RE})\s*)?"
    r"(?:hours?|hrs?|h\b|stunden?|std\.?)(?![- ]?(?:drive|away|by car|train|fahrt))"
)
_HALF_DAY = re.compile(r"\bhalf[- ]day\b|\bhalbtag\w*|\bhalben tag\b")
_FULL_DAY = re.compile(r"\b(?:full|whole|entire)[- ]day\b|\ball[- ]day\b|\bganztag\w*|\bganzen tag\b")
_SHORT = re.compile(
    r"\b(?:short(?:ish)?|kurze?[nrs]?)\b(?! ?(?:drive|trip to|distance|journey|way|hop|ride from|walk from|time"
    r"|train|bus|car|section|stretch|bit|climb|ascent|anfahrt|fahrt|strecke mit dem auto))"
)
_LONG = re.compile(
    r"\b(?:long|lange[nrs]?)\s+(?:walk|hike|hiking|ride|tour|route|loop|trail|run|bike|day out|day'?s? (?:hike|walk|"
    r"ride)|wanderung|tour|radtour|strecke)\b"
)


def _first(t: str, pats: list[tuple[re.Pattern, bool]]) -> tuple[Optional[re.Match], list[str]]:
    """First non-negated match of any pattern, and the phrases of negated matches."""
    negated: list[str] = []
    for pat, own in pats:
        for m in pat.finditer(t):
            if _negated(t, m, own_negation=own):
                negated.append(m.group(0))
                continue
            return m, negated
    return None, negated


def _length_need(t: str, activity: str) -> Optional[Need]:
    """A stated length: "around 8 km" (window), "5-10 km" (range), "no more than 10 km" (upper bound),
    "at least 20 km" (lower bound), "short" / "long" (the chat's bands). Distances followed by "of / from /
    away" (how far the start is) are not lengths."""
    for m in _DIST.finditer(t):
        if _RADIUS_AFTER.search(t[m.end() :]):
            continue
        before = _clause_before(t, m.start())
        if re.search(r"\b(?:drive|driving|travel|from home|fahrt)\b", before):
            continue
        a = _num(m.group(1))
        b = _num(m.group(2)) if m.group(2) else None
        unit = 1.609 if m.group(3).startswith("mi") else 1.0
        if a is None or a <= 0:
            continue
        a *= unit
        b = b * unit if b else None
        phrase = (before.strip().split(" ")[-3:] if before.strip() else []) + [m.group(0)]
        phrase_s = " ".join(phrase).strip()
        about = bool(_ABOUT.search(before)) or bool(re.search(r"\b(?:about|around|roughly|approx|ca|etwa)\b", before))
        if _UPPER.search(before):
            hi = (b or a) * (1.10 if about else 1.05)
            return Need("max_length", phrase_s, params=(("hi", round(hi, 2)), ("stated", b or a)))
        if _LOWER.search(before):
            return Need("length", phrase_s, params=(("lo", round(a * 0.9, 2)), ("stated_lo", a), ("kind", 2)))
        if b and b > a:
            return Need(
                "length",
                phrase_s,
                params=(
                    ("lo", round(0.8 * a, 2)),
                    ("hi", round(1.2 * b, 2)),
                    ("stated_lo", a),
                    ("stated_hi", b),
                    ("kind", 1),
                ),
            )
        return Need(
            "length",
            phrase_s,
            params=(("lo", round(0.5 * a, 2)), ("hi", round(1.5 * a, 2)), ("stated", a), ("kind", 0)),
        )
    m = _SHORT.search(t)
    if m and not _negated(t, m):
        return Need("length", m.group(0), params=(("hi", SHORT_KM[activity]), ("kind", 3)))
    m = _LONG.search(t)
    if m and not _negated(t, m) and not re.search(r"\b(?:too|not too|so)\s*$", _clause_before(t, m.start())):
        return Need("length", m.group(0), params=(("lo", LONG_KM[activity]), ("kind", 4)))
    return None


def _days_need(t: str, ignored: list[dict[str, str]]) -> Optional[Need]:
    for m in _DAYS.finditer(t):
        if _negated(t, m):
            ignored.append({"need": "days", "phrase": m.group(0), "why": "negated"})
            continue
        if m.group(1):
            n = _num(m.group(1))
            if n is None or n < 2 or n > 21:
                continue
            days = n
        else:
            w = m.group(2)
            days = 3.0 if w.startswith("long") or "few" in w else 5.0 if "week" in w or "woche" in w else 2.0
        return Need("days", m.group(0), params=(("days", days),))
    return None


def _duration_need(t: str) -> Optional[Need]:
    for m in _HOURS.finditer(t):
        if _negated(t, m) or _RADIUS_AFTER.search(t[m.end() :]):
            continue
        before = _clause_before(t, m.start())
        if re.search(r"\b(?:drive|driving|travel|train ride|fahrt|anfahrt)\b", before + t[m.end() : m.end() + 15]):
            continue
        g1 = m.group(1)
        a = 0.5 if g1.startswith("half") else _num(g1) if g1 not in ("a couple of", "a few") else _NUMWORDS[g1]
        b = _num(m.group(2)) if m.group(2) else None
        if not a or a > 24:
            continue
        if _UPPER.search(before):
            return Need("duration", m.group(0), params=(("hi_h", round((b or a) * 1.1, 2)),))
        if b and b > a:
            return Need("duration", m.group(0), params=(("lo_h", round(0.8 * a, 2)), ("hi_h", round(1.2 * b, 2))))
        return Need("duration", m.group(0), params=(("lo_h", round(0.5 * a, 2)), ("hi_h", round(1.5 * a, 2))))
    m = _HALF_DAY.search(t)
    if m and not _negated(t, m):
        return Need("duration", m.group(0), params=(("lo_h", 2.0), ("hi_h", 5.0)))
    m = _FULL_DAY.search(t)
    if m and not _negated(t, m):
        return Need("duration", m.group(0), params=(("lo_h", 4.0),))
    return None


SLOT_NEEDS = ("stroller", "no_steps", "train", "public_transport", "mtb", "loop", "one_way")


def parse_slot_need(s: str) -> Optional[Need]:
    """A structured need from a slot: ``"stroller"``, ``"days:3"``, ``"duration_h:2.5"``, ``"max_km:12"``."""
    s = (s or "").strip().lower()
    if s in SLOT_NEEDS:
        return Need(s, f"slot: {s}", "slot")
    key, _, val = s.partition(":")
    v = _num(val) if val else None
    if v is None or v <= 0:
        return None
    if key == "days" and v >= 2:
        return Need("days", f"slot: {s}", "slot", (("days", v),))
    if key == "duration_h":
        return Need("duration", f"slot: {s}", "slot", (("lo_h", round(0.5 * v, 2)), ("hi_h", round(1.5 * v, 2))))
    if key == "max_km":
        return Need("max_length", f"slot: {s}", "slot", (("hi", round(v * 1.05, 2)), ("stated", v)))
    return None


def extract_needs(
    intent: Optional[str], category: Optional[str] = None, slot_needs: tuple[str, ...] = ()
) -> StatedNeeds:
    """Needs stated in ``intent`` (English or German) plus structured ``slot_needs``; deterministic."""
    t = _norm(intent or "")
    cycling = category == "cycling"
    found: dict[str, Need] = {}
    ignored: list[dict[str, str]] = []
    for code, pats in _PATTERNS.items():
        if not t:
            break
        m, neg = _first(t, pats)
        ignored += [{"need": code, "phrase": p, "why": "negated"} for p in neg]
        if m is not None:
            found[code] = Need(code, m.group(0))
    if "mtb" in found and not cycling:
        ignored.append({"need": "mtb", "phrase": found.pop("mtb").phrase, "why": "not the cycling tab"})
    activity = "mtb" if "mtb" in found else ("cycling" if cycling else (category if category in DAY_KM else "hiking"))
    if t:
        for n in (_days_need(t, ignored), _length_need(t, activity), _duration_need(t)):
            if n is not None:
                found[n.code] = n
        if "days" in found and _NOT_MULTIDAY.search(t):
            ignored.append({"need": "days", "phrase": found.pop("days").phrase, "why": "also says one day"})
    if "train" in found and "public_transport" in found:
        found.pop("public_transport")  # the train is the stricter, more specific statement
    if "stroller" in found and "no_steps" in found:
        found.pop("no_steps")  # the stroller check already requires no steps
    for s in slot_needs:
        n = parse_slot_need(s)
        if n is not None and not (n.code == "mtb" and not cycling):
            found[n.code] = n  # a slot the user set wins over the text
    if "loop" in found and "one_way" in found:
        for c in ("loop", "one_way"):
            ignored.append({"need": c, "phrase": found.pop(c).phrase, "why": "both loop and one-way stated"})
    if "days" in found and found.get("length") and found["length"].p("hi") is not None:
        # "a three-day hike of 15 km a day": the km figure is per day, not a window for the whole route
        ignored.append({"need": "length", "phrase": found.pop("length").phrase, "why": "per day with a day count"})
    needs = tuple(found[c] for c in NEED_CODES if c in found)
    return StatedNeeds(needs, tuple(ignored))


# --------------------------------------------------------------------------------------- verification
@dataclass
class _Arrays:
    steps: np.ndarray
    stroller_ok: np.ndarray
    climb: np.ndarray
    train_start: np.ndarray
    train_end: np.ndarray
    pt: np.ndarray
    paved: np.ndarray
    is_mtb: np.ndarray
    is_loop: np.ndarray
    length: np.ndarray
    duration_h: np.ndarray


def _arrays(catalog: Catalog) -> _Arrays:
    def build() -> _Arrays:
        prof = [r.features.get("profile") or {} for r in catalog.routes]
        ns = [r.features.get("near_end") or {} for r in catalog.routes]
        return _Arrays(
            steps=np.array([p.get("steps_m", 0) or 0 for p in prof], dtype=float),
            stroller_ok=np.array([p.get("stroller_ok_share", 0) or 0 for p in prof], dtype=float),
            climb=np.maximum(catalog.ascent_m, catalog.descent_m),
            train_start=np.array(["Train access" in r.tags for r in catalog.routes]),
            train_end=np.array(
                [(not r.is_loop) and bool(e.get("train_station")) for r, e in zip(catalog.routes, ns, strict=True)]
            ),
            pt=np.array(["Public transport access" in r.tags for r in catalog.routes]),
            paved=np.array([p.get("paved_share", np.nan) for p in prof], dtype=float),
            is_mtb=catalog.activity == "mtb",
            is_loop=catalog.is_loop,
            length=catalog.length_km,
            duration_h=catalog.duration_min / 60.0,
        )

    return catalog.memo("needs:arrays", build)


def _window(x: np.ndarray, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    m = np.ones(len(x), dtype=bool)
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    return m


def n_levels(need: Need) -> int:
    return {"stroller": 3, "train": 3, "mtb": 2, "days": 2, "length": 2, "duration": 2, "max_length": 2}.get(
        need.code, 1
    )


def level_mask(a: _Arrays, need: Need, level: int, activity: str) -> np.ndarray:
    """Routes (over the whole catalogue) that meet ``need`` at ``level`` (0 = strict)."""
    c = need.code
    if c == "stroller":
        nostep = a.steps <= 0
        if level == 0:
            return nostep & (a.stroller_ok >= 0.5) & (a.climb <= 200)
        if level == 1:
            return nostep & (a.climb <= 300)
        return nostep
    if c == "no_steps":
        return a.steps <= 0
    if c == "train":
        return a.train_start if level == 0 else (a.train_start | a.train_end) if level == 1 else (a.pt | a.train_end)
    if c == "public_transport":
        return a.pt | a.train_end
    if c == "mtb":
        return a.is_mtb if level == 0 else a.is_mtb | (np.nan_to_num(a.paved, nan=1.0) <= 0.5)
    if c == "loop":
        return a.is_loop
    if c == "one_way":
        return ~a.is_loop
    if c == "days":
        lo = need.p("days", 2) * DAY_KM.get(activity, 5.0) * (1.0 if level == 0 else 0.75)
        return a.length >= lo
    if c in ("length", "max_length"):
        lo, hi = need.p("lo"), need.p("hi")
        if level == 1:
            lo = lo * 2 / 3 if lo is not None else None
            hi = hi * (1.25 if c == "max_length" else 4 / 3) if hi is not None else None
        return _window(a.length, lo, hi)
    if c == "duration":
        lo, hi = need.p("lo_h"), need.p("hi_h")
        if level == 1:
            lo = lo * 2 / 3 if lo is not None else None
            hi = hi * 4 / 3 if hi is not None else None
        return _window(a.duration_h, lo, hi)
    raise ValueError(f"unknown need {c!r}")


def need_activity(category: Optional[str], needs: StatedNeeds) -> str:
    if "mtb" in needs.codes():
        return "mtb"
    return category if category in DAY_KM else "hiking"


@dataclass
class NeedsResult:
    indices: np.ndarray  # eligible catalogue indices after the needs (filter) or unchanged (demote)
    levels: dict[str, Optional[int]]  # need -> level applied (None = dropped)
    info: dict[str, Any] = field(default_factory=dict)
    strict: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # meet every need strictly


def apply_needs(
    catalog: Catalog, elig: np.ndarray, needs: StatedNeeds, category: Optional[str], cfg: NeedsConfig
) -> NeedsResult:
    """Filter mode: restrict ``elig`` to routes meeting every need, relaxing in RELAX_ORDER below the floor."""
    a = _arrays(catalog)
    act = need_activity(category, needs)
    by_code = {n.code: n for n in needs.needs}
    levels: dict[str, Optional[int]] = {c: 0 for c in by_code}
    floor = min(cfg.floor, len(elig))
    steps: list[dict[str, Any]] = []

    def current() -> np.ndarray:
        m = np.ones(len(elig), dtype=bool)
        for c, lv in levels.items():
            if lv is not None:
                m &= level_mask(a, by_code[c], lv, act)[elig]
        return m

    m = current()
    n_strict = int(m.sum())
    strict_idx = elig[m]
    for code in RELAX_ORDER:
        if code not in by_code:
            continue
        while int(m.sum()) < floor and levels[code] is not None:
            before = int(m.sum())
            nxt = levels[code] + 1
            levels[code] = nxt if nxt < n_levels(by_code[code]) else None
            m = current()
            steps.append(
                {
                    "need": code,
                    "to_level": levels[code],
                    "routes_before": before,
                    "routes_after": int(m.sum()),
                    "why": f"only {before} routes met every need (floor {floor})",
                }
            )
        if int(m.sum()) >= floor:
            break
    idx = elig[m]
    info = {
        "mode": cfg.mode,
        "version": NEEDS_VERSION,
        "floor": cfg.floor,
        "extracted": needs.describe(),
        "applied": [c for c, lv in levels.items() if lv == 0],
        "relaxed": {c: lv for c, lv in levels.items() if lv not in (0, None)},
        "dropped": [c for c, lv in levels.items() if lv is None],
        "relax_steps": steps,
        "n_strict": n_strict,
        "n_eligible_before": int(len(elig)),
        "n_eligible_after": int(len(idx)),
    }
    return NeedsResult(idx, levels, info, strict_idx)


def first_level(catalog: Catalog, i: int, need: Need, activity: str) -> Optional[int]:
    """Strictest level at which route ``i`` meets ``need`` (None = not even at the loosest)."""
    a = _arrays(catalog)
    for lv in range(n_levels(need)):
        if level_mask(a, need, lv, activity)[i]:
            return lv
    return None


def demote_key(catalog: Catalog, i: int, needs: StatedNeeds, activity: str) -> float:
    """Demotion key of route ``i``: sum over needs of (strictest level met / number of levels), 1 per need not
    met at any level. 0 = meets every need strictly."""
    k = 0.0
    for n in needs.needs:
        lv = first_level(catalog, i, n, activity)
        k += 1.0 if lv is None else lv / n_levels(n)
    return round(k, 6)


# --------------------------------------------------------------------------------------- explanation
def _km(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def _window_text(lo: Optional[float], hi: Optional[float], unit: str = "km") -> str:
    if lo is not None and hi is not None:
        return f"{_km(lo)}-{_km(hi)} {unit}"
    if hi is not None:
        return f"at most {_km(hi)} {unit}"
    return f"at least {_km(lo)} {unit}"


def _hours(h: float) -> str:
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh} h {mm:02d} min" if hh else f"{mm} min"


def check_route(catalog: Catalog, i: int, need: Need, activity: str, applied_level: Optional[int]) -> dict[str, Any]:
    """One need checked on one route, with the catalogue's numbers and a plain sentence (``fact``)."""
    r = catalog.routes[int(i)]
    prof = r.features.get("profile") or {}
    lv = first_level(catalog, i, need, activity)
    c = need.code
    values: dict[str, Any] = {}
    if c in ("stroller", "no_steps"):
        steps = float(prof.get("steps_m", 0) or 0)
        share = float(prof.get("stroller_ok_share", 0) or 0)
        climb = max(r.ascent_m, r.descent_m)
        values = {"steps_m": steps, "stroller_ok_share": round(share, 3), "climb_m": climb}
        bits = ["no steps" if steps <= 0 else f"{steps:.0f} m of steps"]
        if c == "stroller":
            bits += [f"{100 * share:.0f}% of the route on stroller-suitable surface", f"{climb} m of climbing"]
        fact = ", ".join(bits)
        threshold = (
            ("no steps, >= 50% stroller-suitable surface, <= 200 m climbing (relaxed: <= 300 m; then no steps only)")
            if c == "stroller"
            else "no steps"
        )
    elif c in ("train", "public_transport"):
        ns = r.features.get("near_start") or {}
        ne = r.features.get("near_end") or {}
        st = (ns.get("train_station") or [None])[0]
        en = (ne.get("train_station") or [None])[0] if not r.is_loop else None
        values = {
            "train_station_start": st and st.get("name"),
            "train_station_start_m": st and st.get("dist_m"),
            "train_station_end": en and en.get("name"),
            "public_transport_start": "Public transport access" in r.tags,
        }
        if st:
            fact = f"train station {st.get('name') or ''} {st.get('dist_m')} m from the start".replace("  ", " ")
        elif en:
            fact = f"train station {en.get('name') or ''} near the end of this one-way route".replace("  ", " ")
        elif "Public transport access" in r.tags:
            stop = ((ns.get("bus_stop") or ns.get("tram_stop") or [None])[0]) or {}
            fact = (
                (
                    f"no train station near the start; bus/tram stop {stop.get('name') or ''} {stop.get('dist_m')} m "
                    "from the start"
                ).replace("  ", " ")
                if stop
                else "public transport at the start"
            )
        else:
            fact = "no train station or stop mapped near the start"
        threshold = (
            (
                "train station within 1 km of the start (relaxed: or at the end of a one-way route; then any "
                "public transport)"
            )
            if c == "train"
            else "train within 1 km, tram 500 m or bus 300 m of the start"
        )
    elif c == "mtb":
        paved = prof.get("paved_share")
        values = {"activity": r.activity, "paved_share": paved}
        fact = (
            "a mapped mountain-bike route"
            if r.activity == "mtb"
            else f"a cycling route, {100 * (paved or 0):.0f}% paved"
        )
        threshold = "a mountain-bike route (relaxed: or a cycling route at most 50% paved)"
    elif c in ("loop", "one_way"):
        values = {"is_loop": r.is_loop}
        fact = "a loop" if r.is_loop else "one-way (start and end differ)"
        threshold = "start and end within 300 m" if c == "loop" else "start and end more than 300 m apart"
    elif c == "days":
        lo = need.p("days", 2) * DAY_KM.get(activity, 5.0)
        values = {"length_km": round(r.length_km, 2), "days": need.p("days"), "min_km": lo}
        fact = f"{_km(r.length_km)} km for {need.p('days'):g} days"
        threshold = f"at least {_km(lo)} km ({DAY_KM.get(activity, 5.0):g} km a day; relaxed: {_km(0.75 * lo)} km)"
    elif c in ("length", "max_length"):
        values = {"length_km": round(r.length_km, 2), "lo_km": need.p("lo"), "hi_km": need.p("hi")}
        fact = f"{_km(r.length_km)} km"
        threshold = _window_text(need.p("lo"), need.p("hi"))
    else:  # duration
        values = {"duration_h": round(r.duration_min / 60.0, 2), "lo_h": need.p("lo_h"), "hi_h": need.p("hi_h")}
        fact = f"about {_hours(r.duration_min / 60.0)} moving time"
        threshold = _window_text(need.p("lo_h"), need.p("hi_h"), "h")
    return {
        "need": c,
        "label": LABEL[c],
        "phrase": need.phrase,
        "source": need.source,
        "applied_level": applied_level,
        "met_level": lv,
        "ok": lv is not None and (applied_level is None or lv <= applied_level),
        "strict": lv == 0,
        "fact": fact,
        "threshold": threshold,
        "values": values,
    }
