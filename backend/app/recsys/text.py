"""Text processing for retrieval: tokenising, light stemming, German title glossary, query handling,
and the per-route texts each stage sees (BM25 fields, embedded document, ranker document).

Route titles are mostly German OSM names ("Rundwanderweg Eibsee", "Burgenweg"), descriptions are
generated English text (docs/DATA.md section 14) optionally preceded by a German OSM description, and
intents are English. Options (all measured on dev, eval/results/_cg_dev.json):

- ``stem``: a light English suffix stripper applied to both sides (lakes -> lake, ruins -> ruin);
- ``glossary``: German morphemes inside title / OSM-description tokens add their English concept
  terms (``Eibsee`` -> ``lake``, ``Burgruine`` -> ``castle ruin``). This doubles as a crude
  decompounder: German compounds are matched by contained morphemes, not split exhaustively;
- ``negation``: words in the scope of "no / not / without / avoid" are removed from the query, so
  "no busy roads" does not reward routes whose description mentions roads;
- ``expand``: a few query-side synonyms towards the catalogue's vocabulary (loop -> circular, pub ->
  refreshment, stroller -> pushchair).

Everything here is deterministic and model-free.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import RouteRecord

TEXT_VERSION = "text/v1"

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i i'd i'm id im in is it its me my of on or our
please so some something that the this to us we we'd we're with would you your could around about like
want looking suggest ideally nice good near maybe also there their them they km m d ll re s t ve
can will just very really bit one if than then into over up out all any more most much such only own
same too way somewhere something anything place places route routes let love great lovely ideal
perfect happy keen hoping fine suit suits suitable works work need needs prefer preferably rather
der die das den dem des und am im zum zur von vom bei mit auf an ein eine einer nach über""".split()
)

# German morpheme -> English concept terms (matched as a substring of a title / OSM-description token;
# morphemes shorter than 5 characters only as a suffix or whole token, to limit false hits).
GLOSSARY: dict[str, tuple[str, ...]] = {
    "see": ("lake",),
    "seen": ("lake",),
    "weiher": ("lake", "pond"),
    "teich": ("pond", "lake"),
    "burg": ("castle",),
    "schloss": ("castle", "palace"),
    "schloß": ("castle", "palace"),
    "ruine": ("ruin", "castle"),
    "kloster": ("monastery",),
    "abtei": ("monastery", "abbey"),
    "wasserfall": ("waterfall",),
    "wasserfälle": ("waterfall",),
    "klamm": ("gorge",),
    "schlucht": ("gorge",),
    "tobel": ("gorge",),
    "höhle": ("cave",),
    "hoehle": ("cave",),
    "hütte": ("hut",),
    "huette": ("hut",),
    "alm": ("hut", "alpine", "pasture"),
    "alpe": ("hut", "alpine", "pasture"),
    "gipfel": ("summit", "peak"),
    "berg": ("mountain", "summit"),
    "aussicht": ("viewpoint", "view"),
    "panorama": ("panorama", "view"),
    "rundweg": ("loop", "circular"),
    "rundwanderweg": ("loop", "circular", "hike"),
    "runde": ("loop", "circular"),
    "rundtour": ("loop", "circular"),
    "rund": ("loop", "circular"),
    "wanderweg": ("hike", "trail"),
    "radweg": ("cycle", "bike"),
    "radwanderweg": ("cycle", "bike"),
    "radrunde": ("cycle", "bike", "loop"),
    "kapelle": ("chapel", "church"),
    "kirche": ("church",),
    "kirch": ("church",),
    "wald": ("forest", "wood"),
    "moor": ("moor", "bog"),
    "fluss": ("river",),
    "bach": ("stream", "brook"),
    "tal": ("valley",),
    "quelle": ("spring",),
    "biergarten": ("beer", "garden"),
    "spielplatz": ("playground",),
    "erlebnisweg": ("adventure", "trail", "family"),
    "lehrpfad": ("educational", "trail"),
    "sagen": ("legend",),
    "felsen": ("rock",),
    "fels": ("rock",),
    "höhenweg": ("ridge", "view"),
    "limes": ("roman", "history"),
    "römer": ("roman", "history"),
    "jakobsweg": ("pilgrimage",),
    "pilger": ("pilgrimage",),
    "kneipp": ("kneipp", "water"),
    "kinder": ("children", "family"),
    "familie": ("family",),
    "donau": ("river", "danube"),
    "main": ("river",),
    "altmühl": ("river",),
    "isar": ("river",),
    "inn": ("river",),
}
WHOLE_TOKEN_ONLY = frozenset({"inn", "main", "rund"})
_GLOSS_LONG = sorted((k for k in GLOSSARY if len(k) >= 5), key=len, reverse=True)
_GLOSS_SHORT = sorted((k for k in GLOSSARY if len(k) < 5), key=len, reverse=True)

# query-side expansion towards the catalogue vocabulary (English descriptions and tags)
EXPANSIONS: dict[str, tuple[str, ...]] = {
    "loop": ("circular", "round"),
    "circular": ("loop",),
    "pub": ("refreshment", "restaurant"),
    "inn": ("refreshment", "restaurant"),
    "café": ("refreshment", "cafe"),
    "cafe": ("refreshment",),
    "bite": ("refreshment",),
    "food": ("refreshment", "restaurant"),
    "meal": ("refreshment", "restaurant"),
    "eat": ("refreshment",),
    "drink": ("refreshment",),
    "stroller": ("pushchair",),
    "pram": ("pushchair", "stroller"),
    "buggy": ("pushchair", "stroller"),
    "kid": ("family", "playground"),
    "children": ("family", "playground"),
    "child": ("family", "playground"),
    "swim": ("swimming", "bathing"),
    "woods": ("forest",),
    "wood": ("forest",),
    "woodland": ("forest",),
    "wooded": ("forest",),
    "bike": ("cycle",),
    "ride": ("cycle",),
    "mtb": ("mountain", "bike"),
    "view": ("viewpoint",),
    "views": ("viewpoint",),
    "peak": ("summit",),
    "lakeside": ("lake",),
    "riverside": ("river",),
    "train": ("railway", "station"),
    "station": ("railway",),
    "abbey": ("monastery",),
    "rocks": ("rock", "geological"),
    "rock": ("geological",),
    "cave": ("geological",),
    "animals": ("wildlife", "zoo"),
    "wildlife": ("zoo", "bird"),
}

NEGATORS = frozenset({"no", "not", "without", "avoid", "avoiding", "never", "nothing", "aren", "isn", "don"})
_SCOPE_END = re.compile(r"[,.;:!?()]|\b(?:and|but|or|with|plus|ideally|please|maybe)\b", re.IGNORECASE)


def stem(tok: str) -> str:
    """Light English suffix stripping (both sides): plural and -ing/-ed forms. Not a full Porter stemmer."""
    if len(tok) <= 3 or not tok.isascii():
        return tok
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", "e"), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            if suf == "s" and tok.endswith(("ss", "us", "is")):
                return tok
            out = tok[: -len(suf)] + rep
            if suf == "ing" and len(out) >= 2 and out[-1] == out[-2]:  # running -> run
                out = out[:-1]
            return out
    return tok


def raw_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t]


def normalise(tokens: Iterable[str], *, do_stem: bool) -> list[str]:
    out = []
    for t in tokens:
        if t in STOPWORDS or len(t) < 2 or t.isdigit():
            continue
        out.append(stem(t) if do_stem else t)
    return out


def glossary_terms(tok: str) -> list[str]:
    """English concept terms for one German token, by contained morphemes (a crude decompounder).

    Long morphemes (>= 5 letters) match anywhere; short ones at the start or end of the token or right
    next to a long match ("Staffel|see|rundweg"); a few ambiguous ones (``inn``, ``main``, ``rund``)
    only as a whole token."""
    hits: list[str] = []
    spans: list[tuple[int, int]] = []

    def free(i: int, j: int) -> bool:
        return not any(a < j and i < b for a, b in spans)

    for m in _GLOSS_LONG:
        i = tok.find(m)
        while i >= 0:
            if free(i, i + len(m)):
                spans.append((i, i + len(m)))
                hits.extend(GLOSSARY[m])
                break
            i = tok.find(m, i + 1)
    edges = {0, len(tok)} | {a for a, _ in spans} | {b for _, b in spans}
    for m in _GLOSS_SHORT:
        if m in WHOLE_TOKEN_ONLY:
            if tok == m:
                hits.extend(GLOSSARY[m])
            continue
        i = tok.find(m)
        while i >= 0:
            j = i + len(m)
            if free(i, j) and (i in edges or j in edges) and (tok == m or len(tok) >= len(m) + 3):
                spans.append((i, j))
                hits.extend(GLOSSARY[m])
                break
            i = tok.find(m, i + 1)
    return hits


def split_negated(text: str) -> tuple[str, str]:
    """(text without negated spans, the negated spans): "no busy roads, a lake" -> (", a lake", "no busy roads")."""
    kept, dropped, i = [], [], 0
    for m in re.finditer(r"\b(" + "|".join(sorted(NEGATORS)) + r")\b", text, flags=re.IGNORECASE):
        if m.start() < i:
            continue
        kept.append(text[i : m.start()])
        end = _SCOPE_END.search(text, m.end())
        i = end.start() if end else len(text)
        dropped.append(text[m.start() : i])
    kept.append(text[i:])
    return "".join(kept), " ".join(dropped)


def remove_negated(text: str) -> str:
    """Drop the words in the scope of a negator (up to punctuation or a conjunction)."""
    return split_negated(text)[0]


# --------------------------------------------------------------------------- route texts
def generated_description(r: RouteRecord) -> str:
    """The generated English part of the description (without a leading OSM description)."""
    d = r.description or ""
    if r.osm_description and d.startswith(r.osm_description):
        return d[len(r.osm_description) :].strip()
    return d


def bm25_fields(r: RouteRecord, *, glossary: bool, do_stem: bool) -> dict[str, list[str]]:
    """Token lists per field: title (+ glossary terms), description, tags."""
    title_raw = raw_tokens(r.title)
    osm_raw = raw_tokens(r.osm_description or "")
    title = normalise(title_raw, do_stem=do_stem)
    desc = normalise(raw_tokens(generated_description(r)) + osm_raw, do_stem=do_stem)
    if glossary:
        extra_t = [g for t in title_raw for g in glossary_terms(t)]
        extra_d = [g for t in osm_raw for g in glossary_terms(t)]
        title += normalise(extra_t, do_stem=do_stem)
        desc += normalise(extra_d, do_stem=do_stem)
    tags = normalise(raw_tokens(" ".join(r.tags)), do_stem=do_stem)
    return {"title": title, "description": desc, "tags": tags}


def query_tokens(intent: str, *, do_stem: bool, negation: bool, expand: bool) -> list[str]:
    text = remove_negated(intent) if negation else intent
    toks = normalise(raw_tokens(text), do_stem=False)
    if expand:
        toks = toks + [e for t in toks for e in EXPANSIONS.get(t, ())]
    if do_stem:
        toks = [stem(t) for t in toks]
    return toks


def _feature_names(r: RouteRecord, kinds: tuple[str, ...], n: int = 3) -> list[str]:
    on = (r.features or {}).get("on_route") or {}
    names: list[str] = []
    for k in kinds:
        for item in on.get(k, [])[:n]:
            if item.get("name"):
                names.append(item["name"])
    return names


ACTIVITY_WORDS = {
    "hiking": "hiking trail",
    "cycling": "cycle route",
    "mtb": "mountain bike route",
    "running": "running route",
}


def facts_text(r: RouteRecord) -> str:
    """Compact structured card: what the route is, its numbers, and its named highlights (no prose)."""
    shape = "circular" if r.is_loop else "one-way"
    parts = [
        f"{r.title}.",
        f"{shape.capitalize()} {ACTIVITY_WORDS.get(r.activity, r.activity)} of {r.length_km:.1f} km, "
        f"{r.ascent_m} m ascent, difficulty {['easy', 'moderate', 'difficult', 'very demanding'][r.difficulty]}"
        + (f", hardest grade {r.max_grade}" if r.max_grade else "")
        + f", about {max(r.duration_min, 1) // 60} h {max(r.duration_min, 1) % 60:02d} min.",
        f"Starts near {r.locality or 'unknown'} ({r.admin_district or ''}).",
        "Features: " + ", ".join(r.tags) + ".",
    ]
    nature = _feature_names(r, ("lake", "peak", "waterfall", "gorge", "cave", "river", "viewpoint"))
    culture = _feature_names(r, ("castle", "ruins", "monastery", "museum"))
    food = _feature_names(r, ("alpine_hut", "beer_garden", "restaurant"), n=2)
    if nature:
        parts.append("Nature: " + ", ".join(nature) + ".")
    if culture:
        parts.append("Sights: " + ", ".join(culture) + ".")
    if food:
        parts.append("Food and huts: " + ", ".join(food) + ".")
    return " ".join(parts)


# Embedded document formulations (tested on dev; eval/results/_cg_dev.json)
DOC_FORMULATIONS = ("title_desc", "facts")


def embed_text(r: RouteRecord, formulation: str) -> str:
    if formulation == "title_desc":
        return f"{r.title}. {r.description}"
    if formulation == "facts":
        return facts_text(r)
    raise ValueError(f"unknown document formulation {formulation!r}")


def ranker_text(r: RouteRecord, formulation: str = "facts_desc") -> str:
    """Document text the ranker API sees for a route."""
    if formulation == "facts_desc":
        return facts_text(r) + " " + generated_description(r)
    if formulation == "desc":
        return f"{r.title}. {r.description}"
    raise ValueError(f"unknown ranker document formulation {formulation!r}")
