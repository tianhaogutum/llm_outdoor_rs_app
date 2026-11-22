"""Deterministic English descriptions generated from the record's own facts.

Every sentence is built only from fields of the record (no outside
knowledge). Template variants and sentence order are chosen from a hash of
the relation id so thousands of texts do not share one shape, but the same
record always yields the same text.
"""

from __future__ import annotations

import hashlib

DIFF_WORD = {0: "easy", 1: "moderate", 2: "difficult", 3: "very demanding"}
ACT_NOUN = {
    "Hiking trail": "hiking trail",
    "Theme trail": "themed trail",
    "Cycling": "cycle route",
    "Long distance cycling": "long-distance cycle route",
    "Mountainbiking": "mountain-bike route",
    "Jogging": "running route",
    "Trail running": "trail-running route",
}
TIME_VERB = {"hiking": "walking time", "running": "running time", "cycling": "riding time", "mtb": "riding time"}
SAC_WORD = {
    "T1": "T1 (hiking)",
    "T2": "T2 (mountain hiking)",
    "T3": "T3 (demanding mountain hiking)",
    "T4": "T4 (alpine hiking)",
    "T5": "T5 (demanding alpine hiking)",
    "T6": "T6 (difficult alpine hiking)",
}


def _h(rid: int, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{rid}:{salt}".encode()).digest()[:8], "big")


def _pick(rid: int, salt: str, options: list[str]) -> str:
    return options[_h(rid, salt) % len(options)]


def _km(m: float) -> str:
    km = m / 1000.0
    return f"{km:.1f} km" if km < 100 else f"{km:,.0f} km"


def _m(v: float) -> str:
    return f"{int(round(v)):,} m"


def _dur(minutes: int) -> str:
    h, mi = divmod(int(minutes), 60)
    if h == 0:
        return f"{mi} min"
    if mi == 0:
        return f"{h} h"
    return f"{h} h {mi:02d} min"


def _names(items, n=3, with_ele=False):
    out = []
    for it in items:
        nm = it.get("name")
        if not nm or nm in out:
            continue
        if with_ele and it.get("ele_m"):
            out.append(f"{nm} ({_m(it['ele_m'])})")
        else:
            out.append(nm)
        if len(out) >= n:
            break
    return out


def _join(xs: list[str]) -> str:
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def generate(r: dict) -> str:
    rid = r["id"]
    on = r["features"].get("on_route", {})
    start = r["features"].get("near_start", {})
    prof = r["features"]["profile"]
    noun = ACT_NOUN[r["category_name"]]
    loc = r.get("locality")
    county = r.get("admin_district")
    length = _km(r["length_m"])
    shape = "out-and-back" if "Out and back" in r["tags"] else "circular" if "Round trip" in r["tags"] else "linear"
    parts: list[str] = []

    # --- opening -----------------------------------------------------------
    in_county = f" in {county}" if county and county != loc else ""
    frm, to = r["osm_tags"].get("from"), r["osm_tags"].get("to")
    end_loc = r.get("end_locality")
    if shape == "linear" and frm and to and frm != to:
        parts.append(
            _pick(
                rid,
                "open",
                [
                    f"A {length} linear {noun} from {frm} to {to}.",
                    f"This {noun} runs {length} from {frm} to {to}.",
                    f"Linear {noun} of {length} between {frm} and {to}.",
                ],
            )
        )
    elif shape == "linear" and loc and end_loc and end_loc != loc:
        parts.append(
            _pick(
                rid,
                "open",
                [
                    f"A {length} linear {noun}{in_county} that starts near {loc} and ends near {end_loc}.",
                    f"This {noun} runs {length} from near {loc} to near {end_loc}{in_county}.",
                    f"Linear {noun} of {length}{in_county}, from the {loc} area to the {end_loc} area.",
                ],
            )
        )
    else:
        art = "An" if shape[0] in "aeiou" else "A"
        near = f" near {loc}" if loc else ""
        parts.append(
            _pick(
                rid,
                "open",
                [
                    f"{art} {shape} {noun} of {length} that starts{near}{in_county}.",
                    f"This {length} {shape} {noun} starts{near}{in_county}.",
                    f"{art} {length} {shape} {noun}{in_county}, starting{near}.".replace(", starting.", "."),
                ],
            )
        )

    # --- elevation ---------------------------------------------------------
    body: list[str] = []
    if r["ascent_m"] is not None:
        asc, desc = r["ascent_m"], r["descent_m"]
        lo, hi = r["min_ele_m"], r["max_ele_m"]
        if max(asc, desc) < 60:
            body.append(
                _pick(
                    rid,
                    "ele",
                    [
                        f"It is almost flat, staying between {_m(lo)} and {_m(hi)} above sea level.",
                        f"The profile is nearly level ({_m(lo)} to {_m(hi)} elevation).",
                    ],
                )
            )
        elif desc > 2 * asc + 100 and not r["is_loop"]:
            body.append(
                _pick(
                    rid,
                    "ele",
                    [
                        f"As mapped it runs mostly downhill: about {_m(desc)} of descent and "
                        f"{_m(asc)} of ascent, between {_m(lo)} and {_m(hi)} above sea level.",
                        f"In the mapped direction it descends about {_m(desc)} (ascent {_m(asc)}); "
                        f"the highest point is at {_m(hi)}.",
                    ],
                )
            )
        else:
            body.append(
                _pick(
                    rid,
                    "ele",
                    [
                        f"It climbs about {_m(asc)} and descends {_m(desc)}, between {_m(lo)} and "
                        f"{_m(hi)} above sea level.",
                        f"Expect roughly {_m(asc)} of ascent and {_m(desc)} of descent; the highest "
                        f"point is at {_m(hi)}.",
                        f"The elevation ranges from {_m(lo)} to {_m(hi)}, with about {_m(asc)} of climbing in total.",
                    ],
                )
            )

    # --- natural highlights -------------------------------------------------
    peaks = _names(on.get("peak", []), 3, with_ele=True)
    lakes = _names(on.get("lake", []), 3)
    rivers = _names(on.get("river", []), 2)
    falls = _names(on.get("waterfall", []), 2)
    gorges = _names(on.get("gorge", []), 2)
    views = _names(on.get("viewpoint", []), 2)
    nat = []

    def grp(items, one, many):
        return (one if len(items) == 1 else many) + " " + _join(items)

    if peaks:
        nat.append(grp(peaks, "the summit of", "the summits"))
    if lakes:
        nat.append(grp(lakes, "the lake", "the lakes"))
    if gorges:
        nat.append(grp(gorges, "the gorge", "the gorges"))
    if falls:
        nat.append(grp(falls, "the waterfall", "the waterfalls"))
    if nat:
        # groups contain "and" themselves, so separate them with semicolons
        listed = "; ".join(nat) if len(nat) > 1 else nat[0]
        body.append(
            _pick(
                rid,
                "nat",
                [
                    f"Along the way it passes {listed}.",
                    f"Natural highlights on the route: {listed}.",
                    f"The route takes in {listed}.",
                ],
            )
        )
    elif on.get("waterfall") or on.get("viewpoint"):
        bits = []
        if on.get("waterfall"):
            bits.append("a waterfall")
        if on.get("viewpoint"):
            bits.append("a viewpoint")
        body.append(f"There is {_join(bits)} on the way.")
    if rivers:
        body.append(
            _pick(
                rid,
                "riv",
                [
                    f"Part of it follows the {_join(rivers)}.",
                    f"Sections run along the {_join(rivers)}.",
                ],
            )
        )
    if views and nat:
        body.append(
            _pick(
                rid,
                "view",
                [
                    f"Viewpoints: {_join(views)}.",
                    f"Look out for the viewpoint{'s' if len(views) > 1 else ''} {_join(views)}.",
                ],
            )
        )

    # --- huts and food -------------------------------------------------------
    huts = _names(on.get("alpine_hut", []) + on.get("wilderness_hut", []), 3, with_ele=True)
    food = _names(on.get("beer_garden", []) + on.get("restaurant", []), 3)
    if huts:
        body.append(
            _pick(
                rid,
                "hut",
                [
                    f"Huts on the route: {_join(huts)}.",
                    f"It passes the hut{'s' if len(huts) > 1 else ''} {_join(huts)}.",
                    f"You can stop at {_join(huts)}.",
                ],
            )
        )
    if food:
        body.append(
            _pick(
                rid,
                "food",
                [
                    f"Places to eat or drink along the way include {_join(food)}.",
                    f"Refreshments are available at {_join(food)}.",
                    f"For a break there is {_join(food)}.",
                ],
            )
        )
    elif "Refreshment stops available" in r["tags"] and not huts:
        body.append("There are places for refreshments along the way.")

    # --- culture ------------------------------------------------------------
    cult = _names(
        on.get("castle", [])
        + on.get("monastery", [])
        + on.get("ruins", [])
        + on.get("museum", [])
        + on.get("archaeological_site", [])
        + on.get("monument", []),
        3,
    )
    if cult:
        body.append(
            _pick(
                rid,
                "cult",
                [
                    f"Sights include {_join(cult)}.",
                    f"Cultural stops: {_join(cult)}.",
                    f"On the cultural side it passes {_join(cult)}.",
                ],
            )
        )
    churches = _names(on.get("church", []), 2)
    if churches and not cult:
        body.append(f"It passes {_join(churches)}.")
    prot = [
        p["name"]
        for p in r["features"].get("protected_areas", [])
        if p["kind"] in ("nature_reserve", "national_park") and p["name"]
    ][:2]
    if prot:
        body.append(
            _pick(
                rid,
                "prot",
                [
                    f"Part of the route lies in {_join(prot)}.",
                    f"It crosses the protected area{'s' if len(prot) > 1 else ''} {_join(prot)}.",
                ],
            )
        )
    family_bits = []
    if on.get("playground"):
        family_bits.append("a playground")
    if on.get("swimming"):
        family_bits.append("a swimming spot")
    if family_bits:
        body.append(f"There is {_join(family_bits)} on the way.")

    # --- terrain ------------------------------------------------------------
    terr = []
    if prof["paved_share"] >= 0.7:
        terr.append(f"mostly paved ({round(prof['paved_share'] * 100)}%)")
    elif prof["unpaved_share"] >= 0.7:
        terr.append(f"mostly unpaved ({round(prof['unpaved_share'] * 100)}%)")
    elif prof["paved_share"] + prof["unpaved_share"] > 0:
        terr.append(
            f"a mix of paved ({round(prof['paved_share'] * 100)}%) and unpaved "
            f"({round(prof['unpaved_share'] * 100)}%) surfaces"
        )
    if prof["path_share"] >= 0.3:
        terr.append(f"{round(prof['path_share'] * 100)}% on footpaths")
    elif prof["track_share"] >= 0.3:
        terr.append(f"{round(prof['track_share'] * 100)}% on tracks")
    if prof["busy_road_share"] >= 0.1:
        terr.append(f"{round(prof['busy_road_share'] * 100)}% on busier roads")
    if terr:
        tt = terr[0] + (", with " + _join(terr[1:]) if len(terr) > 1 else "")
        body.append(
            _pick(
                rid,
                "terr",
                [
                    f"The surface is {tt}.",
                    f"Underfoot: {tt}.",
                ],
            )
            if r["activity"] in ("hiking", "running")
            else f"The surface is {tt}."
        )
    if r.get("max_grade") and r["max_grade"].startswith("T") and r["max_grade"] != "T1":
        body.append(f"The hardest sections are graded {SAC_WORD.get(r['max_grade'], r['max_grade'])} on the SAC scale.")
    elif r.get("max_grade") and r["max_grade"].startswith("S"):
        body.append(f"The hardest sections are graded {r['max_grade']} on the MTB scale.")
    if "Via ferrata" in r["tags"]:
        body.append("It includes a via ferrata section.")

    # shuffle the middle deterministically (keep elevation first)
    if len(body) > 2:
        head, rest = body[:1], body[1:]
        rot = _h(rid, "order") % len(rest)
        if _h(rid, "order2") % 3 == 0:
            rest = rest[rot:] + rest[:rot]
        body = head + rest
    parts.extend(body)

    # --- difficulty, duration ----------------------------------------------
    dw = DIFF_WORD[r["difficulty"]]
    dur = _dur(r["duration_min"])
    tv = TIME_VERB[r["activity"]]
    days = r.get("est_days", 1)
    day_txt = f" (about {days} days)" if days >= 2 else ""
    parts.append(
        _pick(
            rid,
            "diff",
            [
                f"Rated {dw}; estimated {tv} {dur}{day_txt}.",
                f"Difficulty: {dw}. Estimated {tv}: {dur}{day_txt}.",
                f"It is rated {dw} and takes about {dur}{day_txt}.",
            ],
        )
    )

    # --- access -------------------------------------------------------------
    acc = []
    if start.get("train_station"):
        s = start["train_station"][0]
        acc.append(
            f"the railway station {s['name']} is {_m(s['dist_m'])} from the start"
            if s["name"]
            else f"a railway station is {_m(s['dist_m'])} from the start"
        )
    if start.get("bus_stop"):
        b = start["bus_stop"][0]
        acc.append(
            f"the bus stop {b['name']} is {_m(b['dist_m'])} away"
            if b["name"]
            else f"a bus stop is {_m(b['dist_m'])} away"
        )
    if start.get("parking"):
        acc.append("there is parking near the start")
    if acc:
        s = _join(acc)
        parts.append(s[0].upper() + s[1:] + ".")

    fam = []
    if "Family-friendly" in r["tags"]:
        fam.append("suitable for families")
    if "Suitable for strollers" in r["tags"]:
        fam.append("passable with a pushchair")
    if fam:
        parts.append(f"By the catalogue's rules it is {_join(fam)}.")

    op = r["osm_tags"].get("operator")
    if op:
        parts.append(_pick(rid, "op", [f"Maintained by {op}.", f"Operator: {op}."]))
    return " ".join(parts)


def compose(r: dict) -> tuple[str, str]:
    """(description, description_source). An OSM description (English
    preferred) is quoted first and the generated text follows."""
    gen = generate(r)
    t = r["osm_tags"]
    osm = t.get("description:en") or t.get("description") or t.get("description:de")
    if osm:
        osm = " ".join(osm.split())
        if not osm.endswith((".", "!", "?")):
            osm += "."
        return f"{osm} {gen}", "osm+generated"
    return gen, "generated"
