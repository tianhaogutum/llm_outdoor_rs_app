"""Deduplication of routes mapped more than once.

Two routes of the same activity are duplicates when each runs along the
other: >= ``min_overlap`` of A's sample points lie within ``tol_m`` of B's
line and vice versa. This catches the same trail mapped as two relations,
forward/backward variants, and a one-stage "superroute" wrapping its stage.
The keeper is the more informative relation (OSM description, more tags),
then the lower id; dropped ids are recorded on the keeper.
"""

from __future__ import annotations

from shapely import STRtree

from .geo import overlap_share


def keeper_key(r: dict):
    """Sort key: smaller is better."""
    t = r["osm_tags"]
    has_desc = any(k in t for k in ("description", "description:en", "description:de"))
    return (0 if has_desc else 1, -len(t), r["id"])


def find_duplicates(items: list[dict], tol_m: float, min_overlap: float):
    """items: dicts with id, activity, line (projected LineString), samples
    (N x 2 projected points), length_m, osm_tags.

    Returns (drop: dict[id -> keeper id], pairs_checked)."""
    if not items:
        return {}, 0
    lines = [it["line"] for it in items]
    tree = STRtree([ln.envelope.buffer(tol_m) for ln in lines])
    order = sorted(range(len(items)), key=lambda i: keeper_key(items[i]))
    rank = {i: k for k, i in enumerate(order)}
    drop: dict[int, int] = {}
    checked = 0
    # union-find-free greedy: process from best keeper down; a route is dropped
    # if it duplicates an already kept route.
    kept: set[int] = set()
    for i in order:
        a = items[i]
        cands = tree.query(lines[i])
        dup_of = None
        for j in sorted(cands.tolist(), key=lambda j: rank[j]):
            if j == i or j not in kept:
                continue
            b = items[j]
            if b["activity"] != a["activity"]:
                continue
            la, lb = a["length_m"], b["length_m"]
            if min(la, lb) / max(la, lb) < 0.75:
                continue
            checked += 1
            if overlap_share(a["samples"], lines[j], tol_m) < min_overlap:
                continue
            if overlap_share(b["samples"], lines[i], tol_m) < min_overlap:
                continue
            dup_of = b["id"]
            break
        if dup_of is None:
            kept.add(i)
        else:
            drop[a["id"]] = dup_of
    return drop, checked
