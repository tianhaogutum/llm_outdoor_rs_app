"""Pure geometry helpers: distances, way stitching, resampling, loop tests.

Coordinate convention everywhere in the pipeline: ``(lon, lat)`` in WGS84
degrees (GeoJSON order). Output records spell the order out as
``start_lat`` / ``start_lon`` so nothing downstream has to guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres. Works on scalars or numpy arrays."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def point_dist_m(p, q) -> float:
    return float(haversine_m(p[0], p[1], q[0], q[1]))


def line_length_m(coords) -> float:
    """Length of a polyline given as a sequence of (lon, lat)."""
    if len(coords) < 2:
        return 0.0
    a = np.asarray(coords, dtype=float)
    return float(haversine_m(a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1]).sum())


def cumulative_m(coords) -> np.ndarray:
    a = np.asarray(coords, dtype=float)
    if len(a) < 2:
        return np.zeros(len(a))
    d = haversine_m(a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1])
    return np.concatenate([[0.0], np.cumsum(d)])


class LocalProjection:
    """Equirectangular projection around a reference latitude (metres).

    Accurate to well under 1% within a few hundred km of the reference
    latitude, which is plenty for proximity tests and simplification inside
    one region. Lengths reported in the catalogue use haversine, not this.
    """

    def __init__(self, lat0: float = 48.0, lon0: float = 11.0):
        self.lat0 = lat0
        self.lon0 = lon0
        self.kx = math.radians(1) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * EARTH_RADIUS_M

    def fwd(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def inv(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x / self.kx + self.lon0, y / self.ky + self.lat0

    def fwd_coords(self, coords) -> np.ndarray:
        a = np.asarray(coords, dtype=float).reshape(-1, 2)
        x, y = self.fwd(a[:, 0], a[:, 1])
        return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------
@dataclass
class WayPiece:
    """One member way: OSM way id, node ids and their (lon, lat) coordinates."""

    way_id: int
    nodes: list[int]
    coords: list[tuple[float, float]]


@dataclass
class _Seg:
    nodes: list[int]
    coords: list[tuple[float, float]]
    ways: list[int]
    order: int  # relation index of the first way, used to break ties

    def reverse(self):
        self.nodes.reverse()
        self.coords.reverse()
        self.ways.reverse()


@dataclass
class StitchResult:
    coords: list[tuple[float, float]]  # main line
    way_ids: list[int]  # ways along the main line, in order (consecutive repeats collapsed)
    way_seq: list[tuple[int, float]]  # (way id, metres of that way used) along the line
    n_input_ways: int
    n_ordered_segments: int  # segments after the relation-order pass
    n_chains: int  # 1 = everything used; >1 = unused chains were dropped
    coverage: float  # main line length (without gaps) / all member length
    n_gaps_bridged: int
    gap_m: float  # total length of bridged gaps in the main line
    length_m: float
    method: str = "ordered"  # "ordered" | "graph"
    dropped_way_ids: list[int] = field(default_factory=list)


def _ordered_pass(pieces: list[WayPiece]) -> list[_Seg]:
    """Chain ways in relation order, flipping as needed. Breaks on gaps."""
    segs: list[_Seg] = []
    cur: _Seg | None = None
    single = False  # current segment still has only one way: orientation open
    for idx, w in enumerate(pieces):
        if len(w.nodes) < 2:
            continue
        if cur is None:
            cur = _Seg(list(w.nodes), list(w.coords), [w.way_id], idx)
            single = True
            continue
        s, e = cur.nodes[0], cur.nodes[-1]
        ws, we = w.nodes[0], w.nodes[-1]
        if ws == e:
            cur.nodes += w.nodes[1:]
            cur.coords += w.coords[1:]
            cur.ways.append(w.way_id)
        elif we == e:
            cur.nodes += w.nodes[-2::-1]
            cur.coords += w.coords[-2::-1]
            cur.ways.append(w.way_id)
        elif single and ws == s:
            cur.reverse()
            cur.nodes += w.nodes[1:]
            cur.coords += w.coords[1:]
            cur.ways.append(w.way_id)
        elif single and we == s:
            cur.reverse()
            cur.nodes += w.nodes[-2::-1]
            cur.coords += w.coords[-2::-1]
            cur.ways.append(w.way_id)
        else:
            segs.append(cur)
            cur = _Seg(list(w.nodes), list(w.coords), [w.way_id], idx)
            single = True
            continue
        single = False
    if cur is not None:
        segs.append(cur)
    return segs


@dataclass
class _Edge:
    u: int
    v: int
    coords: list  # from u to v
    ways: list  # [(way_id, metres)] from u to v
    length: float
    gap: bool = False


def _split_pieces(pieces: list[WayPiece]) -> list[_Edge]:
    """Split member ways at every node shared with another member way, so a
    way that is not split where the route turns (a T-junction on an interior
    node, or an overshooting way) becomes proper graph edges."""
    occ: dict[int, int] = {}
    for p in pieces:
        for n in set(p.nodes):
            occ[n] = occ.get(n, 0) + 1
    edges = []
    for p in pieces:
        cut = [0] + [i for i in range(1, len(p.nodes) - 1) if occ[p.nodes[i]] >= 2] + [len(p.nodes) - 1]
        for a, b in zip(cut[:-1], cut[1:], strict=True):
            c = p.coords[a : b + 1]
            L = line_length_m(c)
            edges.append(_Edge(p.nodes[a], p.nodes[b], list(c), [(p.way_id, L)], L))
    return edges


def _merge_deg2(edges: list[_Edge]) -> list[_Edge]:
    """Merge edges through nodes of degree exactly 2 (like linemerge)."""
    adj: dict[int, list[int]] = {}
    for i, e in enumerate(edges):
        adj.setdefault(e.u, []).append(i)
        adj.setdefault(e.v, []).append(i)
    used = [False] * len(edges)
    out = []

    def oriented(i, start):
        e = edges[i]
        if e.u == start:
            return e.coords, e.ways, e.v
        return e.coords[::-1], e.ways[::-1], e.u

    def walk(i, start):
        coords, ways, end = oriented(i, start)
        coords, ways = list(coords), list(ways)
        used[i] = True
        while len(adj[end]) == 2 and end != start:
            nxt = [j for j in adj[end] if not used[j]]
            if not nxt:
                break
            j = nxt[0]
            c2, w2, end2 = oriented(j, end)
            coords += c2[1:]
            ways += w2
            used[j] = True
            end = end2
        return coords, ways, end

    # start chains at nodes with degree != 2, then leftover pure cycles
    for n in sorted(adj):
        if len(adj[n]) == 2:
            continue
        for i in adj[n]:
            if not used[i]:
                c, w, end = walk(i, n)
                out.append(_Edge(n, end, c, w, sum(L for _, L in w)))
    for i in range(len(edges)):
        if not used[i]:
            n = edges[i].u
            c, w, end = walk(i, n)
            out.append(_Edge(n, end, c, w, sum(L for _, L in w)))
    return out


def _longest_trail(chains: list[_Edge], start_pref: list[int], budget: int = 20000):
    """Longest trail (no edge used twice) over the chain graph by bounded DFS.
    Returns list of (edge index, reversed?)."""
    adj: dict[int, list[int]] = {}
    for i, e in enumerate(chains):
        adj.setdefault(e.u, []).append(i)
        if e.v != e.u:
            adj.setdefault(e.v, []).append(i)
    for n in adj:
        adj[n].sort(key=lambda i: (-chains[i].length, i))
    odd = sorted(n for n, lst in adj.items() if sum(2 if chains[i].u == chains[i].v else 1 for i in lst) % 2 == 1)
    starts = []
    for n in list(start_pref) + odd + sorted(adj):
        if n in adj and n not in starts:
            starts.append(n)
    starts = starts[:40]
    total = sum(e.length for e in chains)
    best = (-1.0, [])
    steps = 0
    used = [False] * len(chains)

    def dfs(node, length, path):
        nonlocal best, steps
        steps += 1
        if length > best[0] + 1e-9:
            best = (length, list(path))
        if steps > budget or best[0] >= total - 1e-6:
            return
        for i in adj.get(node, ()):
            if used[i]:
                continue
            e = chains[i]
            rev = e.u != node
            used[i] = True
            path.append((i, rev))
            dfs(e.u if rev else e.v, length + e.length, path)
            path.pop()
            used[i] = False
            if steps > budget or best[0] >= total - 1e-6:
                return

    import sys

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10000))
    try:
        for s in starts:
            dfs(s, 0.0, [])
            if steps > budget or best[0] >= total - 1e-6:
                break
    finally:
        sys.setrecursionlimit(old)
    return best[1]


def stitch(pieces: list[WayPiece], gap_tol_m: float = 50.0) -> StitchResult | None:
    """Assemble member ways into one line.

    1. Ordered pass: follow the relation order, flipping ways as needed. A
       correctly ordered relation comes out as one segment and is used as is
       (keeps the mapper's direction and start).
    2. Otherwise graph pass: split ways at shared nodes, merge through
       degree-2 nodes, connect dangling ends to other chain ends within
       ``gap_tol_m`` (virtual gap edges), and take the longest trail (no
       edge twice) through the result. Unused chains (overshooting spurs,
       alternatives, disconnected bits) are dropped; the caller decides
       whether the coverage is good enough.
    """
    pieces = [p for p in pieces if len(p.nodes) >= 2]
    if not pieces:
        return None
    segs = _ordered_pass(pieces)
    n_ordered = len(segs)
    wl = {p.way_id: line_length_m(p.coords) for p in pieces}
    if n_ordered == 1:
        s = segs[0]
        L = line_length_m(s.coords)
        return StitchResult(
            coords=s.coords,
            way_ids=_collapse(s.ways),
            way_seq=[(w, wl[w]) for w in s.ways],
            n_input_ways=len(pieces),
            n_ordered_segments=1,
            n_chains=1,
            coverage=1.0,
            n_gaps_bridged=0,
            gap_m=0.0,
            length_m=L,
            method="ordered",
        )

    edges = _split_pieces(pieces)
    chains = _merge_deg2(edges)
    total = sum(e.length for e in chains)

    # gap edges: dangling ends -> nearest other chain end within tolerance
    deg: dict[int, int] = {}
    pos: dict[int, tuple] = {}
    for e in chains:
        deg[e.u] = deg.get(e.u, 0) + 1
        deg[e.v] = deg.get(e.v, 0) + 1
        pos[e.u] = e.coords[0]
        pos[e.v] = e.coords[-1]
    ends = sorted(pos)
    cell = max(gap_tol_m / 111_000.0, 1e-6)
    grid: dict[tuple, list] = {}
    for n in ends:
        c = pos[n]
        grid.setdefault((int(math.floor(c[0] / cell)), int(math.floor(c[1] / cell))), []).append(n)
    cands = []
    for n in ends:
        if deg[n] != 1:
            continue
        c = pos[n]
        gx, gy = int(math.floor(c[0] / cell)), int(math.floor(c[1] / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for m in grid.get((gx + dx, gy + dy), ()):
                    if m == n or (deg[m] == 1 and m < n):
                        continue
                    d = point_dist_m(c, pos[m])
                    if 0 < d <= gap_tol_m:
                        cands.append((d, n, m))
    cands.sort()
    taken = set()
    n_gap_edges = 0
    for d, n, m in cands:
        if n in taken or (deg[m] == 1 and m in taken):
            continue
        taken.add(n)
        if deg[m] == 1:
            taken.add(m)
        chains.append(_Edge(n, m, [pos[n], pos[m]], [], d, gap=True))
        n_gap_edges += 1

    first = pieces[0]
    trail = _longest_trail(chains, [first.nodes[0], first.nodes[-1]])
    coords: list = []
    way_seq: list = []
    gaps = 0
    gap_m = 0.0
    real = 0.0
    used = set()
    for i, rev in trail:
        e = chains[i]
        used.add(i)
        c = e.coords[::-1] if rev else e.coords
        w = e.ways[::-1] if rev else e.ways
        if coords and coords[-1] == c[0]:
            c = c[1:]
        coords.extend(c)
        way_seq.extend(w)
        if e.gap:
            gaps += 1
            gap_m += e.length
        else:
            real += e.length
    if len(coords) < 2:
        return None
    # loops: rotate so the line starts where the relation's first way starts
    if coords[0] == coords[-1]:
        for target in (first.coords[0], first.coords[-1]):
            if target in coords[:-1]:
                k = coords.index(target)
                coords = coords[k:-1] + coords[: k + 1]
                break
    elif point_dist_m(coords[-1], first.coords[0]) < point_dist_m(coords[0], first.coords[0]) and min(
        point_dist_m(coords[-1], first.coords[0]), point_dist_m(coords[-1], first.coords[-1])
    ) < min(point_dist_m(coords[0], first.coords[0]), point_dist_m(coords[0], first.coords[-1])):
        coords.reverse()
        way_seq.reverse()
    used_ways = {w for w, _ in way_seq}
    unused_real = [i for i, e in enumerate(chains) if i not in used and not e.gap]
    return StitchResult(
        coords=coords,
        way_ids=_collapse([w for w, _ in way_seq]),
        way_seq=_merge_seq(way_seq),
        n_input_ways=len(pieces),
        n_ordered_segments=n_ordered,
        n_chains=1 + len(unused_real),
        coverage=(real / total) if total > 0 else 0.0,
        n_gaps_bridged=gaps,
        gap_m=gap_m,
        length_m=line_length_m(coords),
        method="graph",
        dropped_way_ids=sorted({p.way_id for p in pieces} - used_ways),
    )


def _collapse(ws):
    out = []
    for w in ws:
        if not out or out[-1] != w:
            out.append(w)
    return out


def _merge_seq(seq):
    out = []
    for w, L in seq:
        if out and out[-1][0] == w:
            out[-1] = (w, out[-1][1] + L)
        else:
            out.append((w, L))
    return out


# ---------------------------------------------------------------------------
# Resampling, loops, overlap
# ---------------------------------------------------------------------------
def resample(coords, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``spacing_m`` along the line (plus the last vertex).

    Returns (points[N, 2] as lon/lat, distance_along[N] in metres). Linear
    interpolation in lon/lat between vertices is fine at OSM vertex spacing.
    """
    a = np.asarray(coords, dtype=float)
    cum = cumulative_m(a)
    total = cum[-1] if len(cum) else 0.0
    if total <= 0:
        return a[:1].copy(), np.zeros(1)
    n = max(2, int(math.floor(total / spacing_m)) + 1)
    d = np.arange(n) * spacing_m
    if d[-1] < total - 1e-6:
        d = np.append(d, total)
    lon = np.interp(d, cum, a[:, 0])
    lat = np.interp(d, cum, a[:, 1])
    return np.column_stack([lon, lat]), d


def is_loop(coords, max_endpoint_dist_m: float = 300.0) -> bool:
    if len(coords) < 2:
        return False
    return point_dist_m(coords[0], coords[-1]) <= max_endpoint_dist_m


def self_retrace_share(
    coords, tol_m: float = 30.0, min_sep_m: float = 300.0, proj: LocalProjection | None = None
) -> float:
    """Share of the line that runs within ``tol_m`` of another part of itself
    that is at least ``min_sep_m`` away along the line. ~1.0 for an
    out-and-back line, ~0 for a proper loop or a linear route."""
    from shapely import STRtree, linestrings
    from shapely import points as shp_points

    proj = proj or LocalProjection()
    pts, dist = resample(coords, 25.0)
    if len(pts) < 4:
        return 0.0
    xy = proj.fwd_coords(pts)
    segs = linestrings(np.stack([xy[:-1], xy[1:]], axis=1))
    seg_mid_d = (dist[:-1] + dist[1:]) / 2
    tree = STRtree(segs)
    pq = shp_points(xy)
    pi, si = tree.query(pq, predicate="dwithin", distance=tol_m)
    far = np.abs(dist[pi] - seg_mid_d[si]) >= min_sep_m
    hit = np.zeros(len(pts), dtype=bool)
    hit[pi[far]] = True
    return float(hit.mean())


def simplify(coords, tol_m: float, proj: LocalProjection | None = None):
    """Douglas-Peucker simplification in metres; returns (lon, lat) list."""
    from shapely import LineString

    if len(coords) < 3:
        return [tuple(c) for c in coords]
    proj = proj or LocalProjection()
    xy = proj.fwd_coords(coords)
    s = LineString(xy).simplify(tol_m, preserve_topology=False)
    sx, sy = np.asarray(s.coords).T
    lon, lat = proj.inv(sx, sy)
    return list(zip(lon.tolist(), lat.tolist(), strict=True))


def overlap_share(a_xy: np.ndarray, b_line, tol_m: float) -> float:
    """Share of sample points of A (projected, metres) within tol of line B."""
    from shapely import dwithin
    from shapely import points as shp_points

    if len(a_xy) == 0:
        return 0.0
    return float(dwithin(shp_points(a_xy), b_line, tol_m).mean())
