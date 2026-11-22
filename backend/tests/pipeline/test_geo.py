import math

import pytest

from pipeline.geo import WayPiece, haversine_m, is_loop, line_length_m, resample, self_retrace_share, simplify, stitch

# ~111 m per 0.001 deg latitude; build lines along a meridian/parallel near 47.5N
LAT0, LON0 = 47.5, 11.0
DLAT = 0.001  # ~111.2 m


def pt(i, j=0):
    """grid point i steps north, j steps east (0.001 deg each)."""
    return (LON0 + j * DLAT, LAT0 + i * DLAT)


def way(wid, node_ids, pts):
    return WayPiece(wid, list(node_ids), list(pts))


def straight_ways():
    # 3 ways along a meridian: nodes 1-2-3, 3-4-5, 5-6-7
    return [
        way(10, [1, 2, 3], [pt(0), pt(1), pt(2)]),
        way(11, [3, 4, 5], [pt(2), pt(3), pt(4)]),
        way(12, [5, 6, 7], [pt(4), pt(5), pt(6)]),
    ]


def test_haversine_known_distance():
    # 1 degree of latitude ~ 111.2 km
    assert haversine_m(11.0, 47.0, 11.0, 48.0) == pytest.approx(111_195, rel=1e-3)
    # coordinate order matters: (lon, lat)
    d1 = haversine_m(11.0, 47.0, 12.0, 47.0)
    assert d1 == pytest.approx(111_195 * math.cos(math.radians(47)), rel=2e-3)


def test_line_length():
    ws = straight_ways()
    res = stitch(ws)
    assert res.length_m == pytest.approx(6 * 111.2, rel=2e-3)
    assert line_length_m([pt(0)]) == 0.0


def test_stitch_ordered():
    res = stitch(straight_ways())
    assert res.n_ordered_segments == 1
    assert res.n_chains == 1
    assert res.coverage == pytest.approx(1.0)
    assert res.coords[0] == pt(0) and res.coords[-1] == pt(6)
    assert len(res.coords) == 7  # shared nodes not duplicated
    assert res.way_ids == [10, 11, 12]


def test_stitch_reversed_members():
    ws = straight_ways()
    # second way stored in the opposite direction; first way reversed as well
    ws[0] = way(10, [3, 2, 1], [pt(2), pt(1), pt(0)])
    ws[1] = way(11, [5, 4, 3], [pt(4), pt(3), pt(2)])
    res = stitch(ws)
    assert res.n_ordered_segments == 1
    assert {res.coords[0], res.coords[-1]} == {pt(0), pt(6)}
    assert res.coverage == pytest.approx(1.0)


def test_stitch_unordered_members():
    ws = straight_ways()
    res = stitch([ws[2], ws[0], ws[1]])
    assert res.n_ordered_segments > 1
    assert res.n_chains == 1
    assert res.coverage == pytest.approx(1.0)
    assert res.length_m == pytest.approx(6 * 111.2, rel=2e-3)
    assert {res.coords[0], res.coords[-1]} == {pt(0), pt(6)}


def test_stitch_small_gap_is_bridged():
    ws = straight_ways()
    # way 12 starts 20 m north of node 5 (different node id): bridged
    off = (pt(4)[0], pt(4)[1] + 0.00018)
    ws[2] = way(12, [50, 6, 7], [off, pt(5), pt(6)])
    res = stitch(ws, gap_tol_m=50)
    assert res.n_chains == 1
    assert res.n_gaps_bridged == 1
    assert 15 < res.gap_m < 25


def test_stitch_large_gap_splits_and_keeps_longest():
    ws = straight_ways()
    far = [(p[0] + 0.05, p[1]) for p in (pt(4), pt(5), pt(6))]  # ~3.7 km east
    ws[2] = way(12, [50, 60, 70], far)
    res = stitch(ws, gap_tol_m=50)
    assert res.n_chains == 2
    assert res.way_ids == [10, 11]
    assert res.coverage == pytest.approx(4 / 6, rel=1e-3)
    assert res.dropped_way_ids == [12]


def test_stitch_out_and_back_way_listed_twice():
    ws = straight_ways()
    res = stitch([ws[0], ws[1], ws[1], ws[0]])
    assert res.n_ordered_segments == 1
    assert res.coords[0] == res.coords[-1] == pt(0)
    assert res.length_m == pytest.approx(8 * 111.2, rel=2e-3)


def test_stitch_loop_closes():
    # square loop of 4 ways, given unordered
    a, b, c, d = pt(0, 0), pt(0, 5), pt(5, 5), pt(5, 0)
    ws = [way(1, [1, 2], [a, b]), way(3, [3, 4], [c, d]), way(2, [2, 3], [b, c]), way(4, [4, 1], [d, a])]
    res = stitch(ws)
    assert res.n_chains == 1
    assert res.coverage == pytest.approx(1.0)
    assert is_loop(res.coords)


def test_stitch_empty_and_degenerate():
    assert stitch([]) is None
    assert stitch([way(1, [1], [pt(0)])]) is None


def test_loop_detection_threshold():
    assert is_loop([pt(0), pt(10), pt(0, 2)], 300)  # end 150 m east of start
    assert not is_loop([pt(0), pt(10), pt(0, 4)], 300)  # end ~300 m east -> not within 300
    assert not is_loop([pt(0), pt(10)], 300)


def test_resample_spacing_and_end():
    line = [pt(0), pt(10)]  # ~1112 m
    pts, d = resample(line, 100)
    assert d[0] == 0 and d[-1] == pytest.approx(line_length_m(line))
    assert all(abs(x - 100) < 1e-6 for x in (d[1:-1] - d[:-2]))


def test_retrace_share_out_and_back_vs_loop():
    out_back = [pt(i) for i in range(0, 30)] + [pt(i) for i in range(28, -1, -1)]
    loop = [pt(0, 0), pt(0, 30), pt(30, 30), pt(30, 0), pt(0, 0)]
    assert self_retrace_share(out_back) > 0.8
    assert self_retrace_share(loop) < 0.1


def test_simplify_keeps_ends():
    line = [pt(i, (i % 2) * 0.01) for i in range(50)]  # ~1 m zigzag
    s = simplify(line, 10)
    assert len(s) == 2
    assert s[0] == pytest.approx(line[0]) and s[-1] == pytest.approx(line[-1])


def test_stitch_unsplit_way_t_junction_and_overshoot():
    # way 1 runs north rows 0..6 (nodes 1..7) but the route turns east at node 4 (row 3);
    # rows 3..6 of way 1 are an overshoot spur. Way 2 goes east from node 4.
    w1 = way(1, [1, 2, 3, 4, 5, 6, 7], [pt(i) for i in range(7)])
    w2 = way(2, [4, 20, 21, 22, 23, 24], [pt(3, j) for j in range(6)])
    res = stitch([w2, w1])  # also out of order
    assert res.method == "graph"
    main = line_length_m([pt(0), pt(3)]) + line_length_m([pt(3, 0), pt(3, 5)])
    spur = line_length_m([pt(3), pt(6)])
    assert {res.coords[0], res.coords[-1]} == {pt(0), pt(3, 5)}
    assert res.length_m == pytest.approx(main, rel=1e-6)
    assert res.coverage == pytest.approx(main / (main + spur), rel=1e-6)
    assert res.n_chains == 2


def test_stitch_lollipop_with_unsplit_ways():
    # stick: nodes 1-2-3 (rows 0..2); loop from node 3 around a square and back to node 3,
    # loop drawn as ONE closed way starting at node 3
    sq = [pt(2, 0), pt(2, 3), pt(5, 3), pt(5, 0), pt(2, 0)]
    stick = way(1, [1, 2, 3], [pt(0), pt(1), pt(2)])
    loop = way(2, [3, 30, 31, 32, 3], sq)
    res = stitch([loop, stick])
    assert res.coverage == pytest.approx(1.0)
    assert res.length_m == pytest.approx(line_length_m(stick.coords) + line_length_m(sq), rel=1e-6)
    assert sum(L for _, L in res.way_seq) == pytest.approx(res.length_m, rel=1e-6)
