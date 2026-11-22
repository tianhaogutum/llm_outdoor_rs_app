import numpy as np
from shapely.geometry import LineString

from pipeline.dedup import find_duplicates


def item(rid, xy, activity="hiking", tags=None):
    line = LineString(xy)
    n = max(2, int(line.length // 50) + 1)
    samples = np.array([line.interpolate(d).coords[0] for d in np.linspace(0, line.length, n)])
    return {
        "id": rid,
        "activity": activity,
        "line": line,
        "samples": samples,
        "length_m": line.length,
        "osm_tags": tags or {"name": "x"},
    }


def test_same_trail_twice_is_merged_keeping_richer():
    a = item(10, [(0, 0), (5000, 0)])
    b = item(5, [(5000, 10), (0, 10)], tags={"name": "x", "description": "d", "network": "lwn"})  # reversed, 10 m off
    drop, checked = find_duplicates([a, b], 40, 0.9)
    assert drop == {10: 5}
    assert checked >= 1


def test_lower_id_wins_on_tie():
    a = item(10, [(0, 0), (5000, 0)])
    b = item(7, [(0, 5), (5000, 5)])
    drop, _ = find_duplicates([a, b], 40, 0.9)
    assert drop == {10: 7}


def test_different_activity_or_partial_overlap_kept():
    a = item(1, [(0, 0), (5000, 0)])
    b = item(2, [(0, 0), (5000, 0)], activity="cycling")
    c = item(3, [(0, 0), (2500, 0), (2500, 2500)])  # half shared
    d = item(
        4, [(0, 0), (4200, 0)]
    )  # stage covering 84% of a: different route, length ratio ok but overlap one way < 0.9
    drop, _ = find_duplicates([a, b, c, d], 40, 0.9)
    assert drop == {}
