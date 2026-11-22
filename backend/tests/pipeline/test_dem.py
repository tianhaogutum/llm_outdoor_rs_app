import gzip

import numpy as np
import pytest

from pipeline.dem import DEM, VOID, ascent_descent, fill_voids, read_hgt, smooth, tile_name


def synthetic_tile(n=3601):
    """Plane: z = 1000 + 100 * (lon - 11) + 200 * (lat - 47) on tile N47E011.
    Row 0 is the north edge (lat 48), column 0 the west edge (lon 11)."""
    rows = np.arange(n)
    cols = np.arange(n)
    lat = 48 - rows / (n - 1)
    lon = 11 + cols / (n - 1)
    return (1000 + 100 * (lon[None, :] - 11) + 200 * (lat[:, None] - 47)).astype(np.float32)


def test_tile_name():
    assert tile_name(47.5, 11.2) == "N47E011"
    assert tile_name(48.0, 11.999) == "N48E011"


def test_read_hgt_big_endian_gz(tmp_path):
    arr = np.array([[1, 2, 3], [4, -32768, 6], [7, 8, 9]], dtype=">i2")
    p = tmp_path / "N47E011.hgt.gz"
    p.write_bytes(gzip.compress(arr.tobytes()))
    t = read_hgt(p)
    assert t.shape == (3, 3) and t[0, 2] == 3 and t[1, 1] == VOID


def test_bilinear_on_plane_is_exact():
    t = synthetic_tile(361)  # coarse tile: same layout, faster
    dem = DEM(lambda name: t if name == "N47E011" else None)
    lon = np.array([11.0, 11.5, 11.123456, 11.999, 11.0])
    lat = np.array([47.0, 47.5, 47.654321, 47.999, 48.0])
    z = dem.sample(lon, lat)
    expect = 1000 + 100 * (lon - 11) + 200 * (lat - 47)
    assert np.allclose(z, expect, atol=1e-3)


def test_orientation_north_row_first():
    t = np.zeros((3, 3), dtype=np.float32)
    t[0, :] = 900  # north edge
    t[2, :] = 100  # south edge
    t[1, :] = 500
    dem = DEM(lambda name: t)
    z = dem.sample(np.array([11.5, 11.5]), np.array([47.999, 47.001]))
    assert z[0] > 890 and z[1] < 110


def test_missing_tile_and_void_are_nan_and_filled():
    t = synthetic_tile(361).copy()
    t[100:110, 100:110] = VOID
    dem = DEM(lambda name: t if name == "N47E011" else None)
    lon = np.array([11.1, 11.0 + 105 / 360, 12.5])
    lat = np.array([47.1, 48 - 105 / 360, 47.5])
    z = dem.sample(lon, lat)
    assert not np.isnan(z[0]) and np.isnan(z[1]) and np.isnan(z[2])
    assert dem.missing_tile_samples == 1
    zf, n = fill_voids(np.array([1.0, np.nan, 3.0]), np.array([0.0, 1.0, 2.0]))
    assert n == 1 and zf[1] == pytest.approx(2.0)


def test_edge_of_tile_uses_neighbour():
    a = np.full((361, 361), 500.0, dtype=np.float32)
    b = np.full((361, 361), 700.0, dtype=np.float32)
    dem = DEM(lambda name: {"N47E011": a, "N47E012": b}.get(name))
    z = dem.sample(np.array([11.9999, 12.0, 12.0001]), np.array([47.5, 47.5, 47.5]))
    assert z[0] == pytest.approx(500) and z[1] == pytest.approx(700) and z[2] == pytest.approx(700)


def test_ascent_hysteresis_suppresses_noise():
    d = np.arange(0, 5000, 20.0)
    rng = np.random.default_rng(0)
    flat = 600 + rng.normal(0, 2.0, len(d))
    up_raw, _ = ascent_descent(flat, 0)
    up_h, down_h = ascent_descent(flat, 10)
    assert up_raw > 200  # raw noise inflates "ascent"
    # peak-to-peak noise is ~12 m, so at most a swing or two survives a 10 m band
    assert up_h < 0.1 * up_raw and down_h < 0.1 * up_raw
    assert ascent_descent(flat, 15) == (0.0, 0.0)


def test_ascent_real_climb_preserved():
    d = np.arange(0, 10000, 20.0)
    z = np.where(d < 5000, 500 + d * 0.1, 1000 - (d - 5000) * 0.1)  # +500 m then -500 m
    up, down = ascent_descent(z, 5)
    assert up == pytest.approx(500, abs=1e-6) and down == pytest.approx(1000 - z[-1], abs=1e-6)
    # wiggles below the band on top of the climb do not add ascent
    z2 = z + np.where((np.arange(len(z)) % 2) == 0, 2.0, -2.0)
    up2, _ = ascent_descent(z2, 5)
    assert up2 == pytest.approx(500, abs=4)  # true climb +- the 2 m wiggle at each end


def test_smoothing_then_hysteresis_on_noisy_climb():
    d = np.arange(0, 8000, 20.0)
    rng = np.random.default_rng(1)
    z = 700 + 0.05 * d + rng.normal(0, 3, len(d))  # true ascent 400 m
    zs = smooth(z, d, 100)
    up, _ = ascent_descent(zs, 5)
    assert up == pytest.approx(400, rel=0.08)
    assert smooth(z, d, 0) is not None and len(smooth(z, d, 100)) == len(z)
