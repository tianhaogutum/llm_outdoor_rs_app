"""SRTM-style HGT tiles: loading, bilinear sampling, ascent with hysteresis.

HGT layout: a 1x1 degree tile named after its south-west corner
(``N47E011`` covers lat 47..48, lon 11..12). 1 arc-second tiles have
3601 x 3601 big-endian int16 samples; row 0 is the *north* edge (lat 48),
column 0 the *west* edge (lon 11). Edge rows/columns duplicate the
neighbouring tile, so a point on a tile edge can be sampled from either tile.
Value -32768 marks a void.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np

VOID = -32768


def tile_name(lat: float, lon: float) -> str:
    la = math.floor(lat)
    lo = math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"


def read_hgt(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    n = int(round(math.sqrt(len(raw) // 2)))
    if n * n * 2 != len(raw):
        raise ValueError(f"{path}: not a square HGT tile ({len(raw)} bytes)")
    return np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float32)


class DEM:
    """Lazy collection of HGT tiles with bilinear interpolation."""

    def __init__(self, tile_loader):
        """``tile_loader(name) -> np.ndarray | None`` (None = tile missing)."""
        self._loader = tile_loader
        self._tiles: dict[str, np.ndarray | None] = {}
        self.void_samples = 0
        self.missing_tile_samples = 0

    def _tile(self, name: str):
        if name not in self._tiles:
            arr = self._loader(name)
            if arr is not None:
                arr = arr.copy()
                arr[arr == VOID] = np.nan
            self._tiles[name] = arr
        return self._tiles[name]

    def _sample_tile(self, t, la, lo, lon, lat):
        n = t.shape[0] - 1  # 3600 intervals per degree
        r = (la + 1 - lat) * n
        c = (lon - lo) * n
        r0 = np.clip(np.floor(r).astype(int), 0, n - 1)
        c0 = np.clip(np.floor(c).astype(int), 0, n - 1)
        fr = r - r0
        fc = c - c0
        v00 = t[r0, c0]
        v01 = t[r0, c0 + 1]
        v10 = t[r0 + 1, c0]
        v11 = t[r0 + 1, c0 + 1]
        return v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc + v10 * fr * (1 - fc) + v11 * fr * fc

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Bilinear interpolation; NaN where void or tile missing.

        A point exactly on a north/east tile edge belongs (by floor) to the
        next tile; if that tile is missing, the shared edge row/column of the
        tile below/left is used instead."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        out = np.full(lon.shape, np.nan)
        la0 = np.floor(lat).astype(int)
        lo0 = np.floor(lon).astype(int)
        pairs = np.column_stack([la0, lo0])
        for la, lo in np.unique(pairs, axis=0).tolist():
            m = (la0 == la) & (lo0 == lo)
            t = self._tile(tile_name(la + 0.5, lo + 0.5))
            if t is not None:
                out[m] = self._sample_tile(t, la, lo, lon[m], lat[m])
                continue
            # fallback for points on the shared edge
            idx = np.nonzero(m)[0]
            on_s = lat[idx] == la
            on_w = lon[idx] == lo
            done = np.zeros(len(idx), dtype=bool)
            for dla, dlo, sel in ((-1, 0, on_s), (0, -1, on_w), (-1, -1, on_s & on_w)):
                sel = sel & ~done
                if not sel.any():
                    continue
                t2 = self._tile(tile_name(la + dla + 0.5, lo + dlo + 0.5))
                if t2 is None:
                    continue
                ii = idx[sel]
                out[ii] = self._sample_tile(t2, la + dla, lo + dlo, lon[ii], lat[ii])
                done |= sel
            self.missing_tile_samples += int((~done).sum())
        nan = np.isnan(out)
        self.void_samples += int(nan.sum())
        return out


def fill_voids(z: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, int]:
    """Linear interpolation over NaNs along the profile. Returns (z, n_filled)."""
    z = np.asarray(z, dtype=float).copy()
    bad = np.isnan(z)
    nb = int(bad.sum())
    if nb == 0:
        return z, 0
    if nb == len(z):
        return z, nb
    z[bad] = np.interp(d[bad], d[~bad], z[~bad])
    return z, nb


def smooth(z: np.ndarray, d: np.ndarray, window_m: float) -> np.ndarray:
    """Centred moving average over a window of ``window_m`` metres
    (samples assumed evenly spaced except possibly the last)."""
    if window_m <= 0 or len(z) < 3:
        return np.asarray(z, dtype=float)
    spacing = float(np.median(np.diff(d))) if len(d) > 1 else window_m
    k = max(1, int(round(window_m / spacing)))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return np.asarray(z, dtype=float)
    pad = k // 2
    zp = np.pad(np.asarray(z, dtype=float), pad, mode="edge")
    ker = np.ones(k) / k
    return np.convolve(zp, ker, mode="valid")


def ascent_descent(z: np.ndarray, hysteresis_m: float) -> tuple[float, float]:
    """Cumulative ascent/descent with a hysteresis (dead-band) threshold.

    Classic extreme-to-extreme counting: a peak or trough is only confirmed
    once the profile has moved ``hysteresis_m`` back from it. Real climbs are
    counted in full (trough to peak); wiggles smaller than the threshold are
    ignored. With a threshold of 0 this is the plain sum of positive and
    negative differences.
    """
    z = np.asarray(z, dtype=float)
    z = z[~np.isnan(z)]
    if len(z) < 2:
        return 0.0, 0.0
    if hysteresis_m <= 0:
        dz = np.diff(z)
        return float(dz[dz > 0].sum()), float(-dz[dz < 0].sum())
    h = hysteresis_m
    up = down = 0.0
    trend = 0
    lo = hi = z[0]
    last = cand = z[0]
    for v in z[1:]:
        if trend == 0:
            lo = min(lo, v)
            hi = max(hi, v)
            if v - lo >= h:
                trend, last, cand = 1, lo, v
            elif hi - v >= h:
                trend, last, cand = -1, hi, v
        elif trend == 1:
            if v > cand:
                cand = v
            elif cand - v >= h:
                up += cand - last
                trend, last, cand = -1, cand, v
        else:
            if v < cand:
                cand = v
            elif v - cand >= h:
                down += last - cand
                trend, last, cand = 1, cand, v
    if trend == 1:
        up += cand - last
    elif trend == -1:
        down += last - cand
    return float(up), float(down)
