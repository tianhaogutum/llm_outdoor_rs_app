"""Cached downloads of the pinned inputs, with checksums."""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from . import config

USER_AGENT = "outdoor-rs-data-pipeline/1.0 (+https://www.openstreetmap.org/copyright)"


def file_hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, log=print) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)


def osm_extract(name: str = config.OSM_EXTRACT, log=print) -> Path:
    dest = config.OSM_DIR / f"{name}.osm.pbf"
    if not dest.exists():
        url = config.OSM_URL_TEMPLATE.format(name=name)
        try:
            _fetch(url, dest, log)
        except urllib.error.HTTPError as e:
            raise SystemExit(
                f"Could not download {url} ({e.code}). Geofabrik removes dated daily/monthly "
                f"snapshots after some time; only the yearly YY0101 snapshots are kept. "
                f"The committed catalogue + manifest.json remain the artefact of record. "
                f"To rebuild from a newer extract set PIPELINE_OSM_EXTRACT (e.g. bayern-270101) "
                f"and expect different counts."
            ) from e
    expected = config.OSM_EXPECTED_MD5.get(name)
    if expected:
        got = file_hash(dest, "md5")
        if got != expected:
            raise SystemExit(f"{dest}: md5 {got} != expected {expected}; delete it and retry")
    return dest


def dem_tile(name: str, log=print) -> Path | None:
    """Download one skadi tile (e.g. 'N47E011'). Returns None if it does not exist
    (ocean tiles are absent from the bucket)."""
    dest = config.DEM_DIR / f"{name}.hgt.gz"
    if dest.exists():
        return dest
    url = config.DEM_URL_TEMPLATE.format(lat_dir=name[:3], name=name)
    try:
        _fetch(url, dest, log)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    return dest
