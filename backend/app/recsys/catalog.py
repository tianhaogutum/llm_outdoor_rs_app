"""In-memory route catalogue built from ``backend/data/catalog/routes.jsonl.gz`` (docs/DATA.md).

Routes become frozen :class:`RouteRecord` objects (the fields systems and cards need, plus the raw
``features`` dict) and the numeric fields are also held as aligned numpy arrays, so filters and
scorers can work vectorised. Build once per process with :func:`get_catalog`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

DEFAULT_CATALOG_DIR = Path(__file__).resolve().parents[2] / "data" / "catalog"
ROUTES_FILE = "routes.jsonl.gz"


@dataclass(frozen=True)
class RouteRecord:
    id: int
    title: str
    category_name: str
    activity: str
    tags: tuple[str, ...]
    difficulty: int
    difficulty_technical: Optional[int]
    max_grade: Optional[str]
    length_m: int
    ascent_m: int
    descent_m: int
    max_ele_m: int
    duration_min: int
    est_days: int
    start_lat: float
    start_lon: float
    is_loop: bool
    locality: Optional[str]
    end_locality: Optional[str]
    admin_district: Optional[str]
    admin_region: Optional[str]
    description: str
    description_source: str
    max_grade_200m_pct: float = 0.0
    stages: tuple[int, ...] = ()
    stage_of: tuple[int, ...] = ()
    features: dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    osm_description: Optional[str] = field(default=None, compare=False, hash=False, repr=False)

    @property
    def length_km(self) -> float:
        return self.length_m / 1000.0

    @classmethod
    def from_json(cls, r: dict[str, Any]) -> RouteRecord:
        tags = r.get("osm_tags") or {}
        osm_desc = tags.get("description:en") or tags.get("description") or tags.get("description:de")
        return cls(
            id=int(r["id"]),
            title=r["title"],
            category_name=r["category_name"],
            activity=r["activity"],
            tags=tuple(r.get("tags") or ()),
            difficulty=int(r["difficulty"]),
            difficulty_technical=r.get("difficulty_technical"),
            max_grade=r.get("max_grade"),
            length_m=int(r["length_m"]),
            ascent_m=int(r["ascent_m"]),
            descent_m=int(r["descent_m"]),
            max_ele_m=int(r["max_ele_m"]),
            duration_min=int(r["duration_min"]),
            est_days=int(r["est_days"]),
            start_lat=float(r["start_lat"]),
            start_lon=float(r["start_lon"]),
            is_loop=bool(r["is_loop"]),
            locality=r.get("locality"),
            end_locality=r.get("end_locality"),
            admin_district=r.get("admin_district"),
            admin_region=r.get("admin_region"),
            description=r.get("description") or "",
            description_source=r.get("description_source") or "generated",
            max_grade_200m_pct=float(r.get("max_grade_200m_pct") or 0.0),
            stages=tuple(r.get("stages") or ()),
            stage_of=tuple(r.get("stage_of") or ()),
            features=r.get("features") or {},
            osm_description=osm_desc,
        )

    def v0_vector(self) -> dict:
        """Route vector in the shape ``rank_routes_v0`` expects (tags lower-cased, as the DB path does)."""
        return {"difficulty": self.difficulty, "length_km": self.length_km, "tags": [t.lower() for t in self.tags]}


class Catalog:
    """Routes in id order plus aligned numpy arrays (``ids``, ``start_lat``, ``length_km``, ...)."""

    def __init__(self, routes: Sequence[RouteRecord], *, source: str = "<memory>", sha256: str = "") -> None:
        routes = sorted(routes, key=lambda r: r.id)
        self.routes: tuple[RouteRecord, ...] = tuple(routes)
        self.source = source
        self.sha256 = sha256
        self._index = {r.id: i for i, r in enumerate(self.routes)}
        if len(self._index) != len(self.routes):
            raise ValueError("duplicate route ids in catalogue")
        self.ids = np.array([r.id for r in self.routes], dtype=np.int64)
        self.start_lat = np.array([r.start_lat for r in self.routes], dtype=np.float64)
        self.start_lon = np.array([r.start_lon for r in self.routes], dtype=np.float64)
        self.length_km = np.array([r.length_km for r in self.routes], dtype=np.float64)
        self.ascent_m = np.array([r.ascent_m for r in self.routes], dtype=np.float64)
        self.descent_m = np.array([r.descent_m for r in self.routes], dtype=np.float64)
        self.duration_min = np.array([r.duration_min for r in self.routes], dtype=np.float64)
        self.difficulty = np.array([r.difficulty for r in self.routes], dtype=np.int64)
        self.is_loop = np.array([r.is_loop for r in self.routes], dtype=bool)
        self.category_name = np.array([r.category_name for r in self.routes], dtype=object)
        self.activity = np.array([r.activity for r in self.routes], dtype=object)
        self._lock = threading.RLock()
        self._memo: dict[str, Any] = {}

    # -------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.routes)

    def __iter__(self) -> Iterator[RouteRecord]:
        return iter(self.routes)

    def __contains__(self, route_id: object) -> bool:
        return route_id in self._index

    def index_of(self, route_id: int) -> int:
        return self._index[int(route_id)]

    def get(self, route_id: int) -> RouteRecord:
        return self.routes[self._index[int(route_id)]]

    def take(self, route_ids: Iterable[int]) -> list[RouteRecord]:
        return [self.get(i) for i in route_ids]

    def memo(self, key: str, build):
        """Per-catalogue memo for derived structures (e.g. per-tab v0 vectors); thread-safe."""
        with self._lock:
            if key not in self._memo:
                self._memo[key] = build()
            return self._memo[key]

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, catalog_dir: Path | str | None = None) -> Catalog:
        path = Path(catalog_dir or DEFAULT_CATALOG_DIR) / ROUTES_FILE
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        routes = [RouteRecord.from_json(json.loads(line)) for line in gzip.decompress(raw).splitlines() if line]
        return cls(routes, source=str(path), sha256=sha)

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> Catalog:
        """Build from plain dicts shaped like catalogue lines (tests use small synthetic catalogues)."""
        recs = list(records)
        digest = hashlib.sha256(json.dumps(recs, sort_keys=True, default=str).encode()).hexdigest()
        return cls([RouteRecord.from_json(r) for r in recs], source="<records>", sha256=digest)


_catalogs: dict[str, Catalog] = {}
_cat_lock = threading.Lock()


def get_catalog(catalog_dir: Path | str | None = None) -> Catalog:
    """Process-wide catalogue (loaded on first use, about 0.5 s for the 9.3k Bavaria routes)."""
    key = str(Path(catalog_dir or DEFAULT_CATALOG_DIR).resolve())
    with _cat_lock:
        if key not in _catalogs:
            _catalogs[key] = Catalog.load(key)
        return _catalogs[key]
