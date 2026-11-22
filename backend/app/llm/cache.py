"""On-disk cache for LLM / embedding / rank calls.

Two layers, looked up in order:

1. **Local writable cache**: one SQLite file (``<cache_dir>/cache.sqlite3``,
   WAL mode) — safe for concurrent async use within a process (calls are
   short and serialised by a lock) and for a few parallel processes (SQLite
   file locking with a generous busy timeout).
2. **Frozen layers**: read-only, git-friendly directories produced by
   :func:`export_frozen`. Each export writes

   - ``entries-<digest>.jsonl.gz``: one JSON object per line,
     ``{"key", "capability", "payload"}``, sorted by key, gzip with mtime 0
     (byte-identical for identical content);
   - ``vectors-<digest>.npz``: for embeddings, arrays ``d<dim>`` (float32,
     n x dim) and ``d<dim>__keys`` (the cache keys of the rows), written with
     fixed zip timestamps;
   - ``manifest-<digest>.json``: format version and counts (for humans).

   Several exports may live in one directory; all are read.

Cache keys: see :func:`make_key`. They are a SHA-256 over a canonical JSON
document of everything that changes the output and nothing else.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sqlite3
import threading
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

KEY_VERSION = 1
FROZEN_FORMAT = "app.llm.frozen-cache"
FROZEN_FORMAT_VERSION = 1


# ------------------------------------------------------------------- keys
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def make_key(fields: dict[str, Any]) -> str:
    """SHA-256 hex digest of ``{"v": KEY_VERSION, **fields}`` as canonical JSON."""
    doc = canonical_json({"v": KEY_VERSION, **fields})
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


def generate_key(
    *,
    provider: str,
    model: str,
    prompt: str,
    system: Optional[str],
    examples: Sequence[tuple[str, str]],
    temperature: float,
    max_tokens: int,
    json_schema: Optional[dict],
    thinking: Any,
) -> str:
    return make_key(
        {
            "cap": "generate",
            "provider": provider,
            "model": model,
            "system": system,
            "prompt": prompt,
            "examples": [[u, a] for u, a in examples],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "schema": json_schema,
            "thinking": thinking,
        }
    )


def embed_key(*, provider: str, model: str, text: str, task: str, dim: int) -> str:
    return make_key({"cap": "embed", "provider": provider, "model": model, "task": task, "dim": int(dim), "text": text})


def rank_key(*, provider: str, model: str, query: str, documents: Sequence[str]) -> str:
    return make_key({"cap": "rank", "provider": provider, "model": model, "query": query, "documents": list(documents)})


# ---------------------------------------------------------------- entries
@dataclass
class Entry:
    key: str
    capability: str
    payload: dict[str, Any]
    vector: Optional[np.ndarray] = None  # float32, embeddings only
    source: str = "local"  # "local" | "frozen:<dir name>"


class LocalCache:
    """Writable SQLite-backed layer."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "cache.sqlite3"
        self._lock = threading.Lock()
        self._conn_obj: Optional[sqlite3.Connection] = None

    @property
    def _conn(self) -> sqlite3.Connection:
        # opened lazily: constructing a client never touches the disk
        if self._conn_obj is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False, isolation_level=None)
            conn.execute("PRAGMA busy_timeout=60000")
            # Switching a fresh file to WAL takes an exclusive lock that SQLite does not wait for, so several
            # processes opening a new cache at once can get "database is locked" despite the busy timeout.
            for attempt in range(50):
                try:
                    if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                        conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS entries ("
                        " key TEXT PRIMARY KEY, capability TEXT NOT NULL, payload TEXT NOT NULL,"
                        " vector BLOB, dim INTEGER, created_at REAL NOT NULL)"
                    )
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 49:
                        raise
                    time.sleep(0.02 * (attempt + 1))
            self._conn_obj = conn
        return self._conn_obj

    def get_many(self, keys: Sequence[str]) -> dict[str, Entry]:
        out: dict[str, Entry] = {}
        if not keys:
            return out
        uniq = list(dict.fromkeys(keys))
        with self._lock:
            for i in range(0, len(uniq), 500):
                chunk = uniq[i : i + 500]
                marks = ",".join("?" * len(chunk))
                q = f"SELECT key, capability, payload, vector FROM entries WHERE key IN ({marks})"
                for key, cap, payload, blob in self._conn.execute(q, chunk):
                    vec = np.frombuffer(blob, dtype=np.float32).copy() if blob is not None else None
                    out[key] = Entry(key, cap, json.loads(payload), vec, "local")
        return out

    def put_many(self, entries: Iterable[Entry]) -> None:
        rows = []
        now = time.time()
        for e in entries:
            blob = None
            dim = None
            if e.vector is not None:
                v = np.asarray(e.vector, dtype=np.float32)
                blob = v.tobytes()
                dim = int(v.shape[0])
            rows.append((e.key, e.capability, canonical_json(e.payload), blob, dim, now))
        if not rows:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO entries (key, capability, payload, vector, dim, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._conn_obj is not None:
                self._conn_obj.close()
                self._conn_obj = None


class FrozenLayer:
    """Read-only layer loaded (lazily, once) from an export directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._entries: Optional[dict[str, Entry]] = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Entry]:
        with self._lock:
            if self._entries is not None:
                return self._entries
            entries: dict[str, Entry] = {}
            src = f"frozen:{self.directory.name}"
            if self.directory.is_dir():
                for f in sorted(self.directory.glob("entries-*.jsonl.gz")):
                    with gzip.open(f, "rt", encoding="utf-8") as fh:
                        for line in fh:
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            entries[obj["key"]] = Entry(obj["key"], obj["capability"], obj["payload"], None, src)
                for f in sorted(self.directory.glob("vectors-*.npz")):
                    with np.load(f, allow_pickle=False) as z:
                        for name in z.files:
                            if not name.endswith("__keys"):
                                continue
                            mat = np.asarray(z[name[: -len("__keys")]], dtype=np.float32)
                            for i, key in enumerate(z[name].tolist()):
                                if key in entries:
                                    entries[key].vector = mat[i].copy()
            self._entries = entries
            return entries

    def get_many(self, keys: Sequence[str]) -> dict[str, Entry]:
        entries = self._load()
        out = {}
        for k in keys:
            e = entries.get(k)
            if e is not None and (e.capability != "embed" or e.vector is not None):
                out[k] = e
        return out

    def __len__(self) -> int:
        return len(self._load())


class TieredCache:
    """Local writable layer (optional) followed by frozen layers."""

    def __init__(self, local_dir: Optional[Path], frozen_dirs: Sequence[Path] = ()) -> None:
        self.local = LocalCache(local_dir) if local_dir is not None else None
        self.frozen = [FrozenLayer(Path(d)) for d in frozen_dirs]

    def get_many(self, keys: Sequence[str]) -> dict[str, Entry]:
        found: dict[str, Entry] = {}
        missing = list(dict.fromkeys(keys))
        if self.local is not None and missing:
            found.update(self.local.get_many(missing))
            missing = [k for k in missing if k not in found]
        for layer in self.frozen:
            if not missing:
                break
            found.update(layer.get_many(missing))
            missing = [k for k in missing if k not in found]
        return found

    def get(self, key: str) -> Optional[Entry]:
        return self.get_many([key]).get(key)

    def put_many(self, entries: Iterable[Entry]) -> None:
        if self.local is not None:
            self.local.put_many(entries)

    def close(self) -> None:
        if self.local is not None:
            self.local.close()


# ----------------------------------------------------------------- export
def _write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zf.writestr(info, buf.getvalue())


def export_frozen(entries: Iterable[Entry], dest_dir: Path) -> dict[str, Any]:
    """Write ``entries`` as one frozen shard into ``dest_dir``; returns the manifest."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    items = sorted({e.key: e for e in entries}.values(), key=lambda e: e.key)
    digest = hashlib.sha256("\n".join(e.key for e in items).encode()).hexdigest()[:12]

    lines = []
    by_dim: dict[int, list[Entry]] = {}
    caps: dict[str, int] = {}
    for e in items:
        lines.append(canonical_json({"key": e.key, "capability": e.capability, "payload": e.payload}))
        caps[e.capability] = caps.get(e.capability, 0) + 1
        if e.vector is not None:
            by_dim.setdefault(int(e.vector.shape[0]), []).append(e)

    entries_path = dest_dir / f"entries-{digest}.jsonl.gz"
    with open(entries_path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(("\n".join(lines) + "\n").encode("utf-8") if lines else b"")

    files = [entries_path.name]
    if by_dim:
        arrays: dict[str, np.ndarray] = {}
        for dim, es in by_dim.items():
            arrays[f"d{dim}"] = np.stack([np.asarray(e.vector, dtype=np.float32) for e in es])
            arrays[f"d{dim}__keys"] = np.array([e.key for e in es])
        vec_path = dest_dir / f"vectors-{digest}.npz"
        _write_npz_deterministic(vec_path, arrays)
        files.append(vec_path.name)

    manifest = {
        "format": FROZEN_FORMAT,
        "format_version": FROZEN_FORMAT_VERSION,
        "key_version": KEY_VERSION,
        "entries": len(items),
        "by_capability": dict(sorted(caps.items())),
        "vector_dims": sorted(by_dim),
        "files": files,
    }
    (dest_dir / f"manifest-{digest}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
