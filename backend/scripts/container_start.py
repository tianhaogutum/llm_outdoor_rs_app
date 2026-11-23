"""
Container entry point (Dockerfile ``CMD``): make sure the writable database exists and is migrated, then
replace this process with uvicorn (one process, one worker: the demo guard's counters are per process).

The image carries a *pristine* database, seeded from the committed catalogue at build time
(``SEED_DB_DIR/app.db`` plus ``catalogue.sha256``). User data (profiles, feedback, chat, souvenirs) lives in the
writable database at ``DATABASE_URL`` (default ``/data/app.db``; mount a volume on ``/data`` to keep it):

1. No database yet: copy the pristine one (a file copy, ~0.1 s, instead of seeding 9.3k routes at every first
   start).
2. A database from an image with a *different catalogue* (its ``app.db.catalogue.sha256`` marker differs): the
   route ids may not match any more, so with ``RESEED_ON_CATALOGUE_CHANGE=true`` (default) the old file is kept
   as ``app.db.<old-sha>.bak`` and the pristine one is copied in; with ``false`` it is left alone and the API
   reports ``catalogue_mismatch`` and serves the baseline (docs/SERVING.md).
3. Always: ``alembic upgrade head`` (no-op on a fresh copy; migrates a database from an older image).

Nothing here needs credentials or the network.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
SEED_DB_DIR = Path(os.environ.get("SEED_DB_DIR", BACKEND / "data" / "seed-db"))
MARKER_SUFFIX = ".catalogue.sha256"


def log(msg: str) -> None:
    print(f"[container_start] {msg}", flush=True)


def db_path_from_env() -> Path:
    url = os.environ.get("DATABASE_URL", "")
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            return Path(url[len(prefix):])
    if not url:
        return Path("/data/app.db")
    raise SystemExit(f"DATABASE_URL must be a SQLite URL for this image (got scheme {url.split(':', 1)[0]!r})")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip() if p.is_file() else ""


def copy_pristine(db: Path, sha: str) -> None:
    tmp = db.with_name(db.name + ".tmp")
    shutil.copyfile(SEED_DB_DIR / "app.db", tmp)
    os.replace(tmp, db)
    Path(str(db) + MARKER_SUFFIX).write_text(sha + "\n", encoding="utf-8")


def ensure_db() -> None:
    t0 = time.perf_counter()
    db = db_path_from_env()
    db.parent.mkdir(parents=True, exist_ok=True)
    image_sha = read(SEED_DB_DIR / "catalogue.sha256")
    if not (SEED_DB_DIR / "app.db").is_file() or not image_sha:
        raise SystemExit(f"no pristine database in {SEED_DB_DIR}: the image build did not seed it")
    if not db.exists():
        copy_pristine(db, image_sha)
        log(f"new database from the image's seed ({db.stat().st_size / 1e6:.1f} MB)")
    else:
        have = read(Path(str(db) + MARKER_SUFFIX))
        if have != image_sha:
            if os.environ.get("RESEED_ON_CATALOGUE_CHANGE", "true").lower() in ("1", "true", "yes"):
                backup = db.with_name(f"{db.name}.{(have or 'unknown')[:12]}.bak")
                os.replace(db, backup)
                copy_pristine(db, image_sha)
                log(f"catalogue changed ({(have or 'unknown')[:12]} -> {image_sha[:12]}): old database kept as "
                    f"{backup.name}, fresh database copied in")
            else:
                log("catalogue changed but RESEED_ON_CATALOGUE_CHANGE=false: keeping the database; the API will "
                    "serve the baseline (catalogue_mismatch)")
        else:
            log("existing database, same catalogue")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db}"
    from alembic.config import main as alembic_main

    os.chdir(BACKEND)
    alembic_main(argv=["upgrade", "head"])
    log(f"database ready in {(time.perf_counter() - t0) * 1000:.0f} ms")


def main() -> None:
    ensure_db()
    port = os.environ.get("PORT", "8000")
    args = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", os.environ.get("HOST", "0.0.0.0"),
            "--port", port, "--workers", "1", "--no-server-header"]
    log(f"exec uvicorn on port {port}")
    os.chdir(BACKEND)
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
