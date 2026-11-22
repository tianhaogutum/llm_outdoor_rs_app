"""
Fixtures for the app against the OSM catalogue (offline, LLM_PROVIDER=mock via tests/conftest.py).

- ``catalog_subset_dir``: a small catalogue (routes + geometry gz files) cut from the committed one:
  the first routes of every category plus a few well-known routes, so tests stay fast.
- ``osm_db``: a migrated DB seeded from the subset (session-wide, treat as read-only or use a copy).
- ``full_osm_db``: a DB seeded from the full committed catalogue (~2 s, session-wide).
- ``make_client``: a TestClient on a *fresh* app whose ``get_db`` dependency points at a given DB file.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.conftest import BACKEND_DIR, run_alembic

CATALOG_DIR = BACKEND_DIR / "data" / "catalog"
PER_CATEGORY = 25
KNOWN_ROUTES = (1356555, 26937, 551290, 3190325, 223215, 3201006)  # spot checks quoted in docs/DATA.md


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture(scope="session")
def catalog_records() -> list[dict]:
    return read_jsonl_gz(CATALOG_DIR / "routes.jsonl.gz")


@pytest.fixture(scope="session")
def catalog_subset_dir(tmp_path_factory, catalog_records) -> Path:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(catalog_records, key=lambda r: r["id"]):
        by_category[r["category_name"]].append(r)
    chosen = {r["id"]: r for rs in by_category.values() for r in rs[:PER_CATEGORY]}
    chosen.update({r["id"]: r for r in catalog_records if r["id"] in KNOWN_ROUTES})
    records = sorted(chosen.values(), key=lambda r: r["id"])
    geometry = [g for g in read_jsonl_gz(CATALOG_DIR / "geometry.jsonl.gz") if g["id"] in chosen]
    out = tmp_path_factory.mktemp("catalog_subset")
    write_jsonl_gz(out / "routes.jsonl.gz", records)
    write_jsonl_gz(out / "geometry.jsonl.gz", geometry)
    return out


def build_osm_db(db_path: Path, catalog_dir: Path) -> Path:
    from scripts.seed_from_catalog import load_catalog

    db_path.unlink(missing_ok=True)
    run_alembic(db_path, "upgrade", "head")
    load_catalog(db_path, catalog_dir)
    return db_path


@pytest.fixture(scope="session")
def osm_db(tmp_path_factory, catalog_subset_dir) -> Path:
    return build_osm_db(tmp_path_factory.mktemp("osm") / "osm_subset.db", catalog_subset_dir)


@pytest.fixture(scope="session")
def full_osm_db(tmp_path_factory) -> Path:
    return build_osm_db(tmp_path_factory.mktemp("osm_full") / "osm_full.db", CATALOG_DIR)


@pytest.fixture
def osm_db_copy(osm_db: Path, tmp_path: Path) -> Path:
    target = tmp_path / "osm_copy.db"
    shutil.copy(osm_db, target)
    return target


@pytest.fixture
def make_client(monkeypatch) -> Iterator[Callable]:
    """``make_client(db_path)`` -> TestClient on a fresh app bound to ``db_path`` (achievements seeded)."""
    import asyncio

    from fastapi.testclient import TestClient
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.api.v1 import profiles
    from app.database import get_db
    from app.main import create_app
    from app.services.achievement_service import seed_achievements

    async def no_llm_welcome(questionnaire):  # offline whatever genai_service does
        return "Welcome (test stub)"

    monkeypatch.setattr(profiles, "generate_welcome_summary", no_llm_welcome)
    clients = []

    def factory(db_path: Path) -> TestClient:
        url = f"sqlite+aiosqlite:///{db_path}"

        async def seed() -> None:  # own engine: connections never cross event loops (NullPool)
            seed_engine = create_async_engine(url, poolclass=NullPool)
            async with async_sessionmaker(seed_engine, class_=AsyncSession)() as session:
                await seed_achievements(session)
            await seed_engine.dispose()

        asyncio.run(seed())
        engine = create_async_engine(url, poolclass=NullPool)
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

        async def override_get_db():
            async with maker() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app = create_app()
        app.dependency_overrides[get_db] = override_get_db
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
