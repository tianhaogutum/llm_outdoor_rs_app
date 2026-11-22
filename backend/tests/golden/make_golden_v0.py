"""
Generate the v0_rules golden ranking file from the ORIGINAL scorer.

The scorer source is taken from git (default ``main:backend/app/services/recommendation_service.py``)
and executed as a throw-away module, so the golden always reflects the pre-refactor code no matter what
the working tree contains. It runs the real DB path (``get_recommended_routes``) against a copy of a
SQLite database with one profile per questionnaire combination and no feedback rows.

Usage (from ``backend/``)::

    .venv/bin/python tests/golden/make_golden_v0.py --db data/app.db \
        --out tests/golden/v0_rules_golden.json

This is a one-off tool; the regression test only reads the JSON it writes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

FITNESS_LEVELS = ["beginner", "intermediate", "advanced"]
ADVENTURE_TYPES = ["history-culture", "natural-scenery", "family-fun"]
CATEGORIES = [None, "running", "hiking", "cycling"]
SCORER_PATH = "backend/app/services/recommendation_service.py"


def golden_user_vector(fitness: str, adventure_type: str) -> dict:
    """User vector exactly as the questionnaire builds it, minus the narrative fields (unused by scoring)."""
    from app.services.user_profile_service import ADVENTURE_TYPE_TO_TAGS, FITNESS_MAPPING

    cfg = FITNESS_MAPPING[fitness]
    return {
        "difficulty_range": list(cfg["difficulty_range"]),
        "max_distance_km": cfg["max_distance_km"],
        "min_distance_km": cfg["min_distance_km"],
        "preferred_tags": list(dict.fromkeys(ADVENTURE_TYPE_TO_TAGS[adventure_type])),
        "fitness_level": fitness,
    }


RANKING_COLUMNS = ["route_id", "score", "difficulty", "distance", "tags", "breakdown_sha256"]


def breakdown_digest(breakdown: dict) -> str:
    """Stable digest of the full per-route score breakdown (canonical JSON)."""
    canonical = json.dumps(breakdown, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ranking_row(route_id: int, score: float, breakdown: dict) -> list:
    return [
        route_id,
        score,
        breakdown["difficulty"]["score"],
        breakdown["distance"]["score"],
        breakdown["tags"]["score"],
        breakdown_digest(breakdown),
    ]


def dump_golden(golden: dict) -> str:
    """JSON with one ranking row per line so diffs stay readable."""
    head = {k: v for k, v in golden.items() if k != "cases"}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},")
    lines.append('  "cases": [')
    for i, case in enumerate(golden["cases"]):
        meta = {k: v for k, v in case.items() if k != "ranking"}
        lines.append("    {")
        for key, value in meta.items():
            lines.append(f"      {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},")
        lines.append('      "ranking": [')
        rows = [f"        {json.dumps(row)}" for row in case["ranking"]]
        lines.append(",\n".join(rows))
        lines.append("      ]")
        lines.append("    }" + ("," if i < len(golden["cases"]) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def load_scorer_from_git(rev: str):
    source = subprocess.run(
        ["git", "show", f"{rev}:{SCORER_PATH}"],
        cwd=BACKEND_DIR.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", rev], cwd=BACKEND_DIR.parent, check=True, capture_output=True, text=True
    ).stdout.strip()
    tmp = Path(tempfile.mkdtemp()) / "original_recommendation_service.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("original_recommendation_service", tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, commit


async def build(db_path: Path, rev: str) -> dict:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.models.entities import DemoProfile

    scorer, commit = load_scorer_from_git(rev)

    work = Path(tempfile.mkdtemp()) / "golden.db"
    shutil.copy(db_path, work)
    engine = create_async_engine(f"sqlite+aiosqlite:///{work}")
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    cases = []
    async with session_maker() as db:
        for fitness in FITNESS_LEVELS:
            for adventure in ADVENTURE_TYPES:
                vector = golden_user_vector(fitness, adventure)
                profile = DemoProfile(user_vector_json=json.dumps(vector), total_xp=0, level=1)
                db.add(profile)
                await db.commit()
                for category in CATEGORIES:
                    routes = await scorer.get_recommended_routes(
                        db=db, profile_id=profile.id, category=category, limit=10_000
                    )
                    ranking = [
                        ranking_row(r.id, r.recommendation_score, r.recommendation_score_breakdown) for r in routes
                    ]
                    cases.append(
                        {
                            "fitness": fitness,
                            "adventure_type": adventure,
                            "category": category,
                            "user_vector": vector,
                            "ranking": ranking,
                        }
                    )
                    db.expunge_all()
    await engine.dispose()
    return {
        "description": "v0_rules full ranking from the original scorer, no feedback rows",
        "scorer_source": f"{rev}:{SCORER_PATH}",
        "scorer_commit": commit,
        "ranking_columns": RANKING_COLUMNS,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=BACKEND_DIR / "data" / "app.db")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("v0_rules_golden.json"))
    parser.add_argument("--rev", default="main", help="git revision holding the original scorer")
    args = parser.parse_args()
    golden = asyncio.run(build(args.db, args.rev))
    args.out.write_text(dump_golden(golden), encoding="utf-8")
    json.loads(args.out.read_text(encoding="utf-8"))  # sanity: output parses
    n = sum(len(c["ranking"]) for c in golden["cases"])
    print(f"wrote {args.out} ({len(golden['cases'])} cases, {n} ranked rows)")


if __name__ == "__main__":
    main()
