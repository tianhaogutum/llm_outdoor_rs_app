"""
Prove that the runtime dependency set (requirements-runtime.txt) covers what the served app imports.

Walks every ``.py`` file under ``app/`` plus the scripts the image runs (seeding) and collects **every**
import statement, including the lazy ones inside functions (``from google import genai`` in the Vertex
provider, ``from scripts.osm_catalog import ...`` in the chat's place index, ...). Each imported module is
then imported for real. Standard-library modules are skipped; first-party modules (``app``, ``scripts``,
``pipeline``) are imported too, because a lazy first-party import can drag in a third-party one (e.g.
``pipeline.dem`` would need numpy, ``pipeline.osm_extract`` osmium).

Exit status 1 if anything outside :data:`OPTIONAL` fails. Run inside the image (the Dockerfile does) or in
any environment that should be able to serve::

    python scripts/check_runtime_imports.py [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

ROOTS = [BACKEND / "app", BACKEND / "alembic"]
EXTRA_FILES = [BACKEND / "scripts" / n for n in ("seed_from_catalog.py", "seed_io.py", "osm_catalog.py",
                                                 "calculate_route_xp.py", "check_runtime_imports.py")]
FIRST_PARTY = {"app", "scripts", "pipeline"}
# Loaded by name, not by an import statement: the SQLAlchemy async driver (DATABASE_URL), the ASGI server and
# its event loop / HTTP parser, and SQLAlchemy's async bridge.
DYNAMIC = ("aiosqlite", "greenlet", "uvicorn", "uvloop", "httptools", "alembic.config")
# Imported only on a path the image does not serve by default; documented in docs/DEPLOY.md.
_RANK_API = "RANK_PROVIDER=vertex (Vertex Ranking API): install google-cloud-discoveryengine"
OPTIONAL = {"google.cloud": _RANK_API, "google.cloud.discoveryengine_v1": _RANK_API}


def imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pkg_parts = path.relative_to(BACKEND).with_suffix("").parts[:-1]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against the file's package
                base = list(pkg_parts[: len(pkg_parts) - node.level + 1])
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if not mod:
                continue
            out.add(mod)
            # ``from google import genai`` / ``from app.services import narratives``: the names may be modules
            for a in node.names:
                out.add(f"{mod}.{a.name}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    files = [p for root in ROOTS for p in sorted(root.rglob("*.py"))] + [p for p in EXTRA_FILES if p.exists()]
    wanted: set[str] = set()
    for f in files:
        wanted |= imports_in(f)
    wanted |= set(DYNAMIC)
    stdlib = set(sys.stdlib_module_names)
    failures, optional_missing, third_party = [], [], set()
    checked = 0
    for mod in sorted(wanted):
        top = mod.split(".")[0]
        if top in stdlib or top == "__future__":
            continue
        try:
            importlib.import_module(mod)
            checked += 1
            if top not in FIRST_PARTY:
                third_party.add(top)
        except ImportError as e:
            # ``from x import name`` where name is a class/function, not a module: fine if the parent imports
            parent = mod.rsplit(".", 1)[0] if "." in mod else None
            if parent and parent in wanted:
                try:
                    m = importlib.import_module(parent)
                    if hasattr(m, mod.rsplit(".", 1)[1]):
                        continue
                except ImportError:
                    pass
            if mod in OPTIONAL:
                optional_missing.append(mod)
                continue
            failures.append(f"{mod}: {type(e).__name__}: {e}")
    print(f"{len(files)} files, {checked} modules imported; third-party: {', '.join(sorted(third_party))}")
    for mod in optional_missing:
        print(f"optional, not installed: {mod} ({OPTIONAL[mod]})")
    for f in failures:
        print(f"MISSING {f}")
    if args.verbose:
        print("\n".join(sorted(wanted)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
