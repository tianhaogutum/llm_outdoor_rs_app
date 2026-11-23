"""Write one real explanation of the Phase 3 default ``recsys_v1`` (``eval/results/_explanation_example.json``).

    cd backend && python -m evalkit.explain_example

The first test persona's request through ``recsys_v1`` (upstream calls come from the cache / frozen
layer), explained with :func:`app.recsys.explain.explain`; the top route is kept in full plus the
stage summaries. ``make report`` renders it; docs/RECSYS.md describes the schema.
"""

from __future__ import annotations

import json

from app.recsys import get_catalog
from app.recsys.explain import explain
from evalkit import paths
from evalkit.evaluate import dumps, make_system
from evalkit.personas import load_personas

SYSTEM = "recsys_v1"
OUT = "_explanation_example.json"


def build() -> dict:
    catalog = get_catalog()
    p = next(p for p in load_personas() if p.split == "test")
    rec = make_system(SYSTEM, catalog).recommend(p.to_query(), 10)
    ex = json.loads(explain(rec).model_dump_json(by_alias=True))
    for s in ex["stages"]:
        s["local_ms"] = None  # machine-dependent; keep the file deterministic
    top = ex["routes"][0]
    top["title"] = catalog.get(top["route_id"]).title
    return {
        "schema": "evalkit.explanation_example/v1",
        "system": SYSTEM,
        "persona_id": p.persona_id,
        "intent": p.intent,
        "profile_slots": {"category": p.latent["category"], **p.profile},
        "top_route": top,
        "stages": ex["stages"],
        "returned_route_ids": [r["route_id"] for r in ex["routes"]],
    }


def main() -> None:
    (paths.RESULTS_DIR / OUT).write_text(dumps(build()), encoding="utf-8")
    print(f"wrote {paths.RESULTS_DIR / OUT}")


if __name__ == "__main__":
    main()
