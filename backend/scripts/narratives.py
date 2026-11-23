"""
Pre-generate the LLM-backed narratives for the demo (docs/DEPLOY.md, "Pre-generated narratives").

Subcommands (run from ``backend/``)::

    python scripts/narratives.py select              # demo route set -> data/narratives/demo_routes.json
    python scripts/narratives.py generate            # replay from the committed cache (no credentials)
    python scripts/narratives.py generate --live     # call the provider for anything not cached (costs money)
    python scripts/narratives.py check               # store vs catalogue, provenance, grounding checks

``generate`` writes ``welcome.json`` and ``routes.json`` and exports every LLM call it used to
``data/narratives/cache/`` (a frozen cache layer, docs/LLM_CLIENT.md section 4). Without ``--live`` it runs
cache-only against that layer, so the output is reproduced byte for byte without credentials or network;
a cache miss is an error. With ``--live`` the configured provider (``LLM_PROVIDER``, ``backend/.env``) answers
the misses, under ``--budget`` USD.

**Scope.** Welcome texts: all 72 questionnaire combinations. Route texts: the *demo route set*, i.e. the
routes the served pipeline puts in a top 10 for a grid of demo profiles (24 questionnaire vectors x 3 tabs x
{no home, 6 towns at 50 km}, without an intent through the configured system's rule path; plus demo intents
through the no-LLM path, which is what a container without credentials serves). Ranked by how often they
appear, capped at ``--max-routes``. Deterministic: no randomness, no provider call.

**Grounding.** Route texts may only use the facts block built from the catalogue record
(``app.services.narratives.route_facts``). Every text goes through ``check_route_text`` (numbers must occur in
the facts, capitalised words must occur in the facts, no quests/XP/levels); a text that fails is not stored,
and that route falls back to the template at request time.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import narratives as nv  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
CATALOG_DIR = BACKEND_DIR / "data" / "catalog"
OUT_DIR = nv.DEFAULT_NARRATIVES_DIR

# Generation parameters (part of the cache key; changing one regenerates everything).
WELCOME_PARAMS = {"temperature": 0.7, "max_tokens": 400, "thinking": 0}
ROUTE_PARAMS = {"temperature": 0.4, "max_tokens": 400, "thinking": 0}
CONCURRENCY = 6

# Demo grid (deterministic). Towns: the frontend's examples (Garmisch, Tegernsee) and the largest towns of
# the catalogue's region; 50 km is one of the form's radius options.
DEMO_HOMES = ("München", "Garmisch-Partenkirchen", "Tegernsee", "Rosenheim", "Nürnberg", "Regensburg")
DEMO_RADIUS_KM = 50.0
DEMO_INTENTS = {
    "hiking": (
        "easy loop near Tegernsee with a hut, max 12 km",
        "a quiet lake loop with a hut stop, fine for a seven-year-old",
        "summit hike with a great view",
        "castle ruins and a beer garden",
    ),
    "cycling": ("family bike ride along a river", "long cycle tour with beer gardens"),
    "running": ("flat run along the river", "short forest run"),
}
TOP_K = 10


# ------------------------------------------------------------------------------------------- helpers
def load_catalog_records() -> dict[int, dict[str, Any]]:
    with gzip.open(CATALOG_DIR / "routes.jsonl.gz", "rt", encoding="utf-8") as fh:
        return {int(r["id"]): r for r in (json.loads(line) for line in fh if line.strip())}


def catalogue_sha256() -> str:
    return hashlib.sha256((CATALOG_DIR / "routes.jsonl.gz").read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# -------------------------------------------------------------------------------------------- select
def cmd_select(args: argparse.Namespace) -> int:
    from app.api.schemas import ProfileCreate
    from app.chat.places import IndexPlaceResolver
    from app.recsys import UserQuery
    from app.services import recsys_service
    from app.services.user_profile_service import translate_questionnaire_to_vector
    from app.settings import Settings

    # Serving configuration without any credentials: the rule path and the no-LLM path make no provider call.
    s = Settings(_env_file=None, llm_provider="vertex", google_cloud_project=None, openrouter_api_key=None,
                 llm_cache_only=True, llm_cache_enabled=False)
    res = recsys_service.load_resources(s)
    if not res.ok:
        print(f"pipeline not loaded: {res.error}", file=sys.stderr)
        return 1
    places = IndexPlaceResolver.from_catalog(CATALOG_DIR)
    homes = []
    for name in DEMO_HOMES:
        p = places.lookup(name)
        if p is None:
            print(f"home {name!r} not found in the catalogue's places", file=sys.stderr)
            return 1
        homes.append((p.name, (p.lat, p.lon)))

    type_sets = sorted({ts for _, ts, _ in nv.all_welcome_combinations()}, key=lambda t: (len(t), t))
    vectors = {
        (f, ts): translate_questionnaire_to_vector(ProfileCreate(fitness=f, type=list(ts), narrative="adventure"))
        for f in nv.FITNESS_LEVELS for ts in type_sets
    }
    hits: Counter[int] = Counter()
    best_rank: dict[int, int] = {}
    n_queries = 0

    def take(rec) -> None:
        nonlocal n_queries
        n_queries += 1
        for rank, rid in enumerate(rec.route_ids[:TOP_K], start=1):
            hits[int(rid)] += 1
            best_rank[int(rid)] = min(best_rank.get(int(rid), rank), rank)

    async def run() -> None:
        for (f, ts), vec in vectors.items():
            for tab in recsys_service.TABS:
                for home_name, home in [(None, None), *homes]:
                    q = UserQuery(f"demo-{f}-{'+'.join(ts)}-{tab}-{home_name}", vec, tab,
                                  home=home, max_travel_km=DEMO_RADIUS_KM if home else None)
                    take(await res.pipeline.arecommend(q, TOP_K))
        for f in nv.FITNESS_LEVELS:
            vec = vectors[(f, ("natural-scenery",))]
            for tab, intents in DEMO_INTENTS.items():
                for intent in intents:
                    for home_name, home in homes:
                        q = UserQuery(f"demo-intent-{f}-{tab}-{home_name}", vec, tab, home=home,
                                      max_travel_km=DEMO_RADIUS_KM, intent=intent)
                        take(await res.no_llm.arecommend(q, TOP_K))

    asyncio.run(run())
    ranked = sorted(hits, key=lambda r: (-hits[r], best_rank[r], r))
    chosen = ranked[: args.max_routes]
    records = load_catalog_records()
    out = {
        "schema": "narratives.demo_routes/v1",
        "how": "routes in the top 10 of the served pipeline for a grid of demo profiles (scripts/narratives.py "
               "select): questionnaire vectors x tabs x homes without an intent (configured system, rule path), "
               "plus demo intents through the no-LLM path; ranked by number of appearances, then best rank.",
        "system": res.pipeline.name,
        "no_llm_system": res.no_llm.name,
        "homes": [h for h, _ in homes],
        "radius_km": DEMO_RADIUS_KM,
        "intents": DEMO_INTENTS,
        "queries": n_queries,
        "distinct_routes_seen": len(hits),
        "max_routes": args.max_routes,
        "catalogue_sha256": catalogue_sha256(),
        "routes": [{"route_id": r, "title": records[r]["title"], "appearances": hits[r], "best_rank": best_rank[r]}
                   for r in chosen],
    }
    write_json(OUT_DIR / nv.DEMO_ROUTES_FILE, out)
    covered = sum(hits[r] for r in chosen) / max(sum(hits.values()), 1)
    print(f"{n_queries} queries, {len(hits)} distinct routes in a top {TOP_K}; kept {len(chosen)} "
          f"({covered:.1%} of all top-{TOP_K} appearances) -> {OUT_DIR / nv.DEMO_ROUTES_FILE}")
    return 0


# ------------------------------------------------------------------------------------------ generate
def build_client(live: bool, budget: float):
    from app.llm import LLMClient, LLMConfig
    from app.settings import get_settings

    cache_dir = OUT_DIR / nv.CACHE_SUBDIR
    s = get_settings()
    cfg = LLMConfig.from_settings(s)
    if live:
        cfg = cfg.with_(budget_usd=budget, frozen_dirs=(cache_dir,) if cache_dir.is_dir() else (), cache_only=False,
                        cache_enabled=True)
    else:
        if not cache_dir.is_dir():
            raise SystemExit(f"no cache at {cache_dir}: run with --live once (costs money)")
        # Replay: provider and model as recorded (they are part of the cache key), no writable cache, no network.
        rec = read_json(OUT_DIR / nv.ROUTES_FILE) or read_json(OUT_DIR / nv.WELCOME_FILE) or {}
        prov = rec.get("provenance", {})
        over: dict[str, Any] = {"cache_dir": None, "frozen_dirs": (cache_dir,), "cache_only": True,
                                "cache_enabled": True, "budget_usd": 0.0}
        if prov.get("provider"):
            over["llm_provider"] = prov["provider"]
            key = {"vertex": "vertex_llm_model", "openrouter": "openrouter_llm_model",
                   "ollama": "ollama_model"}.get(prov["provider"])
            if key and prov.get("model"):
                over[key] = prov["model"]
        cfg = cfg.with_(**over)
    return LLMClient(cfg)


async def _generate_all(client, jobs: list[dict[str, Any]]) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(job: dict[str, Any]) -> None:
        async with sem:
            p = job["params"]
            try:
                res = await client.generate(job["user"], system=job["system"], max_tokens=p["max_tokens"],
                                            temperature=p["temperature"], thinking=p["thinking"], tag=job["tag"],
                                            cache=True, retries=3)
            except Exception as e:  # recorded, the item is left out (template at request time)
                job["error"] = f"{type(e).__name__}: {e}"
                return
            job["text"] = nv.clean_generated(res.text)
            job["model_reported"] = res.model
            job["cache_key"] = res.cache_key
            job["cost_usd"] = res.cost_usd
            job["nominal_cost_usd"] = res.nominal_cost_usd

    await asyncio.gather(*(one(j) for j in jobs))


def _previous_dates(path: Path, key_field: str) -> dict[str, tuple[str, str]]:
    """key -> (cache_key, generated_at) from the current file, so a replay keeps the original dates."""
    data = read_json(path) or {}
    return {str(it[key_field]): (it.get("cache_key"), it.get("generated_at")) for it in data.get("items", [])}


def cmd_generate(args: argparse.Namespace) -> int:
    client = build_client(args.live, args.budget)
    cfg = client.config
    provider = cfg.llm_provider
    model = cfg.default_model("generate")
    if provider == "mock" and not args.allow_mock:
        print("LLM_PROVIDER=mock would write mock texts; pass --allow-mock for a dry run", file=sys.stderr)
        return 2
    today = datetime.now(UTC).date().isoformat()
    records = load_catalog_records()
    kinds = ("welcome", "routes") if args.kind == "all" else (args.kind,)
    t0 = time.perf_counter()
    jobs: list[dict[str, Any]] = []
    if "welcome" in kinds:
        for f, ts, n in nv.all_welcome_combinations():
            system, user = nv.build_welcome_pregen_prompt(f, ts, n)
            jobs.append({"kind": "welcome", "key": nv.welcome_key(f, ts, n), "fitness": f, "types": list(ts),
                         "narrative": n, "system": system, "user": user, "params": WELCOME_PARAMS,
                         "tag": "narrative.pregen.welcome"})
    if "routes" in kinds:
        demo = read_json(OUT_DIR / nv.DEMO_ROUTES_FILE)
        if not demo:
            print("no demo_routes.json: run `select` first", file=sys.stderr)
            return 1
        for item in demo["routes"]:
            rec = records.get(int(item["route_id"]))
            if rec is None:
                print(f"route {item['route_id']} is not in the catalogue; run `select` again", file=sys.stderr)
                return 1
            system, user = nv.build_route_pregen_prompt(rec)
            jobs.append({"kind": "routes", "key": str(rec["id"]), "route_id": int(rec["id"]), "title": rec["title"],
                         "facts": nv.route_facts(rec), "system": system, "user": user, "params": ROUTE_PARAMS,
                         "tag": "narrative.pregen.route"})
    if args.limit:  # pilot: the first N of each kind
        jobs = [j for k in kinds for j in [x for x in jobs if x["kind"] == k][: args.limit]]
    asyncio.run(_generate_all(client, jobs))

    def checks_of(j: dict[str, Any]) -> dict[str, Any]:
        if j["kind"] == "welcome":
            return nv.check_welcome_text(j["text"])
        return nv.check_route_text(j["text"], j["facts"])

    # One more attempt for a text that fails the checks, at a slightly higher temperature (its own cache key,
    # so a replay is still deterministic). If that fails too, the item is left out (template at request time).
    retry = [{**j, "params": {**j["params"], "temperature": round(j["params"]["temperature"] + 0.1, 2)},
              "attempt": 2, "first_text": j["text"], "first_key": j["cache_key"]}
             for j in jobs if "error" not in j and not checks_of(j)["ok"]]
    for j in retry:
        for k in ("text", "cache_key", "model_reported", "cost_usd", "nominal_cost_usd"):
            j.pop(k, None)
    if retry:
        asyncio.run(_generate_all(client, retry))
        by_key = {(j["kind"], j["key"]): j for j in retry}
        jobs = [by_key.get((j["kind"], j["key"]), j) for j in jobs]
        extra_keys = [j["cache_key"] for j in retry if j.get("cache_key")]
    else:
        extra_keys = []

    report: dict[str, Any] = {}
    for kind in kinds:
        kj = [j for j in jobs if j["kind"] == kind]
        path = OUT_DIR / (nv.WELCOME_FILE if kind == "welcome" else nv.ROUTES_FILE)
        prev = _previous_dates(path, "key" if kind == "welcome" else "route_id")
        items, rejected, errors = [], [], []
        for j in kj:
            if "error" in j:
                errors.append({"key": j["key"], "error": j["error"]})
                continue
            checks = checks_of(j)
            if not checks["ok"]:
                rejected.append({"key": j["key"], "text": j["text"], "first_text": j.get("first_text"),
                                 "checks": checks})
                continue
            old = prev.get(j["key"])
            generated_at = old[1] if old and old[0] == j["cache_key"] and old[1] else today
            base = {"key": j["key"]} if kind == "welcome" else {"route_id": j["route_id"], "title": j["title"]}
            if kind == "welcome":
                base.update({"fitness": j["fitness"], "types": j["types"], "narrative": j["narrative"]})
            items.append({**base, "text": j["text"], "llm_generated": True, "model": j["model_reported"] or model,
                          "prompt_version": nv.WELCOME_PROMPT_VERSION if kind == "welcome" else nv.ROUTE_PROMPT_VERSION,
                          "generated_at": generated_at, "cache_key": j["cache_key"],
                          "temperature": j["params"]["temperature"], "attempt": j.get("attempt", 1),
                          "checks": {k: v for k, v in checks.items() if k != "ok"}})
        if kind == "routes":
            items.sort(key=lambda it: it["route_id"])
        system_sample = kj[0]["system"] if kj else ""
        header = {
            "schema": nv.STORE_SCHEMA,
            "kind": kind,
            "llm_generated": True,
            "notice": "LLM-generated text, produced offline by backend/scripts/narratives.py and served when no live "
                      "LLM is used. Route texts may only use facts from the catalogue record (OpenStreetMap "
                      "contributors, ODbL); each passed the automatic grounding checks in "
                      "app/services/narratives.py.",
            "provenance": {
                "provider": provider,
                "model": model,
                "prompt_version": nv.WELCOME_PROMPT_VERSION if kind == "welcome" else nv.ROUTE_PROMPT_VERSION,
                "system_prompt_sha256": sha(system_sample) if kind == "routes" else
                sha("".join(sorted({j["system"] for j in kj}))),
                "params": WELCOME_PARAMS if kind == "welcome" else ROUTE_PARAMS,
                "generator": "backend/scripts/narratives.py generate",
            },
            "scope": "all questionnaire combinations (3 fitness x 8 type sets x 3 styles)" if kind == "welcome"
            else "the demo route set (data/narratives/demo_routes.json)",
            **({"catalogue_sha256": catalogue_sha256()} if kind == "routes" else {}),
            "counts": {"requested": len(kj), "stored": len(items), "rejected_by_checks": len(rejected),
                       "errors": len(errors)},
            "items": items,
        }
        write_json(path, header)
        if rejected or errors:
            write_json(OUT_DIR / f"_rejected_{kind}.json", {"rejected": rejected, "errors": errors})
        else:
            (OUT_DIR / f"_rejected_{kind}.json").unlink(missing_ok=True)
        report[kind] = header["counts"]

    keys = sorted({j["cache_key"] for j in jobs if j.get("cache_key")} | set(extra_keys)
                  | {j.get("first_key") for j in jobs if j.get("first_key")})
    if args.live:
        cache_dir = OUT_DIR / nv.CACHE_SUBDIR
        tmp = OUT_DIR / (nv.CACHE_SUBDIR + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        manifest = client.export_frozen(tmp, keys)
        if manifest.get("missing"):
            print(f"warning: {manifest['missing']} keys missing from the cache export", file=sys.stderr)
        shutil.rmtree(cache_dir, ignore_errors=True)
        tmp.rename(cache_dir)
    usage = client.usage_report()
    print(json.dumps({"counts": report, "provider": provider, "model": model, "live": args.live,
                      "real_usd": usage["real_usd"], "nominal_usd": usage["nominal_usd"],
                      "seconds": round(time.perf_counter() - t0, 1)}, indent=1))
    return 0


# --------------------------------------------------------------------------------------------- check
def cmd_check(args: argparse.Namespace) -> int:
    records = load_catalog_records()
    store = nv.load_store(OUT_DIR)
    problems = []
    for rid, e in store.routes.items():
        if rid not in records:
            problems.append(f"route {rid}: not in the catalogue")
            continue
        c = nv.check_route_text(e.text, nv.route_facts(records[rid]))
        if not c["ok"]:
            problems.append(f"route {rid}: grounding check failed {c}")
    for key, e in store.welcome.items():
        if not nv.check_welcome_text(e.text)["ok"]:
            problems.append(f"welcome {key}: check failed")
    for kind, entries in (("welcome", store.welcome.values()), ("routes", store.routes.values())):
        for e in entries:
            for f in ("model", "prompt_version", "generated_at", "llm_generated"):
                if not e.meta.get(f):
                    problems.append(f"{kind}: entry without {f}")
    expected = {nv.welcome_key(f, ts, n) for f, ts, n in nv.all_welcome_combinations()}
    missing = expected - set(store.welcome)
    print(json.dumps({**store.summary(), "welcome_missing": len(missing), "problems": len(problems)}, indent=1))
    for p in problems[:50]:
        print(" -", p)
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("select", help="compute the demo route set (no provider call)")
    sp.add_argument("--max-routes", type=int, default=400)
    sp.set_defaults(func=cmd_select)
    gp = sub.add_parser("generate", help="write welcome.json / routes.json (cache replay unless --live)")
    gp.add_argument("--kind", choices=("welcome", "routes", "all"), default="all")
    gp.add_argument("--live", action="store_true", help="call the provider for cache misses (costs money)")
    gp.add_argument("--budget", type=float, default=1.0, help="hard stop on real USD with --live")
    gp.add_argument("--allow-mock", action="store_true")
    gp.add_argument("--limit", type=int, default=0, help="pilot: only the first N items of each kind")
    gp.set_defaults(func=cmd_generate)
    cp = sub.add_parser("check", help="validate the committed store")
    cp.set_defaults(func=cmd_check)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
