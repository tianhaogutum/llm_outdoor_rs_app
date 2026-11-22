"""Frozen replay: export every LLM cache entry the evaluation touches to ``eval/cache/`` (committed).

    cd backend && python -m evalkit.freeze export    # make freeze
    cd backend && python -m evalkit.freeze verify    # make replay-check

``export`` re-issues, from the local cache, every call behind the committed artefacts (persona intents,
the recorded judge batches in ``qrels.jsonl``, the judge pilot, the second judge on the owner set) and
exports exactly those keys. ``verify`` runs in a subprocess with an empty cache directory,
``LLM_CACHE_ONLY=true`` and only the frozen layer, so no credentials are needed and any miss fails: it
regenerates the personas, replays every judge batch, re-runs the pilot and recomputes all results, and
compares them with the committed files (results modulo their ``volatile`` block).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evalkit import paths


def _replay_all(check: bool) -> tuple[list[str], list[str]]:
    """Re-issue every recorded call. Returns (touched keys, problems)."""
    from app.recsys import get_catalog
    from evalkit import intents, owner, pilot
    from evalkit.judge import JUDGES, client_for, replay_batches
    from evalkit.personas import generate, persona_stats  # noqa: F401
    from evalkit.qrels import Qrels

    problems: list[str] = []
    keys: list[str] = []
    catalog = get_catalog()

    # 1. persona intents
    personas = generate()
    wc = intents.make_writer_client()
    intents.add_intents(personas, wc)
    keys += wc.touched_keys()
    committed = {
        json.loads(line)["persona_id"]: json.loads(line) for line in paths.PERSONAS_FILE.read_text().splitlines()
    }
    for p in personas:
        if committed[p.persona_id]["intent"] != p.intent:
            problems.append(f"intent differs for {p.persona_id}")
    pmap = {p.persona_id: p for p in personas}
    for p in personas:  # use the committed objects for the judge (identical if no problem above)
        p.intent = committed[p.persona_id]["intent"]

    # 2. judge batches recorded in the qrels
    qrels = Qrels.load(paths.QRELS_FILE)
    recs = list(qrels.records.values())
    for name in sorted({r["judge"] for r in recs if r["source"] == "llm"}):
        cfg = JUDGES[name]
        c = client_for(cfg)
        out = replay_batches(cfg, catalog, pmap, recs, client=c)
        keys += c.touched_keys()
        got = {(r["persona_id"], r["route_id"]): r["grade"] for r in out.records}
        for r in recs:
            if r["source"] == "llm" and r["judge"] == name:
                k = (r["persona_id"], r["route_id"])
                if got.get(k) != r["grade"]:
                    problems.append(f"judge replay differs for {k}: {got.get(k)} vs {r['grade']}")

    # 3. judge pilot
    if (paths.RESULTS_DIR / "_judge_pilot.json").exists():
        res = pilot.run(argparse.Namespace(only=None, limit=0))
        summ = pilot.summarise(res)
        for name in {*pilot.CANDIDATES, pilot.REFERENCE}:
            keys += client_for(JUDGES[name]).touched_keys()
        if check:
            committed_p = json.loads((paths.RESULTS_DIR / "_judge_pilot.json").read_text())
            if json.dumps(summ, sort_keys=True) != json.dumps(committed_p, sort_keys=True):
                problems.append("judge pilot summary differs")

    # 4. second judge on the owner set
    if (paths.OWNER_DIR / "second_judge.jsonl").exists():
        from evalkit.judge import judge_pairs

        cfg = JUDGES[owner.SECOND_JUDGE]
        pairs = owner.load_pairs()
        c = client_for(cfg)
        out = judge_pairs(
            cfg,
            catalog,
            pmap,
            [(p["persona_id"], p["route_id"]) for p in pairs],
            client=c,
            seed=owner.SEED,
            tag="second-judge",
        )
        keys += c.touched_keys()
        sec = {
            json.loads(line)["pair_id"]: json.loads(line)["grade"]
            for line in (paths.OWNER_DIR / "second_judge.jsonl").read_text().splitlines()
        }
        got = {(r["persona_id"], r["route_id"]): r["grade"] for r in out.records}
        for p in pairs:
            if got.get((p["persona_id"], p["route_id"])) != sec[p["pair_id"]]:
                problems.append(f"second judge differs for {p['pair_id']}")
    # 5. the systems' own upstream calls (query embeddings, ranker API): re-run every stored run of a
    #    system that uses the LLM client and compare its rankings with the committed run file
    from evalkit import evaluate as E

    for name in E.stored_systems():
        from app.recsys import get_system_spec

        if not get_system_spec(name).uses_llm:
            continue
        client = E.system_client()
        client.reset_touched()
        run, _, _ = E.run_system(name, catalog, list(pmap.values()))
        keys += client.touched_keys()
        if check:
            stored, _ = E.load_run(name)
            for pid, r in run.items():
                if r["ids"] != stored[pid]["ids"]:
                    problems.append(f"run differs: {name} {pid}")
                    break
    for extra in EXTRA_REPLAYS:
        k, pr = extra(check)
        keys += k
        problems += pr
    return list(dict.fromkeys(keys)), problems


# Further recorded experiments to replay (registered by evalkit modules: ranker pilot, LTR, MMR sweep).
EXTRA_REPLAYS: list = []


def export() -> None:
    from evalkit.llm import make_client

    keys, problems = _replay_all(check=False)
    if problems:
        sys.exit("replay from the local cache differs from the committed files:\n" + "\n".join(problems[:20]))
    if paths.FROZEN_DIR.exists():
        shutil.rmtree(paths.FROZEN_DIR)
    paths.FROZEN_DIR.mkdir(parents=True)
    man = make_client("freeze", frozen_dirs=()).export_frozen(paths.FROZEN_DIR, keys)
    if man.get("missing"):
        sys.exit(f"{man['missing']} touched keys are not in the local cache")
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))


def _strip(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "volatile"}


def verify_inner() -> None:
    """Runs inside the clean-cache subprocess."""
    from evalkit import evaluate

    keys, problems = _replay_all(check=True)
    saved = {p: p.read_bytes() for p in paths.RESULTS_DIR.glob("*.json")}
    before = {p.name: _strip(json.loads(b)) for p, b in saved.items() if not p.name.startswith("_")}
    qrels_before = paths.QRELS_FILE.read_bytes()
    try:
        evaluate.evaluate_all(judge=False)
        for name, doc in before.items():
            after = _strip(json.loads((paths.RESULTS_DIR / name).read_text()))
            if json.dumps(after, sort_keys=True) != json.dumps(doc, sort_keys=True):
                problems.append(f"results differ: {name}")
        for p, b in saved.items():
            if p.name in ("_qrels.json", "_personas.json") and p.read_bytes() != b:
                problems.append(f"{p.name} differs")
        if paths.QRELS_FILE.read_bytes() != qrels_before:
            problems.append("qrels changed")
    finally:  # leave the committed files (including their volatile blocks) untouched
        for p, b in saved.items():
            p.write_bytes(b)
    print(json.dumps({"replayed_keys": len(keys), "problems": problems}, indent=1))
    if problems:
        sys.exit(1)


def verify() -> None:
    with tempfile.TemporaryDirectory(prefix="frozen-replay-") as tmp:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER", "GOOGLE"))}
        env.update(
            {
                "LLM_CACHE_DIR": str(Path(tmp) / "cache"),
                "LLM_CACHE_ONLY": "true",
                "LLM_CACHE_FROZEN_DIRS": str(paths.FROZEN_DIR),
                # blank every credential (real env vars override backend/.env); any provider call would fail
                "OPENROUTER_API_KEY": "",
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_APPLICATION_CREDENTIALS": str(Path(tmp) / "no-credentials.json"),
                "CLOUDSDK_CONFIG": str(Path(tmp) / "gcloud"),
            }
        )
        r = subprocess.run([sys.executable, "-m", "evalkit.freeze", "_verify_inner"], cwd=paths.BACKEND_DIR, env=env)
        if r.returncode:
            sys.exit("frozen replay FAILED")
        print(
            "frozen replay OK: clean cache, cache-only, no credentials; personas, judge batches, pilot and "
            "all results reproduced"
        )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["export", "verify", "_verify_inner"])
    args = ap.parse_args(argv)
    {"export": export, "verify": verify, "_verify_inner": verify_inner}[args.cmd]()


if __name__ == "__main__":
    main()
