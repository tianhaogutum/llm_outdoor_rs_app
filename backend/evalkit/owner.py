"""Owner checkpoint: blind labelling set, interim second judge, and kappa.

    cd backend && python -m evalkit.owner build      # blind set: eval/owner_labels/{pairs.jsonl, sheet.csv, label.html}
    cd backend && python -m evalkit.owner second     # second judge (other vendor) on the same 150 pairs
    cd backend && python -m evalkit.owner kappa      # make kappa

The owner opens ``eval/owner_labels/label.html`` in a browser, grades each pair 0-3 and exports
``owner_labels.csv`` (columns ``pair_id,label``) into ``eval/owner_labels/``. Neither the sheet nor the
page contains judge grades; the pair -> (persona, route) mapping is in ``pairs.jsonl`` and the grades are
only joined in by ``kappa``.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import sys
from collections import Counter
from typing import Any

import numpy as np

from app.recsys import get_catalog
from evalkit import paths
from evalkit.cards import persona_card, route_card
from evalkit.evaluate import dumps
from evalkit.judge import JUDGES, PRODUCTION_JUDGE, judge_pairs, rubric_text
from evalkit.metrics import confusion_matrix, weighted_kappa
from evalkit.personas import load_personas
from evalkit.qrels import Qrels

N_PAIRS = 150
SEED = 150
SECOND_JUDGE = "second-sonnet55"


def _f(name: str):
    return paths.OWNER_DIR / name


def sample_pairs(qrels: Qrels, personas: dict[str, Any]) -> list[dict[str, Any]]:
    """150 LLM-judged pairs stratified over judge grade x split (rule-graded pairs are excluded: they are 0 by rule).

    Target: an equal share per grade, within each grade dev:test as in the persona split (2:1); a grade with
    too few pairs gives its remainder to the others. At most 2 pairs per persona per grade, for breadth."""
    rng = np.random.default_rng(SEED)
    cells: dict[tuple[int, str], list[tuple[str, int]]] = {}
    for (pid, rid), r in sorted(qrels.records.items()):
        if r["source"] == "llm":
            cells.setdefault((r["grade"], personas[pid].split), []).append((pid, rid))
    for k in cells:
        order = rng.permutation(len(cells[k]))
        seen: Counter = Counter()
        picked = []
        for i in order:
            pid, rid = cells[k][int(i)]
            if seen[pid] < 2:
                picked.append((pid, rid))
                seen[pid] += 1
        cells[k] = picked
    per_grade = {g: N_PAIRS // 4 + (1 if g < N_PAIRS % 4 else 0) for g in range(4)}
    avail = {g: len(cells.get((g, "dev"), [])) + len(cells.get((g, "test"), [])) for g in range(4)}
    # redistribute shortfalls
    short = sum(max(0, per_grade[g] - avail[g]) for g in range(4))
    for g in range(4):
        per_grade[g] = min(per_grade[g], avail[g])
    while short > 0 and any(per_grade[g] < avail[g] for g in range(4)):
        for g in range(4):
            if short > 0 and per_grade[g] < avail[g]:
                per_grade[g] += 1
                short -= 1
    chosen = []
    for g in range(4):
        n_test = round(per_grade[g] / 3)
        test = cells.get((g, "test"), [])[:n_test]
        dev = cells.get((g, "dev"), [])[: per_grade[g] - len(test)]
        if len(dev) + len(test) < per_grade[g]:
            test = cells.get((g, "test"), [])[: per_grade[g] - len(dev)]
        chosen += dev + test
    order = rng.permutation(len(chosen))
    return [
        {"pair_id": f"O{j + 1:03d}", "persona_id": chosen[int(i)][0], "route_id": int(chosen[int(i)][1])}
        for j, i in enumerate(order)
    ]


HTML_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Blind relevance labelling (150 pairs)</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}
h1{font-size:1.4rem} details{margin:1rem 0;padding:.5rem 1rem;background:#f5f5f7;border-radius:8px}
.pair{border:1px solid #d2d2d7;border-radius:10px;margin:1.2rem 0;padding:1rem}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
pre{white-space:pre-wrap;font:13px/1.45 ui-monospace,Menlo,monospace;background:#fafafa;
 padding:.6rem;border-radius:6px;margin:0}
.grade label{margin-right:1.2rem;font-weight:600;cursor:pointer} .grade{margin-top:.8rem}
#bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #d2d2d7;z-index:2}
button{font-size:1rem;padding:.4rem 1rem;margin-right:.5rem}
.done{border-color:#34c759}
</style></head><body>
<h1>Blind relevance labelling: 150 (simulated person, route) pairs</h1>
<p>Grade each route for the person on the 0-3 scale of the rubric below. Your labels are saved in this browser as you go
(localStorage); press <b>Export labels</b> to download <code>owner_labels.csv</code> and save it as
<code>eval/owner_labels/owner_labels.csv</code>, then run <code>make kappa</code>.
The judge's grades are not on this page.</p>
"""

HTML_JS = """
<script>
const KEY='owner_labels_v1';
const store=JSON.parse(localStorage.getItem(KEY)||'{}');
function refresh(){let n=0;document.querySelectorAll('.pair').forEach(el=>{const id=el.dataset.id;
 if(store[id]!==undefined){n++;el.classList.add('done');
  const r=el.querySelector('input[value="'+store[id]+'"]');if(r)r.checked=true;}});
 document.getElementById('count').textContent=n+' / '+document.querySelectorAll('.pair').length+' labelled';}
document.addEventListener('change',e=>{if(e.target.name&&e.target.name.startsWith('g_')){
 store[e.target.name.slice(2)]=e.target.value;
 localStorage.setItem(KEY,JSON.stringify(store));refresh();}});
function exportCsv(){let s='pair_id,label\\n';Object.keys(store).sort().forEach(k=>{s+=k+','+store[k]+'\\n'});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));
 a.download='owner_labels.csv';a.click();}
function clearAll(){if(confirm('Delete all labels saved in this browser?')){
 localStorage.removeItem(KEY);location.reload();}}
refresh();
</script></body></html>
"""


def build() -> None:
    catalog = get_catalog()
    personas = {p.persona_id: p for p in load_personas()}
    qrels = Qrels.load()
    pairs = sample_pairs(qrels, personas)
    paths.OWNER_DIR.mkdir(parents=True, exist_ok=True)
    _f("pairs.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs), encoding="utf-8")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["pair_id", "persona_card", "route_card", "label"])
    rows = []
    for pr in pairs:
        p = personas[pr["persona_id"]]
        pc = persona_card(p)
        rc = route_card(catalog.get(pr["route_id"]), tuple(p.profile["home"]))
        w.writerow([pr["pair_id"], pc, rc, ""])
        rows.append((pr["pair_id"], pc, rc))
    _f("sheet.csv").write_text(buf.getvalue(), encoding="utf-8")
    parts = [
        HTML_HEAD,
        "<details open><summary><b>Rubric</b></summary><pre>",
        html.escape(rubric_text()),
        "</pre></details>",
        '<div id="bar"><button onclick="exportCsv()">Export labels</button>'
        '<button onclick="clearAll()">Clear</button> <span id="count"></span></div>',
    ]
    for pid, pc, rc in rows:
        radios = "".join(f'<label><input type="radio" name="g_{pid}" value="{g}"> {g}</label>' for g in range(4))
        parts.append(
            f'<div class="pair" data-id="{pid}"><b>{pid}</b><div class="cards">'
            f"<div><i>Person</i><pre>{html.escape(pc)}</pre></div>"
            f"<div><i>Route</i><pre>{html.escape(rc)}</pre></div></div>"
            f'<div class="grade">Grade: {radios}</div></div>'
        )
    parts.append(HTML_JS)
    _f("label.html").write_text("\n".join(parts), encoding="utf-8")
    grades = Counter(qrels.records[(p["persona_id"], p["route_id"])]["grade"] for p in pairs)
    splits = Counter(personas[p["persona_id"]].split for p in pairs)
    print(
        f"{len(pairs)} pairs -> {paths.OWNER_DIR} (judge grades {dict(sorted(grades.items()))}, splits {dict(splits)})"
    )


def load_pairs() -> list[dict[str, Any]]:
    return [json.loads(line) for line in _f("pairs.jsonl").read_text(encoding="utf-8").splitlines() if line]


def second() -> None:
    catalog = get_catalog()
    personas = {p.persona_id: p for p in load_personas()}
    pairs = load_pairs()
    cfg = JUDGES[SECOND_JUDGE]
    out = judge_pairs(
        cfg, catalog, personas, [(p["persona_id"], p["route_id"]) for p in pairs], seed=SEED, tag="second-judge"
    )
    by = {(r["persona_id"], r["route_id"]): r for r in out.records}
    with _f("second_judge.jsonl").open("w", encoding="utf-8") as fh:
        for p in pairs:
            r = by[(p["persona_id"], p["route_id"])]
            fh.write(
                json.dumps(
                    {
                        "pair_id": p["pair_id"],
                        "grade": r["grade"],
                        "reason": r["reason"],
                        "judge": cfg.name,
                        "model": r["model"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(
        f"second judge {cfg.name}: {len(out.records)} grades, {out.calls} calls, nominal {out.nominal_cost_usd:.4f} USD"
    )


def _agreement(a: list[int], b: list[int]) -> dict[str, Any]:
    return {
        "n": len(a),
        "qwk": weighted_kappa(a, b),
        "linear_kappa": weighted_kappa(a, b, weights="linear"),
        "exact": float(np.mean([x == y for x, y in zip(a, b, strict=True)])) if a else None,
        "within_one": float(np.mean([abs(x - y) <= 1 for x, y in zip(a, b, strict=True)])) if a else None,
        "confusion_rows_first": confusion_matrix(a, b),
    }


def kappa() -> dict[str, Any]:
    qrels = Qrels.load()
    pairs = load_pairs() if _f("pairs.jsonl").exists() else []
    judge = {p["pair_id"]: qrels.records[(p["persona_id"], p["route_id"])] for p in pairs}
    doc: dict[str, Any] = {"schema": "evalkit.kappa/v1", "production_judge": PRODUCTION_JUDGE, "n_pairs": len(pairs)}
    if _f("owner_labels.csv").exists():
        with _f("owner_labels.csv").open(encoding="utf-8") as fh:
            owner = {r["pair_id"]: int(r["label"]) for r in csv.DictReader(fh) if r.get("label", "").strip() != ""}
        ids = [p["pair_id"] for p in pairs if p["pair_id"] in owner]
        a = [owner[i] for i in ids]
        b = [judge[i]["grade"] for i in ids]
        dis = [
            {
                "pair_id": i,
                "persona_id": judge[i]["persona_id"],
                "route_id": judge[i]["route_id"],
                "owner": owner[i],
                "judge": judge[i]["grade"],
                "judge_reason": judge[i]["reason"],
            }
            for i in ids
            if owner[i] != judge[i]["grade"]
        ]
        doc["owner"] = {
            "status": "ok" if len(ids) == len(pairs) else "partial",
            "labelled": len(ids),
            **_agreement(a, b),
            "disagreements": sorted(dis, key=lambda d: -abs(d["owner"] - d["judge"])),
        }
    else:
        doc["owner"] = {
            "status": "owner labels missing",
            "expected_file": str(_f("owner_labels.csv").relative_to(paths.OWNER_DIR.parent.parent)),
        }
    if _f("second_judge.jsonl").exists():
        sec = {
            json.loads(line)["pair_id"]: json.loads(line)
            for line in _f("second_judge.jsonl").read_text(encoding="utf-8").splitlines()
        }
        ids = [p["pair_id"] for p in pairs if p["pair_id"] in sec]
        a = [judge[i]["grade"] for i in ids]
        b = [sec[i]["grade"] for i in ids]
        models = sorted({sec[i]["model"] for i in ids})
        doc["llm_vs_llm"] = {
            "note": "interim signal only: two LLM judges from different vendors; NOT a human check",
            "first": PRODUCTION_JUDGE,
            "second": SECOND_JUDGE,
            "second_models": models,
            **_agreement(a, b),
        }
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS_DIR / "_kappa.json").write_text(dumps(doc), encoding="utf-8")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "second", "kappa"])
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build()
    elif args.cmd == "second":
        second()
    else:
        doc = kappa()
        o = doc["owner"]
        if o["status"] == "owner labels missing":
            print("owner labels missing: label eval/owner_labels/label.html and save owner_labels.csv there")
        else:
            print(f"owner vs judge ({o['labelled']} pairs): QWK {o['qwk']:.3f}, exact {o['exact']:.2f}")
            print("confusion (rows = owner 0..3, cols = judge 0..3):")
            for row in o["confusion_rows_first"]:
                print("   ", row)
            print(f"{len(o['disagreements'])} disagreements (largest first):")
            for d in o["disagreements"][:40]:
                print(f"  {d['pair_id']} owner {d['owner']} judge {d['judge']}: {d['judge_reason']}")
        if "llm_vs_llm" in doc:
            m = doc["llm_vs_llm"]
            print(
                f"LLM vs LLM ({m['first']} vs {m['second']}, {m['n']} pairs, NOT a human check): "
                f"QWK {m['qwk']:.3f}, exact {m['exact']:.2f}"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
