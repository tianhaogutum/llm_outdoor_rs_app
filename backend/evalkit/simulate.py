"""Feedback-loop simulation (Phase 5): 10 rounds of recommendations and simulated reactions per persona.

    cd backend && python -m evalkit.simulate run         # make simulate: test personas, all SIM_SYSTEMS
    cd backend && python -m evalkit.simulate estimate    # judge volume and cost, proxy grades, writes nothing
    cd backend && python -m evalkit.simulate tune        # dev subset: the feedback parameter grid (TUNING)
    cd backend && python -m evalkit.simulate tune --estimate

Protocol (docs/EVAL.md section 10):

- Round t = 1..10, one week apart. Every system gets ``persona.to_query()`` plus the feedback events of
  rounds 1..t-1 (age = 7 x (t - s) days) and returns its top 10.
- Every shown route is judged before the user reacts: holes go to the production judge (hard-constraint
  violations by rule), batched per persona over all systems of the round. These judgements live in
  ``eval/qrels/sim_qrels.jsonl`` (same format, replayable), not in the pooled ``qrels.jsonl``, so the
  single-request tables keep their top-10 pool.
- The simulated user (:class:`BehaviourModel`, written down in EVAL.md) reacts to the list: at most one
  completion and a few rejections with a reason. It reads the judge's grades and the persona's latent
  profile; no system can read either. Its random draws are keyed by (persona, round, route), so the same
  route at the same round gets the same draw for every system (common random numbers).
- Per-round NDCG@10: routes the user completed or rejected *before* round t have gain 0 in round t and are
  removed from the ideal ranking (a system is not punished for no longer showing them, and gains nothing
  by showing them again). Everything else uses its judged grade.
- Deterministic: personas, systems, judge (cached) and behaviour are all seeded; a rerun with the qrels in
  place makes no LLM call and reproduces ``_simulation.json`` outside its ``volatile`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from app.recsys import Catalog, get_catalog
from app.recsys.constraints import CATEGORY_TABS, haversine_km
from app.recsys.feedback import FeedbackConfig
from app.recsys.query import FeedbackEvent
from evalkit import metrics as M
from evalkit import paths
from evalkit.evaluate import dumps, make_system
from evalkit.personas import Persona, load_personas
from evalkit.qrels import Qrels, fill_holes

ROUNDS = 10
DAYS_PER_ROUND = 7
K = 10
SEED = 20261123
N_BOOT, BOOT_SEED = 1000, 20261122
SIM_QRELS_NAME = "sim_qrels.jsonl"


def sim_qrels_file():
    return paths.QRELS_DIR / SIM_QRELS_NAME


OUT = "_simulation.json"
TUNE_OUT = "_sim_tuning.json"
FIGURE = paths.DOCS_DIR / "convergence.png"
RUNS_FILE = paths.RUNS_DIR / "simulation.jsonl"

# label -> (registered system, config overrides). Order = table order.
SIM_SYSTEMS: dict[str, tuple[str, dict[str, Any]]] = {
    "v0_rules": ("v0_rules", {}),
    "v1_filters": ("v1_filters", {}),
    "recsys_v1": ("recsys_v1", {}),
    "recsys_v1_fb_route": ("recsys_v1_fb_route", {}),
    "recsys_v1_fb": ("recsys_v1_fb", {}),
    "recsys_v1_fb_light": ("recsys_v1_fb_light", {}),
    "recsys_v1_fb_explore": ("recsys_v1_fb_explore", {}),
    "recsys_serve_noexplore": ("recsys_serve_noexplore", {}),
    "recsys_serve": ("recsys_serve", {}),
}
SIM_LABELS = {
    "v0_rules": "baseline rule scorer, its own feedback (negative reasons only)",
    "v1_filters": "hard filters + rule scorer, v0's feedback mechanism",
    "recsys_v1": "default system, feedback off (control)",
    "recsys_v1_fb_route": "default + route effects only (completed / rejected not shown again)",
    "recsys_v1_fb": "default + feedback loop as chosen on dev (route effects + slot steps)",
    "recsys_v1_fb_light": "default + full light loop (slots, Rocchio, re-scoring, near-duplicates)",
    "recsys_v1_fb_explore": "default + feedback loop + exploration slot",
    "recsys_serve_noexplore": "served ranker (no length_cap_ratio) + feedback loop, no exploration",
    "recsys_serve": "SERVED DEFAULT: served ranker + feedback loop + exploration slot (D25)",
}

# Dev tuning grid (feedback parameters; half-life and route effects fixed, see EVAL.md section 10)
_ZERO = dict(beta=0.0, gamma=0.0, eta_emb=0.0, eta_tag=0.0, nu_dup=0.0, difficulty_step=0.0, cap_step=0.0,
             radius_step=0.0)
_LIGHT = dict(beta=0.25, gamma=0.1, eta_emb=0.1, eta_tag=0.1, nu_dup=0.15, difficulty_step=0.25, cap_step=0.25,
              radius_step=0.25)
TUNING: dict[str, FeedbackConfig] = {
    "route_only": FeedbackConfig(**_ZERO),
    "slots_only": FeedbackConfig(**{**_ZERO, "difficulty_step": 0.25, "cap_step": 0.25, "radius_step": 0.25}),
    "rocchio_only": FeedbackConfig(**{**_ZERO, "beta": 0.25, "gamma": 0.1}),
    "rescore_only": FeedbackConfig(**{**_LIGHT, "difficulty_step": 0.0, "cap_step": 0.0, "radius_step": 0.0}),
    "light": FeedbackConfig(**_LIGHT),
    "medium": FeedbackConfig(beta=0.5, gamma=0.25, eta_emb=0.3, eta_tag=0.2, nu_dup=0.3, difficulty_step=0.5,
                             cap_step=0.5, radius_step=0.5),
    "strong": FeedbackConfig(beta=1.0, gamma=0.5, eta_emb=0.6, eta_tag=0.4, nu_dup=0.5, difficulty_step=1.0,
                             cap_step=1.0, radius_step=1.0),
}
TUNE_PERSONAS = 40
TUNE_SEED = 7


REASONS_ALL = ("completed", "too-hard", "too-easy", "too-far", "not-interested")


# --------------------------------------------------------------------------- the simulated user
@dataclass(frozen=True)
class BehaviourModel:
    """How a simulated user reacts to one list (docs/EVAL.md section 10). Never visible to a system."""

    examine_decay: float = 0.95  # P(look at rank i) = examine_decay ** (i - 1)
    p_complete: tuple[float, float, float, float] = (0.0, 0.03, 0.25, 0.5)  # by judged grade 0..3
    p_reject: tuple[float, float, float, float] = (0.5, 0.2, 0.0, 0.0)
    max_completions: int = 1  # one outing per week
    max_rejections: int = 3
    too_hard_factor: float = 1.2  # length / ascent above the latent limit by this factor
    too_easy_length: float = 0.7  # length below this share of the band's lower end ...
    too_easy_ascent: float = 0.5  # ... and ascent below this share of the ascent tolerance
    seed: int = SEED

    def describe(self) -> dict[str, Any]:
        return asdict(self)

    def _u(self, *parts: Any) -> float:
        h = hashlib.sha256("|".join(map(str, (self.seed, *parts))).encode()).digest()[:8]
        return int.from_bytes(h, "big") / 2**64

    def reason(self, p: Persona, catalog: Catalog, route_id: int) -> str:
        """Why this person rejects this route: from the latent profile vs the route's facts."""
        x, r = p.latent, catalog.get(route_id)
        home = x["home"]
        if haversine_km(home["lat"], home["lon"], r.start_lat, r.start_lon) > x["max_travel_km"]:
            return "too-far"
        if r.category_name not in CATEGORY_TABS.get(x["category"], ()):
            return "not-interested"
        if (x["style"] == "mtb") != (r.activity == "mtb") and x["category"] == "cycling":
            return "not-interested"
        asc = max(r.ascent_m, r.descent_m)
        tech = r.difficulty_technical or 0
        lo, hi = x["distance_band_km"]
        if (
            r.length_km > hi * self.too_hard_factor
            or asc > x["max_ascent_m"] * self.too_hard_factor
            or tech > x["technical_max"]
        ):
            return "too-hard"
        if r.length_km < lo * self.too_easy_length and asc < x["max_ascent_m"] * self.too_easy_ascent:
            return "too-easy"
        return "not-interested"

    def react(
        self,
        p: Persona,
        catalog: Catalog,
        shown: list[int],
        grades: dict[int, int],
        round_: int,
        consumed: frozenset[int] = frozenset(),
    ) -> list[tuple[int, str]]:
        """Events for one shown list: [(route_id, reason)], completions as ``completed``. Routes the user
        already completed or rejected are passed over (no repeat outing, no second thumbs-down)."""
        out: list[tuple[int, str]] = []
        done = rej = 0
        for i, rid in enumerate(shown[:K]):
            if rid in consumed:
                continue
            if self._u(p.persona_id, round_, rid, "examine") >= self.examine_decay**i:
                continue
            g = int(grades[rid])
            if done < self.max_completions and self._u(p.persona_id, round_, rid, "complete") < self.p_complete[g]:
                out.append((rid, "completed"))
                done += 1
                continue
            if rej < self.max_rejections and self._u(p.persona_id, round_, rid, "reject") < self.p_reject[g]:
                out.append((rid, self.reason(p, catalog, rid)))
                rej += 1
        return out


# --------------------------------------------------------------------------- metric
def round_ndcg(shown: list[int], grades: dict[int, int], consumed: set[int]) -> Optional[float]:
    """NDCG@10 of one round: routes consumed (completed or rejected) in earlier rounds have gain 0 and are
    not part of the ideal ranking. None if nothing relevant is left (undefined, excluded)."""
    g = {r: (0 if r in consumed else v) for r, v in grades.items()}
    ideal_pool = {r: v for r, v in grades.items() if r not in consumed}
    ideal = M.dcg(sorted(ideal_pool.values(), reverse=True)[:K])
    if ideal == 0:
        return None
    return M.dcg([g[r] for r in shown[:K]]) / ideal


# --------------------------------------------------------------------------- running
def _query(p: Persona, events: list[tuple[int, int, str]], round_: int):
    from dataclasses import replace

    fb = tuple(FeedbackEvent(rid, reason, float(DAYS_PER_ROUND * (round_ - s))) for s, rid, reason in events)
    return replace(p.to_query(), feedback=fb)


def _recommend_all(system, queries: list) -> list[list[int]]:
    if hasattr(system, "arecommend"):
        from app.llm import run_sync

        async def go():
            sem = asyncio.Semaphore(8)

            async def one(q):
                async with sem:
                    return await system.arecommend(q, K)

            return await asyncio.gather(*(one(q) for q in queries))

        recs = run_sync(go())
    else:
        recs = [system.recommend(q, K) for q in queries]
    return [[int(x) for x in r.route_ids[:K]] for r in recs]


def proxy_grade(p: Persona, catalog: Catalog, rid: int) -> int:
    """Estimate mode only: a stand-in grade for an unjudged route (never used in reported numbers)."""
    from evalkit.pool import latent_utility

    if p.constraints().violations(catalog.get(rid)):
        return 0
    u = latent_utility(p, catalog.get(rid))
    return 3 if u >= 0.75 else 2 if u >= 0.45 else 1 if u >= 0.15 else 0


class Grades:
    """Main qrels + simulation qrels; fills holes per round (or proxies them in estimate mode)."""

    def __init__(self, catalog: Catalog, personas: list[Persona], mode: str) -> None:
        self.catalog, self.mode = catalog, mode
        self.pm = {p.persona_id: p for p in personas}
        self.main = Qrels.load()
        self.sim = Qrels.load(sim_qrels_file())
        self.all = Qrels([*self.main.records.values(), *self.sim.records.values()])
        self.proxy: dict[tuple[str, int], int] = {}
        self.counts = {"holes": 0, "rule": 0, "llm_pairs": 0, "llm_calls_est": 0, "reports": []}
        self._bp = {pid: g for pid, g in self.all.by_persona().items() if pid in self.pm}

    def grades(self, pid: str) -> dict[int, int]:
        return self._bp.setdefault(pid, {})

    def fill(self, pairs: set[tuple[str, int]]) -> None:
        holes = [h for h in self.all.holes(pairs) if h not in self.proxy]
        if not holes:
            return
        from evalkit.judge import rule_grade

        llm = [h for h in holes if rule_grade(self.pm[h[0]], self.catalog, h[1]) is None]
        self.counts["holes"] += len(holes)
        self.counts["rule"] += len(holes) - len(llm)
        self.counts["llm_pairs"] += len(llm)
        per_p: dict[str, int] = {}
        for pid, _ in llm:
            per_p[pid] = per_p.get(pid, 0) + 1
        self.counts["llm_calls_est"] += sum(math.ceil(n / 8) for n in per_p.values())
        if self.mode == "proxy":
            for pid, rid in holes:
                self.proxy[(pid, rid)] = g = proxy_grade(self.pm[pid], self.catalog, rid)
                self._bp.setdefault(pid, {})[rid] = g
            return
        if self.mode == "replay":
            raise SystemExit(f"{len(holes)} unjudged simulation pairs (replay mode judges nothing)")
        before = set(self.all.records)
        rep = fill_holes(self.all, holes, self.pm, self.catalog)
        self.counts["reports"].append(rep)
        for k in set(self.all.records) - before:
            self.sim.records[k] = self.all.records[k]
            self._bp.setdefault(k[0], {})[k[1]] = self.all.records[k]["grade"]
        self.sim.save(sim_qrels_file())
        sim_qrels_file().with_suffix(".txt").unlink(missing_ok=True)  # the TREC copy is only for the main pool
        print(f"[simulate] judged {rep}", file=sys.stderr)


def simulate(
    systems: dict[str, tuple[str, dict[str, Any]]],
    personas: list[Persona],
    *,
    mode: str = "judge",
    behaviour: BehaviourModel = BehaviourModel(),
    rounds: int = ROUNDS,
    catalog: Optional[Catalog] = None,
) -> tuple[dict[str, Any], Grades]:
    """Returns ({label: {persona_id: [round records]}}, grades). ``mode``: judge | proxy | replay."""
    catalog = catalog or get_catalog()
    built = {lab: make_system(name, catalog, **ov) for lab, (name, ov) in systems.items()}
    G = Grades(catalog, personas, mode)
    hist: dict[str, dict[str, list[tuple[int, int, str]]]] = {
        lab: {p.persona_id: [] for p in personas} for lab in built
    }
    out: dict[str, dict[str, list[dict[str, Any]]]] = {lab: {p.persona_id: [] for p in personas} for lab in built}
    for t in range(1, rounds + 1):
        shown = {lab: dict(zip([p.persona_id for p in personas],
                               _recommend_all(sysm, [_query(p, hist[lab][p.persona_id], t) for p in personas]),
                               strict=True))
                 for lab, sysm in built.items()}
        G.fill({(pid, r) for lists in shown.values() for pid, ids in lists.items() for r in ids})
        for lab in built:
            for p in personas:
                pid = p.persona_id
                ids = shown[lab][pid]
                g = G.grades(pid)
                consumed = {rid for _, rid, _ in hist[lab][pid]}
                ev = behaviour.react(p, catalog, ids, g, t, frozenset(consumed))
                hist[lab][pid] += [(t, rid, reason) for rid, reason in ev]
                out[lab][pid].append(
                    {
                        "round": t,
                        "ids": ids,
                        "reshown_consumed": sum(1 for r in ids if r in consumed),
                        "events": [[rid, reason] for rid, reason in ev],
                    }
                )
        print(f"[simulate] round {t}/{rounds} done", file=sys.stderr)
    # The metric is computed after the last round, against every judgement made in the run (the ideal ranking
    # of round 1 must not depend on which routes happened to be judged by round 1; a replay sees them all).
    for lab in out:
        for pid, rounds_ in out[lab].items():
            g = G.grades(pid)
            consumed: set[int] = set()
            for r in rounds_:
                r["ndcg_at_10"] = round_ndcg(r["ids"], g, consumed)
                consumed |= {rid for rid, _ in r["events"]}
    return out, G


# --------------------------------------------------------------------------- summary
def summarise(out: dict[str, Any], catalog_size: int, labels: list[str]) -> dict[str, Any]:
    pids = sorted(next(iter(out.values())))
    n_rounds = len(out[labels[0]][pids[0]])
    res: dict[str, Any] = {}
    for lab in labels:
        rows = []
        seen: set[int] = set()
        for t in range(n_rounds):
            nd = [out[lab][pid][t]["ndcg_at_10"] for pid in pids]
            lists = [out[lab][pid][t]["ids"] for pid in pids]
            seen |= {r for ids in lists for r in ids[:K]}
            evs = [e for pid in pids for e in out[lab][pid][t]["events"]]
            rows.append(
                {
                    "round": t + 1,
                    "ndcg_at_10": M.bootstrap_ci(nd, n_boot=N_BOOT, seed=BOOT_SEED),
                    "ndcg_undefined": sum(1 for v in nd if v is None),
                    "coverage_at_10": M.coverage(lists, catalog_size, K),
                    "cumulative_coverage": len(seen) / catalog_size,
                    "completion_rate": sum(1 for e in evs if e[1] == "completed") / len(pids),
                    "rejections_per_persona": sum(1 for e in evs if e[1] != "completed") / len(pids),
                    "reshown_consumed_per_list": float(np.mean([out[lab][pid][t]["reshown_consumed"] for pid in pids])),
                }
            )
        mean_nd = [_mean([out[lab][pid][t]["ndcg_at_10"] for t in range(1, n_rounds)]) for pid in pids]
        reasons: dict[str, int] = {}
        for pid in pids:
            for r in out[lab][pid]:
                for _, reason in r["events"]:
                    reasons[reason] = reasons.get(reason, 0) + 1
        res[lab] = {
            "rounds": rows,
            "ndcg_rounds_2_10": M.bootstrap_ci(mean_nd, n_boot=N_BOOT, seed=BOOT_SEED),
            "completions_total": sum(1 for pid in pids for r in out[lab][pid] for e in r["events"]
                                     if e[1] == "completed") / len(pids),
            "events_by_reason": dict(sorted(reasons.items())),
        }
    return res


def _mean(xs: list[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return float(np.mean(v)) if v else None


def paired(out: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """b - a: per round, and over the mean of rounds 2-10 (paired over personas)."""
    pids = sorted(out[a])
    per_round = []
    for t in range(ROUNDS):
        per_round.append(
            M.paired_bootstrap(
                [out[a][pid][t]["ndcg_at_10"] for pid in pids],
                [out[b][pid][t]["ndcg_at_10"] for pid in pids],
                n_boot=N_BOOT,
                seed=BOOT_SEED,
            )
        )
    ma = [_mean([out[a][pid][t]["ndcg_at_10"] for t in range(1, ROUNDS)]) for pid in pids]
    mb = [_mean([out[b][pid][t]["ndcg_at_10"] for t in range(1, ROUNDS)]) for pid in pids]
    return {"a": a, "b": b, "per_round": per_round,
            "rounds_2_10": M.paired_bootstrap(ma, mb, n_boot=N_BOOT, seed=BOOT_SEED)}


PAIRS = [
    ("recsys_v1", "recsys_v1_fb_route"),
    ("recsys_v1", "recsys_v1_fb"),
    ("recsys_v1_fb_route", "recsys_v1_fb"),
    ("recsys_v1_fb", "recsys_v1_fb_light"),
    ("recsys_v1_fb", "recsys_v1_fb_explore"),
    ("recsys_serve_noexplore", "recsys_serve"),
    ("recsys_v1", "recsys_serve"),
    ("recsys_v1_fb", "recsys_serve_noexplore"),
    ("v1_filters", "recsys_v1_fb"),
    ("v0_rules", "recsys_v1_fb"),
]


def _config(systems) -> dict[str, Any]:
    catalog = get_catalog()
    return {lab: make_system(name, catalog, **ov).config() for lab, (name, ov) in systems.items()}


def run(mode: str = "judge") -> dict[str, Any]:
    personas = [p for p in load_personas() if p.split == "test"]
    out, G = simulate(SIM_SYSTEMS, personas, mode=mode)
    catalog = get_catalog()
    labels = list(SIM_SYSTEMS)
    doc = {
        "schema": "evalkit.simulation/v1",
        "split": "test",
        "n_personas": len(personas),
        "rounds": ROUNDS,
        "days_per_round": DAYS_PER_ROUND,
        "behaviour": BehaviourModel().describe(),
        "systems": {lab: {"system": SIM_SYSTEMS[lab][0], "label": SIM_LABELS[lab]} for lab in labels},
        "config": _config(SIM_SYSTEMS),
        "sim_qrels": {"pairs": len(G.sim), "sha256": G.sim.sha256()},
        "qrels_sha256": G.main.sha256(),
        "metrics": summarise(out, len(catalog), labels),
        "paired": [paired(out, a, b) for a, b in PAIRS],
        "exploration": _exploration(out, len(catalog)),
    }
    return {"doc": doc, "out": out, "counts": G.counts}


EXPLORE_PAIRS = [("recsys_v1_fb", "recsys_v1_fb_explore"), ("recsys_serve_noexplore", "recsys_serve")]


def _exploration(out: dict[str, Any], n_cat: int) -> dict[str, Any]:
    """Coverage gain of the exploration slot per round, for each (without, with) pair that was simulated."""
    res: dict[str, Any] = {}
    for a, b in EXPLORE_PAIRS:
        if a not in out or b not in out:
            continue
        pids = sorted(out[a])
        rows = []
        for t in range(ROUNDS):
            la = [out[a][pid][t]["ids"] for pid in pids]
            lb = [out[b][pid][t]["ids"] for pid in pids]
            rows.append({"round": t + 1, "coverage_gain": M.coverage(lb, n_cat, K) - M.coverage(la, n_cat, K)})
        res[f"{b} - {a}"] = {"a": a, "b": b, "per_round_coverage_gain": rows}
    return res


def write_runs(out: dict[str, Any]) -> None:
    paths.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("w", encoding="utf-8") as fh:
        for lab in out:
            for pid in sorted(out[lab]):
                fh.write(json.dumps({"system": lab, "persona_id": pid, "rounds": out[lab][pid]}, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- tuning (dev)
def tune_personas() -> list[Persona]:
    dev = [p for p in load_personas() if p.split == "dev"]
    rng = np.random.default_rng(TUNE_SEED)
    pick = sorted(rng.choice(len(dev), size=min(TUNE_PERSONAS, len(dev)), replace=False))
    return [dev[i] for i in pick]


def tune(mode: str = "judge") -> dict[str, Any]:
    personas = tune_personas()
    systems = {"control": ("recsys_v1", {})}
    systems.update({name: ("recsys_v1", {"feedback": cfg}) for name, cfg in TUNING.items()})
    out, G = simulate(systems, personas, mode=mode)
    catalog = get_catalog()
    met = summarise(out, len(catalog), list(systems))
    best = max(TUNING, key=lambda n: (met[n]["ndcg_rounds_2_10"]["mean"], -list(TUNING).index(n)))
    doc = {
        "schema": "evalkit.sim_tuning/v1",
        "split": "dev",
        "personas": [p.persona_id for p in personas],
        "grid": {n: c.describe() for n, c in TUNING.items()},
        "rule": "highest mean NDCG@10 over rounds 2-10 on the dev subset",
        "metrics": {n: {"ndcg_rounds_2_10": m["ndcg_rounds_2_10"],
                        "per_round": [r["ndcg_at_10"]["mean"] for r in m["rounds"]],
                        "completions": m["completions_total"]} for n, m in met.items()},
        "vs_control": {n: paired(out, "control", n)["rounds_2_10"] for n in TUNING},
        "vs_route_only": {n: paired(out, "route_only", n)["rounds_2_10"] for n in TUNING if n != "route_only"},
        "best": best,
        "sim_qrels": {"pairs": len(G.sim), "sha256": G.sim.sha256()},
    }
    return {"doc": doc, "counts": G.counts}


# --------------------------------------------------------------------------- figure
def figure(doc: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"v0_rules": "#9e9e9e", "v1_filters": "#6d6875", "recsys_v1": "#2a6f97",
               "recsys_v1_fb_route": "#8ecae6", "recsys_v1_fb": "#e76f51", "recsys_v1_fb_light": "#9b5de5",
               "recsys_serve_noexplore": "#f4a261", "recsys_serve": "#d00000",
               "recsys_v1_fb_explore": "#2a9d8f"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    ax = axes[0]
    for lab, m in doc["metrics"].items():
        xs = [r["round"] for r in m["rounds"]]
        ys = [r["ndcg_at_10"]["mean"] for r in m["rounds"]]
        lo = [r["ndcg_at_10"]["lo"] for r in m["rounds"]]
        hi = [r["ndcg_at_10"]["hi"] for r in m["rounds"]]
        ax.plot(xs, ys, "o-", ms=3, color=colours.get(lab), label=lab)
        ax.fill_between(xs, lo, hi, color=colours.get(lab), alpha=0.12)
    ax.set_xlabel("round (one week apart)")
    ax.set_ylabel("NDCG@10, test (95% CI)")
    ax.set_title("Per-round NDCG@10 (consumed routes: gain 0, out of the ideal)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for lab, m in doc["metrics"].items():
        ax.plot([r["round"] for r in m["rounds"]], [100 * r["cumulative_coverage"] for r in m["rounds"]], "o-",
                ms=3, color=colours.get(lab), label=lab)
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative coverage@10 (% of catalogue)")
    ax.set_title("Distinct routes shown so far, 100 test personas", fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Simulated users and a simulated behaviour model; LLM-judged relevance (judge not yet "
             "validated)", fontsize=6, color="#555")
    fig.tight_layout()
    fig.savefig(FIGURE, metadata={"Software": None})
    plt.close(fig)


# --------------------------------------------------------------------------- replay
def replay(check: bool) -> tuple[list[str], list[str]]:
    """Frozen replay: re-run the simulation (and the tuning) without judging; compare with the files."""
    from evalkit.evaluate import system_client

    keys: list[str] = []
    probs: list[str] = []
    client = system_client()
    for fname, fn in ((OUT, run), (TUNE_OUT, tune)):
        f = paths.RESULTS_DIR / fname
        if not f.exists():
            continue
        client.reset_touched()
        doc = fn(mode="replay")["doc"]
        keys += client.touched_keys()
        if check:
            stored = json.loads(f.read_text(encoding="utf-8"))
            stored.pop("volatile", None)
            if json.loads(dumps(doc)) != stored:
                probs.append(f"{fname} differs")
    return keys, probs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "estimate", "tune"])
    ap.add_argument("--estimate", action="store_true", help="with tune: proxy grades, count only")
    args = ap.parse_args(argv)
    if args.cmd == "estimate" or (args.cmd == "tune" and args.estimate):
        res = (tune if args.cmd == "tune" else run)(mode="proxy")
        c = res["counts"]
        c.pop("reports")
        c["est_judge_usd"] = round(c["llm_calls_est"] * 0.0025 + c["llm_pairs"] * 0.0009, 2)
        print(json.dumps(c, indent=1))
        return
    if args.cmd == "tune":
        res = tune()
        (paths.RESULTS_DIR / TUNE_OUT).write_text(dumps(res["doc"]), encoding="utf-8")
        print(json.dumps({n: m["ndcg_rounds_2_10"]["mean"] for n, m in res["doc"]["metrics"].items()}, indent=1))
        print("best:", res["doc"]["best"])
        return
    res = run()
    doc = res["doc"]
    write_runs(res["out"])
    full = {**doc, "volatile": {"judging": res["counts"]}}
    (paths.RESULTS_DIR / OUT).write_text(dumps(full), encoding="utf-8")
    figure(doc)
    for lab, m in doc["metrics"].items():
        print(f"{lab}: " + " ".join(f"{r['ndcg_at_10']['mean']:.3f}" for r in m["rounds"]))


if __name__ == "__main__":
    main()
