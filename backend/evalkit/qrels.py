"""Judged relevance (``eval/qrels/qrels.jsonl``): one line per (persona, route).

Each line: ``persona_id``, ``route_id``, ``grade`` (0-3), ``source`` (``rule`` or ``llm``), ``reason``,
``judge`` (config name), ``model`` (id reported by the provider), ``rubric``; LLM lines also record the
batch they were graded in (``batch_id``, ``batch_pos``, ``batch_ids``) so the exact calls can be replayed.
The file is sorted and written deterministically. A TREC-style ``qrels.txt`` is written next to it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from app.recsys import Catalog
from evalkit import paths
from evalkit.personas import Persona


class Qrels:
    def __init__(self, records: Iterable[dict[str, Any]] = ()) -> None:
        self.records: dict[tuple[str, int], dict[str, Any]] = {}
        for r in records:
            self.records[(r["persona_id"], int(r["route_id"]))] = r

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Qrels:
        path = path or paths.QRELS_FILE
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as fh:
            return cls(json.loads(line) for line in fh if line.strip())

    def _lines(self) -> str:
        return "".join(
            json.dumps(self.records[k], ensure_ascii=False, sort_keys=True) + "\n" for k in sorted(self.records)
        )

    def save(self, path: Optional[Path] = None) -> str:
        path = path or paths.QRELS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._lines()
        path.write_text(data, encoding="utf-8")
        trec = "".join(f"{pid} 0 {rid} {self.records[(pid, rid)]['grade']}\n" for pid, rid in sorted(self.records))
        path.with_suffix(".txt").write_text(trec, encoding="utf-8")
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def sha256(self) -> str:
        return hashlib.sha256(self._lines().encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, key: tuple[str, int]) -> bool:
        return (key[0], int(key[1])) in self.records

    def add(self, recs: Iterable[dict[str, Any]]) -> int:
        n = 0
        for r in recs:
            k = (r["persona_id"], int(r["route_id"]))
            if k not in self.records:
                self.records[k] = r
                n += 1
        return n

    def grades(self, persona_id: str) -> dict[int, int]:
        return {rid: r["grade"] for (pid, rid), r in self.records.items() if pid == persona_id}

    def by_persona(self) -> dict[str, dict[int, int]]:
        out: dict[str, dict[int, int]] = {}
        for (pid, rid), r in self.records.items():
            out.setdefault(pid, {})[rid] = r["grade"]
        return out

    def holes(self, pairs: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
        return sorted({(p, int(r)) for p, r in pairs if (p, int(r)) not in self.records})

    def stats(self, personas: Optional[list[Persona]] = None) -> dict[str, Any]:
        recs = list(self.records.values())
        n = len(recs)
        out: dict[str, Any] = {
            "pairs": n,
            "personas": len({r["persona_id"] for r in recs}),
            "by_source": dict(sorted(Counter(r["source"] for r in recs).items())),
            "rule_share": round(sum(r["source"] == "rule" for r in recs) / n, 4) if n else None,
            "grades_all": {str(g): c for g, c in sorted(Counter(r["grade"] for r in recs).items())},
            "grades_llm": {
                str(g): c for g, c in sorted(Counter(r["grade"] for r in recs if r["source"] == "llm").items())
            },
            "rules": dict(sorted(Counter(r.get("rule") for r in recs if r["source"] == "rule").items())),
            "judges": dict(sorted(Counter(r["judge"] for r in recs if r["source"] == "llm").items())),
            "models": dict(sorted(Counter(r["model"] for r in recs if r["source"] == "llm").items())),
            "rubrics": dict(sorted(Counter(r["rubric"] for r in recs).items())),
        }
        rel = Counter(r["persona_id"] for r in recs if r["grade"] >= 2)
        pids = [p.persona_id for p in personas] if personas else sorted({r["persona_id"] for r in recs})
        counts = sorted(rel.get(p, 0) for p in pids)
        if counts:
            out["relevant_per_persona"] = {
                "min": counts[0],
                "median": counts[len(counts) // 2],
                "max": counts[-1],
                "personas_without_relevant": sum(1 for c in counts if c == 0),
            }
        return out


def fill_holes(
    qrels: Qrels,
    pairs: Iterable[tuple[str, int]],
    personas: dict[str, Persona],
    catalog: Catalog,
    *,
    judge_name: Optional[str] = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Grade every pair not yet in ``qrels``: hard-constraint violations by rule, the rest with the LLM judge."""
    from evalkit.judge import JUDGES, judge_pairs, production_judge, rule_grade

    holes = qrels.holes(pairs)
    rule_recs, llm_pairs = [], []
    for pid, rid in holes:
        rec = rule_grade(personas[pid], catalog, rid)
        if rec is None:
            llm_pairs.append((pid, rid))
        else:
            rule_recs.append(rec)
    qrels.add(rule_recs)
    report: dict[str, Any] = {"holes": len(holes), "rule_graded": len(rule_recs), "llm_pairs": len(llm_pairs)}
    if llm_pairs:
        cfg = JUDGES[judge_name] if judge_name else production_judge()
        out = judge_pairs(cfg, catalog, personas, llm_pairs, seed=seed, tag="judge")
        qrels.add(out.records)
        report.update(
            {"llm_calls": out.calls, "llm_failures": out.failures, "nominal_cost_usd": round(out.nominal_cost_usd, 6)}
        )
    return report
