"""The multi-stage recommender (Phase 3): hard filter -> candidate generation -> ranking -> diversity.

    pipe = Pipeline(catalog, PipelineConfig(...), doc_embeddings=..., ltr_model=..., llm=client)
    rec = pipe.recommend(query, k)            # sync (scripts, offline evaluation)
    rec = await pipe.arecommend(query, k)     # async (FastAPI handlers)

Every stage can be switched by config, and every stage writes its per-route contribution into the
trace (``explain.py`` defines the schema). Design and measurements: docs/RECSYS.md.

Stages, in order:

1. ``filter`` (``filters``): category tab, radius from home, distance cap (``filters.py``).
2. candidate generation over the eligible set with the intent as query: ``bm25`` and/or ``dense``
   (exact cosine), each keeping its top ``gen_depth``; ``fusion`` = RRF when both run. The candidate
   list is the top ``cand_depth`` (50) of the generator or of the fused list.
   Without generators the candidates are the whole eligible set (``ranker="v0"``: the baseline scorer).
3. ranking: ``none`` (keep the candidate order), ``v0`` (rule scorer), ``api`` (semantic ranker through
   the LLM client) or ``ltr`` (learned ranker; it calls the ranker API only if its model uses that feature).
4. ``feedback`` (Phase 5, if ``feedback`` is configured and the query carries events): the slots are
   adjusted once from the original query before stage 1, the intent embedding moves by Rocchio before
   dense retrieval, and this stage removes recently completed / rejected routes and re-scores the rest
   (``feedback.py``).
5. ``mmr`` re-ranking if ``mmr_lambda`` is set.
6. ``explore`` (Phase 5, if ``explore`` is configured): one reserved slot filled by Thompson sampling over
   tag clusters while the user has little feedback (``explore.py``).

**No intent** (questionnaire-only request): BM25, dense, fusion, the ranker API and the learned ranker
need a text query, so they are skipped and the candidates are the eligible set ordered by the rule
scorer (``v0``); MMR still applies. The trace says ``"fallback": "no_intent"``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

import numpy as np

from .bm25 import BM25Config, get_index, matched_terms
from .catalog import Catalog
from .constraints import CATEGORY_TABS, HardConstraints
from .dense import DOC_EMBEDDINGS_FILE, DocEmbeddings, abuild_doc_embeddings, build_doc_embeddings
from .explore import ExploreConfig, tag_clusters
from .explore import pick as explore_pick
from .features import FEATURES, build_features, parse_intent, stage_arrays
from .feedback import NEGATIVE_REASONS, FeedbackConfig, FeedbackState, adjust_query, rescore, rocchio
from .filters import FilterConfig, FilterResult, apply_filters
from .fusion import rrf
from .ltr import TreeEnsemble, load_model
from .mmr import mmr as mmr_select
from .query import UserQuery
from .result import Recommendation, StageTrace
from .text import ranker_text

RANKERS = ("none", "v0", "api", "ltr")
GENERATORS = ("bm25", "dense")


def _rule_contribution(entry: Any) -> dict[str, Any]:
    """The rule scorer's three weighted sub-scores for one route, as reported in the stage trace."""
    out = {
        "difficulty": round(entry.breakdown["difficulty"]["weighted_score"], 6),
        "distance": round(entry.breakdown["distance"]["weighted_score"], 6),
        "tags": round(entry.breakdown["tags"]["weighted_score"], 6),
        "total": round(entry.score, 6),
    }
    if "feedback_penalty" in entry.breakdown:
        out["feedback_penalty"] = entry.breakdown["feedback_penalty"]
    return out


@dataclass(frozen=True)
class PipelineConfig:
    filters: bool = True
    filter: FilterConfig = field(default_factory=FilterConfig)
    generators: tuple[str, ...] = ("bm25", "dense")
    rrf_k: int = 30
    gen_depth: int = 100
    cand_depth: int = 50
    ranker: str = "none"
    bm25: BM25Config = field(default_factory=BM25Config)
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    rank_model: str = "mongodb/rerank-3"  # chosen by the dev pilot (eval/results/_rank_pilot.json)
    rank_doc: str = "facts_desc"
    ltr_model: Optional[str] = None
    mmr_lambda: Optional[float] = None
    feedback: Optional[FeedbackConfig] = None  # None = feedback ignored (v0 ranker: its own mechanism)
    explore: Optional[ExploreConfig] = None

    def __post_init__(self) -> None:
        if self.ranker not in RANKERS:
            raise ValueError(f"ranker must be one of {RANKERS}")
        if any(g not in GENERATORS for g in self.generators):
            raise ValueError(f"generators must be a subset of {GENERATORS}")
        if self.ranker == "ltr" and not self.ltr_model:
            raise ValueError("ranker 'ltr' needs ltr_model")
        if self.ranker in ("api", "ltr") and not self.generators:
            raise ValueError(f"ranker {self.ranker!r} needs a candidate generator")
        if self.mmr_lambda is not None and not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be in [0, 1]")

    def with_(self, **changes: Any) -> PipelineConfig:
        return replace(self, **changes)

    def describe(self) -> dict[str, Any]:
        d = asdict(self)
        d["generators"] = list(self.generators)
        d["bm25"] = self.bm25.describe()
        d["feedback"] = self.feedback.describe() if self.feedback else None
        d["explore"] = self.explore.describe() if self.explore else None
        return d


@dataclass
class _Retrieval:
    hc: HardConstraints
    fr: FilterResult
    elig: np.ndarray
    intent: str
    traces: list[StageTrace]
    stage: dict[str, np.ndarray]
    per_term: dict[str, np.ndarray]
    lists: dict[str, list[int]]
    fused_inputs: dict[int, dict[str, int]]
    cand: Optional[list[int]]
    fallback: Optional[str]


def _committed_doc_meta() -> dict[str, Any]:
    if not DOC_EMBEDDINGS_FILE.exists():
        return {}
    import hashlib
    import json
    import zipfile

    with zipfile.ZipFile(DOC_EMBEDDINGS_FILE) as zf, zf.open("meta.npy") as fh:
        meta = json.loads(str(np.load(fh, allow_pickle=False)))
    meta["sha256"] = hashlib.sha256(DOC_EMBEDDINGS_FILE.read_bytes()).hexdigest()[:16]
    return meta


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks by descending score; ties by position (stable)."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


class Pipeline:
    """One configured pipeline. Thread-safe for concurrent requests (no per-request state on self)."""

    def __init__(
        self,
        catalog: Catalog,
        config: PipelineConfig,
        *,
        name: str = "pipeline",
        doc_embeddings: Optional[DocEmbeddings] = None,
        ltr_model: Optional[TreeEnsemble] = None,
        llm=None,
    ) -> None:
        self.catalog = catalog
        self.cfg = config
        self.name = name
        self._llm = llm
        self._docs = doc_embeddings.aligned_to(catalog) if doc_embeddings is not None else None
        self._docs_explicit = doc_embeddings is not None
        self._ltr = ltr_model
        if config.ranker == "ltr" and self._ltr is None:
            self._ltr = load_model(config.ltr_model)
        if self._ltr is not None:
            unknown = set(self._ltr.feature_names) - set(FEATURES)
            if unknown:
                raise ValueError(f"model uses unknown features {sorted(unknown)}")
        self._bm25 = get_index(catalog, config.bm25) if "bm25" in config.generators else None

    # ------------------------------------------------------------ resources
    @property
    def llm(self):
        if self._llm is None:
            from app.llm import get_client

            self._llm = get_client()
        return self._llm

    def _provider(self, cap: str) -> str:
        return self.llm.config.provider_for(cap)

    @property
    def needs_embeddings(self) -> bool:
        return "dense" in self.cfg.generators or self.cfg.mmr_lambda is not None

    @property
    def uses_rank_api(self) -> bool:
        if self.cfg.ranker == "api":
            return True
        return self.cfg.ranker == "ltr" and self._ltr is not None and "rank_api" in self._ltr.feature_names

    def doc_embeddings(self) -> DocEmbeddings:
        """The committed matrix; with the mock embedding provider (tests, CI) a mock matrix built in memory."""
        if self._docs is None:
            if self._provider("embed") == "mock":
                self._docs = self.catalog.memo(
                    "dense:mock-doc-embeddings",
                    lambda: build_doc_embeddings(
                        self.catalog,
                        self.llm,
                        formulation="title_desc",
                        model=None,
                        dim=self.cfg.embed_dim,
                        tag="doc-embed",
                    ),
                )
            else:
                self._docs = self.catalog.memo(
                    f"dense:{DOC_EMBEDDINGS_FILE}",
                    lambda: DocEmbeddings.load(DOC_EMBEDDINGS_FILE).aligned_to(self.catalog),
                )
        d = self._docs
        prov = self._provider("embed")
        if prov != "mock" and (d.meta.get("provider") != prov or d.meta.get("model") != self.cfg.embed_model):
            raise ValueError(
                f"document embeddings are {d.meta.get('provider')}:{d.meta.get('model')}, "
                f"but queries would be embedded with {prov}:{self.cfg.embed_model}"
            )
        if d.dim != self.cfg.embed_dim:
            raise ValueError(f"document embeddings have dim {d.dim}, config says {self.cfg.embed_dim}")
        return d

    async def _adoc_embeddings(self) -> DocEmbeddings:
        """Async variant for use inside the event loop (the mock matrix is built with an awaited embed call)."""
        if self._docs is None and self._provider("embed") == "mock":
            key = "dense:mock-doc-embeddings"
            if not self.catalog.has_memo(key):
                d = await abuild_doc_embeddings(
                    self.catalog,
                    self.llm,
                    formulation="title_desc",
                    model=None,
                    dim=self.cfg.embed_dim,
                    tag="doc-embed",
                )
                self.catalog.memo(key, lambda: d)
        return self.doc_embeddings()

    def config(self) -> dict[str, Any]:
        c = self.cfg.describe()
        models = []
        out: dict[str, Any] = {"system": self.name, "pipeline": c}
        if self.needs_embeddings:
            models.append(f"embed:{self.cfg.embed_model}")
            if self._docs_explicit:
                meta = self._docs.meta
            elif self._provider("embed") == "mock":
                meta = {
                    "provider": "mock",
                    "formulation": "title_desc",
                    "dim": self.cfg.embed_dim,
                    "precision": "float32",
                }
            else:
                meta = _committed_doc_meta()
            out["doc_embeddings"] = {
                k: meta.get(k)
                for k in ("provider", "model", "dim", "formulation", "precision", "text_version", "sha256")
            }
        if self.uses_rank_api:
            models.append(f"rank:{self.cfg.rank_model}")
        if self._ltr is not None:
            models.append(f"ltr:{self._ltr.name}")
            out["ltr"] = {
                "name": self._ltr.name,
                "features": self._ltr.feature_names,
                "trees": self._ltr.n_trees,
                "trained": self._ltr.meta.get("trained_on"),
            }
        out["models"] = models
        return out

    # ------------------------------------------------------------ entry points
    def recommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        from app.llm import run_sync

        return run_sync(self.arecommend(query, k))

    async def _retrieve(self, query: UserQuery, fb: Optional[FeedbackState] = None) -> _Retrieval:
        """Stages 1-2: filter, generators, fusion. ``cand`` = eligible-set positions, or None (rule fallback)."""
        cfg = self.cfg
        cat = self.catalog
        traces: list[StageTrace] = []
        intent = (query.intent or "").strip()

        # 1. filter ------------------------------------------------------
        t0 = time.perf_counter()
        hc = HardConstraints.from_query(query)
        if cfg.filters:
            fr = apply_filters(cat, hc, cfg.filter)
        else:
            tab = np.isin(cat.category_name, list(CATEGORY_TABS.get(query.category, ()))) if query.category else None
            idx = np.flatnonzero(tab) if tab is not None else np.arange(len(cat))
            fr = FilterResult(idx, hc.home_distance_km(cat), None, {"applied": "category tab only (no filter)"})
        elig = fr.indices
        traces.append(StageTrace("filter", len(cat), len(elig), (time.perf_counter() - t0) * 1000, info=dict(fr.info)))

        use_text = bool(intent) and bool(cfg.generators)
        embed_task = None
        if use_text and "dense" in cfg.generators and len(elig):
            embed_task = asyncio.ensure_future(self._embed_query(intent))

        n_e = len(elig)
        stage = {k_: np.full(n_e, np.nan) for k_ in ("bm25", "bm25_rank", "dense", "dense_rank", "rrf", "fused_rank")}
        per_term: dict[str, np.ndarray] = {}
        lists: dict[str, list[int]] = {}

        # 2a. BM25 --------------------------------------------------------
        if use_text and "bm25" in cfg.generators:
            t0 = time.perf_counter()
            terms = self._bm25.query_terms(intent)
            allsc, per_term = self._bm25.score_all(terms)
            sc = allsc[elig]
            rk = _ranks(sc)
            stage["bm25"], stage["bm25_rank"] = sc, rk.astype(float)
            lists["bm25"] = list(np.argsort(-sc, kind="stable")[: cfg.gen_depth])
            traces.append(
                StageTrace(
                    "bm25",
                    n_e,
                    len(lists["bm25"]),
                    (time.perf_counter() - t0) * 1000,
                    info={"terms": terms, "matched_routes": int((sc > 0).sum())},
                )
            )

        # 2b. dense -------------------------------------------------------
        if embed_task is not None:
            qv, up_lat, up_cost = await embed_task
            t0 = time.perf_counter()
            docs = await self._adoc_embeddings()
            rocchio_info = None
            if fb is not None:
                qv, rocchio_info = rocchio(qv, docs.matrix, fb)
            sims = docs.cosine(qv, elig)
            stage["dense"], stage["dense_rank"] = sims.astype(float), _ranks(sims).astype(float)
            lists["dense"] = list(np.argsort(-sims, kind="stable")[: cfg.gen_depth])
            traces.append(
                StageTrace(
                    "dense",
                    n_e,
                    len(lists["dense"]),
                    (time.perf_counter() - t0) * 1000,
                    up_lat,
                    up_cost,
                    info={
                        "exact": True,
                        "dim": docs.dim,
                        "formulation": docs.meta.get("formulation"),
                        **({"rocchio": rocchio_info} if rocchio_info else {}),
                    },
                )
            )

        # 2c. candidates (fusion / single generator / rule fallback) -----
        fused_inputs: dict[int, dict[str, int]] = {}
        if use_text and len(lists) == 2:
            t0 = time.perf_counter()
            fused = rrf(lists, k=cfg.rrf_k)
            for pos, (j, s, ins) in enumerate(fused, start=1):
                stage["rrf"][j], stage["fused_rank"][j] = s, pos
                fused_inputs[j] = ins
            cand = [j for j, _, _ in fused[: cfg.cand_depth]]
            traces.append(
                StageTrace(
                    "fusion",
                    sum(len(v) for v in lists.values()),
                    len(cand),
                    (time.perf_counter() - t0) * 1000,
                    info={"method": "rrf", "k": cfg.rrf_k},
                )
            )
        elif use_text and lists:
            ((gname, gl),) = lists.items()
            cand = gl[: cfg.cand_depth]
        else:
            cand = None  # rule scorer over the eligible set
        fallback = None
        if cand is None and cfg.generators and not intent:
            fallback = "no_intent"

        return _Retrieval(hc, fr, elig, intent, traces, stage, per_term, lists, fused_inputs, cand, fallback)

    async def arecommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        cfg = self.cfg
        cat = self.catalog
        contrib: dict[str, dict[int, Any]] = {}
        fb: Optional[FeedbackState] = None
        if cfg.feedback is not None and query.feedback:
            t0 = time.perf_counter()
            fb = adjust_query(cat, query, cfg.feedback)  # once, from the original query
            query = fb.query
            fb_ms = (time.perf_counter() - t0) * 1000
        r = await self._retrieve(query, fb)
        hc, fr, elig, intent, traces = r.hc, r.fr, r.elig, r.intent, r.traces
        stage, per_term, lists, fused_inputs, cand, fallback = (
            r.stage,
            r.per_term,
            r.lists,
            r.fused_inputs,
            r.cand,
            r.fallback,
        )

        # 3. ranking -------------------------------------------------------
        ranker = cfg.ranker if cand is not None else "v0"
        rule_contrib: dict[int, Any] = {}
        if cand is None or ranker == "v0":
            t0 = time.perf_counter()
            pool = elig if cand is None else elig[np.array(cand, dtype=np.int64)]
            depth = max(k, cfg.cand_depth) if (cfg.mmr_lambda is not None or fb is not None or cfg.explore) else k
            ranked = self._v0(query, pool, limit=depth if cand is None else None)
            pos_of = {int(cat.ids[i]): j for j, i in enumerate(elig)}
            cand = [pos_of[e.route_id] for e in ranked]
            order_scores = [float(e.score) for e in ranked]
            for e in ranked:
                rule_contrib[pos_of[e.route_id]] = _rule_contribution(e)
            traces.append(
                StageTrace(
                    "v0_rules",
                    len(pool),
                    len(cand),
                    (time.perf_counter() - t0) * 1000,
                    info={"fallback": fallback} if fallback else {},
                )
            )
            final_scores = np.array(order_scores)
        else:
            cand_arr = np.array(cand, dtype=np.int64)
            if "fusion" in [t.stage for t in traces]:
                final_scores = stage["rrf"][cand_arr]
            elif cand:
                final_scores = stage["bm25" if "bm25" in lists else "dense"][cand_arr]
            else:
                final_scores = np.array([])
            api_scores = np.full(len(cand), np.nan)
            if self.uses_rank_api and cand:
                scores, up_lat, up_cost, t_local = await self._rank_api(intent, elig[cand_arr])
                api_scores = scores
                traces.append(
                    StageTrace(
                        "rank_api",
                        len(cand),
                        len(cand),
                        t_local,
                        up_lat,
                        up_cost,
                        info={"model": self.cfg.rank_model, "doc": self.cfg.rank_doc},
                    )
                )
                if ranker == "api":
                    order = np.argsort(-api_scores, kind="stable")
                    cand = [cand[o] for o in order]
                    final_scores = api_scores[order]
                    api_scores = final_scores
            ltr_info = None
            if ranker == "ltr" and cand:
                t0 = time.perf_counter()
                ca = np.array(cand, dtype=np.int64)
                st = stage_arrays(len(cand))
                for key_ in ("bm25", "bm25_rank", "dense", "dense_rank", "rrf", "fused_rank"):
                    st[key_] = stage[key_][ca]
                st["rank_api"] = api_scores
                X = build_features(cat, query, elig[ca], st, fr.home_km, parse_intent(intent))
                cols = [FEATURES.index(f) for f in self._ltr.feature_names]
                Xm = X[:, cols]
                pred, C, bias = self._ltr.predict_contrib(Xm)
                order = np.argsort(-pred, kind="stable")
                ltr_info = {}
                for rnk, o in enumerate(order, start=1):
                    feats = [
                        {
                            "name": f,
                            "value": None if np.isnan(Xm[o, j]) else round(float(Xm[o, j]), 6),
                            "contribution": round(float(C[o, j]), 6),
                        }
                        for j, f in enumerate(self._ltr.feature_names)
                    ]
                    feats.sort(key=lambda d: (-abs(d["contribution"]), d["name"]))
                    ltr_info[cand[o]] = {
                        "score": round(float(pred[o]), 6),
                        "rank": rnk,
                        "bias": round(bias, 6),
                        "model": self._ltr.name,
                        "features": feats,
                    }
                api_scores = api_scores[order]
                cand = [cand[o] for o in order]
                final_scores = pred[order]
                traces.append(
                    StageTrace(
                        "ltr",
                        len(order),
                        len(order),
                        (time.perf_counter() - t0) * 1000,
                        info={"model": self._ltr.name, "trees": self._ltr.n_trees},
                    )
                )
            if self.uses_rank_api and cand:
                api_rank = _ranks(np.nan_to_num(api_scores, nan=-np.inf))
                contrib["rank_api"] = {
                    cand[j]: {
                        "score": round(float(api_scores[j]), 6),
                        "rank": int(api_rank[j]),
                        "model": self.cfg.rank_model,
                    }
                    for j in range(len(cand))
                    if not np.isnan(api_scores[j])
                }
            if ltr_info:
                contrib["ltr"] = ltr_info

        # 4. feedback ----------------------------------------------------
        fb_contrib: dict[int, Any] = {}
        if fb is not None and cand:
            t0 = time.perf_counter()
            docs = await self._adoc_embeddings()
            n_before = len(cand)
            kept, final_scores, c_by_idx, fb_info = rescore(
                cat, [int(elig[j]) for j in cand], final_scores, fb, docs.matrix
            )
            pos_of = {int(i): j for j, i in enumerate(elig)}
            cand = [pos_of[i] for i in kept]
            fb_contrib = {pos_of[i]: c for i, c in c_by_idx.items()}
            traces.append(
                StageTrace(
                    "feedback",
                    n_before,
                    len(cand),
                    (time.perf_counter() - t0) * 1000 + fb_ms,
                    info={**fb.info(), **fb_info},
                )
            )

        # 5. MMR ---------------------------------------------------------
        mmr_contrib: dict[int, Any] = {}
        if cfg.mmr_lambda is not None and len(cand) > 1:
            t0 = time.perf_counter()
            docs = await self._adoc_embeddings()
            E = docs.matrix[elig[np.array(cand, dtype=np.int64)]]
            picks, det = mmr_select(final_scores, E @ E.T, cfg.mmr_lambda)
            for after, (p, d) in enumerate(zip(picks, det, strict=True), start=1):
                mmr_contrib[cand[p]] = {"lambda": cfg.mmr_lambda, **d, "rank_before": p + 1, "rank_after": after}
            final_scores = np.array([det_["mmr_score"] for det_ in det])
            cand = [cand[p] for p in picks]
            traces.append(
                StageTrace(
                    "mmr",
                    len(picks),
                    len(picks),
                    (time.perf_counter() - t0) * 1000,
                    info={"lambda": cfg.mmr_lambda, "similarity": "document-embedding cosine"},
                )
            )

        # 6. exploration -------------------------------------------------
        explore_contrib: dict[int, Any] = {}
        if cfg.explore is not None and cand and k >= cfg.explore.position:
            t0 = time.perf_counter()
            events = fb.events if fb is not None else []
            if cfg.feedback is None and query.feedback:  # posteriors still need the events' weights
                from .feedback import weigh

                events = weigh(cat, query.feedback, FeedbackConfig())
            sim = stage["dense"] if not np.all(np.isnan(stage["dense"])) else None
            choice = explore_pick(
                cat,
                tag_clusters(cat, cfg.explore),
                cfg.explore,
                request_id=query.request_id,
                events=events,
                ranked_idx=[int(elig[j]) for j in cand],
                eligible_idx=elig,
                excluded_ids=set(fb.excluded) if fb is not None else set(),
                similarity=sim,
            )
            info: dict[str, Any] = {"active": choice is not None, "events": len(events)}
            if choice is not None:
                pos_of = {int(i): j for j, i in enumerate(elig)}
                j = pos_of[choice.pop("index")]
                p = cfg.explore.position - 1
                scores_l = list(np.asarray(final_scores, dtype=float))
                own = scores_l[cand.index(j)] if j in cand else float(min(scores_l[:p] or [0.0]))
                if j in cand:
                    del scores_l[cand.index(j)]
                    cand.remove(j)
                cand.insert(p, j)
                scores_l.insert(p, own)
                final_scores = np.array(scores_l)
                explore_contrib[j] = choice
                if rule_contrib and j not in rule_contrib:
                    # the pick can come from beyond the rule scorer's top list: explain it like the others
                    rule_contrib[j] = _rule_contribution(self._v0(query, np.array([elig[j]]), limit=None)[0])
                info.update({"cluster": choice["cluster"], "label": choice["label"], "position": p + 1})
            traces.append(StageTrace("explore", len(cand), len(cand), (time.perf_counter() - t0) * 1000, info=info))

        # output + per-route contributions -------------------------------
        cand = cand[:k]
        final_scores = np.asarray(final_scores, dtype=float)[:k]
        out_ids = [int(cat.ids[elig[j]]) for j in cand]
        by_stage: dict[str, dict[int, Any]] = {t.stage: {} for t in traces}
        for j, rid in zip(cand, out_ids, strict=True):
            i = int(elig[j])
            if cfg.filters:
                by_stage["filter"][rid] = fr.verdict(cat, hc, i)
            if "bm25" in by_stage:
                by_stage["bm25"][rid] = {
                    "score": round(float(stage["bm25"][j]), 6),
                    "rank": int(stage["bm25_rank"][j]),
                    "matched_terms": matched_terms(per_term, i),
                }
            if "dense" in by_stage:
                by_stage["dense"][rid] = {
                    "similarity": round(float(stage["dense"][j]), 6),
                    "rank": int(stage["dense_rank"][j]),
                }
            if "fusion" in by_stage and not np.isnan(stage["rrf"][j]):  # an explored route may not be fused
                by_stage["fusion"][rid] = {
                    "rrf_score": round(float(stage["rrf"][j]), 6),
                    "rank": int(stage["fused_rank"][j]),
                    "k": cfg.rrf_k,
                    "inputs": fused_inputs.get(j, {}),
                }
            if "v0_rules" in by_stage and j in rule_contrib:
                by_stage["v0_rules"][rid] = rule_contrib[j]
            for sname in ("rank_api", "ltr"):
                if sname in contrib and j in contrib[sname]:
                    by_stage[sname][rid] = contrib[sname][j]
            if j in mmr_contrib:
                by_stage["mmr"][rid] = mmr_contrib[j]
            if j in fb_contrib:
                by_stage["feedback"][rid] = fb_contrib[j]
            if j in explore_contrib:
                by_stage["explore"][rid] = explore_contrib[j]
        for t in traces:
            t.contrib = by_stage.get(t.stage, {})
        if fallback:
            traces[0].info["fallback"] = fallback
        return Recommendation(query.request_id, self.name, out_ids, [round(float(s), 6) for s in final_scores], traces)

    async def afeatures(self, query: UserQuery, *, with_api: bool = True) -> tuple[list[int], np.ndarray]:
        """Candidate route ids and their full feature matrix (over ``FEATURES``), as the learned ranker sees
        them; used to build training data. ``with_api`` adds the ranker-API score (one rank call)."""
        r = await self._retrieve(query)
        if not r.cand:
            return [], np.zeros((0, len(FEATURES)))
        ca = np.array(r.cand, dtype=np.int64)
        st = stage_arrays(len(ca))
        for key_ in ("bm25", "bm25_rank", "dense", "dense_rank", "rrf", "fused_rank"):
            st[key_] = r.stage[key_][ca]
        if with_api:
            st["rank_api"] = (await self._rank_api(r.intent, r.elig[ca]))[0]
        X = build_features(self.catalog, query, r.elig[ca], st, r.fr.home_km, parse_intent(r.intent))
        return [int(self.catalog.ids[r.elig[j]]) for j in ca], X

    # ------------------------------------------------------------ stage helpers
    def _v0(self, query: UserQuery, pool: np.ndarray, limit: Optional[int]):
        from datetime import timedelta

        from app.services.recommendation_service import rank_routes_v0

        from .systems.baselines import REFERENCE_NOW

        routes = [{"id": int(self.catalog.ids[i]), **self.catalog.routes[int(i)].v0_vector()} for i in pool]
        # v0's own mechanism knows only the four negative reasons (completions are not v0 feedback)
        fb = [
            {"route_id": f.route_id, "reason": f.reason, "created_at": REFERENCE_NOW - timedelta(days=f.age_days)}
            for f in query.feedback
            if f.reason in NEGATIVE_REASONS
        ]
        return rank_routes_v0(query.user_vector, routes, fb, now=REFERENCE_NOW, limit=limit)

    async def _embed_query(self, intent: str) -> tuple[np.ndarray, float, float]:
        prov = self._provider("embed")
        res = await self.llm.embed(
            [intent],
            task="query",
            model=None if prov == "mock" else self.cfg.embed_model,
            dim=self.cfg.embed_dim,
            tag="query-embed",
        )
        return np.asarray(res.vectors[0], dtype=np.float32), float(res.nominal_latency_ms), float(res.nominal_cost_usd)

    async def _rank_api(self, intent: str, cand_idx: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        t0 = time.perf_counter()
        docs = [ranker_text(self.catalog.routes[int(i)], self.cfg.rank_doc) for i in cand_idx]
        t_local = (time.perf_counter() - t0) * 1000
        prov = self._provider("rank")
        res = await self.llm.rank(intent, docs, model=None if prov == "mock" else self.cfg.rank_model, tag="rerank")
        return np.asarray(res.scores, dtype=float), float(res.nominal_latency_ms), float(res.nominal_cost_usd), t_local
