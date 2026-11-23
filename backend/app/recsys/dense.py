"""Document embedding matrix and exact cosine search.

The catalogue matrix is committed in reduced precision (``artifacts/doc_embeddings.npz``): float16,
half of float32. int8 with a per-row scale is also supported (a quarter of float32) but reorders more
near-ties (eval/results/_cg_dev.json). On load the matrix is converted to float32 and re-normalised;
cosine = dot product. Exact search over the eligible rows is a single matrix-vector product
(docs/RECSYS.md has the measured time and the ANN threshold).

File layout (npz): ``f16`` float16 (n, d) (or ``q`` int8 (n, d) + ``scale`` float32 (n,)), ``ids`` int64
(n,), ``meta`` (JSON string: provider, model, dim, task, formulation, text version, catalogue sha256,
precision).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import Catalog

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DOC_EMBEDDINGS_FILE = ARTIFACTS_DIR / "doc_embeddings.npz"


def quantise_int8(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(m).max(axis=1).astype(np.float32) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(m / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantise(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    m = q.astype(np.float32) * scale[:, None]
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


@dataclass
class DocEmbeddings:
    ids: np.ndarray  # route ids, aligned with the catalogue
    matrix: np.ndarray  # float32 (n, d), L2-normalised
    meta: dict[str, Any]

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def aligned_to(self, catalog: Catalog) -> DocEmbeddings:
        if len(self.ids) == len(catalog) and np.array_equal(self.ids, catalog.ids):
            return self
        pos = {int(r): i for i, r in enumerate(self.ids)}
        missing = [int(r) for r in catalog.ids if int(r) not in pos]
        if missing:
            raise ValueError(f"{len(missing)} catalogue routes have no embedding (e.g. {missing[:3]})")
        order = np.array([pos[int(r)] for r in catalog.ids], dtype=np.int64)
        return DocEmbeddings(catalog.ids.copy(), self.matrix[order], self.meta)

    def cosine(self, q: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Exact cosine similarity of the (normalised) query vector with the rows ``idx``."""
        return self.matrix[idx] @ np.asarray(q, dtype=np.float32)

    # ------------------------------------------------------------------ io
    def save(self, path: Path | str, *, precision: str = "float16") -> dict[str, Any]:
        meta = {**self.meta, "precision": precision}
        arrays: dict[str, np.ndarray] = {"ids": self.ids.astype(np.int64)}
        if precision == "int8":
            arrays["q"], arrays["scale"] = quantise_int8(self.matrix)
        elif precision == "float16":
            arrays["f16"] = self.matrix.astype(np.float16)
        else:
            arrays["f32"] = self.matrix.astype(np.float32)
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # deterministic zip: fixed timestamps, sorted members
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(arrays):
                buf = io.BytesIO()
                np.save(buf, arrays[name], allow_pickle=False)
                zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, buf.getvalue())
        return meta

    @classmethod
    def load(cls, path: Path | str = DOC_EMBEDDINGS_FILE) -> DocEmbeddings:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if "q" in z:
                m = dequantise(z["q"], z["scale"])
            elif "f16" in z:
                m = z["f16"].astype(np.float32)
                m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
            else:
                m = z["f32"].astype(np.float32)
            return cls(z["ids"].astype(np.int64), m, meta)


def build_doc_embeddings(catalog: Catalog, client, *, formulation: str, model: str | None, dim: int, tag: str):
    """Embed every catalogue route through the LLM client (cached per text); returns DocEmbeddings."""
    from app.llm import run_sync

    return run_sync(abuild_doc_embeddings(catalog, client, formulation=formulation, model=model, dim=dim, tag=tag))


async def abuild_doc_embeddings(
    catalog: Catalog, client, *, formulation: str, model: str | None, dim: int, tag: str
) -> DocEmbeddings:
    from .text import TEXT_VERSION, embed_text

    texts = [embed_text(r, formulation) for r in catalog.routes]
    res = await client.embed(texts, task="document", model=model, dim=dim, tag=tag)
    meta = {
        "provider": res.provider,
        "model": model or res.model,
        "reported_model": res.model,
        "dim": dim,
        "task": "document",
        "formulation": formulation,
        "text_version": TEXT_VERSION,
        "catalogue_sha256": catalog.sha256,
        "n": len(texts),
        "nominal_cost_usd": round(res.nominal_cost_usd, 6),
    }
    return DocEmbeddings(catalog.ids.copy(), np.asarray(res.vectors, dtype=np.float32), meta)
