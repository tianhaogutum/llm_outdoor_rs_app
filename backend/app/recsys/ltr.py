"""Learned ranker at serving time: a LightGBM LambdaMART model evaluated in numpy.

Training (``python -m evalkit.ltr``) needs LightGBM; serving does not. The trained booster is exported
from ``Booster.dump_model()`` into a flat, versioned JSON artefact (``artifacts/ltr_<name>.json``):
one array per node attribute over all trees, the feature names, the training metadata and a reference
batch of feature rows with LightGBM's own predictions. :class:`TreeEnsemble` evaluates all trees for
all rows at once (vectorised traversal, depth iterations), so the API needs no native library and no
OpenMP runtime.

Per-feature contributions use Saabas' path attribution: walking a row from the root to its leaf, each
split's change in node value (``internal_value`` of the child minus that of the parent) is credited to
the split feature. ``bias + sum(contributions) == prediction`` exactly (the sum telescopes); unlike
TreeSHAP the attribution depends on the order of splits, which is acceptable for an explanation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dense import ARTIFACTS_DIR

FORMAT = "recsys.tree_ensemble/v1"
MODELS_COMMITTED = ("v1", "v1_noapi")  # artefacts trained by ``python -m evalkit.ltr train``
MISSING_NONE, MISSING_ZERO, MISSING_NAN = 0, 1, 2
_MISSING = {"None": MISSING_NONE, "Zero": MISSING_ZERO, "NaN": MISSING_NAN}
K_ZERO = 1e-35


def model_path(name: str) -> Path:
    return ARTIFACTS_DIR / f"ltr_{name}.json"


def flatten_lightgbm_dump(dump: dict[str, Any]) -> dict[str, Any]:
    """Convert ``Booster.dump_model()`` into flat node arrays (numerical splits only)."""
    feat, thr, left, right, dleft, miss, value, roots = [], [], [], [], [], [], [], []

    def add(node: dict[str, Any]) -> int:
        i = len(feat)
        feat.append(-1)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        dleft.append(False)
        miss.append(MISSING_NONE)
        if "leaf_value" in node:
            value.append(float(node["leaf_value"]))
            return i
        if node.get("decision_type", "<=") != "<=":
            raise ValueError("only numerical '<=' splits are supported")
        value.append(float(node["internal_value"]))
        feat[i] = int(node["split_feature"])
        thr[i] = float(node["threshold"])
        dleft[i] = bool(node["default_left"])
        miss[i] = _MISSING[node.get("missing_type", "None")]
        left[i] = add(node["left_child"])
        right[i] = add(node["right_child"])
        return i

    for t in dump["tree_info"]:
        roots.append(add(t["tree_structure"]))
    return {
        "feature": feat,
        "threshold": thr,
        "left": left,
        "right": right,
        "default_left": dleft,
        "missing_type": miss,
        "value": value,
        "roots": roots,
    }


@dataclass
class TreeEnsemble:
    feature_names: list[str]
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    default_left: np.ndarray
    missing_type: np.ndarray
    value: np.ndarray
    roots: np.ndarray
    meta: dict[str, Any]
    reference: dict[str, Any]
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], name: str = "") -> TreeEnsemble:
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} artefact")
        t = d["trees"]
        return cls(
            feature_names=list(d["feature_names"]),
            feature=np.asarray(t["feature"], dtype=np.int64),
            threshold=np.asarray(t["threshold"], dtype=np.float64),
            left=np.asarray(t["left"], dtype=np.int64),
            right=np.asarray(t["right"], dtype=np.int64),
            default_left=np.asarray(t["default_left"], dtype=bool),
            missing_type=np.asarray(t["missing_type"], dtype=np.int64),
            value=np.asarray(t["value"], dtype=np.float64),
            roots=np.asarray(t["roots"], dtype=np.int64),
            meta=d.get("meta", {}),
            reference=d.get("reference", {}),
            name=name,
        )

    @classmethod
    def load(cls, path: Path | str) -> TreeEnsemble:
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), name=p.stem.removeprefix("ltr_"))

    @property
    def n_trees(self) -> int:
        return int(len(self.roots))

    def _walk(self, X: np.ndarray, contrib: bool) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected (n, {len(self.feature_names)}) features, got {X.shape}")
        n = X.shape[0]
        node = np.broadcast_to(self.roots, (n, self.n_trees)).copy()
        rows = np.broadcast_to(np.arange(n)[:, None], node.shape)
        C = np.zeros((n, X.shape[1]), dtype=np.float64) if contrib else None
        while True:
            f = self.feature[node]
            active = f >= 0
            if not active.any():
                break
            fa = np.where(active, f, 0)
            x = X[rows, fa]
            mt = self.missing_type[node]
            isnan = np.isnan(x)
            # LightGBM numerical decision: NaN -> 0 unless missing_type is NaN; missing values go default
            xz = np.where(isnan & (mt != MISSING_NAN), 0.0, x)
            is_missing = ((mt == MISSING_NAN) & isnan) | ((mt == MISSING_ZERO) & (np.abs(xz) <= K_ZERO))
            go_left = np.where(is_missing, self.default_left[node], xz <= self.threshold[node])
            nxt = np.where(go_left, self.left[node], self.right[node])
            nxt = np.where(active, nxt, node)
            if C is not None:
                delta = np.where(active, self.value[nxt] - self.value[node], 0.0)
                np.add.at(C, (rows[active], fa[active]), delta[active])
            node = nxt
        return self.value[node].sum(axis=1), C

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._walk(X, contrib=False)[0]

    def predict_contrib(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """(predictions, per-feature contributions (n, f), bias); bias + contrib.sum(1) == predictions."""
        pred, C = self._walk(X, contrib=True)
        bias = float(self.value[self.roots].sum())
        return pred, C, bias

    def check_reference(self, atol: float = 1e-6) -> float:
        """Max absolute difference to LightGBM's own predictions on the stored reference rows."""
        ref = self.reference
        if not ref:
            return 0.0
        X = np.array([[np.nan if v is None else v for v in row] for row in ref["X"]], dtype=np.float64)
        d = float(np.max(np.abs(self.predict(X) - np.asarray(ref["pred"], dtype=np.float64))))
        if d > atol:
            raise ValueError(f"tree ensemble disagrees with LightGBM on the reference rows (max diff {d:.3g})")
        return d


_models: dict[str, TreeEnsemble] = {}


def load_model(name: str) -> TreeEnsemble:
    """Load (once per process) and self-check a committed model artefact."""
    if name not in _models:
        m = TreeEnsemble.load(model_path(name))
        m.check_reference()
        _models[name] = m
    return _models[name]
