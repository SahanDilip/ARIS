"""Content corpus construction.

This is a *methodological* step, not data loading (04_EVALUATION.md 4.1).
If popularity, affect intensity, recency and velocity are mutually correlated in
the candidate pool, then oracles O1, O2 and O5 are confounded *by construction*
and no probe can separate them -- any identifiability finding would be a fact
about the corpus rather than about the policies. We therefore enforce
approximate decorrelation and report the correlation matrix as evidence (paper
Table: corpus validity).

Marginals are held fixed while dependence is removed: we decorrelate on ranks
and then map each column back onto its own empirical marginal, so the
distributional shape (e.g. heavy-tailed popularity) survives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ATTRS = ["popularity", "arousal", "valence_abs", "recency", "velocity"]

_TOPIC_NAMES = [
    "politics", "health", "technology", "sports", "finance", "entertainment",
    "science", "education", "local news", "travel", "food", "environment",
]
_FRAMES = [
    "Report: {t} debate widens as new figures land",
    "Explainer: what the latest {t} shift actually means",
    "Opinion: the {t} story nobody is telling",
    "Breaking: officials respond to {t} developments",
    "Analysis: five numbers that reframe the {t} picture",
    "Feature: inside the quiet turn in {t}",
    "Update: {t} outlook revised after weekend data",
    "Interview: a practitioner's view on {t}",
]
_TONE = {0: "measured", 1: "urgent", 2: "alarming", 3: "hopeful", 4: "sceptical"}


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return (order + 0.5) / len(x)


def _decorrelate(M: np.ndarray, max_abs: float, iters: int, rng) -> np.ndarray:
    """Remove linear dependence among columns while preserving each marginal.

    Works in the latent Gaussian space of the rank transform (a Gaussian copula
    with identity correlation), then re-imposes the original marginals by rank.
    """
    from scipy.special import erfinv
    from scipy.stats import norm

    n, d = M.shape
    marginals = [np.sort(M[:, j]) for j in range(d)]
    Z = np.column_stack([np.sqrt(2) * erfinv(2 * _ranks(M[:, j]) - 1) for j in range(d)])
    for _ in range(iters):
        Z = Z - Z.mean(0)
        C = np.corrcoef(Z, rowvar=False)
        off = np.abs(C - np.eye(d))
        if off.max() < max_abs / 2:
            break
        # whiten: Z C^{-1/2}
        w, V = np.linalg.eigh(C)
        Z = Z @ (V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-9))) @ V.T)
        Z += 1e-3 * rng.standard_normal(Z.shape)  # break exact ties
    out = np.empty_like(M)
    for j in range(d):
        r = np.argsort(np.argsort(norm.cdf(Z[:, j])))
        out[:, j] = marginals[j][r]
    return out


@dataclass
class Corpus:
    """Immutable item corpus. Shared by the platform and (read-only) by agents."""

    item_ids: np.ndarray
    embed: np.ndarray                 # (n, d) unit-norm topical embeddings
    topic: np.ndarray                 # (n,) int topic index
    popularity: np.ndarray            # (n,) in [0,1], heavy-tailed
    arousal: np.ndarray               # (n,) in [0,1]
    valence: np.ndarray               # (n,) in [-1,1]
    recency: np.ndarray               # (n,) in [0,1]; 1 = brand new
    velocity: np.ndarray              # (n,) in [0,1]; engagement acceleration
    text: list[str] = field(default_factory=list)
    topic_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.item_ids)

    @property
    def dim(self) -> int:
        return self.embed.shape[1]

    def attr_matrix(self) -> np.ndarray:
        return np.column_stack(
            [self.popularity, self.arousal, np.abs(self.valence), self.recency, self.velocity]
        )

    def correlations(self, method: str = "pearson") -> np.ndarray:
        M = self.attr_matrix()
        if method == "spearman":
            M = np.column_stack([_ranks(M[:, j]) for j in range(M.shape[1])])
        return np.corrcoef(M, rowvar=False)

    def max_offdiag_corr(self) -> float:
        out = 0.0
        for m in ("pearson", "spearman"):
            C = self.correlations(m)
            out = max(out, float(np.abs(C - np.eye(C.shape[0])).max()))
        return out

    def features(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Item-side features visible to *agents* (no platform internals)."""
        return {
            "embed": self.embed[idx],
            "arousal": self.arousal[idx],
            "valence": self.valence[idx],
            "recency": self.recency[idx],
            "topic": self.topic[idx],
        }


def build_corpus(cfg: dict, seed: int) -> Corpus:
    c = cfg["corpus"]
    n, d, T = int(c["n_items"]), int(c["embed_dim"]), int(c["n_topics"])
    rng = np.random.default_rng(seed)

    topic = rng.integers(0, T, size=n)
    centres = rng.standard_normal((T, d))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    embed = centres[topic] + 0.45 * rng.standard_normal((n, d))
    embed /= np.linalg.norm(embed, axis=1, keepdims=True)

    raw = np.column_stack([
        rng.lognormal(0.0, 1.1, n),        # popularity: heavy tail
        rng.beta(2.0, 2.0, n),             # arousal
        rng.beta(2.0, 2.0, n),             # |valence|
        rng.beta(1.6, 2.4, n),             # recency
        rng.gamma(2.0, 0.5, n),            # velocity
    ])
    dec = _decorrelate(raw, float(c["max_abs_corr"]), int(c["decorrelation_max_iters"]), rng)

    def unit(x):
        return (x - x.min()) / (np.ptp(x) + 1e-12)

    popularity = unit(np.log1p(dec[:, 0]))
    arousal, val_abs = unit(dec[:, 1]), unit(dec[:, 2])
    recency, velocity = unit(dec[:, 3]), unit(dec[:, 4])
    valence = val_abs * rng.choice([-1.0, 1.0], n)

    # Residualise popularity against the dominant topical direction, so that
    # "popular" is not a proxy for "matches the average interest vector".
    mean_dir = embed.mean(0)
    mean_dir /= np.linalg.norm(mean_dir) + 1e-12
    proj = embed @ mean_dir
    beta = np.cov(popularity, proj)[0, 1] / (np.var(proj) + 1e-12)
    popularity = unit(popularity - beta * proj)

    topic_names = (_TOPIC_NAMES * ((T // len(_TOPIC_NAMES)) + 1))[:T]
    text = []
    for i in range(n):
        frame = _FRAMES[int(rng.integers(0, len(_FRAMES)))]
        tone = _TONE[int(min(4, arousal[i] * 5))]
        text.append(f"[{tone}] " + frame.format(t=topic_names[topic[i]]))

    return Corpus(
        item_ids=np.arange(n),
        embed=embed, topic=topic, popularity=popularity, arousal=arousal,
        valence=valence, recency=recency, velocity=velocity,
        text=text, topic_names=topic_names,
    )
