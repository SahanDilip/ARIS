"""Tier-T1 oracle ranking policies -- the hidden ground truth.

BLACK BOX. Nothing under aris/population/ or aris/investigator/ may import this
module; tests/test_barrier.py enforces that by AST inspection, which is what
makes the black-box claim in the paper verifiable rather than asserted.

Each family implements Eqs. (1)-(2) of 02_METHODOLOGY.tex:
    score()        ~ pi_theta : user rep x candidates -> ranking distribution
    update_state() ~ g_theta  : (u_t, a_t, S_t) -> u_{t+1}

Families (methodology Table "Oracle policy families"):
  O1 popularity-biased matrix factorisation   gamma, latent_dim
  O2 engagement objective with affect boost   alpha, arousal_gain
  O3 epsilon-greedy explore/exploit feed      epsilon, decay
  O4 reinforcing profile update (bubble)      kappa, lambda_d
  O5 recency and velocity trending ranker     half_life, velocity_weight
  O6 non-stationary composite of O1..O5       mixture weights, switch_round
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTION_WEIGHT = {
    "skip": -0.15, "dwell": 0.5, "like": 1.0, "share": 1.5,
    "follow": 1.2, "comment": 1.2, "exit": 0.0,
}


@dataclass
class UserState:
    """The platform's *internal* representation of a user. Never exposed."""

    vec: np.ndarray
    round: int = 0
    n_actions: int = 0
    seen: set = field(default_factory=set)

    def copy(self) -> "UserState":
        return UserState(self.vec.copy(), self.round, self.n_actions, set(self.seen))


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s + 1e-9)


class RankingPolicy:
    """Base class. Subclasses override score() and optionally rank()."""

    family: str = "base"

    def __init__(self, corpus, params: dict, dim: int):
        self.corpus, self.p, self.dim = corpus, dict(params), dim

    # --- pi_theta ------------------------------------------------------------
    def _core(self, st: UserState, cand: np.ndarray) -> np.ndarray:
        """The family's characteristic term. Implemented by subclasses."""
        raise NotImplementedError

    def _bg(self, cand: np.ndarray) -> np.ndarray:
        """Shared nuisance signals present in *every* policy.

        Real rankers use popularity, affect, recency and velocity all at once;
        families differ in how they weight them, not in which they possess. With
        nuisance weights at zero the families are cartoonishly distinct and the
        presence of an effect is itself diagnostic, which makes mechanism
        classification trivial and leaves active probe design nothing to win.
        Non-zero weights force the investigator to resolve *magnitudes*.
        """
        c, out = self.corpus, 0.0
        for key, arr in (("nuis_pop", c.popularity), ("nuis_aro", c.arousal),
                         ("nuis_rec", c.recency), ("nuis_vel", c.velocity)):
            w = float(self.p.get(key, 0.0))
            if w:
                out = out + w * _z(arr[cand])
        return out if not isinstance(out, float) else np.zeros(len(cand))

    def score(self, st: UserState, cand: np.ndarray) -> np.ndarray:
        return self._core(st, cand) + self._bg(cand)

    def rank(self, st: UserState, cand: np.ndarray, k: int,
             rng: np.random.Generator, noise: float) -> np.ndarray:
        """Default: noisy top-k (Gumbel perturbation = Plackett-Luce sampling)."""
        s = self.score(st, cand)
        if noise > 0:
            s = s + noise * rng.gumbel(size=s.shape)
        return cand[np.argsort(-s)[:k]]

    # --- g_theta -------------------------------------------------------------
    def update_state(self, st: UserState, item: int, action: str) -> UserState:
        kappa = float(self.p.get("kappa", 0.25))
        w = ACTION_WEIGHT.get(action, 0.0)
        v = self.corpus.embed[item]
        st.vec = 0.97 * st.vec + kappa * w * v
        n = np.linalg.norm(st.vec)
        if n > 1e-9:
            st.vec = st.vec / n
        st.n_actions += 1
        st.seen.add(int(item))
        return st

    def ground_truth_params(self) -> dict:
        """Analysis-time only. Read from the ground-truth ledger, never served."""
        return {"family": self.family, **self.p}

    # convenience
    def _rel(self, st: UserState, cand: np.ndarray) -> np.ndarray:
        return self.corpus.embed[cand] @ st.vec


class O1PopularityMF(RankingPolicy):
    """Relevance from a latent factor model, additively boosted by popularity."""

    family = "O1"

    def __init__(self, corpus, params, dim):
        super().__init__(corpus, params, dim)
        ld = int(self.p.get("latent_dim", dim))
        self.ld = min(ld, dim)

    def _core(self, st, cand):
        rel = self.corpus.embed[cand][:, : self.ld] @ st.vec[: self.ld]
        return _z(rel) + float(self.p["gamma"]) * _z(np.log1p(20 * self.corpus.popularity[cand]))


class O2AffectEngagement(RankingPolicy):
    """Engagement objective: relevance plus an arousal-driven affect boost."""

    family = "O2"

    def _core(self, st, cand):
        a = self.corpus.arousal[cand] ** float(self.p["arousal_gain"])
        return _z(self._rel(st, cand)) + float(self.p["alpha"]) * _z(a)


class O3EpsilonGreedy(RankingPolicy):
    """Exploit relevance, but reserve a decaying fraction of slots for exploration."""

    family = "O3"

    def eps(self, t: int) -> float:
        return float(self.p["epsilon"]) * float(np.exp(-t / float(self.p["decay"])))

    def _core(self, st, cand):
        return _z(self._rel(st, cand))

    def rank(self, st, cand, k, rng, noise):
        s = self.score(st, cand)
        if noise > 0:
            s = s + noise * rng.gumbel(size=s.shape)
        order = list(cand[np.argsort(-s)])
        n_exp = int(rng.binomial(k, min(0.95, self.eps(st.round))))
        exploit = order[: k - n_exp]
        pool = [c for c in cand if c not in set(exploit)]
        explore = list(rng.choice(pool, size=min(n_exp, len(pool)), replace=False)) if n_exp else []
        slate = exploit + explore
        rng.shuffle(slate)  # exploration slots are not pinned to the tail
        return np.array(slate[:k], dtype=int)


class O4Reinforcing(RankingPolicy):
    """Strong profile reinforcement plus an in-slate diversity penalty (MMR)."""

    family = "O4"

    def _core(self, st, cand):
        return _z(self._rel(st, cand))

    def rank(self, st, cand, k, rng, noise):
        s = self.score(st, cand)
        if noise > 0:
            s = s + noise * rng.gumbel(size=s.shape)
        lam = float(self.p["lambda_d"])
        E = self.corpus.embed[cand]
        chosen, avail = [], list(range(len(cand)))
        while len(chosen) < min(k, len(cand)):
            if not chosen:
                j = int(np.argmax(s))
            else:
                sim = (E[avail] @ E[chosen].T).max(axis=1)
                j = avail[int(np.argmax(s[avail] - lam * sim))]
            chosen.append(j)
            avail.remove(j)
        return cand[np.array(chosen, dtype=int)]


class O5RecencyVelocity(RankingPolicy):
    """Trending ranker: exponential recency decay plus engagement velocity."""

    family = "O5"

    def _core(self, st, cand):
        age = (1.0 - self.corpus.recency[cand]) * 30.0
        rec = np.exp(-np.log(2.0) * age / float(self.p["half_life"]))
        vel = self.corpus.velocity[cand]
        return 0.5 * _z(self._rel(st, cand)) + _z(rec) + float(self.p["velocity_weight"]) * _z(vel)


class O6NonStationary(RankingPolicy):
    """Mixture of O1..O5 whose weights switch at an undisclosed round."""

    family = "O6"

    def __init__(self, corpus, params, dim):
        super().__init__(corpus, params, dim)
        base = {"kappa": self.p.get("kappa", 0.4)}
        base.update({k: v for k, v in self.p.items() if k.startswith("nuis_")})
        self.parts = [
            O1PopularityMF(corpus, {"gamma": 0.9, "latent_dim": dim, **base}, dim),
            O2AffectEngagement(corpus, {"alpha": 1.1, "arousal_gain": 1.4, **base}, dim),
            O3EpsilonGreedy(corpus, {"epsilon": 0.3, "decay": 40.0, **base}, dim),
            O4Reinforcing(corpus, {"lambda_d": 0.9, **base}, dim),
            O5RecencyVelocity(corpus, {"half_life": 12.0, "velocity_weight": 1.2, **base}, dim),
        ]

    def weights(self, t: int) -> np.ndarray:
        key = "weights_a" if t < int(self.p["switch_round"]) else "weights_b"
        w = np.asarray(self.p[key], dtype=float)
        return w / (w.sum() + 1e-12)

    def _core(self, st, cand):
        w = self.weights(st.round)
        return sum(wi * _z(part._core(st, cand)) for wi, part in zip(w, self.parts) if wi > 0)


REGISTRY = {
    "O1": O1PopularityMF, "O2": O2AffectEngagement, "O3": O3EpsilonGreedy,
    "O4": O4Reinforcing, "O5": O5RecencyVelocity, "O6": O6NonStationary,
}


def make_oracle(name: str, corpus, cfg: dict) -> RankingPolicy:
    return REGISTRY[name](corpus, cfg["oracles"][name], int(cfg["platform"]["state_dim"]))
