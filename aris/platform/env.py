"""The black-box platform interface.

Two invariants, both load-bearing for the paper's claims:

D1  The candidate pool is NEVER revealed. A real platform does not tell you what
    it chose from, so the investigator must marginalise over an unknown pool.
    (The earlier scaffold leaked the pool via get_feed and that single leak made
    the inference task trivially easy -- see README of aris_scaffold.)

D2  Ground truth is written to a separate ledger that only analysis code reads.
    Agents receive item ids, text and timestamps. No scores, no features, no
    policy identity. `policy_version` is an opaque hash: you can detect a switch
    but not read it, which mirrors platforms shipping visible changes.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from .oracles import RankingPolicy, UserState


@dataclass
class ServedItem:
    """Exactly what a scraped feed would give you."""

    item_id: int
    text: str
    ts: float


@dataclass
class Session:
    session_id: str
    persona_id: int
    state: UserState
    rng: np.random.Generator
    last_slate: list[int] = field(default_factory=list)
    n_queries: int = 0


class BlackBoxPlatform:
    """In-process implementation. server.py exposes exactly this over HTTP."""

    def __init__(self, corpus, policy: RankingPolicy, cfg: dict, seed: int = 0):
        self.corpus, self.policy, self.cfg = corpus, policy, cfg
        p = cfg["platform"]
        self.k = int(p["slate_size"])
        self.pool = int(p["candidate_pool"])
        self.noise = float(p["noise_scale"])
        self.reveal_pool = bool(p.get("reveal_pool", False))
        self.dim = int(p["state_dim"])
        self._rng = np.random.default_rng(seed)
        self.sessions: dict[str, Session] = {}
        self.query_count = 0
        # --- ground-truth ledger: write-only from here, read-only at analysis ---
        self.gt_log: list[dict] = []
        self._pop_p = corpus.popularity / corpus.popularity.sum()

    # ------------------------------------------------------------------ public
    @property
    def policy_version(self) -> str:
        blob = f"{self.policy.family}|{sorted(self.policy.p.items())}".encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def session(self, persona_id: int, seed: int, interest: np.ndarray | None = None) -> str:
        """Open a session. `interest` seeds the platform's cold-start guess only."""
        sid = f"s{persona_id}-{seed}"
        rng = np.random.default_rng(abs(hash(sid)) % (2**32))
        if interest is None:
            v = rng.standard_normal(self.dim)
        else:
            v = np.asarray(interest, dtype=float)[: self.dim] + 0.3 * rng.standard_normal(self.dim)
        v = v / (np.linalg.norm(v) + 1e-12)
        self.sessions[sid] = Session(sid, persona_id, UserState(v), rng)
        return sid

    def get_feed(self, sid: str, k: int | None = None) -> list[ServedItem]:
        s = self.sessions[sid]
        k = k or self.k
        cand = self._sample_pool(s)
        slate = self.policy.rank(s.state, cand, k, s.rng, self.noise)
        s.last_slate = [int(i) for i in slate]
        s.n_queries += 1
        self.query_count += 1
        ts = time.time()
        self.gt_log.append({
            "session_id": sid, "persona_id": s.persona_id, "round": s.state.round,
            "pool": [int(c) for c in cand], "slate": s.last_slate,
            "state_vec": s.state.vec.copy(), "policy_family": self.policy.family,
            "eps": getattr(self.policy, "eps", lambda t: None)(s.state.round),
        })
        return [ServedItem(int(i), self.corpus.text[int(i)], ts) for i in slate]

    def interact(self, sid: str, item_id: int, action: str, dwell_ms: int = 0) -> dict:
        s = self.sessions[sid]
        if item_id not in s.last_slate:
            return {"ack": False, "reason": "item not in last served slate"}
        s.state = self.policy.update_state(s.state, int(item_id), action)
        return {"ack": True}

    def end_round(self, sid: str) -> None:
        self.sessions[sid].state.round += 1

    def health(self) -> dict:
        return {"policy_version": self.policy_version, "slate_size": self.k}

    # ---------------------------------------------------------------- analysis
    def ground_truth(self) -> dict:
        return {"params": self.policy.ground_truth_params(), "log": self.gt_log}

    # ----------------------------------------------------------------- private
    def _sample_pool(self, s: Session) -> np.ndarray:
        """Half uniform, half popularity-weighted retrieval -- hidden from callers."""
        n = len(self.corpus)
        half = self.pool // 2
        a = s.rng.choice(n, size=half, replace=False)
        b = s.rng.choice(n, size=self.pool - half, replace=False, p=self._pop_p)
        cand = np.unique(np.concatenate([a, b]))
        if len(cand) < self.pool:
            extra = s.rng.choice(n, size=self.pool - len(cand), replace=False)
            cand = np.unique(np.concatenate([cand, extra]))
        return cand
