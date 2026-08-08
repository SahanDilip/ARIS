"""Ranking surrogate: the S in 01_ARCHITECTURE.md's investigator roster.

Learns to predict, from a served slate and item attributes, which item a
persona will engage with. This is the paper's listwise recovery target
(04_EVALUATION.md 4.3: Top-1/Top-5 agreement, NDCG-agreement, Kendall's tau).

Design note, found empirically before this was written correctly: a first
version used only generic item attributes (arousal, valence, recency,
velocity, popularity) and scored BELOW random-chance HitRatio@1. Diagnosis:
under the population policy, choice is driven almost entirely by
persona-specific relevance (interest_vec . item_embed), which generic item
attributes cannot see at all -- popularity, for instance, showed essentially
zero correlation with the chosen item (mean difference ~ -0.02 against the
slate average). This mirrors exactly the bug found and fixed in Step 2's Gate
1 (population/validate.py): any predictor of engagement must condition on the
persona, not just the item. The surrogate below therefore takes an optional
per-row persona profile and adds topical_match as a feature. In a real T2/T3
deployment the investigator would not know which "real user" IDs slates came
from, but it can and should observe repeated interaction with a SESSION (this
is exactly what the interaction-history-based probing in the survey's
candidate framework, Section 9, assumes): the surrogate conditions on a
running estimate of the session's own revealed preference (the mean embedding
of items it has previously engaged with), not on any hidden persona identity.

Deviation from the original plan (docs/DEVIATIONS.md D4): LightGBM's
`lambdarank` objective is replaced by a pairwise decomposition -- for each
served slate, every (engaged_item, other_item) pair becomes one training
example for a binary classifier (does the left item outrank the right one?).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _item_row(corpus, item: int) -> np.ndarray:
    """Generic item attributes -- content-observable, not platform internals."""
    return np.array([
        float(corpus.arousal[item]), float(np.abs(corpus.valence[item])),
        float(corpus.recency[item]), float(corpus.velocity[item]),
        float(corpus.popularity[item]),
    ])


def _session_relevance(corpus, item: int, revealed_pref: np.ndarray | None) -> float:
    """Topical match against a RUNNING ESTIMATE of session preference (the mean
    embedding of items this session has engaged with so far), not a hidden
    persona identity. This is legitimately observable by an outside auditor
    watching one session unfold, unlike a persona's true interest_vec.
    """
    if revealed_pref is None:
        return 0.0
    return float(corpus.embed[item] @ revealed_pref)


def build_oracle_trace(rows: list[dict], corpus, personas: dict) -> dict:
    """B7-style oracle-informed ceiling (04_EVALUATION.md 4.2): the SAME trace
    structure as build_session_trace, but using each persona's TRUE interest_vec
    (constant across rounds) instead of a revealed-preference estimate built
    from limited session history. A real investigator never has this; it
    exists only to quantify the headroom between a realistic, content-only
    proxy and perfect persona knowledge. `personas` is {persona_id: PersonaProfile}.
    """
    trace: dict[int, list[tuple[tuple, np.ndarray]]] = {}
    for r in rows:
        pid = r["persona_id"]
        if pid not in personas:
            continue
        trace.setdefault(pid, []).append(((pid, r["round"]), personas[pid].interest_vec))
    return trace


def build_session_trace(rows: list[dict], corpus) -> dict[int, list[tuple[tuple, np.ndarray | None]]]:
    """For each persona (proxy for "session"), walk its rows in round order and
    record, at the time of each decision, the running revealed-preference
    vector built ONLY from that session's own prior engaged items -- so this
    never looks ahead and never uses another session's history.

    Call this ONCE on the full row history per persona. Keys are
    (persona_id, round) tuples rather than id(row), because id() is only
    guaranteed unique among simultaneously-live objects -- a row dict that
    goes out of scope and is garbage-collected can have its id reused by an
    unrelated later object, which is a real, if rare, correctness hazard when
    rows are filtered into new lists (as train_rows/test_rows are).
    """
    by_pid: dict[int, list[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["persona_id"], []).append(r)

    trace: dict[int, list[tuple[tuple, np.ndarray | None]]] = {}
    for pid, prows in by_pid.items():
        prows = sorted(prows, key=lambda r: r["round"])
        history: list[np.ndarray] = []
        seq = []
        for r in prows:
            pref = np.mean(history, axis=0) if history else None
            if pref is not None:
                pref = pref / (np.linalg.norm(pref) + 1e-12)
            seq.append(((pid, r["round"]), pref))
            if r["action"] != "skip" and r["action_item"] in r["slate"]:
                history.append(corpus.embed[r["action_item"]])
                if len(history) > 20:
                    history.pop(0)
        trace[pid] = seq
    return trace


def _pairwise_examples(rows: list[dict], corpus, rng: np.random.Generator,
                       max_pairs_per_round: int = 4, trace: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise training examples: (engaged - other) feature difference plus
    position difference and revealed-preference-match difference; label 1 if
    the left item should outrank the right. Both orderings included so the
    classifier cannot learn a trivial sign shortcut.

    `trace`, if given, must be pre-built by build_session_trace() over the FULL
    row history (not just `rows`) -- building it fresh from a subset would
    reset every session's revealed-preference history to empty at the subset's
    first row, discarding real prior context. This was a real bug: held-out
    fidelity collapsed toward zero at full corpus scale because the "test"
    evaluation lost the preference signal accumulated during "train".
    """
    if trace is None:
        trace = build_session_trace(rows, corpus)
    row_pref = {rid: pref for seq in trace.values() for rid, pref in seq}

    X, y = [], []
    for r in rows:
        if r["action"] == "skip" or r["action_item"] not in r["slate"]:
            continue
        slate = r["slate"]
        pos_engaged = slate.index(r["action_item"])
        others = [it for it in slate if it != r["action_item"]]
        if not others:
            continue
        pref = row_pref.get((r["persona_id"], r["round"]))
        k = min(max_pairs_per_round, len(others))
        chosen_others = rng.choice(others, size=k, replace=False)
        fe = _item_row(corpus, r["action_item"])
        rel_e = _session_relevance(corpus, r["action_item"], pref)
        for other in chosen_others:
            other = int(other)
            fo = _item_row(corpus, other)
            rel_o = _session_relevance(corpus, other, pref)
            pos_other = slate.index(other)
            diff = np.concatenate([fe - fo, [pos_engaged - pos_other, rel_e - rel_o]])
            if rng.random() < 0.5:
                X.append(diff); y.append(1)
            else:
                X.append(-diff); y.append(0)
    return np.array(X), np.array(y)


@dataclass
class RankingSurrogate:
    """Wraps a pairwise classifier; scores items via the same feature space."""

    clf: object = None
    feature_dim: int = 7  # 5 item attrs + position + revealed-preference match

    def fit(self, rows: list[dict], corpus, seed: int = 0, trace: dict | None = None) -> dict:
        from sklearn.ensemble import HistGradientBoostingClassifier

        rng = np.random.default_rng(seed)
        X, y = _pairwise_examples(rows, corpus, rng, trace=trace)
        if len(X) < 10 or len(set(y)) < 2:
            return {"fitted": False, "n_pairs": len(X), "reason": "insufficient pairwise data"}
        self.clf = HistGradientBoostingClassifier(max_depth=3, random_state=seed).fit(X, y)
        return {"fitted": True, "n_pairs": len(X)}

    def _beats(self, corpus, item_a: int, pos_a: int, item_b: int, pos_b: int,
              revealed_pref: np.ndarray | None) -> float:
        """P(item_a outranks item_b), using exactly the feature construction the
        classifier was trained on (a real-item-vs-real-item difference). This
        is the piece that was wrong before: comparing a real item against a
        synthetic all-zero "null" vector asks a gradient-boosted classifier to
        extrapolate to a point far outside its training distribution, which
        tree models handle unreliably -- it produced a ranking signal so weak
        that an oracle-informed ceiling (true interest_vec) scored identically
        to a weak realistic proxy on HitRatio@1. Every comparison here is
        between two actual candidates, exactly matching training.
        """
        fa, fb = _item_row(corpus, item_a), _item_row(corpus, item_b)
        rel_a = _session_relevance(corpus, item_a, revealed_pref)
        rel_b = _session_relevance(corpus, item_b, revealed_pref)
        diff = np.concatenate([fa - fb, [pos_a - pos_b, rel_a - rel_b]])
        return float(self.clf.predict_proba(diff.reshape(1, -1))[0, 1])

    def rank_candidates(self, corpus, items: list[int], positions: list[int],
                        revealed_pref: np.ndarray | None) -> list[int]:
        """Round-robin (Copeland-style) ranking: each candidate's score is its
        mean win probability against every other candidate. Stays entirely
        within the classifier's training distribution -- no synthetic anchor.

        Vectorised: all n*(n-1) pairwise feature-difference rows are built into
        one matrix and scored with a single predict_proba call, rather than
        n*(n-1) individual calls. The unvectorised version made a full
        driver run (fit + 3 evaluate_fidelity calls x 2 oracles) exceed a
        10-minute execution budget; this reduces the O(n^2) part to one batched
        inference call, keeping the algorithm identical.
        """
        if self.clf is None or len(items) <= 1:
            return list(items)
        n = len(items)
        feats = np.stack([_item_row(corpus, it) for it in items])
        rels = np.array([_session_relevance(corpus, it, revealed_pref) for it in items])
        pos = np.array(positions, dtype=float)

        idx_i, idx_j = np.where(~np.eye(n, dtype=bool))
        diffs = np.column_stack([
            feats[idx_i] - feats[idx_j],
            (pos[idx_i] - pos[idx_j]).reshape(-1, 1),
            (rels[idx_i] - rels[idx_j]).reshape(-1, 1),
        ])
        probs = self.clf.predict_proba(diffs)[:, 1]
        wins = np.zeros(n)
        np.add.at(wins, idx_i, probs)
        wins /= max(1, n - 1)
        order = np.argsort(-wins)
        return [items[k] for k in order]

    def rank_slate(self, corpus, slate: list[int], revealed_pref: np.ndarray | None) -> list[int]:
        return self.rank_candidates(corpus, slate, list(range(len(slate))), revealed_pref)


def evaluate_fidelity(surrogate: RankingSurrogate, rows: list[dict], corpus,
                      seed: int = 1, trace: dict | None = None) -> dict:
    """Top-1 agreement, HitRatio@1 on a 1-positive/19-negative candidate set
    (04_EVALUATION.md 4.3, matching the protocol in Cai et al. [5]), and
    Kendall's tau between the surrogate's predicted order and slate order.

    `trace`, if given, must be built by build_session_trace() over the FULL
    row history (see that function's docstring for why -- rebuilding it from
    only `rows` silently discards prior context and was a real bug here).
    """
    from scipy.stats import kendalltau

    if trace is None:
        trace = build_session_trace(rows, corpus)
    row_pref = {rid: pref for seq in trace.values() for rid, pref in seq}

    top1_hits, hr1_hits, taus = 0, 0, []
    n_eval = 0
    rng = np.random.default_rng(seed)
    for r in rows:
        if r["action"] == "skip" or r["action_item"] not in r["slate"]:
            continue
        pref = row_pref.get((r["persona_id"], r["round"]))
        slate = r["slate"]
        predicted_order = surrogate.rank_slate(corpus, slate, pref)
        n_eval += 1
        if predicted_order[0] == r["action_item"]:
            top1_hits += 1

        negatives = rng.choice([i for i in range(len(corpus)) if i not in slate],
                               size=min(19, len(corpus) - len(slate)), replace=False)
        cand = [r["action_item"]] + list(negatives)
        cand_order = surrogate.rank_candidates(corpus, cand, [0] * len(cand), pref)
        if cand_order and cand_order[0] == r["action_item"]:
            hr1_hits += 1

        true_rank = [slate.index(it) for it in predicted_order]
        if len(set(true_rank)) > 1:
            tau, _ = kendalltau(true_rank, list(range(len(predicted_order))))
            if not np.isnan(tau):
                taus.append(tau)

    if n_eval == 0:
        return {"n_eval": 0, "top1_agreement": None, "hitratio_1": None, "kendall_tau_mean": None}
    return {
        "n_eval": n_eval,
        "top1_agreement": round(top1_hits / n_eval, 3),
        "hitratio_1": round(hr1_hits / n_eval, 3),
        "kendall_tau_mean": round(float(np.mean(taus)), 3) if taus else None,
    }