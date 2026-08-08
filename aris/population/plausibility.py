"""Plausibility discriminator (02_METHODOLOGY.tex Eq. rho / "The plausibility
model" in 01_ARCHITECTURE.md 2.4).

Deviation from the original plan (see docs/DEVIATIONS.md): the plan trained
this against public interaction logs (MovieLens/LastFM/Steam or MIND click
logs). No such dataset is downloaded in this build. The substitute reference
distribution is a simple stochastic browsing model with realistic session
statistics (heavier-tailed dwell times, position-decayed engagement, bounded
topical persistence) rather than pure uniform-random clicking -- pure-random is
too easy a foil and would make every method look "plausible" by comparison.

AUC near 0.5 => agent trajectories indistinguishable from the reference =>
maximally plausible. AUC -> 1.0 => trivially distinguishable => off-manifold.
"""
from __future__ import annotations

import numpy as np

TRAJ_FEATURE_NAMES = [
    "mean_dwell_ms", "dwell_cv", "session_len", "action_mix_entropy",
    "topic_entropy", "repeat_rate", "inter_action_gap_s", "burstiness",
]


def trajectory_features(rows: list[dict], corpus, persona_id: int) -> np.ndarray:
    prows = sorted((r for r in rows if r["persona_id"] == persona_id), key=lambda r: r["round"])
    if not prows:
        return np.zeros(len(TRAJ_FEATURE_NAMES))
    dwell = np.array([r["dwell_ms"] for r in prows], dtype=float)
    ts = np.array([r["ts"] for r in prows], dtype=float)
    actions = [r["action"] for r in prows]
    topics = corpus.topic[[r["action_item"] for r in prows]]

    mean_dwell = float(dwell.mean())
    dwell_cv = float(dwell.std() / (dwell.mean() + 1e-6))
    session_len = float(len(prows))

    _, counts = np.unique(actions, return_counts=True)
    p = counts / counts.sum()
    action_entropy = float(-(p * np.log(p)).sum())

    _, tcounts = np.unique(topics, return_counts=True)
    pt = tcounts / tcounts.sum()
    topic_entropy = float(-(pt * np.log(pt + 1e-12)).sum())

    seen: set[int] = set()
    reps = 0
    for r in prows:
        ss = set(r["slate"])
        reps += len(ss & seen)
        seen |= ss
    repeat_rate = reps / max(1, sum(len(r["slate"]) for r in prows))

    gaps = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    inter_gap = float(np.mean(np.abs(gaps))) if gaps.size else 0.0
    burstiness = float((gaps.std() - gaps.mean()) / (gaps.std() + gaps.mean() + 1e-9)) if gaps.size > 1 else 0.0

    return np.array([mean_dwell, dwell_cv, session_len, action_entropy,
                     topic_entropy, repeat_rate, inter_gap, burstiness])


def reference_trajectory(corpus, n_rounds: int, slate_size: int, seed: int,
                         candidate_pool: int = 60) -> dict:
    """A structured (not uniform-random) synthetic 'real user' session, used as
    the plausibility reference in place of a downloaded public dataset.

    Two-stage retrieval-then-rank, matching the real platform (platform/env.py):
    sample a larger candidate pool (half uniform, half popularity-weighted),
    then rank by relevance and keep the top `slate_size`. An earlier version
    sampled directly at slate_size, which under-represented repeat exposure of
    popular items relative to the real platform's larger retrieval pool and
    made `repeat_rate` a perfectly-separating (and spurious) feature.
    """
    rng = np.random.default_rng(seed)
    rows = []
    interest = rng.standard_normal(corpus.dim)
    interest /= np.linalg.norm(interest)
    clock = 0.0
    pop_p = corpus.popularity / corpus.popularity.sum()
    n = len(corpus)
    pool = min(candidate_pool, n)
    for t in range(n_rounds):
        half = pool // 2
        a = rng.choice(n, size=half, replace=False)
        b = rng.choice(n, size=pool - half, replace=False, p=pop_p)
        cand = np.unique(np.concatenate([a, b]))
        rel = corpus.embed[cand] @ interest
        # Gumbel-perturbed top-k (Plackett-Luce sampling), matching the real
        # platform's RankingPolicy.rank() (platform/oracles.py). Deterministic
        # top-k here concentrated on the same handful of highest-relevance
        # items every round given a fixed interest vector, over-suppressing
        # repeat_rate relative to the (noisy) real platform and then
        # overshooting the opposite direction once popularity-weighting alone
        # was added -- the noise term is the piece that was still missing.
        noisy = rel + 0.6 * rng.gumbel(size=rel.shape)
        order = cand[np.argsort(-noisy)][:slate_size]
        # position-decayed engagement probability, log-normal dwell -- structured,
        # not uniform, so the discriminator has a real foil rather than a trivial one
        pos_p = 0.5 * np.exp(-np.arange(slate_size) / 3.0)
        engage = rng.random(slate_size) < pos_p
        j = int(np.argmax(engage)) if engage.any() else int(rng.integers(0, slate_size))
        item = int(order[j])
        action = "skip" if not engage.any() else rng.choice(
            ["dwell", "like", "share", "follow", "comment"], p=[0.55, 0.25, 0.08, 0.05, 0.07])
        dwell = int(rng.lognormal(6.5, 0.6)) if action != "skip" else 0
        # clock accumulates a per-round gap ONCE, matching population_driver's
        # cumulative session clock. Multiplying the round index by a freshly
        # redrawn factor each round (the original version of this line) is not
        # a clock: it is non-monotonic in expectation and inflates apparent
        # gaps and burstiness as t grows, for no behavioural reason.
        clock += dwell / 1000.0 + rng.exponential(3.0) + 2.0
        rows.append({"persona_id": -1, "round": t, "slate": order.tolist(),
                    "action": action, "action_item": item, "dwell_ms": dwell,
                    "ts": clock})
    return rows


def discriminator_auc(agent_rows: list[dict], corpus, persona_ids: list[int],
                      slate_size: int, seed: int) -> dict:
    """AUC of a simple classifier separating agent trajectories from the reference.

    The reference trajectory for each persona is generated with the SAME
    session length as that persona's actual observed history. An earlier
    version used a fixed, separately-configured reference length that did not
    match the population run's actual round count, making `session_len` alone
    a perfect (and spurious) discriminating feature -- AUC=1.0 for a reason
    that had nothing to do with behavioural plausibility.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    by_pid: dict[int, int] = {}
    for r in agent_rows:
        by_pid[r["persona_id"]] = by_pid.get(r["persona_id"], 0) + 1

    X_agent = np.array([trajectory_features(agent_rows, corpus, pid) for pid in persona_ids])
    ref_rows_by_id = {}
    for i, pid in enumerate(persona_ids):
        n_rounds = max(4, by_pid.get(pid, 0))
        ref = reference_trajectory(corpus, n_rounds, slate_size, seed + 1000 + i)
        for r in ref:
            r["persona_id"] = -1 - i
        ref_rows_by_id[-1 - i] = ref
    X_ref = np.array([trajectory_features(ref_rows_by_id[-1 - i], corpus, -1 - i)
                     for i in range(len(persona_ids))])

    X = np.nan_to_num(np.vstack([X_agent, X_ref]))
    y = np.r_[np.ones(len(X_agent)), np.zeros(len(X_ref))]
    if len(X) < 6:
        return {"auc": None, "note": "too few personas for a stable AUC estimate"}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=min(5, len(X) // 2), shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    return {"auc": round(auc, 3),
           "interpretation": ("plausible (near 0.5)" if abs(auc - 0.5) < 0.15
                              else "off-manifold (agents are distinguishable)")}