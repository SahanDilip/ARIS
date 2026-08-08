"""Audit-style summary statistics and the Step-1 separability diagnostic.

Why this belongs in Step 1 rather than Step 10
----------------------------------------------
The earlier scaffold's honest failure was that all methods recovered the true
policy family on every seed, so ARIS, random probing and the unconstrained
ablation were indistinguishable: the task was too easy and the headline
experiments (E2, E3) had nothing to show. The opposite failure is just as bad --
oracles so confounded that nothing separates them at any budget.

This diagnostic measures, in week 1, where the testbed sits between those two
failures. Read it as a proto-E5 identifiability map obtained with *random*
probes. The target is a middle band:

  * pair separability near 1.0 under random probing  -> too easy; active design
    cannot beat random, so RQ2 has no headroom. Harden (raise noise, shrink K,
    shrink rounds).
  * pair separability near 0.5                       -> unidentifiable at this
    budget. Fine for one or two pairs (that is your finding), fatal if all.
  * 0.6-0.9                                          -> healthy: signal exists,
    but efficient probing should be needed to extract it.

The feature vector is deliberately the same audit-style statistics the
Surrogate Agent's mechanism classifier will use in Step 4, so this code is
reused rather than discarded.
"""
from __future__ import annotations

import numpy as np

from .platform.env import BlackBoxPlatform
from .platform.oracles import make_oracle
from .population.random_agent import RandomAgent, browse

FEATURE_NAMES = [
    "pop_lift", "arousal_lift", "recency_lift", "velocity_lift",
    "pos_corr_pop", "pos_corr_arousal", "pos_corr_recency", "pos_corr_velocity",
    "exposure_gini", "slate_topic_entropy", "intra_slate_sim",
    "repeat_rate", "novelty_rate", "churn", "drift_slope", "topic_concentration",
]


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def summary_statistics(rows: list[dict], corpus) -> np.ndarray:
    """Audit-style statistics computable from slates alone. No platform internals."""
    slates = [r["slate"] for r in rows]
    flat = np.array([i for s in slates for i in s], dtype=int)
    pos = np.array([j for s in slates for j in range(len(s))], dtype=float)
    if len(flat) == 0:
        return np.zeros(len(FEATURE_NAMES))

    pop, aro, rec, vel = (corpus.popularity, corpus.arousal, corpus.recency, corpus.velocity)
    base = np.array([pop.mean(), aro.mean(), rec.mean(), vel.mean()])
    lift = np.array([pop[flat].mean(), aro[flat].mean(), rec[flat].mean(), vel[flat].mean()]) - base

    pos_corr = [_safe_corr(pos, arr[flat]) for arr in (pop, aro, rec, vel)]

    counts = np.bincount(flat, minlength=len(corpus)).astype(float)
    gini = _gini(counts[counts > 0])

    ents, sims, tconc = [], [], []
    for s in slates:
        s = np.asarray(s, dtype=int)
        tp = np.bincount(corpus.topic[s], minlength=len(corpus.topic_names)).astype(float)
        p = tp / tp.sum()
        ents.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        tconc.append(float(p.max()))
        E = corpus.embed[s]
        G = E @ E.T
        m = ~np.eye(len(s), dtype=bool)
        sims.append(float(G[m].mean()) if m.any() else 0.0)

    seen: set[int] = set()
    rep, nov = [], []
    for s in slates:
        ss = set(int(i) for i in s)
        rep.append(len(ss & seen) / len(ss))
        nov.append(len(ss - seen) / len(ss))
        seen |= ss

    # slate-to-slate churn and topical drift, per persona
    churn, drift = [], []
    by_p: dict[int, list[dict]] = {}
    for r in rows:
        by_p.setdefault(r["persona_id"], []).append(r)
    for _, rs in by_p.items():
        rs = sorted(rs, key=lambda r: r["round"])
        for a, b in zip(rs, rs[1:]):
            sa, sb = set(a["slate"]), set(b["slate"])
            churn.append(1.0 - len(sa & sb) / max(1, len(sa | sb)))
        if len(rs) >= 4:
            c0 = corpus.embed[np.asarray(rs[0]["slate"], dtype=int)].mean(0)
            cT = corpus.embed[np.asarray(rs[-1]["slate"], dtype=int)].mean(0)
            c0 /= np.linalg.norm(c0) + 1e-12
            cT /= np.linalg.norm(cT) + 1e-12
            drift.append(1.0 - float(c0 @ cT))

    return np.array([
        *lift, *pos_corr, gini, float(np.mean(ents)), float(np.mean(sims)),
        float(np.mean(rep)), float(np.mean(nov)),
        float(np.mean(churn)) if churn else 0.0,
        float(np.mean(drift)) if drift else 0.0,
        float(np.mean(tconc)),
    ])


def observe_oracle(oracle: str, corpus, cfg: dict, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Run one random-probe replicate against one oracle; return (features, rows).

    One returned row == one platform query, and `browse` emits rows in
    round-major order, so truncating the row list is exactly a query budget cut.
    """
    s = cfg["separability"]
    plat = BlackBoxPlatform(corpus, make_oracle(oracle, corpus, cfg), cfg, seed)
    agents = [RandomAgent(i, int(cfg["platform"]["state_dim"]), seed * 1000 + i)
              for i in range(int(s["n_agents"]))]
    rows = browse(plat, agents, int(s["n_rounds"]))
    return summary_statistics(rows, corpus), rows


def _loo_nearest_centroid_bacc(Z: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy of a leave-one-out nearest-centroid rule. Pure numpy.

    Vectorised over the held-out index: for point i the centroid of its own class
    is recomputed without it. Equivalent to sklearn's NearestCentroid under
    LeaveOneOut, but ~3 orders of magnitude faster, which matters because the
    permutation test calls this tens of thousands of times.
    """
    n = len(Z)
    classes = (0.0, 1.0)
    sums = {c: Z[y == c].sum(0) for c in classes}
    cnts = {c: int((y == c).sum()) for c in classes}
    d = np.empty((n, 2))
    for j, c in enumerate(classes):
        own = (y == c)
        cnt = np.where(own, cnts[c] - 1, cnts[c]).astype(float)
        cen = (sums[c][None, :] - np.where(own[:, None], Z, 0.0)) / np.maximum(cnt, 1)[:, None]
        d[:, j] = ((Z - cen) ** 2).sum(1)
    pred = d.argmin(1).astype(float)
    recalls = [float((pred[y == c] == c).mean()) for c in classes if cnts[c]]
    return float(np.mean(recalls))


def _pair_accuracy(Xa: np.ndarray, Xb: np.ndarray, seed: int,
                   n_perm: int = 199, alpha: float = 0.05) -> tuple[float, float]:
    """Leave-one-out balanced accuracy, gated by a label-permutation test.

    With 16 summary features and a handful of replicates per class we are firmly
    in the p >> n regime, where a flexible classifier separates *any* two groups
    perfectly and reports a meaningless 1.0. Three guards:

      1. a nearest-centroid rule -- one centroid per class, so it cannot carve
         arbitrary boundaries;
      2. leave-one-out cross-validation, so nothing is scored in-sample;
      3. a permutation test. Comparing against the null *mean* is not enough:
         with twelve LOO folds the null accuracy has a standard deviation near
         0.14, so a single draw reaching 0.75 is unremarkable. We require the
         observed accuracy to beat the (1 - alpha) quantile of the null before
         reporting any separability at all.

    Returns (separability, p_value). `separability` is exactly 0.5 when the
    permutation test does not reach significance, and otherwise the excess over
    the null mean shifted back onto [0.5, 1.0] so that it reads as an accuracy.

    n_perm must exceed 1/alpha for significance to be attainable at all.
    """
    X = np.nan_to_num(np.vstack([Xa, Xb]), nan=0.0, posinf=0.0, neginf=0.0)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]

    observed = _loo_nearest_centroid_bacc(Z, y)
    rng = np.random.default_rng(seed)
    null = np.array([_loo_nearest_centroid_bacc(Z, rng.permutation(y))
                     for _ in range(n_perm)])
    p_value = float((1 + int((null >= observed).sum())) / (n_perm + 1))
    if p_value > alpha:
        return 0.5, round(p_value, 4)
    return round(min(1.0, 0.5 + max(0.0, observed - float(null.mean()))), 3), round(p_value, 4)


#: absolute platform-query budgets; a real curve needs to start small
BUDGET_GRID = (10, 25, 50, 100, 250, 500, 1000, 2000)

#: the parameter whose recovery each family's paper claim actually rests on
PRIMARY_PARAM = {
    "O1": "gamma", "O2": "alpha", "O3": "epsilon",
    "O4": "lambda_d", "O5": "velocity_weight", "O6": "switch_round",
}


def parameter_resolution(corpus, cfg: dict, seed: int,
                         deltas=(0.30, 0.80), budget_small: int = 60):
    """How small a parameter difference can *random* probing resolve, and at what cost?

    Family classification saturates quickly: O2 boosts arousal, O5 boosts
    recency, and given a few hundred slates no amount of noise hides that. The
    quantity the paper's recovery claim actually rests on is resolution of theta
    *within* a family. For each family we perturb its primary parameter by
    +/- delta/2 in relative terms and test whether the two settings are
    distinguishable from slate statistics alone.

    Read it as follows:
      * random probes already resolve delta = 0.30 at the small budget
            -> no headroom for RQ2 on that family; harden it
      * random probes cannot resolve delta = 0.80 even at the full budget
            -> candidate non-identifiability result for RQ5
      * resolves 0.80 but not 0.30
            -> the healthy case: active design has a measurable gap to close
    """
    import copy

    R = int(cfg["separability"]["n_replicates"])
    rows = []
    for fam in cfg["oracles"]["active"]:
        key = PRIMARY_PARAM.get(fam)
        if key is None or key not in cfg["oracles"][fam]:
            continue
        base = float(cfg["oracles"][fam][key])
        is_int = isinstance(cfg["oracles"][fam][key], int)
        for delta in deltas:
            traj = {}
            for tag, mult in (("lo", 1 - delta / 2), ("hi", 1 + delta / 2)):
                c = copy.deepcopy(cfg)
                v = base * mult
                c["oracles"][fam][key] = int(round(v)) if is_int else v
                traj[tag] = [observe_oracle(fam, corpus, c, seed + 131 * r)[1]
                             for r in range(R)]
            full = min(len(t) for ts in traj.values() for t in ts)
            for label, bud in (("small", min(budget_small, full)), ("full", full)):
                F = {k: np.array([summary_statistics(t[:bud], corpus) for t in ts])
                     for k, ts in traj.items()}
                acc, pval = _pair_accuracy(F["lo"], F["hi"], seed)
                rows.append({
                    "family": fam, "param": key, "rel_delta": delta,
                    "budget": label, "queries": bud,
                    "low": round(base * (1 - delta / 2), 3),
                    "high": round(base * (1 + delta / 2), 3),
                    "separability": acc, "perm_p": pval,
                })
    return rows


def separability_curve(corpus, cfg: dict, seed: int, fractions=None):
    """Pairwise separability as a function of query budget, under random probing.

    Returns
    -------
    names     : oracle names
    budgets   : list of platform-query counts
    mats      : dict budget -> (len x len) lower-triangular balanced-accuracy matrix
    q80       : (len x len) queries needed to reach 0.80 accuracy; inf if never
    """
    names = list(cfg["oracles"]["active"])
    R = int(cfg["separability"]["n_replicates"])

    # Collect trajectories once per (oracle, replicate); reuse across budgets.
    traj = {o: [observe_oracle(o, corpus, cfg, seed + 97 * r)[1] for r in range(R)]
            for o in names}
    total = min(len(rs) for o in names for rs in traj[o])
    if fractions is not None:
        budgets = sorted({max(10, int(f * total)) for f in fractions})
    else:
        budgets = sorted({b for b in BUDGET_GRID if b <= total} | {total})

    feats = {b: {o: np.array([summary_statistics(rs[:b], corpus) for rs in traj[o]])
                 for o in names} for b in budgets}

    mats: dict[int, np.ndarray] = {}
    for b in budgets:
        M = np.full((len(names), len(names)), np.nan)
        for i, a in enumerate(names):
            for j, c in enumerate(names):
                if j < i:
                    M[i, j] = _pair_accuracy(feats[b][a], feats[b][c], seed)[0]
        mats[b] = M

    q80 = np.full((len(names), len(names)), np.inf)
    for i in range(len(names)):
        for j in range(i):
            for b in budgets:
                if mats[b][i, j] >= 0.80:
                    q80[i, j] = b
                    break
    return names, budgets, mats, q80


def separability_matrix(corpus, cfg: dict, seed: int):
    """Backwards-compatible single-budget view (full budget)."""
    names, budgets, mats, _ = separability_curve(corpus, cfg, seed, fractions=(1.0,))
    return names, mats[budgets[-1]], None
