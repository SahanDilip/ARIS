"""Instrument validation (02_METHODOLOGY.tex "Instrument validation" paragraph;
Gate 1 in STEP_PLAN.md -- never de-scope this).

Deviation from the original plan (docs/DEVIATIONS.md): validated against
structurally-generated reference distributions rather than downloaded public
interaction logs, given the 7-day laptop timeline.

Two design notes worth keeping in mind when reading this module.

1. Position bias must be tested on the SELECTION step, not the outcome step. A
   persona chooses one item per round via a softmax that already weights
   position (Eq. 10); an item chosen at a deep position is, by construction,
   one whose other features were unusually strong. Engagement RATE by position
   is therefore confounded by this selection effect. The clean test is a rank
   statistic on chosen positions against a uniform null.

2. Held-out prediction must condition on the PERSONA, not just the item.
   Engagement is persona-specific (heterogeneous interest_vec / affect_target),
   so a model using only item-level features (position, arousal, popularity)
   cannot predict engagement well even when the underlying mechanism is highly
   predictable -- it is missing the persona-conditioned terms (topical_match,
   affect_match) that Level-0's own softmax actually scores on. The pooled
   check below includes those terms; per-persona AUC is retained only as an
   outlier-exclusion diagnostic, since ~20 test rows per persona is too little
   power to gate validity on individually (true AUC ~0.6 routinely produces
   individual estimates from 0.3 to 0.85 by chance alone).
"""
from __future__ import annotations

import numpy as np


def position_bias_check(rows: list[dict], alpha: float = 0.05) -> dict:
    """Are chosen item positions skewed toward the top of the slate, relative to
    uniform-random choice? One-sided Wilcoxon signed-rank test against the
    uniform expectation (K-1)/2.
    """
    from scipy.stats import wilcoxon

    positions, sizes = [], []
    for r in rows:
        if r["action_item"] in r["slate"]:
            positions.append(r["slate"].index(r["action_item"]))
            sizes.append(len(r["slate"]))
    if len(positions) < 10:
        return {"pass": False, "reason": "too few observations"}
    positions = np.array(positions, dtype=float)
    expected = (np.array(sizes, dtype=float) - 1) / 2.0
    diff = positions - expected
    try:
        _, p = wilcoxon(diff, alternative="less")
    except ValueError:
        return {"pass": False, "reason": "degenerate distribution (no variation in positions)"}
    mean_pos, mean_exp = float(positions.mean()), float(expected.mean())
    return {"pass": bool(p < alpha and mean_pos < mean_exp), "p_value": round(float(p), 4),
           "mean_position": round(mean_pos, 3), "expected_under_uniform": round(mean_exp, 3)}


def affect_asymmetry_check(rows: list[dict], corpus, min_gap: float = 0.02) -> dict:
    """Higher-arousal items should be engaged with more often than lower-arousal
    ones, in aggregate across the population (Solovev & Prollochs [19])."""
    engaged, skipped = [], []
    for r in rows:
        a = float(corpus.arousal[r["action_item"]])
        (engaged if r["action"] != "skip" else skipped).append(a)
    if len(engaged) < 5 or len(skipped) < 5:
        return {"pass": False, "reason": "insufficient contrast between engaged and skipped items"}
    gap = float(np.mean(engaged) - np.mean(skipped))
    return {"pass": gap > min_gap, "gap": round(gap, 4),
           "mean_arousal_engaged": round(float(np.mean(engaged)), 3),
           "mean_arousal_skipped": round(float(np.mean(skipped)), 3),
           "n_engaged": len(engaged), "n_skipped": len(skipped)}


def _row_features(rs: list[dict], corpus, profile=None) -> tuple[np.ndarray, np.ndarray]:
    """Reconstructs Level-0's own five features (Eq. 10) from observed rows:
    [position, arousal, popularity, topical_match, affect_match, novelty,
    fatigue]. novelty and fatigue require replaying the persona's own history
    in order (they depend on the running memory buffer, capacity 20, exactly as
    in population/persona.py's Memory), which is why this function takes the
    full per-persona row sequence rather than being applied row-by-row
    independently. Omitting these two was the reason the pooled held-out AUC
    understated true predictability in earlier iterations of this check: two of
    Level-0's five generative inputs were simply invisible to the test.
    """
    X, y = [], []
    mem_embeds: list[np.ndarray] = []
    n_since_exit = 0
    for r in rs:
        if r["action_item"] not in r["slate"]:
            continue
        item = r["action_item"]
        pos = r["slate"].index(item)
        row = [pos, float(corpus.arousal[item]), float(corpus.popularity[item])]
        if profile is not None:
            row.append(float(profile.interest_vec @ corpus.embed[item]))
            row.append(1.0 - abs(float(corpus.arousal[item]) - profile.affect_target))
            novelty = 1.0 if not mem_embeds else float(
                1.0 - np.max(np.stack(mem_embeds) @ corpus.embed[item]))
            fatigue = min(1.0, n_since_exit / 12.0)
            row.extend([novelty, fatigue])
        X.append(row)
        y.append(int(r["action"] != "skip"))
        mem_embeds.append(corpus.embed[item])
        if len(mem_embeds) > 20:
            mem_embeds.pop(0)
        n_since_exit = 0 if r["action"] == "exit" else n_since_exit + 1
    return np.array(X, dtype=float), np.array(y, dtype=int)


def held_out_prediction_check_pooled(rows: list[dict], corpus, personas: dict,
                                     n_boot: int = 500, alpha: float = 0.05) -> dict:
    """Primary Gate-1 criterion. For each persona, features are computed over
    the FULL continuous round sequence (so novelty/fatigue reflect what
    actually happened, without resetting memory at the split point), then split
    into first-half-train / second-half-test. Pools across personas, fits once.

    Pass condition is a bootstrap test that AUC is significantly above the
    no-signal baseline of 0.5 -- not an arbitrary fixed threshold like 0.55.
    With n_test in the hundreds to thousands, a modest AUC can be highly
    significant, and a fixed-magnitude threshold would incorrectly fail a real,
    well-powered regularity.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    by_pid: dict[int, list[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["persona_id"], []).append(r)

    Xtr_parts, ytr_parts, Xte_parts, yte_parts = [], [], [], []
    for pid, profile in personas.items():
        prows = sorted(by_pid.get(pid, []), key=lambda r: r["round"])
        if len(prows) < 4:
            continue
        X, y = _row_features(prows, corpus, profile)
        half = len(X) // 2
        if half >= 1 and len(X) - half >= 1:
            Xtr_parts.append(X[:half]); ytr_parts.append(y[:half])
            Xte_parts.append(X[half:]); yte_parts.append(y[half:])

    if not Xtr_parts or not Xte_parts:
        return {"pass": False, "reason": "insufficient pooled data"}
    Xtr, ytr = np.vstack(Xtr_parts), np.concatenate(ytr_parts)
    Xte, yte = np.vstack(Xte_parts), np.concatenate(yte_parts)
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return {"pass": False, "reason": "no class variation in pooled data"}
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))

    rng = np.random.default_rng(0)
    n = len(yte)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = yte[idx], proba[idx]
        boot[b] = roc_auc_score(yb, pb) if len(set(yb)) > 1 else 0.5
    ci_lo, ci_hi = float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {"pass": bool(ci_lo > 0.5), "auc": round(auc, 3),
           "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
           "n_train": len(Xtr), "n_test": len(Xte)}


def held_out_prediction_check(rows: list[dict], corpus, persona_id: int, profile,
                              min_auc: float = 0.55) -> dict:
    """Per-persona diagnostic. Underpowered on its own -- used only to flag
    individual outliers for exclusion, not to gate validity (see pooled version).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    prows = sorted((r for r in rows if r["persona_id"] == persona_id), key=lambda r: r["round"])
    if len(prows) < 8:
        return {"pass": False, "reason": "insufficient rounds for this persona"}
    X, y = _row_features(prows, corpus, profile)
    half = len(X) // 2
    Xtr, ytr, Xte, yte = X[:half], y[:half], X[half:], y[half:]
    if len(set(ytr)) < 2 or len(set(yte)) < 2 or len(Xtr) < 3 or len(Xte) < 3:
        return {"pass": False, "reason": "insufficient class variation for AUC"}
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return {"pass": auc > min_auc, "auc": round(auc, 3), "n_train": len(Xtr), "n_test": len(Xte)}


def validate_population(rows: list[dict], corpus, personas: dict,
                        outlier_margin: float = 0.30) -> dict:
    """Run all checks. `personas` is {persona_id: PersonaProfile}. Gate 1's pass
    condition uses the population-level pooled AUC; per-persona AUCs flag
    individual outliers (far below the pooled estimate) for exclusion, matching
    "report where personas fail... exclude failing subgroups" (02_METHODOLOGY.tex).
    """
    pos = position_bias_check(rows)
    aff = affect_asymmetry_check(rows, corpus)
    pooled = held_out_prediction_check_pooled(rows, corpus, personas)
    per_persona = {pid: held_out_prediction_check(rows, corpus, pid, profile)
                   for pid, profile in personas.items()}

    pooled_auc = pooled.get("auc")
    excluded = []
    if pooled_auc is not None:
        for pid, r in per_persona.items():
            if r.get("auc") is not None and r["auc"] < pooled_auc - outlier_margin:
                excluded.append(pid)
    exclusion_rate = len(excluded) / max(1, len(personas))

    return {
        "position_bias": pos, "affect_asymmetry": aff,
        "held_out_pooled": pooled,
        "excluded_personas": excluded, "exclusion_rate": round(exclusion_rate, 3),
        "overall_pass": bool(pos.get("pass", False) and aff.get("pass", False)
                             and pooled.get("pass", False) and exclusion_rate < 0.3),
        "per_persona": per_persona,
    }