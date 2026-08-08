"""Probe Designer (D in 01_ARCHITECTURE.md's roster): decides WHICH persona
cohort to probe next, given a disagreement signal computed elsewhere.

MUST NOT import aris.platform (enforced by tests/test_barrier.py). All
simulation needed to COMPUTE disagreement for a candidate cohort happens in
aris/probe_driver.py (outside this package, same pattern as
population_driver.py / investigator_driver.py); this module only scores and
selects among precomputed candidates.

Deviation from the original plan (docs/DEVIATIONS.md D12): Algorithm 2's full
score is disagreement - cost - implausibility (Eq. 12). This MVP implements
the disagreement term only; cost is a placeholder constant and implausibility
is not scored at all (the population tier's own plausibility discriminator
from Step 2 remains the project's only plausibility check). This is a scope
cut under the study's timeline, not an oversight, and is recorded so it can be
extended later without re-deriving the design.
"""
from __future__ import annotations

import numpy as np


def disagreement_score(feats_a: np.ndarray, feats_b: np.ndarray,
                       feature_weights: np.ndarray | None = None) -> float:
    """Scalar disagreement between two summary-statistic vectors describing
    the SAME candidate cohort's simulated behaviour under two competing
    hypotheses (e.g. a low vs high parameter value). Normalised Euclidean
    distance in standardised feature space, computed relative to a shared
    scale so that no single feature with a larger natural range dominates.

    `feature_weights`, if given, downweights features not mechanistically
    linked to the parameter under test. Found necessary empirically: an
    unweighted 16-feature distance diluted the signal from the one or two
    statistics that actually respond to a given parameter (e.g. lambda_d, a
    diversity-reranking penalty, mainly moves intra_slate_sim) among many
    unrelated ones, and active cohort selection showed no reliable advantage
    over random until the score was focused on the relevant statistic(s).
    """
    a, b = np.asarray(feats_a, dtype=float), np.asarray(feats_b, dtype=float)
    scale = np.maximum(np.abs(a) + np.abs(b), 1e-6)
    w = np.ones_like(a) if feature_weights is None else np.asarray(feature_weights, dtype=float)
    return float(np.linalg.norm(w * (a - b) / scale))


def select_cohort(candidates: list[dict], cost_weight: float = 0.0) -> dict:
    """Eq. 12 (disagreement term only, see module docstring): pick the
    candidate with the highest disagreement score minus a cost penalty
    proportional to cohort size. `candidates` is a list of
    {"cohort_seed": int, "disagreement": float, "n_personas": int, ...}.
    Returns the chosen candidate dict unchanged (plus its rank among peers,
    for logging).
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    scored = [
        (c["disagreement"] - cost_weight * c.get("n_personas", 0), c)
        for c in candidates
    ]
    scored.sort(key=lambda t: -t[0])
    best = dict(scored[0][1])
    best["utility"] = scored[0][0]
    best["rank_among_candidates"] = 1
    return best