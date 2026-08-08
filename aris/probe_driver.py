"""Platform-facing simulation for the Probe Designer. Lives outside
aris/investigator/ for the same reason population_driver.py and
investigator_driver.py do: this is the one place allowed to import both the
platform and the (platform-free) probe_designer scoring logic.

Implements the RQ2 experiment (04_EVALUATION.md E2): does actively choosing
which persona cohort to probe next -- by pilot-simulating expected disagreement
under two competing parameter hypotheses -- reach statistical separability
faster (fewer total queries) than choosing cohorts uniformly at random, at a
matched query budget?
"""
from __future__ import annotations

import copy

import numpy as np

from .diagnostics import FEATURE_NAMES, _pair_accuracy, summary_statistics
from .investigator.probe_designer import disagreement_score, select_cohort
from .platform.env import BlackBoxPlatform
from .platform.oracles import make_oracle
from .population.persona import make_personas
from .population.policy import PopulationPolicy
from .population_driver import browse_population

#: which audit statistic each oracle's primary parameter is expected to move
#: most directly (see aris/diagnostics.py FEATURE_NAMES for the full list).
#: Used to focus the Probe Designer's disagreement scoring on the feature(s)
#: mechanistically linked to the hypothesis under test, rather than an
#: unweighted distance across all 16 statistics -- found necessary
#: empirically (docs/DEVIATIONS.md D13): the unweighted score diluted signal
#: from the relevant statistic among many unrelated ones and active cohort
#: selection showed no reliable advantage over random until focused.
_PARAM_RELEVANT_FEATURES = {
    ("O1", "gamma"): ["pop_lift", "pos_corr_pop"],
    ("O2", "alpha"): ["arousal_lift", "pos_corr_arousal"],
    ("O3", "epsilon"): ["exposure_gini", "novelty_rate"],
    ("O4", "lambda_d"): ["intra_slate_sim"],
    ("O5", "velocity_weight"): ["velocity_lift", "pos_corr_velocity"],
    ("O6", "switch_round"): ["drift_slope", "churn"],
}


def _feature_weights_for(oracle: str, param_key: str, focus_gain: float = 4.0) -> np.ndarray:
    """Weight vector emphasising the statistic(s) linked to (oracle, param_key)
    by `focus_gain`, all others at 1.0. Falls back to uniform weights (no
    targeting) if the pair isn't in the map, which is a graceful degradation
    rather than an error -- new oracle/parameter pairs simply get the
    unweighted behaviour until someone adds an entry.
    """
    w = np.ones(len(FEATURE_NAMES))
    relevant = _PARAM_RELEVANT_FEATURES.get((oracle, param_key), [])
    for name in relevant:
        if name in FEATURE_NAMES:
            w[FEATURE_NAMES.index(name)] = focus_gain
    return w


def _hypothesis_configs(cfg: dict, oracle: str, param_key: str, lo: float, hi: float) -> tuple[dict, dict]:
    c_lo, c_hi = copy.deepcopy(cfg), copy.deepcopy(cfg)
    is_int = isinstance(cfg["oracles"][oracle][param_key], int)
    c_lo["oracles"][oracle][param_key] = int(round(lo)) if is_int else lo
    c_hi["oracles"][oracle][param_key] = int(round(hi)) if is_int else hi
    return c_lo, c_hi


def _run_cohort(corpus, cfg: dict, oracle: str, cohort_seed: int, n_personas: int,
                n_rounds: int) -> np.ndarray:
    """One population run against one hypothesis's oracle config; returns the
    audit-style summary-statistic vector for that run.
    """
    plat = BlackBoxPlatform(corpus, make_oracle(oracle, corpus, cfg), cfg, cohort_seed)
    personas = make_personas(corpus, n_personas, cohort_seed)
    policy = PopulationPolicy()
    rows, _ = browse_population(plat, corpus, personas, policy, n_rounds, None, cohort_seed)
    return summary_statistics(rows, corpus)


def pilot_disagreement(corpus, cfg_lo: dict, cfg_hi: dict, oracle: str, cohort_seed: int,
                       n_personas: int, pilot_rounds: int,
                       feature_weights: np.ndarray | None = None) -> float:
    """Run the SAME cohort (same persona interest vectors, via the shared
    cohort_seed) for a short pilot against both hypotheses' oracle configs,
    and score how differently they behaved. This never touches the real,
    secretly-fixed ground-truth oracle -- it is a posterior-predictive check
    under each candidate hypothesis, which is legitimate active design and
    does not require knowing which hypothesis is actually true.
    """
    fa = _run_cohort(corpus, cfg_lo, oracle, cohort_seed, n_personas, pilot_rounds)
    fb = _run_cohort(corpus, cfg_hi, oracle, cohort_seed, n_personas, pilot_rounds)
    return disagreement_score(fa, fb, feature_weights=feature_weights)


def run_comparison(corpus, cfg: dict, oracle: str, param_key: str, lo: float, hi: float,
                   mode: str, total_budget_replicates: int, replicate_personas: int,
                   replicate_rounds: int, seed: int, n_candidate_cohorts: int = 5,
                   pilot_personas: int = 6, pilot_rounds: int = 5,
                   check_every: int = 1) -> dict:
    """Run `mode` in {"active", "random"} for `total_budget_replicates`
    replicates, each replicate contributing one labelled sample per class
    (lo, hi) to a running two-sample separability test. Returns a budget curve
    of (n_replicates, separability, p_value) plus the query count spent,
    including pilot-simulation cost for the active arm (tracked separately,
    not charged against the final-data query budget -- see D12 in
    docs/DEVIATIONS.md for why this simplification was made under the timeline).
    """
    cfg_lo, cfg_hi = _hypothesis_configs(cfg, oracle, param_key, lo, hi)
    rng = np.random.default_rng(seed)
    weights = _feature_weights_for(oracle, param_key)

    feats_lo: list[np.ndarray] = []
    feats_hi: list[np.ndarray] = []
    curve: list[dict] = []
    pilot_query_cost = 0
    final_query_cost = 0

    for rep in range(total_budget_replicates):
        if mode == "active":
            candidates = []
            for _ in range(n_candidate_cohorts):
                cohort_seed = int(rng.integers(0, 2**31))
                d = pilot_disagreement(corpus, cfg_lo, cfg_hi, oracle, cohort_seed,
                                       pilot_personas, pilot_rounds, feature_weights=weights)
                candidates.append({"cohort_seed": cohort_seed, "disagreement": d,
                                   "n_personas": pilot_personas})
                pilot_query_cost += pilot_personas * pilot_rounds
            chosen = select_cohort(candidates)
            cohort_seed = chosen["cohort_seed"]
        elif mode == "random":
            cohort_seed = int(rng.integers(0, 2**31))
        else:
            raise ValueError(f"unknown mode {mode!r}")

        feats_lo.append(_run_cohort(corpus, cfg_lo, oracle, cohort_seed,
                                    replicate_personas, replicate_rounds))
        feats_hi.append(_run_cohort(corpus, cfg_hi, oracle, cohort_seed,
                                    replicate_personas, replicate_rounds))
        final_query_cost += 2 * replicate_personas * replicate_rounds

        if (rep + 1) % check_every == 0 or rep == total_budget_replicates - 1:
            if len(feats_lo) >= 2:
                sep, pval = _pair_accuracy(np.array(feats_lo), np.array(feats_hi), seed)
            else:
                sep, pval = 0.5, 1.0
            curve.append({"n_replicates": rep + 1, "separability": sep, "p_value": pval,
                         "final_query_cost": final_query_cost,
                         "pilot_query_cost": pilot_query_cost if mode == "active" else 0})

    return {
        "mode": mode, "oracle": oracle, "param_key": param_key, "lo": lo, "hi": hi,
        "curve": curve,
        "queries_to_significance": next(
            (c["n_replicates"] for c in curve if c["p_value"] < 0.05), None),
        "final_separability": curve[-1]["separability"] if curve else None,
        "final_p_value": curve[-1]["p_value"] if curve else None,
        "total_final_query_cost": final_query_cost,
        "total_pilot_query_cost": pilot_query_cost,
    }