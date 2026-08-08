from __future__ import annotations

import numpy as np
import pytest

from aris.config import load
from aris.corpus import build_corpus
from aris.investigator.probe_designer import disagreement_score, select_cohort
from aris.probe_driver import (_feature_weights_for, _hypothesis_configs,
                               pilot_disagreement, run_comparison)


@pytest.fixture(scope="module")
def cfg():
    c = load("realistic")
    c["corpus"]["n_items"] = 200
    return c


@pytest.fixture(scope="module")
def corpus(cfg):
    return build_corpus(cfg, int(cfg["run"]["seed"]))


# --------------------------------------------------------------------- barrier
def test_probe_designer_does_not_import_platform():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "aris" / "investigator" / "probe_designer.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = "." * (node.level or 0) + (node.module or "")
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and "platform" in mod.lstrip(".").split("."):
            pytest.fail(f"probe_designer.py imports platform: {mod}")


# --------------------------------------------------------------- probe_designer
def test_disagreement_score_zero_for_identical_vectors():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert disagreement_score(v, v) == pytest.approx(0.0, abs=1e-9)


def test_disagreement_score_increases_with_difference():
    a = np.array([1.0, 1.0, 1.0])
    b_close = np.array([1.1, 1.0, 1.0])
    b_far = np.array([3.0, 1.0, 1.0])
    assert disagreement_score(a, b_far) > disagreement_score(a, b_close)


def test_disagreement_score_feature_weighting_emphasises_target_dim():
    a = np.array([1.0, 1.0])
    b = np.array([2.0, 2.0])  # equal absolute difference on both dims
    unweighted = disagreement_score(a, b)
    weighted = disagreement_score(a, b, feature_weights=np.array([4.0, 1.0]))
    assert weighted > unweighted


def test_select_cohort_picks_highest_disagreement():
    candidates = [
        {"cohort_seed": 1, "disagreement": 0.2, "n_personas": 5},
        {"cohort_seed": 2, "disagreement": 0.9, "n_personas": 5},
        {"cohort_seed": 3, "disagreement": 0.5, "n_personas": 5},
    ]
    chosen = select_cohort(candidates)
    assert chosen["cohort_seed"] == 2
    assert chosen["utility"] == pytest.approx(0.9)


def test_select_cohort_applies_cost_penalty():
    candidates = [
        {"cohort_seed": 1, "disagreement": 0.5, "n_personas": 2},
        {"cohort_seed": 2, "disagreement": 0.6, "n_personas": 100},
    ]
    # with a large enough cost weight, the smaller cohort should win despite
    # slightly lower raw disagreement
    chosen = select_cohort(candidates, cost_weight=0.1)
    assert chosen["cohort_seed"] == 1


def test_select_cohort_raises_on_empty_list():
    with pytest.raises(ValueError):
        select_cohort([])


# ------------------------------------------------------------------- driver
def test_hypothesis_configs_differ_only_in_target_param(cfg):
    lo, hi = _hypothesis_configs(cfg, "O4", "lambda_d", 0.3, 0.9)
    assert lo["oracles"]["O4"]["lambda_d"] == 0.3
    assert hi["oracles"]["O4"]["lambda_d"] == 0.9
    # every other oracle's config must be untouched
    for name in ("O1", "O2", "O3", "O5", "O6"):
        assert lo["oracles"][name] == cfg["oracles"][name]
        assert hi["oracles"][name] == cfg["oracles"][name]


def test_hypothesis_configs_preserve_int_type_for_integer_params(cfg):
    lo, hi = _hypothesis_configs(cfg, "O6", "switch_round", 17.5, 32.5)
    assert isinstance(lo["oracles"]["O6"]["switch_round"], int)
    assert isinstance(hi["oracles"]["O6"]["switch_round"], int)


def test_feature_weights_emphasise_mapped_feature():
    from aris.diagnostics import FEATURE_NAMES

    w = _feature_weights_for("O4", "lambda_d", focus_gain=4.0)
    idx = FEATURE_NAMES.index("intra_slate_sim")
    assert w[idx] == 4.0
    assert (w[np.arange(len(w)) != idx] == 1.0).all()


def test_feature_weights_default_uniform_for_unknown_pair():
    w = _feature_weights_for("O99", "nonexistent")
    assert (w == 1.0).all()


def test_pilot_disagreement_is_deterministic(cfg, corpus):
    lo, hi = _hypothesis_configs(cfg, "O4", "lambda_d", 0.3, 0.9)
    d1 = pilot_disagreement(corpus, lo, hi, "O4", cohort_seed=7, n_personas=4, pilot_rounds=4)
    d2 = pilot_disagreement(corpus, lo, hi, "O4", cohort_seed=7, n_personas=4, pilot_rounds=4)
    assert d1 == pytest.approx(d2)


def test_pilot_disagreement_zero_when_hypotheses_are_identical(cfg, corpus):
    lo, hi = _hypothesis_configs(cfg, "O4", "lambda_d", 0.55, 0.55)
    d = pilot_disagreement(corpus, lo, hi, "O4", cohort_seed=3, n_personas=4, pilot_rounds=4)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_run_comparison_random_mode_produces_valid_curve(cfg, corpus):
    base = cfg["oracles"]["O4"]["lambda_d"]
    result = run_comparison(corpus, cfg, "O4", "lambda_d", base * 0.7, base * 1.3,
                            mode="random", total_budget_replicates=4,
                            replicate_personas=4, replicate_rounds=4, seed=0, check_every=2)
    assert result["mode"] == "random"
    assert len(result["curve"]) == 2
    for c in result["curve"]:
        assert 0.5 <= c["separability"] <= 1.0
        assert 0.0 <= c["p_value"] <= 1.0
    assert result["total_final_query_cost"] > 0
    assert result["total_pilot_query_cost"] == 0  # random mode never pilots


def test_run_comparison_active_mode_tracks_pilot_cost(cfg, corpus):
    base = cfg["oracles"]["O4"]["lambda_d"]
    result = run_comparison(corpus, cfg, "O4", "lambda_d", base * 0.7, base * 1.3,
                            mode="active", total_budget_replicates=3,
                            replicate_personas=4, replicate_rounds=4, seed=0,
                            n_candidate_cohorts=3, pilot_personas=3, pilot_rounds=3, check_every=3)
    assert result["mode"] == "active"
    assert result["total_pilot_query_cost"] > 0
    assert result["total_pilot_query_cost"] == 3 * 3 * 3 * 3  # cohorts x reps x personas x rounds


def test_run_comparison_rejects_unknown_mode(cfg, corpus):
    with pytest.raises(ValueError):
        run_comparison(corpus, cfg, "O4", "lambda_d", 0.3, 0.9, mode="bogus",
                       total_budget_replicates=1, replicate_personas=2, replicate_rounds=2, seed=0)


def test_run_comparison_is_deterministic(cfg, corpus):
    base = cfg["oracles"]["O4"]["lambda_d"]
    r1 = run_comparison(corpus, cfg, "O4", "lambda_d", base * 0.7, base * 1.3,
                        mode="active", total_budget_replicates=3, replicate_personas=4,
                        replicate_rounds=4, seed=5, n_candidate_cohorts=3,
                        pilot_personas=3, pilot_rounds=3, check_every=3)
    r2 = run_comparison(corpus, cfg, "O4", "lambda_d", base * 0.7, base * 1.3,
                        mode="active", total_budget_replicates=3, replicate_personas=4,
                        replicate_rounds=4, seed=5, n_candidate_cohorts=3,
                        pilot_personas=3, pilot_rounds=3, check_every=3)
    assert r1["curve"] == r2["curve"]