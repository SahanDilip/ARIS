from __future__ import annotations

import numpy as np
import pytest

from aris.config import load
from aris.corpus import build_corpus
from aris.llm.client import StubLLM
from aris.platform.env import BlackBoxPlatform
from aris.platform.oracles import make_oracle
from aris.population.persona import Memory, make_personas
from aris.population.plausibility import (discriminator_auc, reference_trajectory,
                                          trajectory_features)
from aris.population.policy import PopulationPolicy
from aris.population.validate import (affect_asymmetry_check, position_bias_check,
                                      validate_population)
from aris.population_driver import browse_population, heterogeneity_entropy


@pytest.fixture(scope="module")
def cfg():
    c = load("realistic")
    c["corpus"]["n_items"] = 300
    c["population"]["n_personas"] = 20
    c["population"]["rounds"] = 20
    return c


@pytest.fixture(scope="module")
def corpus(cfg):
    return build_corpus(cfg, int(cfg["run"]["seed"]))


@pytest.fixture(scope="module")
def browsed(cfg, corpus):
    plat = BlackBoxPlatform(corpus, make_oracle("O4", corpus, cfg), cfg, 1)
    personas = make_personas(corpus, int(cfg["population"]["n_personas"]), 1)
    policy = PopulationPolicy()
    rows, telem = browse_population(plat, corpus, personas, policy,
                                     int(cfg["population"]["rounds"]), None, 1)
    return rows, telem, personas, policy


# --------------------------------------------------------------------- driver
def test_browse_population_closes_loop(browsed, cfg):
    rows, telem, personas, policy = browsed
    assert len(rows) == len(personas) * cfg["population"]["rounds"]
    assert all(r["level"] == 0 for r in rows), "no LLM passed -> must never escalate"
    assert policy.escalation_rate() == 0.0


def test_browse_population_is_deterministic(cfg, corpus):
    plat1 = BlackBoxPlatform(corpus, make_oracle("O4", corpus, cfg), cfg, 5)
    p1 = make_personas(corpus, 10, 5)
    r1, _ = browse_population(plat1, corpus, p1, PopulationPolicy(), 10, None, 5)

    plat2 = BlackBoxPlatform(corpus, make_oracle("O4", corpus, cfg), cfg, 5)
    p2 = make_personas(corpus, 10, 5)
    r2, _ = browse_population(plat2, corpus, p2, PopulationPolicy(), 10, None, 5)

    assert [r["action_item"] for r in r1] == [r["action_item"] for r in r2], (
        "same seed must reproduce identical trajectories -- this is the exact "
        "bug class fixed in platform/env.py's session() seeding")


def test_timestamps_are_monotonic_per_persona(browsed):
    rows, *_ = browsed
    by_pid: dict[int, list[float]] = {}
    for r in rows:
        by_pid.setdefault(r["persona_id"], []).append(r["ts"])
    for pid, ts in by_pid.items():
        assert ts == sorted(ts), f"persona {pid} timestamps are not monotonic"
        assert ts[0] > 0, "clock should have advanced past zero after one round"


def test_heterogeneity_entropy_is_bounded(cfg, corpus):
    personas = make_personas(corpus, 30, 2)
    h = heterogeneity_entropy(personas)
    assert 0.0 <= h <= np.log(2 ** 8)


# -------------------------------------------------------------------- policy
def test_escalation_uses_llm_when_provided(cfg, corpus):
    plat = BlackBoxPlatform(corpus, make_oracle("O4", corpus, cfg), cfg, 3)
    personas = make_personas(corpus, 6, 3)
    policy = PopulationPolicy(eta=0.0, novelty_escalate=0.0)  # force escalation every round
    llm = StubLLM(seed=3)
    rows, _ = browse_population(plat, corpus, personas, policy, 5, llm, 3)
    assert policy.n_l1 > 0
    assert any(r["level"] == 1 for r in rows)


def test_distill_updates_beta_and_clears_cache():
    policy = PopulationPolicy()
    before = policy.beta.copy()
    policy.cache = [(np.array([1.0, 0.5, 0.2, 0.8, 0.1]), 0) for _ in range(5)]
    report = policy.distill()
    assert report["n_cache"] == 5
    assert not policy.cache
    assert not np.allclose(policy.beta, before)


def test_item_probs_sum_to_one():
    policy = PopulationPolicy()
    feats = np.random.default_rng(0).standard_normal((8, 5))
    p = policy.item_probs(feats)
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()


# ------------------------------------------------------------------ validate
def test_position_bias_check_detects_uniform_null():
    """Uniform-random positions must NOT show significant top-bias."""
    rng = np.random.default_rng(0)
    rows = [{"slate": list(range(8)), "action_item": int(rng.integers(0, 8))} for _ in range(200)]
    r = position_bias_check(rows)
    assert r["pass"] is False


def test_position_bias_check_detects_real_bias():
    rng = np.random.default_rng(0)
    rows = [{"slate": list(range(8)), "action_item": int(rng.choice(8, p=[.3,.2,.15,.1,.1,.07,.05,.03]))}
            for _ in range(200)]
    r = position_bias_check(rows)
    assert r["pass"] is True


def test_affect_asymmetry_check(corpus):
    rows = [{"action_item": i, "action": "like" if corpus.arousal[i] > 0.5 else "skip"}
            for i in range(len(corpus))]
    r = affect_asymmetry_check(rows, corpus)
    assert r["pass"] is True
    assert r["gap"] > 0


def test_validate_population_end_to_end(browsed, corpus):
    rows, _, personas, _ = browsed
    val = validate_population(rows, corpus, {p.persona_id: p for p in personas})
    assert set(val) == {"position_bias", "affect_asymmetry", "held_out_pooled",
                        "excluded_personas", "exclusion_rate", "overall_pass", "per_persona"}
    assert isinstance(val["overall_pass"], bool)


# --------------------------------------------------------------- plausibility
def test_reference_trajectory_shape(corpus):
    ref = reference_trajectory(corpus, n_rounds=15, slate_size=8, seed=0)
    assert len(ref) == 15
    assert all(len(r["slate"]) == 8 for r in ref)
    ts = [r["ts"] for r in ref]
    assert ts == sorted(ts), "reference clock must be monotonic (the non-cumulative bug)"


def test_trajectory_features_shape(browsed, corpus):
    rows, *_ = browsed
    f = trajectory_features(rows, corpus, 0)
    assert f.shape == (8,)
    assert np.isfinite(f).all()


def test_discriminator_auc_runs(browsed, corpus, cfg):
    rows, _, personas, _ = browsed
    pcfg = cfg["population"]["plausibility"]
    r = discriminator_auc(rows, corpus, [p.persona_id for p in personas],
                          int(pcfg["slate_size"]), 0)
    assert r["auc"] is not None
    assert 0.0 <= r["auc"] <= 1.0