from __future__ import annotations

import numpy as np
import pytest

from aris.config import config_hash, load
from aris.corpus import build_corpus
from aris.diagnostics import (PRIMARY_PARAM, _pair_accuracy, parameter_resolution,
                              separability_curve)
from aris.llm.bench import recommend
from aris.llm.client import BudgetExceeded, OllamaLLM, StubLLM, make_llm


@pytest.fixture(scope="module")
def cfg():
    c = load("realistic")
    c["corpus"]["n_items"] = 300
    c["oracles"]["active"] = ["O1", "O2", "O5"]
    c["separability"] = {"n_agents": 4, "n_rounds": 5, "n_replicates": 3}
    c["llm"]["backend"] = "stub"
    return c


# --------------------------------------------------------------------- config
def test_overrides_and_include():
    base = load("base")
    real = load("realistic")
    assert real["platform"]["slate_size"] != base["platform"]["slate_size"]
    assert real["corpus"]["n_items"] == base["corpus"]["n_items"], "include should merge"
    ov = load("base", ["platform.noise_scale=1.25", "run.seed=5"])
    assert ov["platform"]["noise_scale"] == 1.25 and ov["run"]["seed"] == 5


def test_config_hash_is_stable_and_sensitive():
    a = load("base")
    assert config_hash(a) == config_hash(load("base"))
    b = load("base", ["run.seed=999"])
    assert config_hash(a) != config_hash(b)


# ------------------------------------------------------------------ diagnostics
def test_pair_accuracy_null_on_identical_distributions():
    """Two samples from the same distribution must not look separable."""
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((6, 16)), rng.standard_normal((6, 16))
    acc, null = _pair_accuracy(A, B, seed=1, n_perm=40)
    assert acc < 0.75, f"permutation adjustment failed: {acc} (null {null})"


def test_pair_accuracy_detects_a_real_shift():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 16))
    B = rng.standard_normal((6, 16)) + 4.0
    acc, _ = _pair_accuracy(A, B, seed=1, n_perm=40)
    assert acc > 0.85


def test_separability_curve_shape(cfg):
    corpus = build_corpus(cfg, 3)
    names, budgets, mats, q80 = separability_curve(corpus, cfg, 3)
    assert names == cfg["oracles"]["active"]
    assert len(budgets) >= 2 and budgets == sorted(budgets)
    M = mats[budgets[-1]]
    assert np.isnan(M[0, 1]) and not np.isnan(M[1, 0]), "should be lower-triangular"
    assert ((M[~np.isnan(M)] >= 0.5) & (M[~np.isnan(M)] <= 1.0)).all()
    assert q80.shape == M.shape


def test_parameter_resolution_covers_families(cfg):
    corpus = build_corpus(cfg, 3)
    rows = parameter_resolution(corpus, cfg, 3, deltas=(0.8,), budget_small=10)
    fams = {r["family"] for r in rows}
    assert fams == set(cfg["oracles"]["active"])
    for r in rows:
        assert r["param"] == PRIMARY_PARAM[r["family"]]
        assert 0.5 <= r["separability"] <= 1.0
        assert r["low"] < r["high"]


# -------------------------------------------------------------------------- llm
def test_stub_is_deterministic_and_json_valid():
    import json
    a, b = StubLLM(seed=1), StubLLM(seed=1)
    out = a.chat("sys", "user", json_mode=True)
    assert out == b.chat("sys", "user", json_mode=True)
    d = json.loads(out)
    assert set(d) == {"choice_index", "action", "dwell_ms", "reason"}
    assert StubLLM(seed=2).chat("sys", "user") != a.chat("sys", "user")


def test_make_llm_falls_back_to_stub_when_unreachable():
    cfg = load("base", ["llm.host=http://127.0.0.1:1"])
    assert isinstance(make_llm(cfg), StubLLM)


def test_ollama_cache_and_budget(tmp_path, monkeypatch):
    cfg = load("base", [f"llm.cache_dir={tmp_path}"])
    c = OllamaLLM(cfg, role="population", max_calls=1)
    calls = {"n": 0}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            calls["n"] += 1
            return {"message": {"content": "hi"}, "eval_count": 5, "prompt_eval_count": 9}

    monkeypatch.setattr("aris.llm.client.requests.post", lambda *a, **k: FakeResp())
    assert c.chat("s", "u") == "hi"
    assert c.chat("s", "u") == "hi", "second identical call must hit the cache"
    assert calls["n"] == 1 and c.stats.cache_hits == 1
    with pytest.raises(BudgetExceeded):
        c.chat("s", "different prompt")


def test_recommend_handles_missing_model():
    r = recommend({"backend": "stub"})
    assert "commands" in r and r["status"].startswith("no local model")


def test_recommend_scales_with_latency():
    fast = recommend({"backend": "ollama", "latency_s_median": 0.8, "model": "m"})
    slow = recommend({"backend": "ollama", "latency_s_median": 12.0, "model": "m"})
    assert fast["l1_calls_per_20min"] > slow["l1_calls_per_20min"]
    assert slow["status"] == "tight"
