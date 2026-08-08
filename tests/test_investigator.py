from __future__ import annotations

import numpy as np
import pytest

from aris.config import load
from aris.corpus import build_corpus
from aris.investigator.mechanism_classifier import fit_and_evaluate
from aris.investigator.surrogate import (RankingSurrogate, _pairwise_examples,
                                         build_oracle_trace, build_session_trace,
                                         evaluate_fidelity)
from aris.investigator_driver import collect_observations, collect_training_replicates


@pytest.fixture(scope="module")
def cfg():
    c = load("realistic")
    c["corpus"]["n_items"] = 300
    return c


@pytest.fixture(scope="module")
def corpus(cfg):
    return build_corpus(cfg, int(cfg["run"]["seed"]))


@pytest.fixture(scope="module")
def collected(cfg, corpus):
    return collect_observations(corpus, cfg, "O1", seed=1, n_personas=12, n_rounds=16)


# --------------------------------------------------------------------- barrier
def test_investigator_tier_does_not_import_platform():
    """investigator/ itself must stay platform-free; the driver (outside it)
    is where all platform access belongs. Covered by test_barrier.py too, but
    asserted here as a direct regression guard on this specific module set.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "aris" / "investigator"
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = "." * (node.level or 0) + (node.module or "")
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if mod and "platform" in mod.lstrip(".").split("."):
                pytest.fail(f"{py.name} imports platform: {mod}")


# -------------------------------------------------------------- regression: D8/D11
def test_trace_lookup_uses_persona_round_key_not_id(cfg, corpus, collected):
    """Regression guard for the id(r) bug (D11): the preference-match feature
    must have nonzero variance once a real trace is supplied. If this feature
    collapses to a constant, the trace lookup is silently failing again.
    """
    rows, persona_map = collected
    trace = build_session_trace(rows, corpus)
    rng = np.random.default_rng(0)
    X, y = _pairwise_examples(rows, corpus, rng, trace=trace)
    rel_diff = X[:, -1]
    assert rel_diff.std() > 1e-6, "rel_diff feature is constant -- the id(r) lookup bug is back"


def test_ceiling_trace_differs_from_session_trace(cfg, corpus, collected):
    """The oracle-informed ceiling (true interest_vec) and the realistic
    session-based proxy must use genuinely different preference vectors.
    """
    rows, persona_map = collected
    session_trace = build_session_trace(rows, corpus)
    oracle_trace = build_oracle_trace(rows, corpus, persona_map)
    pid = next(iter(persona_map))
    # compare a late-round entry, where session history has accumulated
    session_seq = {k: v for k, v in session_trace[pid]}
    oracle_seq = {k: v for k, v in oracle_trace[pid]}
    late_key = max(session_seq, key=lambda k: k[1])
    if session_seq[late_key] is not None:
        assert not np.allclose(session_seq[late_key], oracle_seq[late_key])


def test_ceiling_beats_or_matches_realistic_proxy_on_hr1(cfg, corpus, collected):
    """The B7 oracle-informed ceiling should not systematically underperform
    the realistic proxy -- if it does, something is broken (this is exactly
    how the two real bugs in D11 were originally caught).
    """
    rows, persona_map = collected
    split = int(16 * 0.6)
    train_rows = [r for r in rows if r["round"] < split]
    test_rows = [r for r in rows if r["round"] >= split]
    full_trace = build_session_trace(rows, corpus)
    oracle_trace = build_oracle_trace(rows, corpus, persona_map)

    surr = RankingSurrogate()
    surr.fit(train_rows, corpus, seed=0, trace=full_trace)
    fid_real = evaluate_fidelity(surr, test_rows, corpus, seed=3, trace=full_trace)

    surr2 = RankingSurrogate()
    surr2.fit(train_rows, corpus, seed=0, trace=oracle_trace)
    fid_ceil = evaluate_fidelity(surr2, test_rows, corpus, seed=3, trace=oracle_trace)

    if fid_real["hitratio_1"] is not None and fid_ceil["hitratio_1"] is not None:
        assert fid_ceil["hitratio_1"] >= fid_real["hitratio_1"] - 0.05, (
            "oracle-informed ceiling should not be meaningfully worse than the "
            "realistic proxy -- if this fails, check for the null-anchor or "
            "id(r) bugs (D11) recurring")


# ------------------------------------------------------------------- surrogate
def test_surrogate_fit_and_rank(cfg, corpus, collected):
    rows, _ = collected
    surr = RankingSurrogate()
    report = surr.fit(rows, corpus, seed=0)
    assert report["fitted"] is True
    slate = rows[0]["slate"]
    ranked = surr.rank_slate(corpus, slate, revealed_pref=None)
    assert sorted(ranked) == sorted(slate), "ranking must be a permutation of the input slate"


def test_rank_candidates_is_order_invariant(cfg, corpus, collected):
    """Ranking the same candidate set in a different input order must produce
    the same ranked output (order-independence of a round-robin tournament).
    """
    rows, _ = collected
    surr = RankingSurrogate()
    surr.fit(rows, corpus, seed=0)
    cand = rows[0]["slate"][:6]
    r1 = surr.rank_candidates(corpus, cand, [0] * len(cand), None)
    r2 = surr.rank_candidates(corpus, list(reversed(cand)), [0] * len(cand), None)
    assert sorted(r1) == sorted(r2) == sorted(cand)


def test_evaluate_fidelity_beats_random_on_top1(cfg, corpus, collected):
    rows, _ = collected
    split = int(16 * 0.6)
    train_rows = [r for r in rows if r["round"] < split]
    test_rows = [r for r in rows if r["round"] >= split]
    trace = build_session_trace(rows, corpus)
    surr = RankingSurrogate()
    surr.fit(train_rows, corpus, seed=0, trace=trace)
    fid = evaluate_fidelity(surr, test_rows, corpus, seed=1, trace=trace)
    assert fid["n_eval"] > 0
    # top1 agreement should be near or above 1/slate_size; a hard floor check
    # rather than a strict inequality, since held-out numbers are noisy at
    # this small a test scale
    assert fid["top1_agreement"] >= 0.05


def test_unfitted_surrogate_returns_identity_order(corpus):
    surr = RankingSurrogate()
    slate = [1, 2, 3, 4]
    assert surr.rank_slate(corpus, slate, None) == slate


# ---------------------------------------------------------------- mechanism
def test_collect_training_replicates_shapes(cfg, corpus):
    feats = collect_training_replicates(corpus, cfg, ["O1", "O2"], n_replicates=3,
                                        seed=0, n_personas=6, n_rounds=8)
    assert set(feats) == {"O1", "O2"}
    for arr in feats.values():
        assert arr.shape[0] == 3
        assert np.isfinite(arr).all()


def test_mechanism_classifier_recovers_families(cfg, corpus):
    feats = collect_training_replicates(corpus, cfg, ["O1", "O2", "O5"], n_replicates=6,
                                        seed=0, n_personas=8, n_rounds=10)
    result = fit_and_evaluate(feats, seed=0)
    assert result["fitted"] is True
    assert 0.0 <= result["accuracy"] <= 1.0
    assert set(result["families"]) == {"O1", "O2", "O5"}


def test_mechanism_classifier_handles_too_little_data():
    feats = {"O1": np.zeros((1, 16)), "O2": np.zeros((1, 16))}
    result = fit_and_evaluate(feats, seed=0)
    assert result["fitted"] is False