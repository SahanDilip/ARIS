from __future__ import annotations

import numpy as np
import pytest

from aris.config import config_hash, load
from aris.corpus import build_corpus
from aris.diagnostics import FEATURE_NAMES, summary_statistics
from aris.ledger.schema import ObservationRecord, RunRecord
from aris.ledger.store import Ledger, new_run_id
from aris.platform.env import BlackBoxPlatform, ServedItem
from aris.platform.oracles import REGISTRY, make_oracle
from aris.population.random_agent import RandomAgent, browse


@pytest.fixture(scope="module")
def cfg():
    c = load("base")
    c["corpus"]["n_items"] = 400
    c["separability"] = {"n_agents": 4, "n_rounds": 6, "n_replicates": 2}
    return c


@pytest.fixture(scope="module")
def corpus(cfg):
    return build_corpus(cfg, 7)


# --------------------------------------------------------------------- corpus
def test_corpus_shapes(corpus, cfg):
    assert len(corpus) == cfg["corpus"]["n_items"]
    assert corpus.embed.shape == (len(corpus), cfg["corpus"]["embed_dim"])
    assert np.allclose(np.linalg.norm(corpus.embed, axis=1), 1.0, atol=1e-6)
    assert len(corpus.text) == len(corpus)


def test_corpus_decorrelated(corpus, cfg):
    """The validity precondition for every identifiability claim in the paper."""
    assert corpus.max_offdiag_corr() < cfg["corpus"]["max_abs_corr"] * 1.5


def test_corpus_deterministic(cfg):
    a, b = build_corpus(cfg, 11), build_corpus(cfg, 11)
    assert np.array_equal(a.popularity, b.popularity)
    assert not np.array_equal(build_corpus(cfg, 12).popularity, a.popularity)


# -------------------------------------------------------------------- oracles
@pytest.mark.parametrize("name", list(REGISTRY))
def test_oracle_contract(name, corpus, cfg):
    o = make_oracle(name, corpus, cfg)
    plat = BlackBoxPlatform(corpus, o, cfg, seed=3)
    sid = plat.session(0, 0)
    slate = plat.get_feed(sid)
    assert len(slate) == cfg["platform"]["slate_size"]
    assert len({s.item_id for s in slate}) == len(slate), "duplicate items in slate"
    assert plat.interact(sid, slate[0].item_id, "like")["ack"] is True
    assert plat.interact(sid, -999, "like")["ack"] is False
    assert "family" in o.ground_truth_params()


def test_oracles_are_distinguishable_at_all(corpus, cfg):
    """Different families must produce different slates from an identical state."""
    sig = {}
    for name in REGISTRY:
        plat = BlackBoxPlatform(corpus, make_oracle(name, corpus, cfg), cfg, seed=5)
        cfg2 = dict(cfg)
        plat.noise = 0.0
        sid = plat.session(0, 0)
        sig[name] = tuple(s.item_id for s in plat.get_feed(sid))
    assert len(set(sig.values())) >= 5, f"oracles collapse onto each other: {sig}"


def test_state_update_moves_representation(corpus, cfg):
    o = make_oracle("O4", corpus, cfg)
    plat = BlackBoxPlatform(corpus, o, cfg, seed=1)
    sid = plat.session(0, 0)
    v0 = plat.sessions[sid].state.vec.copy()
    for _ in range(5):
        slate = plat.get_feed(sid)
        plat.interact(sid, slate[0].item_id, "share")
        plat.end_round(sid)
    assert np.linalg.norm(plat.sessions[sid].state.vec - v0) > 1e-3


# ------------------------------------------------------------- black-box leaks
def test_served_items_expose_nothing_internal(corpus, cfg):
    plat = BlackBoxPlatform(corpus, make_oracle("O1", corpus, cfg), cfg, seed=2)
    sid = plat.session(0, 0)
    item = plat.get_feed(sid)[0]
    assert set(item.__dict__) == {"item_id", "text", "ts"}
    assert set(plat.health()) == {"policy_version", "slate_size"}
    assert "O1" not in plat.policy_version


def test_candidate_pool_is_hidden(corpus, cfg):
    """D1: the pool is the single biggest difficulty lever; it must not leak."""
    plat = BlackBoxPlatform(corpus, make_oracle("O1", corpus, cfg), cfg, seed=2)
    sid = plat.session(0, 0)
    slate = plat.get_feed(sid)
    assert len(slate) < cfg["platform"]["candidate_pool"]
    # the pool exists only in the ground-truth log, which agents never read
    assert len(plat.gt_log[-1]["pool"]) >= cfg["platform"]["candidate_pool"] - 5


def test_policy_version_changes_with_params(corpus, cfg):
    import copy
    c2 = copy.deepcopy(cfg)
    c2["oracles"]["O1"]["gamma"] = 0.1
    a = BlackBoxPlatform(corpus, make_oracle("O1", corpus, cfg), cfg, 0).policy_version
    b = BlackBoxPlatform(corpus, make_oracle("O1", corpus, c2), c2, 0).policy_version
    assert a != b


# --------------------------------------------------------------------- agents
def test_browse_closes_the_loop(corpus, cfg):
    plat = BlackBoxPlatform(corpus, make_oracle("O2", corpus, cfg), cfg, seed=4)
    agents = [RandomAgent(i, cfg["platform"]["state_dim"], i) for i in range(3)]
    rows = browse(plat, agents, rounds=5)
    assert len(rows) == 15
    assert plat.query_count == 15
    f = summary_statistics(rows, corpus)
    assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()


# --------------------------------------------------------------------- ledger
def test_ledger_roundtrip(tmp_path, cfg):
    led = Ledger(tmp_path / "l.duckdb")
    rid = new_run_id()
    led.append("runs", RunRecord(run_id=rid, config_hash=config_hash(cfg), seed=1,
                                 oracle="O1", started_at=0.0))
    led.append("observations", [
        ObservationRecord(run_id=rid, probe_id="P0", persona_id=i, round=0,
                          slate_json=[1, 2, 3], action="like", action_item=1)
        for i in range(4)
    ])
    assert led.counts()["observations"] == 4
    df = led.q("SELECT count(*) n FROM observations WHERE run_id = ?", rid)
    assert int(df["n"][0]) == 4
    led.close()
