"""Drives the persona population against the black-box platform for T rounds
(Algorithm 3). Lives outside aris/population/ because it is the one place
allowed to import both the platform and the population tier -- the barrier
test only forbids population/investigator from reaching into platform/, not
the other way around.
"""
from __future__ import annotations

import numpy as np

from .llm.client import LLMClient
from .platform.env import BlackBoxPlatform
from .population.persona import Memory, PersonaProfile, make_personas
from .population.policy import PopulationPolicy


def browse_population(
    plat: BlackBoxPlatform, corpus, personas: list[PersonaProfile], policy: PopulationPolicy,
    rounds: int, llm: LLMClient | None, seed: int, distill_every: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Run every persona for `rounds`. Returns (observation_rows, telemetry_rows).

    `ts` is a SIMULATED per-persona session clock, not wall-clock time. This
    matters for the plausibility discriminator (population/plausibility.py),
    which compares inter-action timing and burstiness against a reference
    browsing model: real wall-clock time between simulated rounds reflects how
    fast this laptop executes Python, not how fast a browsing session unfolds,
    and comparing the two produced a spurious AUC=1.0 ("agents are trivially
    distinguishable") that was really just a units mismatch. The clock below
    advances by dwell time plus a between-item gap drawn from the same kind of
    process the reference model uses, so the two are comparable.
    """
    rng = np.random.default_rng(seed)
    sids = [plat.session(p.persona_id, seed + p.persona_id, p.interest_vec) for p in personas]
    memories = {p.persona_id: Memory() for p in personas}
    clocks = {p.persona_id: 0.0 for p in personas}
    rows: list[dict] = []
    telemetry: list[dict] = []

    for t in range(rounds):
        for p, sid in zip(personas, sids):
            slate = plat.get_feed(sid)
            ids = [s.item_id for s in slate]
            embeds = corpus.embed[ids]
            arousals = corpus.arousal[ids]
            texts = [s.text for s in slate]
            mem = memories[p.persona_id]

            d = policy.decide(p, mem, ids, embeds, arousals, texts, t, rng, llm)
            plat.interact(sid, d.item_id, d.action, d.dwell_ms)
            mem.add(d.item_id, d.action, t, corpus.embed[d.item_id], float(corpus.arousal[d.item_id]))
            plat.end_round(sid)

            gap = d.dwell_ms / 1000.0 + rng.exponential(3.0) + 2.0
            clocks[p.persona_id] += gap
            rows.append({"persona_id": p.persona_id, "round": t, "slate": ids,
                        "action": d.action, "action_item": d.item_id,
                        "dwell_ms": d.dwell_ms, "level": d.level, "ts": clocks[p.persona_id]})

            if d.level == 1 and llm is not None and t > 0 and t % policy.reflect_every == 0:
                policy.reflect(p, mem, llm)

        if (t + 1) % distill_every == 0:
            report = policy.distill()
            telemetry.append({"round": t, "event": "distill", **report})

        telemetry.append({
            "round": t, "event": "round_end",
            "heterogeneity_entropy": heterogeneity_entropy(personas),
            "l1_escalation_rate": policy.escalation_rate(),
        })
    return rows, telemetry


def heterogeneity_entropy(personas: list[PersonaProfile]) -> float:
    """Population heterogeneity telemetry (RQ6): entropy of the distribution of
    interest vectors' dominant topic. Collapsing toward one value signals
    homogenisation (03_IMPLEMENTATION_PLAN.md 3.9, mitigation 1).
    """
    vecs = np.stack([p.interest_vec for p in personas])
    # cluster-free proxy: entropy of the sign pattern over a random projection
    # basis is fast and needs no external clustering dependency
    rng = np.random.default_rng(0)
    proj = vecs @ rng.standard_normal((vecs.shape[1], 8))
    bits = (proj > 0).astype(int)
    _, counts = np.unique(bits, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())