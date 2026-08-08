"""Platform-facing data collection for the investigator tier. Lives outside
aris/investigator/ for the same reason population_driver.py lives outside
aris/population/: this is the one place allowed to import both the platform
and the investigator's (platform-free) analysis code.
"""
from __future__ import annotations

import numpy as np

from .diagnostics import summary_statistics
from .platform.env import BlackBoxPlatform
from .platform.oracles import make_oracle
from .population.persona import make_personas
from .population.policy import PopulationPolicy
from .population_driver import browse_population


def collect_training_replicates(corpus, cfg: dict, oracles: list[str],
                                n_replicates: int, seed: int,
                                n_personas: int = 20, n_rounds: int = 20) -> dict:
    """Run the ACTUAL population policy (not Step 1's random-probe agent)
    against each oracle family, `n_replicates` times, returning
    {oracle: (n_replicates, n_features) summary-statistic matrix}.

    Using the real population here rather than diagnostics.py's RandomAgent
    means the mechanism classifier is trained on the same probing distribution
    Step 4's probe designer will act on, not an idealised random baseline.
    """
    out: dict[str, np.ndarray] = {}
    for oracle in oracles:
        feats = []
        for rep in range(n_replicates):
            rep_seed = seed + 977 * rep
            plat = BlackBoxPlatform(corpus, make_oracle(oracle, corpus, cfg), cfg, rep_seed)
            personas = make_personas(corpus, n_personas, rep_seed)
            policy = PopulationPolicy()
            rows, _ = browse_population(plat, corpus, personas, policy, n_rounds, None, rep_seed)
            feats.append(summary_statistics(rows, corpus))
        out[oracle] = np.array(feats)
    return out


def collect_observations(corpus, cfg: dict, oracle: str, seed: int,
                         n_personas: int, n_rounds: int) -> tuple[list[dict], dict]:
    """One population run against one oracle. Returns (rows, {persona_id: profile})
    -- the persona map lets analysis code build the B7 oracle-informed ceiling
    (investigator/surrogate.py::build_oracle_trace) without the investigator
    tier itself ever seeing persona identities (this function lives outside
    aris/investigator/, same as collect_training_replicates above).
    """
    plat = BlackBoxPlatform(corpus, make_oracle(oracle, corpus, cfg), cfg, seed)
    personas = make_personas(corpus, n_personas, seed)
    policy = PopulationPolicy()
    rows, _ = browse_population(plat, corpus, personas, policy, n_rounds, None, seed)
    return rows, {p.persona_id: p for p in personas}