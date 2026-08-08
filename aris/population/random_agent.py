"""Random-action agent: the Step-1 placeholder for the population tier.

Purpose is narrow: prove the closed interaction loop works over the black-box
interface, and provide the *uninformed probing* condition used by the Step-1
separability diagnostic. In the paper this is a component of baseline B1
(random-probe extraction), so it is not throwaway code.

MUST NOT import anything from aris.platform -- enforced by tests/test_barrier.py.
"""
from __future__ import annotations

import numpy as np

ACTIONS = ["skip", "dwell", "like", "share", "follow", "comment"]
_P = np.array([0.55, 0.20, 0.12, 0.05, 0.03, 0.05])


class RandomAgent:
    def __init__(self, persona_id: int, dim: int, seed: int):
        self.persona_id = persona_id
        self.rng = np.random.default_rng(seed)
        self.interest = self.rng.standard_normal(dim)
        self.interest /= np.linalg.norm(self.interest)
        self.seen: set[int] = set()

    def act(self, slate: list) -> tuple[int, str, int]:
        """Return (item_id, action, dwell_ms). Uniform over positions: no position bias."""
        j = int(self.rng.integers(0, len(slate)))
        item = slate[j].item_id if hasattr(slate[j], "item_id") else int(slate[j]["item_id"])
        action = ACTIONS[int(self.rng.choice(len(ACTIONS), p=_P))]
        self.seen.add(item)
        return item, action, int(200 + self.rng.exponential(1200))


def browse(platform, agents, rounds: int, k: int | None = None) -> list[dict]:
    """Drive `agents` through `platform` for `rounds`. Returns observation rows."""
    sids = [platform.session(a.persona_id, a.persona_id, a.interest) for a in agents]
    rows: list[dict] = []
    for t in range(rounds):
        for a, sid in zip(agents, sids):
            slate = platform.get_feed(sid, k)
            item, action, dwell = a.act(slate)
            platform.interact(sid, item, action, dwell)
            rows.append({
                "persona_id": a.persona_id, "round": t,
                "slate": [s.item_id for s in slate],
                "action": action, "action_item": item, "dwell_ms": dwell,
                "ts": slate[0].ts if slate else 0.0,
            })
            platform.end_round(sid)
    return rows
