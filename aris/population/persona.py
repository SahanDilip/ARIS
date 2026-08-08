"""Persona profiles and episodic memory. State only -- no policy logic here
(that is policy.py) and no platform access (enforced by test_barrier.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TRAIT_NAMES = ["openness", "extraversion", "neuroticism"]


@dataclass
class PersonaProfile:
    persona_id: int
    age: int
    traits: dict[str, float]           # openness, extraversion, neuroticism in [0,1]
    interest_vec: np.ndarray           # d-dim, same space as corpus.embed
    affect_target: float               # preferred arousal level in [0,1]
    narrative: str = ""                # free-text self-description, updated by reflection

    def to_json(self) -> dict:
        return {
            "persona_id": self.persona_id, "age": self.age, "traits": self.traits,
            "affect_target": round(self.affect_target, 3), "narrative": self.narrative,
        }


@dataclass
class Memory:
    """Fixed-capacity episodic buffer. Oldest item drops off (Alg. 3's M_i)."""

    capacity: int = 20
    items: list[dict] = field(default_factory=list)  # {item_id, action, round, embed, arousal}
    n_actions_since_break: int = 0

    def add(self, item_id: int, action: str, round_: int, embed: np.ndarray, arousal: float) -> None:
        self.items.append({"item_id": int(item_id), "action": action, "round": round_,
                           "embed": embed, "arousal": float(arousal)})
        if len(self.items) > self.capacity:
            self.items.pop(0)
        self.n_actions_since_break = 0 if action in ("exit",) else self.n_actions_since_break + 1

    def embeds(self) -> np.ndarray | None:
        if not self.items:
            return None
        return np.stack([m["embed"] for m in self.items])

    def seen_ids(self) -> set[int]:
        return {m["item_id"] for m in self.items}

    def fatigue(self, cap: int = 12) -> float:
        return min(1.0, self.n_actions_since_break / cap)

    def recent_text_summary(self, corpus_text: list[str] | None = None, k: int = 5) -> str:
        tail = self.items[-k:]
        if corpus_text is not None:
            parts = [f"{m['action']}: {corpus_text[m['item_id']]}" for m in tail]
        else:
            parts = [f"{m['action']} on item {m['item_id']}" for m in tail]
        return "; ".join(parts) if parts else "no prior activity"


def make_personas(corpus, n: int, seed: int) -> list[PersonaProfile]:
    """Seed profiles from the corpus's own topic centroids -- a documented stand-in
    for seeding from real log clusters (03_IMPLEMENTATION_PLAN.md 3.9, mitigation 1).
    Interest vectors are drawn *away* from a single shared centroid so the
    population starts heterogeneous rather than mode-collapsed.
    """
    rng = np.random.default_rng(seed)
    d = corpus.dim
    profiles = []
    for i in range(n):
        topic = int(rng.integers(0, len(corpus.topic_names)))
        mask = corpus.topic == topic
        base = corpus.embed[mask].mean(0) if mask.any() else rng.standard_normal(d)
        vec = base + 0.6 * rng.standard_normal(d)
        vec /= np.linalg.norm(vec) + 1e-12
        traits = {t: float(rng.beta(2, 2)) for t in _TRAIT_NAMES}
        profiles.append(PersonaProfile(
            persona_id=i, age=int(rng.integers(18, 70)), traits=traits,
            interest_vec=vec,
            # skewed right: population-level preference for higher arousal on average,
            # matching the documented affect-intensity regularity (Solovev & Prollochs)
            # while individual personas still vary in how strongly they show it
            affect_target=float(rng.beta(3.0, 1.5)),
            narrative=f"Interested in {corpus.topic_names[topic]}; "
                      f"{'curious' if traits['openness'] > 0.5 else 'cautious'} browser.",
        ))
    return profiles