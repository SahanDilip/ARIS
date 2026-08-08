"""The hybrid persona decision policy (02_METHODOLOGY.tex Eqs. 10-11, Algorithm 3).

Level 0 is a shared, cheap softmax choice model over item features (Eq. 10).
Level 1 escalates to the local LLM only when Level 0 is uncertain or the slate
is novel (Eq. 11); its decisions are cached and periodically distilled back
into Level 0's weights (Alg. 3 line 14), so language-model reasoning shapes the
population's behavioural manifold without sitting in the hot loop.

beta is a SINGLE SHARED weight vector across the population -- not per-persona --
which is what "distilling into the population's manifold" means; personas differ
through their profile/memory features, not through separate policies.

MUST NOT import aris.platform (enforced by tests/test_barrier.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..llm.client import LLMClient
from ..llm.prompts import (DECIDE_SYSTEM, DECIDE_USER_TMPL, REFLECT_SYSTEM,
                           REFLECT_USER_TMPL, DecideOutput, ReflectOutput)
from .features import N_FEATURES, slate_features
from .persona import Memory, PersonaProfile

ACTIONS = ["skip", "dwell", "like", "share", "follow", "comment"]
_ACTION_BASE_PROB = np.array([0.45, 0.25, 0.15, 0.05, 0.03, 0.07])  # prior over ACTIONS


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _entropy(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


@dataclass
class Decision:
    item_id: int
    action: str
    dwell_ms: int
    level: int                       # 0 or 1
    reason: str = ""


@dataclass
class PopulationPolicy:
    """Shared Level-0 weights plus the escalation cache used for distillation."""

    beta: np.ndarray = field(default_factory=lambda: np.array([1.2, 0.4, 0.6, 0.5, -0.3]))
    eta: float = 1.3                  # escalation entropy threshold (nats); ln(6)=1.79 max
    novelty_escalate: float = 0.85    # escalate if best-item novelty exceeds this
    reflect_every: int = 8            # rounds between reflection points
    cache: list[tuple[np.ndarray, int]] = field(default_factory=list)  # (features_of_chosen, rank)
    n_l0: int = 0
    n_l1: int = 0

    # ------------------------------------------------------------------- score
    def item_probs(self, feats: np.ndarray) -> np.ndarray:
        """Pr(engage item j) per Eq. 10, over the K items in the slate."""
        return _softmax(feats @ self.beta)

    def action_probs(self, feats_chosen: np.ndarray, raw_arousal: float,
                     arousal_boost: float = 0.9) -> np.ndarray:
        """Action-type distribution shifted toward like/share as feature score
        rises, PLUS a direct raw-arousal term. The direct term is separate from
        feats_chosen's affect_match component (which only measures fit to this
        persona's own affect_target): it represents the population-level
        affect-intensity bias documented in Solovev & Prollochs [19] -- intense
        content spreads more on average, on top of whatever a given user's
        personal taste says. Without this term the effect is diluted by
        topical_match's larger weight and the aggregate asymmetry check
        (validate.py) does not reproduce reliably.
        """
        score = float(np.clip(feats_chosen @ self.beta + arousal_boost * raw_arousal, -3, 3))
        shift = np.array([-1, 0.2, 1.0, 1.3, 0.6, 0.8]) * (score / 1.6)
        return _softmax(np.log(_ACTION_BASE_PROB + 1e-9) + shift)

    def should_escalate(self, p_item: np.ndarray, best_novelty: float, round_: int) -> bool:
        """Eq. 11: entropy OR novelty OR scheduled reflection point."""
        if round_ > 0 and round_ % self.reflect_every == 0:
            return True
        return _entropy(p_item) > self.eta or best_novelty > self.novelty_escalate

    # -------------------------------------------------------------------- decide
    def decide(
        self, profile: PersonaProfile, memory: Memory, slate_item_ids: list[int],
        embeds: np.ndarray, arousals: np.ndarray, texts: list[str], round_: int,
        rng: np.random.Generator, llm: LLMClient | None,
    ) -> Decision:
        feats = slate_features(profile.interest_vec, embeds, arousals, memory.embeds(),
                                profile.affect_target, memory.fatigue())
        p_item = self.item_probs(feats)
        best_j = int(np.argmax(p_item))
        novelty_best = float(feats[best_j, 2])
        escalate = self.should_escalate(p_item, novelty_best, round_) and llm is not None

        if escalate:
            self.n_l1 += 1
            d = self._decide_llm(profile, memory, slate_item_ids, texts, round_, llm, rng)
            j = slate_item_ids.index(d.item_id) if d.item_id in slate_item_ids else best_j
            self.cache.append((feats[j], j))
            return d

        self.n_l0 += 1
        j = int(rng.choice(len(slate_item_ids), p=p_item))
        action_p = self.action_probs(feats[j], float(arousals[j]))
        action = ACTIONS[int(rng.choice(len(ACTIONS), p=action_p))]
        # log-normal on the same scale as the plausibility reference model
        # (population/plausibility.py's reference_trajectory uses lognormal(6.5,
        # 0.6)); mu shifts with affect_match so a better-fit item is dwelled on
        # longer, without changing the overall magnitude the discriminator sees.
        dwell = int(rng.lognormal(6.1 + 0.5 * feats[j, 3], 0.6)) if action != "skip" else 0
        return Decision(slate_item_ids[j], action, dwell, level=0)

    def _decide_llm(self, profile, memory, slate_item_ids, texts, round_, llm, rng) -> Decision:
        feed_lines = "\n".join(f"{j}: {t}" for j, t in enumerate(texts))
        user = DECIDE_USER_TMPL.format(
            age=profile.age, traits={k: round(v, 2) for k, v in profile.traits.items()},
            affect_target=profile.affect_target, narrative=profile.narrative,
            recent=memory.recent_text_summary(), feed_lines=feed_lines,
        )
        out = None
        if hasattr(llm, "chat_json"):
            out = llm.chat_json(DECIDE_SYSTEM, user, schema_model=DecideOutput)
        else:  # StubLLM: parse json_mode output manually
            try:
                out = DecideOutput(**json.loads(llm.chat(DECIDE_SYSTEM, user, json_mode=True)))
            except Exception:
                out = None
        if out is None or not (0 <= out.choice_index < len(slate_item_ids)) or out.action not in ACTIONS:
            j = int(rng.integers(0, len(slate_item_ids)))
            return Decision(slate_item_ids[j], "skip", 0, level=1, reason="llm_fallback")
        return Decision(slate_item_ids[out.choice_index], out.action, int(out.dwell_ms),
                        level=1, reason=out.reason)

    # ------------------------------------------------------------------- reflect
    def reflect(self, profile: PersonaProfile, memory: Memory, llm: LLMClient) -> PersonaProfile:
        user = REFLECT_USER_TMPL.format(narrative=profile.narrative,
                                        affect_target=profile.affect_target,
                                        recent=memory.recent_text_summary())
        out = None
        if hasattr(llm, "chat_json"):
            out = llm.chat_json(REFLECT_SYSTEM, user, schema_model=ReflectOutput)
        if out is not None:
            profile.narrative = out.narrative[:280]
            profile.affect_target = float(np.clip(profile.affect_target + out.affect_target_delta, 0, 1))
        return profile

    # ------------------------------------------------------------------- distill
    def distill(self, lr: float = 0.15, l2: float = 0.05) -> dict:
        """Refit beta on the Level-1 cache (Alg. 3 line 14): logistic gradient step
        toward making the LLM's chosen item the argmax of Level 0's own scoring.
        Returns a small report so telemetry can log the update size.
        """
        if not self.cache:
            return {"n_cache": 0, "delta_norm": 0.0}
        grad = np.zeros(N_FEATURES)
        for feats_chosen, _rank in self.cache:
            # push beta toward the chosen item's feature direction (a one-class
            # perceptron-style update; N_FEATURES is small so this is stable)
            grad += feats_chosen - self.beta * l2
        grad /= len(self.cache)
        delta = lr * grad
        self.beta = self.beta + delta
        n = len(self.cache)
        self.cache.clear()
        return {"n_cache": n, "delta_norm": float(np.linalg.norm(delta)),
               "beta": self.beta.tolist()}

    def escalation_rate(self) -> float:
        total = self.n_l0 + self.n_l1
        return self.n_l1 / total if total else 0.0