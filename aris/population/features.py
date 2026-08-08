"""Feature function f(a, S, p_i, M_i) used by the Level-0 softmax policy (Eq. 10)
of 02_METHODOLOGY.tex, and reused unchanged by the distillation step so that
what Level-0 is scored on during inference is exactly what it is refit on.

MUST NOT import aris.platform -- enforced by tests/test_barrier.py. Callers
(driver scripts) fetch slates from the platform and pass plain (item_id, embed,
arousal, valence, recency, topic) tuples in; this module never touches the
black box itself.
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 5
FEATURE_NAMES = ["topical_match", "position_bias", "novelty", "affect_match", "fatigue"]


def item_features(
    interest_vec: np.ndarray,
    item_embed: np.ndarray,
    position: int,
    slate_size: int,
    memory_embeds: np.ndarray | None,
    item_arousal: float,
    affect_target: float,
    fatigue: float,
) -> np.ndarray:
    """One row of f(.) for a single candidate item at a given slate position."""
    topical_match = float(interest_vec @ item_embed)
    position_bias = 1.0 - (position / max(1, slate_size - 1))  # top of slate favoured
    if memory_embeds is None or len(memory_embeds) == 0:
        novelty = 1.0
    else:
        novelty = float(1.0 - np.max(memory_embeds @ item_embed))
    affect_match = 1.0 - abs(item_arousal - affect_target)
    return np.array([topical_match, position_bias, novelty, affect_match, fatigue], dtype=float)


def slate_features(
    interest_vec: np.ndarray,
    embeds: np.ndarray,           # (K, d) item embeddings in slate order
    arousals: np.ndarray,         # (K,)
    memory_embeds: np.ndarray | None,
    affect_target: float,
    fatigue: float,
) -> np.ndarray:
    """(K, N_FEATURES) feature matrix for every item in a slate, in slate order."""
    K = len(embeds)
    return np.stack([
        item_features(interest_vec, embeds[j], j, K, memory_embeds,
                      float(arousals[j]), affect_target, fatigue)
        for j in range(K)
    ])