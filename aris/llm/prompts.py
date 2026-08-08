"""Prompt templates for Level-1 persona decisions and reflection.

Kept in one place so the wording is versioned alongside the paper's methodology
description, and so distillation/logging code never has to reconstruct a prompt
to know what was asked.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

DECIDE_SYSTEM = (
    "You are simulating one social media user for a research study on recommender "
    "behaviour. Stay strictly in character. Reply with JSON only, matching this "
    'schema: {"choice_index": int, "action": one of '
    '["skip","dwell","like","share","follow","comment"], "dwell_ms": int, '
    '"reason": short string}. choice_index is the 0-based position in the feed below.'
)

DECIDE_USER_TMPL = """Persona: {age}-year-old, traits {traits}, prefers content with \
arousal level around {affect_target:.2f} (0=calm, 1=intense).
Self-description: {narrative}
Recent activity: {recent}

Feed (position: text):
{feed_lines}

Pick ONE item by its position and an action. Skip if nothing fits.
"""

REFLECT_SYSTEM = (
    "You are updating a simulated social media user's self-description after a "
    "browsing session. Reply with JSON only: "
    '{"narrative": short first-person-style description, '
    '"affect_target_delta": float in [-0.2, 0.2], '
    '"topic_shift": one of ["toward_recent","no_change","away_from_recent"]}.'
)

REFLECT_USER_TMPL = """Previous self-description: {narrative}
Previous preferred arousal level: {affect_target:.2f}
Recent activity: {recent}

Update the self-description to reflect any drift in interest.
"""


class DecideOutput(BaseModel):
    choice_index: int
    action: str
    dwell_ms: int = Field(ge=0, le=60000)
    reason: str = ""


class ReflectOutput(BaseModel):
    narrative: str
    affect_target_delta: float = Field(ge=-0.2, le=0.2)
    topic_shift: str = "no_change"