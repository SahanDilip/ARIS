"""Typed records for the append-only Experiment Ledger.

Every inter-agent message is a schema-validated record (01_ARCHITECTURE.md 2.3:
blackboard over a typed message bus, not free-form agent chat). Consequences we
want: the whole run is reproducible, auditable, and the ledger doubles as the
paper's artefact.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Action = Literal["skip", "dwell", "like", "share", "follow", "comment", "exit"]


class RunRecord(BaseModel):
    run_id: str
    config_hash: str
    git_sha: str = "unknown"
    seed: int
    oracle: str
    started_at: float
    notes: str = ""


class PersonaRecord(BaseModel):
    run_id: str
    persona_id: int
    profile_json: dict[str, Any]
    seed_source: str = "synthetic"
    validation_json: dict[str, Any] = Field(default_factory=dict)


class ProbeRecord(BaseModel):
    run_id: str
    probe_id: str
    spec_json: dict[str, Any]
    targets_json: list[str] = Field(default_factory=list)
    est_cost: float = 0.0
    chosen_score: float = 0.0


class ObservationRecord(BaseModel):
    run_id: str
    probe_id: str
    persona_id: int
    round: int
    slate_json: list[int]
    action: Action
    action_item: int
    dwell_ms: int = 0
    ts: float = 0.0


class HypothesisRecord(BaseModel):
    run_id: str
    iter: int
    hypothesis_json: dict[str, Any]
    posterior_weight: float = 0.0
    provenance_json: list[str] = Field(default_factory=list)


class SurrogateRecord(BaseModel):
    run_id: str
    iter: int
    model_path: str = ""
    fidelity_json: dict[str, Any] = Field(default_factory=dict)
    params_posterior_json: dict[str, Any] = Field(default_factory=dict)


class VerdictRecord(BaseModel):
    run_id: str
    iter: int
    verdict: Literal["accept", "reject", "more_data", "instrument_failure"]
    checks_json: dict[str, Any] = Field(default_factory=dict)
    reasons: str = ""


class TelemetryRecord(BaseModel):
    run_id: str
    round: int
    heterogeneity_entropy: float
    persona_consistency: float = 0.0
    drift_flag: bool = False
    l1_escalation_rate: float = 0.0
    tokens_used: int = 0


class GroundTruthRecord(BaseModel):
    """Written by the platform, read only at analysis time. Never served."""

    run_id: str
    policy_family: str
    params_json: dict[str, Any]


TABLES = {
    "runs": RunRecord, "personas": PersonaRecord, "probes": ProbeRecord,
    "observations": ObservationRecord, "hypotheses": HypothesisRecord,
    "surrogates": SurrogateRecord, "verdicts": VerdictRecord,
    "telemetry": TelemetryRecord, "ground_truth": GroundTruthRecord,
}
