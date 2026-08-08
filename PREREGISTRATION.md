# Pre-registration (ARIS)

Cheap to write, and a real credibility signal for a paper whose subject is
auditing rigour (`04_EVALUATION.md` §4.6). Commit this **before** the final
sweeps, with the date, and do not edit predictions afterwards — add dated
amendments instead.

- Date created: `TODO`
- Git SHA at creation: `TODO`
- Author: `TODO`

## Primary hypotheses

**H-A (RQ2, query efficiency).** Information-gain-driven probe design attains a
given parameter-recovery accuracy with fewer platform queries than random probe
assignment, on the families identified in the Step-1 report as having
parameter-level headroom. Predicted effect: ≥1.5× reduction in
queries-to-threshold.

**H-B (RQ4, the plausibility trade-off — headline).** As the plausibility penalty
λ_p decreases from its default toward 0, surrogate fidelity measured on the probe
distribution `D_probe` increases while fidelity on the real distribution `D_real`
decreases. Predicted: the two curves cross.

**H-C (RQ3, value of LLM realism).** A population with Level-1 language-model
decisions attains higher `D_real` fidelity and lower plausibility-discriminator
AUC than a rule-based population at equal query budget. *Explicitly registered as
falsifiable*: if H-C fails everywhere, the paper reframes rather than reports a
null quietly.

**H-D (RQ5, identifiability).** At least one oracle pair remains unseparable at
the largest budget studied. Step-1 candidates (fill in from your own report,
before running Step 4): `TODO`.

**H-E (RQ6, instrument drift).** Population heterogeneity entropy declines over
sustained probing without re-anchoring, and the decline correlates negatively
with surrogate fidelity.

## Registered analysis decisions

- Seeds: 5 per configuration, 10 for H-A and H-B. Seeds control persona
  sampling, language-model sampling, platform noise, and probe tie-breaking.
- Primary metrics fixed in advance: queries-to-threshold (H-A); paired fidelity
  gap `D_probe − D_real` (H-B); plausibility AUC (H-C); pairwise separability
  adjusted against a label-permutation null (H-D); heterogeneity entropy (H-E).
- Paired comparisons throughout: same seeds, personas and corpus across methods.
  Paired bootstrap for intervals; Wilcoxon signed-rank for the ARIS-vs-baseline
  claims.
- Multiple comparisons: Holm–Bonferroni across oracle families in E1.
- Exclusion rule, fixed now: persona subgroups failing instrument validation are
  excluded from probing and reported as excluded.
- Stopping rule for the loop is the one in Algorithm 1; it is not tuned per
  oracle.

## What would falsify the paper's central claim

If recovery accuracy under active design is statistically indistinguishable from
random probing across all families at all budgets, then plausibility-constrained
active extraction confers no measurable benefit in this testbed, and that is what
we report.

## Amendments

| Date | Change | Reason |
|---|---|---|
| | | |
