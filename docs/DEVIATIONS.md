# Deviations from the original plan, and how to write them up

The plan in `03_IMPLEMENTATION_PLAN.md` assumed a GPU and a frontier API model.
This build targets a single laptop with Ollama and no paid services. Every
substitution is recorded here with the sentence you should put in the paper, so
that Section 9 (Limitations) writes itself and no reviewer discovers an
undisclosed shortcut.

Keep this file updated as you go. A reviewer who sees a candid deviations log
trusts the rest of the paper more, not less.

---

## D1 — Candidate pool hidden (this is a *strengthening*, not a compromise)

The earlier scaffold returned the candidate pool from `get_feed`, which let the
surrogate condition on what the platform chose *from*. Real platforms do not
reveal that. `platform.reveal_pool` is `false` by default and the pool exists
only in the ground-truth ledger.

> Paper: "The platform interface returns only the ranked slate; the candidate
> pool from which it was selected is withheld, so the surrogate must marginalise
> over an unobserved retrieval stage."

Keep `reveal_pool: true` available as the `-HiddenPool` ablation — it quantifies
how much of the difficulty comes from the unobserved pool, which is a clean
extra row in the ablation table.

---

## D2 — Investigator tier runs a local 7B rather than a frontier model

The plan called for a frontier API model for the Hypothesis Agent and the
Critic's narration, on the grounds that capability matters and volume is low.
Without paid access, both tiers are local.

Mitigation, and why the claim survives: the Hypothesis Agent's job is
deliberately narrow. It proposes *structure* over a Pydantic schema and a
discriminating prediction; it never estimates parameters, and schema-invalid
hypotheses are rejected without consuming budget. A 7B model is adequate for
selecting and instantiating from a bounded mechanism vocabulary. The enumerated,
LLM-free hypothesis generator is retained as the `-RAG` / `-LLM(investigator)`
ablation, which means you can *measure* how much the language model contributes
rather than assuming it.

> Paper: "Both agent tiers use locally served open-weight models
> (Qwen2.5-3B-Instruct for the population, Qwen2.5-7B-Instruct for the
> investigator). The investigator's language model proposes schema-constrained
> hypothesis structure only; all parameter estimation is numerical. We report an
> ablation in which hypothesis generation is replaced by exhaustive enumeration,
> which bounds the contribution of the language model."

---

## D3 — Simulation-based inference replaced by grid ABC

`sbi`'s neural posterior estimation needs PyTorch and is slow to train on CPU.
Step 5 uses approximate Bayesian computation over the enumerated parameter grid
with bootstrap credible intervals instead. At this parameter dimensionality the
grid is tractable and the intervals are honest; NPE buys smoothness, not
correctness.

> Paper: "Parameter posteriors are obtained by likelihood-free rejection
> sampling over a discretised parameter grid, with credible intervals from
> bootstrap resampling of the ledger. We verify calibration by coverage of the
> 90% intervals."

Report the coverage check. If 90% intervals contain the truth ~90% of the time,
the grid approximation is defensible and no reviewer will press further.

---

## D4 — LightGBM replaced by scikit-learn where wheels are awkward

`HistGradientBoostingRegressor` / `GradientBoostingClassifier` stand in for
LightGBM `lambdarank`. Listwise ranking is approximated by pairwise logistic
loss on within-slate item pairs.

> Paper: "The ranking surrogate is a gradient-boosted pairwise ranker; we use a
> pairwise decomposition of the listwise objective."

If `pip install lightgbm` works on your machine, use it — it is faster and gives
`lambdarank` directly. The interface in `investigator/surrogate.py` is the same.

---

## D5 — T2 learned recommender is a small in-house sequential model

RecBole plus SASRec/BERT4Rec on GPU is out of scope. Step 6 trains a compact
sequential recommender (item-embedding + GRU or a matrix-factorisation baseline)
on MovieLens-100K/1M on CPU. Its internals are learned rather than
hand-specified, which is what T2 is for; the weights are still yours, so ground
truth is preserved.

> Paper: "Tier T2 substitutes a compact GRU-based sequential recommender trained
> on MovieLens-1M for the transformer architectures used in prior extraction
> work. The tier's purpose — a target whose mechanism is learned rather than
> specified — is preserved, though absolute fidelity numbers are not comparable
> to published SASRec extraction results."

---

## D6 — Item corpus is synthetic rather than MIND

MIND is a ~3 GB download. The Step-1 corpus is synthetic but *constructed*:
popularity, affect, recency and velocity are explicitly decorrelated, which MIND
is not. For the identifiability argument the synthetic corpus is arguably
*better*, because confounding is controlled rather than inherited.

> Paper: "The item corpus is synthetic, with popularity, affect intensity,
> recency and engagement velocity decorrelated by construction (max absolute
> pairwise correlation 0.03). This is a deliberate validity choice: it ensures
> that reported non-identifiabilities are properties of the ranking policies
> rather than of correlations in the content pool."

Swap in MIND later if you want an external-validity row; the `Corpus` dataclass
is the only thing that needs a new constructor.

---

## D7 — LLM determinism is partial

Ollama honours `seed` and `temperature`, and every call is cached on disk keyed
by (model, messages, options), so a rerun is byte-identical. But a *cold* cache
on different hardware or a different Ollama build can produce different tokens.

> Paper: "Language-model calls are cached and seeded; reruns from a populated
> cache are exact. Cold-start reproduction on different hardware may differ at
> the token level, and we therefore report results over multiple seeds rather
> than single runs."

Ship the `.llm_cache/` directory with the artefact if size permits. It makes the
artefact exactly reproducible, which reviewers notice.


## D8 — A real reproducibility bug, found and fixed during Step 2

`platform/env.py`'s `session()` originally derived its cold-start RNG seed via
`abs(hash(sid)) % (2**32)`, where `sid` is a string. Python's built-in `hash()`
on strings is **randomized per process** since Python 3.3 (`PYTHONHASHSEED`)
unless explicitly disabled. This silently broke reproducibility across separate
runs with an identical config and seed: the platform's cold-start randomness --
and therefore every downstream observation -- differed run to run, while every
*other* seed in the codebase was correctly fixed. It was caught by the same seed
producing different held-out AUCs and exclusion counts across three consecutive
runs of `step2_report.py`. Fixed by deriving the seed from a direct integer mix
of `persona_id` and `seed` instead of hashing a string. `tests/test_population.py
::test_browse_population_is_deterministic` now guards this regression directly.

> Paper: nothing to disclose here -- this is exactly the kind of bug the
> reproducibility infrastructure (fixed seeds, the ledger, the barrier test) is
> supposed to catch, and it did. Worth a sentence in the methodology section on
> why determinism was verified by rerunning rather than assumed from seeding
> alone: "we verified exact reproducibility empirically (identical trajectories
> across repeated runs at a fixed seed) rather than relying on seed-setting
> alone, which surfaced and let us fix a process-level randomization bug."

---

## D9 — Plausibility discriminator remains off-manifold (AUC≈1.0) after calibration

The plausibility reference model (`population/plausibility.py`) was calibrated
against the actual population's trajectory statistics in four real respects:
matched session length (was previously a perfectly-separating giveaway feature),
a corrected cumulative session clock (the original multiplied round index by a
freshly-redrawn factor each round, which is not a clock), matched dwell-time
scale, and a two-stage retrieval-then-rank candidate process with Gumbel noise
matching the real platform's Plackett-Luce sampling. After these fixes, seven
of eight trajectory features show substantial marginal overlap between agent
and reference distributions; the discriminator nonetheless still reports
AUC≈1.0 on the O4 population at n=60/class with 8 features.

This was not chased further within the Step-2 time budget. Two candidate
explanations, either or both may be true: (a) genuine multivariate separability
at this sample size even with overlapping marginals (a linear classifier can
separate two 60-point clouds in 8 dimensions on a combination of features even
when no single feature does), or (b) a residual, real behavioural difference
not yet identified. This check is a WARN, not a gating criterion, so it did not
block Gate 1. Revisit before relying on plausibility AUC as an RQ3/RQ4 metric
in Step 4 -- either diagnose further with a larger reference sample and a
regularised classifier, or substitute a genuine public dataset (MovieLens
session logs) once time permits.

> Paper: report this honestly if RQ3/RQ4 depend on plausibility AUC as
> constructed here: "the plausibility discriminator, calibrated against a
> structured synthetic reference process, reports agent trajectories as
> distinguishable from the reference at the population scale tested; we did
> not have time within the study's timeline to determine whether this reflects
> a genuine behavioural gap or a synthetic-reference calibration limit, and we
> flag it as a limitation rather than a validated plausibility guarantee."


## D10 — Level-1 LLM path validated structurally, not used in reported results

Ollama's HTTP API responded in 1-3s under direct manual testing (`ollama run`,
and raw API calls via PowerShell), but the same calls issued from the Step-2
driver script intermittently stalled past a 180s timeout on first invocation
after model idle-unload, most likely due to Windows Defender real-time
scanning of the model weights file on reload. This was not resolved within
the study's timeline.

Consequence: reported Step 2 results use Level-0 only (the shared softmax
policy, beta-driven), with Level-1 escalation validated structurally only
(schema parsing, fallback behaviour, cache correctness -- see
tests/test_population.py::test_escalation_uses_llm_when_provided) rather than
exercised in the reported run. This is equivalent to running the paper's own
"-LLM(population)" ablation (B4, 04_EVALUATION.md) as the primary condition.

> Paper: "Due to local LLM serving latency on the target hardware, reported
> population-tier results use the Level-0 policy only; the hybrid escalation
> mechanism (Eq. 11) was validated for correctness but not exercised at scale.
> This is structurally equivalent to the -LLM(population) ablation condition."

## D11 — Ranking surrogate HitRatio@1: two real bugs found and fixed

Initial HitRatio@1 results (1-positive/19-negative protocol matching Cai et al.
[5]) were stuck near or below the random baseline regardless of oracle family,
and an oracle-informed "ceiling" using each persona's TRUE interest_vec
(B7-style upper bound, 04_EVALUATION.md 4.2) scored IDENTICALLY to a weak,
realistic session-based proxy -- perfect ground-truth knowledge should not
match a weak guess, which was the signal that something was structurally wrong
rather than the task simply being hard.

Two real bugs, found by direct inspection rather than parameter tuning:

1. **Null-anchor scoring.** `RankingSurrogate.score()` originally compared each
   real item against a synthetic all-zero "null" feature vector that never
   appears during training. Gradient-boosted trees extrapolate unreliably to
   points far outside their training distribution, which made the resulting
   ranking signal too weak to be useful. Fixed by replacing null-anchor scoring
   with a vectorised round-robin (Copeland-style) tournament: every comparison
   is between two actual candidates, exactly matching what the classifier was
   trained on.

2. **The exact `id(r)` lookup bug from D8, recurring in a second location.**
   `_pairwise_examples` (the TRAINING data builder) still used `row_pref.get(id(r))`
   even after the same bug was fixed in `evaluate_fidelity`. Since `id()` almost
   never matched the `(persona_id, round)`-keyed trace dictionary, the
   preference-match feature (`rel_diff`) was always exactly zero (std=0.0,
   confirmed directly) during training -- the classifier never had a chance to
   learn from persona-conditioned relevance at all, regardless of what was
   supplied at evaluation time. This explains the ceiling/realistic parity
   completely on its own. Fixed by using the same `(persona_id, round)` key
   everywhere.

After both fixes: on O1 (no diversity confound), held-out HR@1=0.097 (vs random
0.05), ceiling HR@1=0.194, headroom=0.097 -- a coherent, interpretable pattern.
On O4 (strong MMR diversity reranking, lambda_d=0.9), the realistic proxy drops
to HR@1=0.011 while the ceiling still clears baseline at 0.074 -- a genuinely
informative, oracle-specific finding: O4's diversity mechanism makes passive
recovery harder even with a strong content proxy, consistent with Step 1's own
diagnostic flagging O4 as the hardest family to resolve (parameter_resolution.csv).

> Paper: no caveat needed here -- this is the intended headline surrogate
> result. "The ranking surrogate's fidelity depends materially on the target
> oracle's mechanism: on O1, held-out HitRatio@1 exceeds the random baseline by
> ~2x (0.097 vs 0.05); on O4, whose strong diversity-reranking penalty
> decorrelates relevance from what is served, the same content-only proxy
> falls below baseline while an oracle-informed ceiling using ground-truth
> preference still clears it (0.074), quantifying the value of persona-level
> information the investigator does not have."
