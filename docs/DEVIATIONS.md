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
