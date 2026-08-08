# ARIS — Step 1: Foundation

This is weeks 1–3 of `03_IMPLEMENTATION_PLAN.md`, built for **one laptop, Ollama,
no paid services**. It is deliberately the part of the system with *no* language
model dependency, so it is pure low-risk progress: you can complete and validate
all of it before spending a single token.

What exists after Step 1: a decorrelated item corpus, six oracle ranking
policies, a black-box platform (in-process and over HTTP), a DuckDB experiment
ledger, an enforced information barrier, a local-LLM client with disk caching,
and a diagnostic that tells you whether your testbed is measurable at all.

---

## 1. Install (5 minutes)

```bash
cd aris_step1
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11+ recommended; 3.10 works. No GPU required. Nothing here downloads a
dataset or a model.

Then Ollama, for Step 2 (Step 1 runs fine without it):

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh
# Windows: install from https://ollama.com/download

ollama pull qwen2.5:3b-instruct       # population tier, ~2 GB, fine on 8 GB RAM
ollama pull qwen2.5:7b-instruct       # investigator tier, ~5 GB, wants 16 GB RAM
```

If you have 8 GB of RAM, pull only the 3B and set
`llm.investigator_model: qwen2.5:3b-instruct` in `conf/base.yaml`. That is a
documented deviation (D2), not a failure — the investigator's job is narrow by
design.

---

## 2. Run

```bash
make test              # 18 tests, ~2 s. Includes the information-barrier test.
make smoke             # ~45 s, no LLM. Confirms the whole loop closes.
make step1-realistic   # the real run, ~4-8 min depending on your CPU
```

`make step1-realistic` writes everything into `outputs/step1_realistic/`:

| File | What it is |
|---|---|
| **`STEP1_REPORT.md`** | **the thing you paste back to me** |
| `corpus_correlations.csv` | paper table: corpus validity |
| `separability.csv`, `separability_curve.csv`, `queries_to_0.80.csv` | proto-E5 identifiability data |
| `parameter_resolution.csv` | where active probe design has headroom |
| `summary_stats.csv` | per-oracle audit statistics (features for Step 4's classifier) |
| `ollama_bench.json` | your machine's cost model |
| `ledger.duckdb` | the artefact |
| `fig_corpus.png`, `fig_separability.png` | draft figures |

Three configs are provided. Start with `realistic`.

| Config | Use |
|---|---|
| `conf/base.yaml` | strong, well-separated mechanisms. Easiest; good for debugging |
| **`conf/realistic.yaml`** | **recommended.** Every policy uses all four signals and families differ only in weighting, so "which signal is present" stops being diagnostic |
| `conf/hard.yaml` | subtle parameters, high noise, short slates |

Override anything without editing files:

```bash
python -m scripts.step1_report --conf realistic --set platform.noise_scale=1.2 run.seed=7
```

To see the HTTP tier (optional):

```bash
make serve                                    # terminal 1
curl -s -X POST localhost:8000/session -H 'content-type: application/json' \
     -d '{"persona_id":1,"seed":0}'           # terminal 2
curl -s 'localhost:8000/feed?session_id=s1-0&k=8'
curl -s localhost:8000/health
```

Note what `/health` returns: an opaque `policy_version` hash. You can detect that
the policy changed; you cannot read what it is. That is the point.

---

## 3. What each piece is for in the paper

| Module | Paper role |
|---|---|
| `aris/corpus.py` | Corpus construction as a *validity argument*. Without decorrelation, O1/O2/O5 are confounded by construction and every identifiability claim is a fact about your corpus rather than the policies. Report the correlation matrix. |
| `aris/platform/oracles.py` | Table T2, "Oracle policy families and free parameters". |
| `aris/platform/env.py` | Tier T1 of the opacity ladder. Enforces the black box; withholds the candidate pool. |
| `aris/platform/server.py` | The architectural information barrier (§2.1), over HTTP so T1→T2→T3 swaps need no agent changes. |
| `aris/ledger/` | Reproducibility: every table in the results section is one SQL query. |
| `tests/test_barrier.py` | Makes the black-box claim *verifiable* rather than asserted. Worth a sentence in §6. |
| `aris/diagnostics.py` | Proto-E5. Also the feature extractor the Step-4 mechanism classifier reuses. |
| `aris/llm/bench.py` | Table T6, cost accounting. Your survey's §9 cost argument becomes constructive rather than apologetic. |

---

## 4. Read section 7 of the report first

`STEP1_REPORT.md` §7 interprets its own numbers and tells you which knob to pull.
The three outcomes it distinguishes:

- **Family classification saturates** (likely). Every oracle pair separable at
  ~10–25 queries. This is *not* a bug — it says mechanism *class* is cheap to
  identify and mechanism *magnitude* is not. Consequence: **frame RQ2 over
  parameter recovery, not family recovery.** Report the family result as
  saturated; it is an honest finding and it protects you from a reviewer noticing
  it first.
- **Parameter-level headroom exists.** Some families' primary parameter is not
  resolvable by random probing at a small budget. Those families carry your RQ2
  experiment. §7 names them.
- **Candidate non-identifiability.** Some family's parameter stays unresolvable
  even at a large relative difference and full budget. Those are your RQ5
  results, if active design fails to rescue them in Step 4.

The single most useful habit: run this diagnostic again whenever you change the
testbed. It is cheap, and it is the thing that stops you discovering in week 13
that your experiments could never have separated anything.

---

## 5. Known-good verdicts

A healthy `make step1-realistic` looks like this:

```
PASS   corpus decorrelation: max |off-diagonal r| = 0.029 (target < 0.1)
PASS   information barrier (served fields): served keys = ['item_id','text','ts']
PASS   candidate pool hidden: pool size 100 sampled per request, slate K = 8
PASS   closed interaction loop: 50 observations, 51 platform queries
WARN   testbed difficulty band: ... 11 already >0.95 ...
PASS   parameter-level headroom: 2/6 families ...
PASS   ledger tables populated: runs=1, personas=5, ... ground_truth=1
PASS/WARN local LLM reachable
PASS   figures written
```

A `WARN` on the difficulty band is expected and is discussed above. A **`FAIL`**
on the information barrier or the ledger is a real problem — tell me the output.

---

## 6. What to send back

Paste these four sections of `outputs/step1_realistic/STEP1_REPORT.md`:

- **§0 Verdicts** — confirms the foundation is sound
- **§2 Corpus validity** — the decorrelation table
- **§4 Separability + parameter resolution** — sets the difficulty knobs and tells
  us which families carry RQ2
- **§6 Local LLM benchmark** — sets personas × rounds × escalation rate for Step 2

Also tell me:

1. **Target venue and deadline.** Under 10 weeks ⇒ MVP scope only (T1 oracles,
   ~60 personas, RQ1/RQ2/RQ4).
2. **RAM, CPU, and whether you have a GPU.** Determines the population model.
3. Which Ollama models actually pulled successfully.

Then Step 2 (the population tier: persona agents, hybrid Level-0/Level-1 policy,
instrument validation, plausibility discriminator) comes back as **file-by-file
content to add to this same repo** — no new ZIP.

---

## 7. Also in this package

- `docs/DEVIATIONS.md` — every laptop-driven substitution, each with the sentence
  to put in the paper's limitations section. Keep it updated; a candid deviations
  log makes reviewers trust the rest of the paper more.
- `PREREGISTRATION.md` — fill in and commit **before** the final sweeps. Cheap,
  and a real credibility signal for a paper about auditing rigour.
