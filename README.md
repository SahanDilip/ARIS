# ARIS

Adversarial recommender introspection/surrogate research testbed: a black-box
recommendation platform with known ground-truth oracles, built to study how well
an outside agent can extract and audit a recommender's mechanism under a
plausibility constraint.

Built in seven steps, one laptop, Ollama, no paid services. See
[STEP_PLAN.md](STEP_PLAN.md) for the full roadmap and the two gates that decide
the paper.

## Status

**Step 1 (Foundation) is complete**: item corpus with decorrelation, oracles
O1-O6, a black-box platform (in-process and over HTTP), a DuckDB experiment
ledger, an enforced information barrier, a local-LLM client with disk caching,
and a testbed-difficulty diagnostic.

Start here: [README_STEP1.md](README_STEP1.md).

## Layout

| Path | Contents |
|---|---|
| `aris/` | Library code: corpus, platform/oracles, ledger, LLM client, population/investigator (later steps) |
| `conf/` | YAML configs (`base`, `realistic`, `hard`) — every reported number must be regenerable from a config + seed |
| `scripts/` | Entry points, e.g. `step1_report.py` |
| `tests/` | Test suite, including the information-barrier test |
| `docs/DEVIATIONS.md` | Every laptop-driven substitution from the original plan, with the limitations-section sentence for each |
| `PREREGISTRATION.md` | Hypotheses and analysis decisions, committed before final sweeps |
| `outputs/` | Generated reports/figures/ledgers (gitignored — regenerate with `make step1-realistic`) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test              # 18 tests, ~2 s
make smoke             # ~45 s, no LLM
make step1-realistic   # the real run, ~4-8 min
```

Optional, for Step 2 onward: install [Ollama](https://ollama.com) and pull
`qwen2.5:3b-instruct` / `qwen2.5:7b-instruct`. Details in
[README_STEP1.md](README_STEP1.md).
