"""Step 3 driver: ranking surrogate + mechanism classifier, both held-out.

    python -m scripts.step3_report --conf realistic --oracle O4
    python -m scripts.step3_report --conf realistic --oracle O4 --tiny

Writes outputs/step3_<conf>_<oracle>/:
    STEP3_REPORT.md
    surrogate_fidelity.json
    mechanism_recovery.json
    confusion_matrix.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aris.config import config_hash, load                                     # noqa: E402
from aris.corpus import build_corpus                                           # noqa: E402
from aris.investigator.mechanism_classifier import fit_and_evaluate           # noqa: E402
from aris.investigator.surrogate import (RankingSurrogate, build_oracle_trace,      # noqa: E402
                                         build_session_trace, evaluate_fidelity)
from aris.investigator_driver import collect_observations, collect_training_replicates  # noqa: E402

OK, WARN, BAD = "PASS", "WARN", "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="realistic")
    ap.add_argument("--oracle", default="O4")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    cfg = load(a.conf, a.set)
    n_personas, n_rounds, n_replicates = 30, 25, 8
    if a.tiny:
        cfg["corpus"]["n_items"] = 300
        n_personas, n_rounds, n_replicates = 10, 12, 4

    seed = int(cfg["run"]["seed"])
    out = Path(f"outputs/step3_{a.conf}_{a.oracle}")
    out.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, str, str]] = []
    t0 = time.time()

    print("[1/4] corpus ...", flush=True)
    corpus = build_corpus(cfg, seed)

    # ------------------------------------------------------- surrogate fidelity
    print(f"[2/4] collecting observations against {a.oracle} for surrogate training ...", flush=True)
    rows, persona_map = collect_observations(corpus, cfg, a.oracle, seed=1,
                                             n_personas=n_personas, n_rounds=n_rounds)
    split_round = int(n_rounds * 0.7)
    train_rows = [r for r in rows if r["round"] < split_round]
    test_rows = [r for r in rows if r["round"] >= split_round]

    # Build the session trace ONCE over the FULL row history. Rebuilding it
    # separately from train_rows and test_rows would reset every session's
    # revealed-preference history to empty at the start of test_rows, silently
    # discarding the context accumulated during train_rows -- this was a real
    # bug that collapsed held-out fidelity toward zero at full corpus scale.
    full_trace = build_session_trace(rows, corpus)
    oracle_trace = build_oracle_trace(rows, corpus, persona_map)

    surr = RankingSurrogate()
    fit_report = surr.fit(train_rows, corpus, seed=0, trace=full_trace)
    checks.append(("surrogate fit", OK if fit_report.get("fitted") else BAD,
                   f"{fit_report.get('n_pairs', 0)} pairwise training examples"))

    fid_train = evaluate_fidelity(surr, train_rows, corpus, seed=2, trace=full_trace)
    fid_test = evaluate_fidelity(surr, test_rows, corpus, seed=3, trace=full_trace)

    # B7 oracle-informed ceiling: same surrogate architecture, but conditioned
    # on the TRUE persona interest_vec instead of a session-revealed estimate.
    # A real investigator never has this; it exists only to quantify how much
    # headroom persona-level information would buy over a realistic proxy
    # (04_EVALUATION.md 4.2, baseline B7).
    surr_ceiling = RankingSurrogate()
    fit_ceiling = surr_ceiling.fit(train_rows, corpus, seed=0, trace=oracle_trace)
    fid_ceiling = evaluate_fidelity(surr_ceiling, test_rows, corpus, seed=3, trace=oracle_trace)
    n_slate = int(np.mean([len(r["slate"]) for r in rows]))
    random_top1 = round(1.0 / n_slate, 3)
    random_hr1 = 0.05  # 1-of-20 by construction

    surrogate_report = {
        "oracle": a.oracle, "n_train_rows": len(train_rows), "n_test_rows": len(test_rows),
        "random_baseline": {"top1_agreement": random_top1, "hitratio_1": random_hr1},
        "fidelity_train_D_probe": fid_train, "fidelity_test_held_out": fid_test,
        "fidelity_ceiling_B7_oracle_informed": fid_ceiling,
    }
    (out / "surrogate_fidelity.json").write_text(json.dumps(surrogate_report, indent=2, default=str))

    hr1_test = fid_test.get("hitratio_1")
    hr1_ceiling = fid_ceiling.get("hitratio_1")
    checks.append(("surrogate held-out HR@1 beats random",
                   OK if (hr1_test is not None and hr1_test > random_hr1) else WARN,
                   f"HR@1={hr1_test} vs random baseline {random_hr1} "
                   f"(train/D_probe HR@1={fid_train.get('hitratio_1')})"))
    checks.append(("surrogate held-out top-1 vs random",
                   OK if (fid_test.get('top1_agreement') is not None
                          and fid_test['top1_agreement'] >= random_top1) else WARN,
                   f"top1={fid_test.get('top1_agreement')} vs random {random_top1}"))
    checks.append(("B7 oracle-informed ceiling (ground-truth interest_vec)",
                   OK if hr1_ceiling is not None else WARN,
                   f"HR@1={hr1_ceiling} -- headroom over the realistic proxy = "
                   f"{round((hr1_ceiling or 0) - (hr1_test or 0), 3)}"))

    # ----------------------------------------------------- mechanism recovery
    print(f"[3/4] mechanism classifier: {n_replicates} replicates x "
          f"{len(cfg['oracles']['active'])} families ...", flush=True)
    feats = collect_training_replicates(corpus, cfg, cfg["oracles"]["active"],
                                        n_replicates=n_replicates, seed=10,
                                        n_personas=max(10, n_personas // 2),
                                        n_rounds=max(10, n_rounds // 2))
    mech = fit_and_evaluate(feats, seed=0)
    (out / "mechanism_recovery.json").write_text(json.dumps(mech, indent=2, default=str))
    if mech.get("fitted"):
        pd.DataFrame(mech["confusion_matrix"], index=mech["families"],
                    columns=mech["families"]).to_csv(out / "confusion_matrix.csv")
        checks.append(("mechanism (family) recovery", OK,
                       f"accuracy={mech['accuracy']}, macro_F1={mech['macro_f1']} "
                       f"(Step 1 predicted this saturates near 1.0 -- expected, see STEP1_REPORT.md 4a)"))
    else:
        checks.append(("mechanism (family) recovery", WARN, mech.get("reason", "not fitted")))

    # -------------------------------------------------------------- write report
    print("[4/4] writing report ...", flush=True)
    elapsed = time.time() - t0
    lines = [
        "# ARIS Step 3 - Surrogate + Mechanism Recovery Report", "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  oracle: `{a.oracle}`  conf: `{a.conf}`",
        f"- config hash: `{config_hash(cfg)}`  wall clock: {elapsed:.1f}s{'  (tiny mode)' if a.tiny else ''}",
        "", "## 0. Verdicts", "", "| Check | Status | Detail |", "|---|---|---|",
    ]
    for name, st, det in checks:
        lines.append(f"| {name} | **{st}** | {det} |")
    lines += [
        "",
        "## 1. Ranking surrogate fidelity (04_EVALUATION.md 4.3)",
        "",
        "Trained on the first 70% of rounds (D_probe), evaluated held-out on the",
        "remaining 30% of rounds (a temporal split, never seen during fitting).",
        "HitRatio@1 is measured against 1 positive + 19 random negatives, matching",
        "the protocol in Cai et al. [5] as your survey's Section 7 describes.",
        "",
        "```json", json.dumps(surrogate_report, indent=2, default=str), "```",
        "",
        "## 2. Mechanism (policy-family) recovery",
        "",
        "```json", json.dumps(mech, indent=2, default=str), "```",
        "",
        "## 3. Interpretation",
        "",
        "- If HR@1 (held-out) clears the random baseline: the surrogate recovers",
        "  a genuine, generalizing behavioural signal from observation alone --",
        "  this is your first RQ1 evidence.",
        "- Mechanism recovery accuracy near 1.0 is EXPECTED (Step 1 predicted",
        "  family classification saturates) -- report it as a finding, not a",
        "  disappointment, and keep the paper's headline claims on parameter-level",
        "  recovery (Step 4), where Step 1 identified real headroom on O4/O6.",
        "",
        "## 4. What to paste back",
        "",
        "Paste section 0 in full, plus the fidelity_test_held_out block from",
        "section 1 and the accuracy/macro_f1 line from section 2.",
    ]
    (out / "STEP3_REPORT.md").write_text("\n".join(lines))
    print("\n".join(f"  {st:5s}  {n}: {d}" for n, st, d in checks))
    print(f"\nwrote {out/'STEP3_REPORT.md'}  ({elapsed:.1f}s)")
    return 0 if not any(st == BAD for _, st, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())