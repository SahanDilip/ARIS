"""Step 2 driver. Gate 1 (03_IMPLEMENTATION_PLAN.md week 6): if personas fail
validation badly, fix it here -- everything downstream depends on it.

    python -m scripts.step2_report --conf realistic --oracle O4
    python -m scripts.step2_report --conf realistic --oracle O4 --tiny --no-llm
    python -m scripts.step2_report --conf realistic --oracle O4 --set population.n_personas=30

Writes outputs/step2_<conf>/:
    STEP2_REPORT.md
    observations.csv, telemetry.csv
    validation.json, plausibility.json
    ledger.duckdb (appended run)
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

from aris.config import config_hash, load                                    # noqa: E402
from aris.corpus import build_corpus                                          # noqa: E402
from aris.ledger.schema import (ObservationRecord, PersonaRecord, ProbeRecord,  # noqa: E402
                                RunRecord, TelemetryRecord)
from aris.ledger.store import Ledger, git_sha, new_run_id                     # noqa: E402
from aris.llm.client import make_llm                                          # noqa: E402
from aris.platform.env import BlackBoxPlatform                                # noqa: E402
from aris.platform.oracles import make_oracle                                 # noqa: E402
from aris.population.persona import make_personas                            # noqa: E402
from aris.population.plausibility import discriminator_auc                    # noqa: E402
from aris.population.policy import PopulationPolicy                          # noqa: E402
from aris.population.validate import validate_population                    # noqa: E402
from aris.population_driver import browse_population                        # noqa: E402

OK, WARN, BAD = "PASS", "WARN", "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="realistic")
    ap.add_argument("--oracle", default="O4", help="O4 or O6 recommended per Step-1 report")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    cfg = load(a.conf, a.set)
    if a.tiny:
        cfg["corpus"]["n_items"] = 400
        cfg["population"]["n_personas"] = 8
        cfg["population"]["rounds"] = 6
        cfg["population"]["policy"]["distill_every"] = 3
        cfg["population"]["policy"]["reflect_every"] = 3
        cfg["population"]["plausibility"]["n_rounds"] = 5

    seed = int(cfg["population"]["seed"])
    pol_cfg = cfg["population"]["policy"]
    out = Path(f"outputs/step2_{a.conf}")
    out.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, str, str]] = []
    t0 = time.time()

    print("[1/5] corpus + platform + personas ...", flush=True)
    corpus = build_corpus(cfg, int(cfg["run"]["seed"]))
    plat = BlackBoxPlatform(corpus, make_oracle(a.oracle, corpus, cfg), cfg, seed)
    personas = make_personas(corpus, int(cfg["population"]["n_personas"]), seed)
    llm = None if a.no_llm else make_llm(cfg, role="population")
    llm_backend = "stub" if a.no_llm or llm is None else type(llm).__name__
    checks.append(("LLM backend", OK if llm_backend != "StubLLM" or a.no_llm else WARN,
                   f"{llm_backend} (model={getattr(llm, 'model', 'n/a')})"))

    print(f"[2/5] browsing: {len(personas)} personas x {cfg['population']['rounds']} rounds "
          f"against oracle {a.oracle} ...", flush=True)
    policy = PopulationPolicy(eta=float(pol_cfg["eta"]), novelty_escalate=float(pol_cfg["novelty_escalate"]),
                              reflect_every=int(pol_cfg["reflect_every"]))
    rows, telem = browse_population(plat, corpus, personas, policy,
                                     int(cfg["population"]["rounds"]), llm, seed,
                                     distill_every=int(pol_cfg["distill_every"]))
    pd.DataFrame(rows).to_csv(out / "observations.csv", index=False)
    pd.DataFrame(telem).to_csv(out / "telemetry.csv", index=False)
    esc = policy.escalation_rate()
    checks.append(("closed population loop", OK if len(rows) == len(personas) * cfg["population"]["rounds"] else BAD,
                   f"{len(rows)} observations, L1 escalation rate = {esc:.1%}"))

    print("[3/5] instrument validation (Gate 1) ...", flush=True)
    vcfg = cfg["population"]["validation"]
    persona_map = {p.persona_id: p for p in personas}
    val = validate_population(rows, corpus, persona_map)
    (out / "validation.json").write_text(json.dumps(val, indent=2, default=str))
    checks.append(("position-bias reproduction",
                   OK if val["position_bias"].get("pass") else WARN,
                   f"p={val['position_bias'].get('p_value', 'n/a')}, "
                   f"mean_pos={val['position_bias'].get('mean_position', 'n/a')} vs "
                   f"uniform={val['position_bias'].get('expected_under_uniform', 'n/a')}"))
    checks.append(("affect-driven asymmetry",
                   OK if val["affect_asymmetry"].get("pass") else WARN,
                   f"gap={val['affect_asymmetry'].get('gap', 'n/a')} "
                   f"(need > {vcfg['min_affect_gap']})"))
    checks.append(("held-out prediction (pooled, population-level)",
                   OK if val["held_out_pooled"].get("pass") else BAD,
                   f"AUC={val['held_out_pooled'].get('auc', 'n/a')}, "
                   f"95% CI={val['held_out_pooled'].get('ci_95', 'n/a')} "
                   f"(pass iff CI lower bound > 0.5; n_test={val['held_out_pooled'].get('n_test','n/a')})"))
    checks.append(("outlier persona exclusion",
                   OK if val["exclusion_rate"] < 0.3 else WARN,
                   f"{len(val['excluded_personas'])}/{len(personas)} personas excluded "
                   f"({val['exclusion_rate']:.0%}, threshold 30%)"))
    checks.append(("GATE 1: instrument validation", OK if val["overall_pass"] else BAD,
                   "all three criteria met" if val["overall_pass"]
                   else "at least one criterion failed -- do not proceed to Step 3/4 until fixed"))

    print("[4/5] plausibility discriminator ...", flush=True)
    pcfg = cfg["population"]["plausibility"]
    plaus = discriminator_auc(rows, corpus, [p.persona_id for p in personas],
                              int(pcfg["slate_size"]), seed)
    (out / "plausibility.json").write_text(json.dumps(plaus, indent=2, default=str))
    if plaus.get("auc") is None:
        checks.append(("plausibility AUC", WARN, plaus.get("note", "not computed")))
    else:
        checks.append(("plausibility AUC", OK if abs(plaus["auc"] - 0.5) < 0.25 else WARN,
                       f"AUC={plaus['auc']} ({plaus['interpretation']})"))

    print("[5/5] ledger ...", flush=True)
    cfg.setdefault("ledger", {})["path"] = str(out / "ledger.duckdb")
    led = Ledger(cfg["ledger"]["path"])
    rid = new_run_id()
    led.append("runs", RunRecord(run_id=rid, config_hash=config_hash(cfg), git_sha=git_sha(),
                                 seed=seed, oracle=a.oracle, started_at=t0, notes="step2 population"))
    led.append("personas", [PersonaRecord(run_id=rid, persona_id=p.persona_id,
                                          profile_json=p.to_json()) for p in personas])
    led.append("probes", ProbeRecord(run_id=rid, probe_id="P-step2",
                                     spec_json={"kind": "population_browse", "oracle": a.oracle,
                                               "rounds": cfg["population"]["rounds"]},
                                     est_cost=float(plat.query_count)))
    led.append("observations", [
        ObservationRecord(run_id=rid, probe_id="P-step2", persona_id=r["persona_id"],
                          round=r["round"], slate_json=[int(i) for i in r["slate"]],
                          action=r["action"], action_item=int(r["action_item"]),
                          dwell_ms=int(r["dwell_ms"]), ts=float(r["ts"]))
        for r in rows])
    for tr in telem:
        if tr.get("event") == "round_end":
            led.append("telemetry", TelemetryRecord(
                run_id=rid, round=int(tr["round"]),
                heterogeneity_entropy=float(tr["heterogeneity_entropy"]),
                l1_escalation_rate=float(tr["l1_escalation_rate"])))
    counts = {t: int(led.q(f'SELECT count(*) n FROM "{t}" WHERE run_id = ?', rid)["n"][0])
              for t in ("runs", "personas", "probes", "observations", "telemetry")}
    led.close()
    checks.append(("ledger populated", OK if counts["observations"] == len(rows) else BAD,
                   ", ".join(f"{k}={v}" for k, v in counts.items())))

    elapsed = time.time() - t0
    het = [tr["heterogeneity_entropy"] for tr in telem if tr.get("event") == "round_end"]
    lines = [
        "# ARIS Step 2 - Population Tier Report", "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  oracle: `{a.oracle}`  conf: `{a.conf}`",
        f"- config hash: `{config_hash(cfg)}`  wall clock: {elapsed:.1f}s"
        f"{'  (tiny mode)' if a.tiny else ''}{'  (no LLM)' if a.no_llm else ''}",
        "", "## 0. Verdicts", "", "| Check | Status | Detail |", "|---|---|---|",
    ]
    for name, st, det in checks:
        lines.append(f"| {name} | **{st}** | {det} |")
    lines += [
        "", "## 1. Escalation and heterogeneity",
        "",
        f"- Level-0 decisions: {policy.n_l0}, Level-1 (LLM) decisions: {policy.n_l1} "
        f"({esc:.1%} escalation rate)",
        f"- heterogeneity entropy: start {het[0]:.3f} -> end {het[-1]:.3f}" if het else "- no telemetry",
        "", "## 2. Instrument validation (Gate 1)", "",
        "```json", json.dumps(val, indent=2, default=str), "```",
        "", "## 3. Plausibility discriminator", "",
        "```json", json.dumps(plaus, indent=2, default=str), "```",
        "", "## 4. What to paste back", "",
        "Paste section 0 in full, plus the escalation rate and heterogeneity line from"
        " section 1. If GATE 1 is FAIL, also paste section 2 in full -- fix validation"
        " before Step 3.",
    ]
    (out / "STEP2_REPORT.md").write_text("\n".join(lines))
    print("\n".join(f"  {st:5s}  {n}: {d}" for n, st, d in checks))
    print(f"\nwrote {out/'STEP2_REPORT.md'}  ({elapsed:.1f}s)")
    return 0 if not any(st == BAD for _, st, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())