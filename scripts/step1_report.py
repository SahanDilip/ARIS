"""Step 1 driver. One command, one report.

    python -m scripts.step1_report                 # full
    python -m scripts.step1_report --tiny          # ~30 s, for a first smoke test
    python -m scripts.step1_report --no-llm        # skip the Ollama benchmark

Writes outputs/step1/:
    STEP1_REPORT.md          <- paste this back for Step 2 calibration
    corpus_correlations.csv  <- paper table: corpus validity
    separability.csv         <- proto-E5 identifiability map
    summary_stats.csv        <- per-oracle audit statistics
    ollama_bench.json        <- cost model for the population tier
    ledger.duckdb            <- the artefact
    fig_corpus.png, fig_separability.png
"""
from __future__ import annotations

import argparse
import json
import platform as pyplatform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aris.config import config_hash, load                                    # noqa: E402
from aris.corpus import build_corpus                                          # noqa: E402
from aris.diagnostics import (FEATURE_NAMES, observe_oracle,                  # noqa: E402
                              parameter_resolution, separability_curve)
from aris.ledger.schema import (GroundTruthRecord, ObservationRecord,         # noqa: E402
                                PersonaRecord, ProbeRecord, RunRecord,
                                TelemetryRecord)
from aris.ledger.store import Ledger, git_sha, new_run_id                     # noqa: E402
from aris.platform.env import BlackBoxPlatform                                # noqa: E402
from aris.platform.oracles import make_oracle                                 # noqa: E402
from aris.population.random_agent import RandomAgent, browse                  # noqa: E402

OK, WARN, BAD = "PASS", "WARN", "FAIL"


def env_block() -> dict:
    import importlib
    vers = {}
    for m in ("numpy", "scipy", "sklearn", "pandas", "duckdb", "pydantic",
              "yaml", "fastapi", "matplotlib"):
        try:
            vers[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:
            vers[m] = "MISSING"
    return {
        "python": sys.version.split()[0],
        "os": f"{pyplatform.system()} {pyplatform.release()}",
        "machine": pyplatform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": vers,
        "git_sha": git_sha(),
    }


def interpret(curve: pd.DataFrame, wf: pd.DataFrame, bench: dict,
              budgets: list, cfg: dict) -> list[str]:
    """Turn the diagnostics into instructions. This is the point of Step 1."""
    out: list[str] = []
    lo_b, hi_b = budgets[0], budgets[-1]
    R = int(cfg["separability"]["n_replicates"])
    if R < 5:
        out.append(
            f"**Underpowered run ({R} replicates).** The permutation test cannot reach "
            f"significance with this few replicates, so every separability figure below is "
            f"floored at 0.5 and section 4 should be ignored. This is expected in `--tiny` "
            f"mode. Rerun without `--tiny` (or set `separability.n_replicates=6`) before "
            f"drawing any conclusion about testbed difficulty.")
        return out
    easy_lo = int(curve.iloc[0]["pairs_above_0.95"])
    n_pairs = int(0.5 * len(cfg["oracles"]["active"]) * (len(cfg["oracles"]["active"]) - 1))
    hard_hi = int(curve.iloc[-1]["pairs_below_0.60"])

    if easy_lo >= 0.6 * n_pairs:
        out.append(
            f"**Family classification saturates.** {easy_lo}/{n_pairs} oracle pairs are already "
            f"separable at {lo_b} platform queries. Do not frame RQ2 (value of active design) "
            f"over *family* recovery -- random probing solves it and there is no headroom. "
            f"Frame RQ2 over **parameter** recovery and over fidelity-per-query, and report the "
            f"family result honestly as saturated. This is a finding, not a bug: it says "
            f"mechanism *class* is cheap to identify and mechanism *magnitude* is not.")
    elif hard_hi >= 0.5 * n_pairs:
        out.append(
            f"**Too hard.** {hard_hi}/{n_pairs} pairs remain unseparable at {hi_b} queries. "
            f"Weaken `platform.noise_scale`, raise `platform.slate_size`, or increase the "
            f"characteristic parameters in `conf/*.yaml` until a majority separate.")
    else:
        out.append(
            f"**Healthy difficulty band.** Separability rises from "
            f"{curve.iloc[0]['mean_separability']:.2f} at {lo_b} queries to "
            f"{curve.iloc[-1]['mean_separability']:.2f} at {hi_b}. Keep this config.")

    if not wf.empty:
        d_min, d_max = wf.rel_delta.min(), wf.rel_delta.max()
        small = wf[(wf.budget == "small") & (wf.rel_delta == d_min)]
        full = wf[(wf.budget == "full") & (wf.rel_delta == d_max)]
        gap = sorted(small[small.separability < 0.70].family.tolist())
        never = sorted(full[full.separability < 0.60].family.tolist())
        if gap:
            out.append(
                f"**Use {', '.join(gap)} as the primary RQ2 families.** A {d_min:.0%} difference "
                f"in their primary parameter is not resolvable by random probing at "
                f"{int(small.queries.max())} queries, so an information-gain probe designer has a "
                f"measurable gap to close. Report the fidelity-vs-budget curves on these.")
        else:
            out.append(
                f"**No parameter-level headroom.** Random probing resolves a {d_min:.0%} "
                f"difference on every family. Shrink `deltas` in parameter_resolution() to "
                f"(0.10, 0.30), and if that still saturates, raise `platform.noise_scale` and "
                f"shrink `platform.slate_size` -- the probe designer needs something to win.")
        if never:
            out.append(
                f"**Candidate RQ5 non-identifiability: {', '.join(never)}.** Even a {d_max:.0%} "
                f"parameter difference is unresolvable at full budget under random probing. "
                f"Check in Step 4 whether active design rescues them; if not, they belong in the "
                f"identifiability map as B-indistinguishable, which is a reportable result.")

    if bench.get("backend") == "ollama":
        rec = bench.get("recommendation", {}).get("suggested_step2_config", {})
        out.append(
            f"**Step-2 population budget.** Measured median latency "
            f"{bench.get('latency_s_median')} s/call on `{bench.get('model')}`. "
            f"Suggested: {rec}. Escalation rate is the lever -- keep total Level-1 calls per "
            f"probe under {bench.get('recommendation', {}).get('l1_calls_per_20min', '?')}.")
    else:
        out.append(
            "**Install Ollama before Step 2.** `curl -fsSL https://ollama.com/install.sh | sh` "
            "then `ollama pull qwen2.5:3b-instruct`, then rerun this report so the population "
            "budget can be calibrated to your hardware. Everything above is unaffected.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="base")
    ap.add_argument("--tiny", action="store_true", help="fast smoke run")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="key.sub=value overrides")
    a = ap.parse_args()

    cfg = load(a.conf, a.set)
    if a.tiny:
        cfg["corpus"]["n_items"] = 400
        cfg["separability"] = {"n_agents": 6, "n_rounds": 8, "n_replicates": 3}
        cfg["llm"]["bench"]["n_calls"] = 2
    seed = int(cfg["run"]["seed"])
    out = Path(cfg["run"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("ledger", {})["path"] = str(out / "ledger.duckdb")
    checks: list[tuple[str, str, str]] = []
    t_start = time.time()

    # ---------------------------------------------------------------- 1 corpus
    print("[1/6] building corpus ...", flush=True)
    corpus = build_corpus(cfg, seed)
    names = ["popularity", "arousal", "|valence|", "recency", "velocity"]
    Cp = pd.DataFrame(corpus.correlations("pearson"), index=names, columns=names)
    Cs = pd.DataFrame(corpus.correlations("spearman"), index=names, columns=names)
    Cp.to_csv(out / "corpus_correlations.csv")
    Cs.to_csv(out / "corpus_correlations_spearman.csv")
    mx = corpus.max_offdiag_corr()
    checks.append(("corpus decorrelation",
                   OK if mx < cfg["corpus"]["max_abs_corr"] else WARN,
                   f"max |off-diagonal r| = {mx:.3f} (target < {cfg['corpus']['max_abs_corr']})"))

    # ------------------------------------------------------------- 2 black box
    print("[2/6] black-box smoke test ...", flush=True)
    plat = BlackBoxPlatform(corpus, make_oracle("O2", corpus, cfg), cfg, seed)
    agents = [RandomAgent(i, int(cfg["platform"]["state_dim"]), seed + i) for i in range(5)]
    rows = browse(plat, agents, rounds=10)
    served_fields = set(vars(plat.get_feed(plat.session(999, 0))[0]))
    leak = served_fields - {"item_id", "text", "ts"}
    checks.append(("information barrier (served fields)", OK if not leak else BAD,
                   f"served keys = {sorted(served_fields)}"))
    checks.append(("candidate pool hidden", OK if cfg["platform"]["reveal_pool"] is False else BAD,
                   f"pool size {cfg['platform']['candidate_pool']} sampled per request, "
                   f"slate K = {cfg['platform']['slate_size']}"))
    checks.append(("closed interaction loop", OK if len(rows) == 50 else BAD,
                   f"{len(rows)} observations, {plat.query_count} platform queries"))

    # ------------------------------------------------------------ 3 statistics
    print("[3/6] per-oracle audit statistics ...", flush=True)
    stat_rows = {}
    for o in cfg["oracles"]["active"]:
        f, _ = observe_oracle(o, corpus, cfg, seed)
        stat_rows[o] = f
    S = pd.DataFrame(stat_rows, index=FEATURE_NAMES).T
    S.to_csv(out / "summary_stats.csv")

    # ---------------------------------------------------------- 4 separability
    print("[4/6] separability vs. query budget under random probing ...", flush=True)
    onames, budgets, mats, q80 = separability_curve(corpus, cfg, seed)
    Sep = pd.DataFrame(mats[budgets[-1]], index=onames, columns=onames)
    Sep.to_csv(out / "separability.csv")
    curve = pd.DataFrame(
        {"queries": budgets,
         "mean_separability": [float(np.nanmean(mats[b])) for b in budgets],
         "pairs_above_0.95": [int((mats[b][~np.isnan(mats[b])] > 0.95).sum()) for b in budgets],
         "pairs_below_0.60": [int((mats[b][~np.isnan(mats[b])] < 0.60).sum()) for b in budgets]})
    curve.to_csv(out / "separability_curve.csv", index=False)
    Q80 = pd.DataFrame(q80, index=onames, columns=onames)
    Q80.to_csv(out / "queries_to_0.80.csv")

    vals = mats[budgets[-1]][~np.isnan(mats[budgets[-1]])]
    lo = mats[budgets[0]][~np.isnan(mats[budgets[0]])]
    n_easy_lo = int((lo > 0.95).sum())          # separable already at the SMALLEST budget
    n_hard_hi = int((vals < 0.60).sum())        # never separable at the LARGEST budget
    finite_q80 = q80[np.isfinite(q80)]
    verdict = OK
    if n_easy_lo == len(lo):
        verdict = BAD          # random probing solves it instantly: no headroom for RQ2
    elif n_easy_lo > 0.6 * len(lo) or n_hard_hi > 0.5 * len(vals):
        verdict = WARN
    checks.append((
        "testbed difficulty band", verdict,
        f"{len(vals)} pairs | at min budget ({budgets[0]} q): {n_easy_lo} already >0.95 | "
        f"at max budget ({budgets[-1]} q): mean {vals.mean():.2f}, {n_hard_hi} still <0.60 | "
        f"median queries-to-0.80 = "
        f"{int(np.median(finite_q80)) if finite_q80.size else 'n/a'}"))

    print("[4b/6] parameter resolution under random probing ...", flush=True)
    wf = pd.DataFrame(parameter_resolution(corpus, cfg, seed))
    if not wf.empty:
        wf.to_csv(out / "parameter_resolution.csv", index=False)
        small_tight = wf[(wf.budget == "small") & (wf.rel_delta == wf.rel_delta.min())]
        full_wide = wf[(wf.budget == "full") & (wf.rel_delta == wf.rel_delta.max())]
        headroom = int((small_tight.separability < 0.70).sum())
        unident = int((full_wide.separability < 0.60).sum())
        n_fam = max(1, len(small_tight))
        verdict = OK if headroom >= 2 else WARN
        checks.append((
            "parameter-level headroom", verdict,
            f"{headroom}/{n_fam} families: a {wf.rel_delta.min():.0%} parameter difference is "
            f"NOT resolvable by random probing at {int(small_tight.queries.max())} queries "
            f"(these are where active design can win RQ2); "
            f"{unident}/{n_fam} unresolvable even at {wf.rel_delta.max():.0%} and full budget "
            f"(candidate RQ5 non-identifiability)"))

    # ---------------------------------------------------------------- 5 ledger
    print("[5/6] ledger round-trip ...", flush=True)
    led = Ledger(cfg["ledger"]["path"])
    rid = new_run_id()
    led.append("runs", RunRecord(run_id=rid, config_hash=config_hash(cfg), git_sha=git_sha(),
                                 seed=seed, oracle="O2", started_at=t_start,
                                 notes="step1 foundation report"))
    led.append("personas", [PersonaRecord(run_id=rid, persona_id=ag.persona_id,
                                          profile_json={"kind": "random_agent"})
                            for ag in agents])
    led.append("probes", ProbeRecord(run_id=rid, probe_id="P-step1",
                                     spec_json={"kind": "random", "rounds": 10},
                                     est_cost=float(plat.query_count)))
    led.append("observations", [
        ObservationRecord(run_id=rid, probe_id="P-step1", persona_id=r["persona_id"],
                          round=r["round"], slate_json=[int(i) for i in r["slate"]],
                          action=r["action"], action_item=int(r["action_item"]),
                          dwell_ms=int(r["dwell_ms"]), ts=float(r["ts"]))
        for r in rows])
    led.append("telemetry", TelemetryRecord(run_id=rid, round=0,
                                            heterogeneity_entropy=float(np.log(len(agents))),
                                            l1_escalation_rate=0.0))
    led.append("ground_truth", GroundTruthRecord(run_id=rid, policy_family="O2",
                                                 params_json=plat.policy.ground_truth_params()))
    counts = {t: int(led.q(f'SELECT count(*) n FROM "{t}" WHERE run_id = ?', rid)["n"][0])
              for t in ("runs", "personas", "probes", "observations", "telemetry", "ground_truth")}
    ex = led.q("""SELECT round, count(*) n, avg(dwell_ms) mean_dwell
                  FROM observations WHERE run_id = ? GROUP BY round ORDER BY round LIMIT 3""", rid)
    led.close()
    checks.append(("ledger tables populated", OK if counts["observations"] == len(rows) else BAD,
                   ", ".join(f"{k}={v}" for k, v in counts.items())))

    # ------------------------------------------------------------------- 6 llm
    bench = {"skipped": True}
    if not a.no_llm:
        print("[6/6] local LLM benchmark (may take a minute on first call) ...", flush=True)
        from aris.llm.bench import run_bench
        bench = run_bench(cfg)
        (out / "ollama_bench.json").write_text(json.dumps(bench, indent=2, default=str))
        if bench.get("backend") == "ollama":
            checks.append(("local LLM reachable", OK,
                           f"{bench['model']} @ {bench.get('latency_s_median','?')} s/call median"))
        else:
            checks.append(("local LLM reachable", WARN,
                           "no Ollama model detected -- Step 1 still valid, Step 2 blocked"))
    else:
        checks.append(("local LLM reachable", WARN, "skipped by --no-llm"))

    # ----------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        im = ax[0].imshow(Cp.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax[0].set_xticks(range(5), names, rotation=45, ha="right", fontsize=7)
        ax[0].set_yticks(range(5), names, fontsize=7)
        ax[0].set_title("Corpus attribute correlations (Pearson)", fontsize=9)
        for i in range(5):
            for j in range(5):
                ax[0].text(j, i, f"{Cp.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].hist(corpus.popularity, bins=40, alpha=.7, label="popularity")
        ax[1].hist(corpus.arousal, bins=40, alpha=.5, label="arousal")
        ax[1].set_title("Marginals preserved after decorrelation", fontsize=9)
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig_corpus.png", dpi=150)
        plt.close(fig)

        Mmax = mats[budgets[-1]]
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        im = ax[0].imshow(Mmax, vmin=0.5, vmax=1.0, cmap="viridis")
        ax[0].set_xticks(range(len(onames)), onames)
        ax[0].set_yticks(range(len(onames)), onames)
        for i in range(len(onames)):
            for j in range(len(onames)):
                if not np.isnan(Mmax[i, j]):
                    ax[0].text(j, i, f"{Mmax[i, j]:.2f}", ha="center", va="center",
                               color="w", fontsize=7)
        ax[0].set_title(f"Pairwise separability at {budgets[-1]} queries\n(proto-E5)", fontsize=9)
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        for i in range(len(onames)):
            for j in range(i):
                ax[1].plot(budgets, [mats[b][i, j] for b in budgets], marker="o",
                           lw=1, alpha=.55, label=f"{onames[i]}/{onames[j]}")
        ax[1].axhline(0.8, ls="--", c="k", lw=.8)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("platform queries (log)")
        ax[1].set_ylabel("balanced accuracy")
        ax[1].set_title("Separability vs. budget, random probing", fontsize=9)
        ax[1].legend(fontsize=5, ncol=3, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "fig_separability.png", dpi=150)
        plt.close(fig)
        figs = OK
    except Exception as e:
        figs = f"{WARN} ({type(e).__name__})"
    checks.append(("figures written", figs if isinstance(figs, str) else OK,
                   "fig_corpus.png, fig_separability.png"))

    # ------------------------------------------------------------------ report
    elapsed = time.time() - t_start
    env = env_block()
    lines = [
        "# ARIS Step 1 - Foundation Report",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- config hash: `{config_hash(cfg)}`  seed: `{seed}`  git: `{env['git_sha']}`",
        f"- wall clock: {elapsed:.1f} s{'  (tiny mode)' if a.tiny else ''}",
        "",
        "## 0. Verdicts",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for name, st, det in checks:
        lines.append(f"| {name} | **{st}** | {det} |")

    lines += [
        "",
        "## 1. Environment",
        "",
        "```json",
        json.dumps(env, indent=2),
        "```",
        "",
        "## 2. Corpus validity (paper table: corpus construction)",
        "",
        "Pearson correlations among the confoundable attributes. Values near zero are"
        " the precondition for claiming that later identifiability results are facts"
        " about the *policies* rather than about the corpus.",
        "",
        Cp.round(3).to_markdown(),
        "",
        f"Max |off-diagonal| Pearson or Spearman: **{mx:.3f}**",
        "",
        "## 3. Per-oracle audit statistics",
        "",
        "Summary statistics observable from slates alone, one random-probe replicate"
        " per oracle. These are the features the mechanism classifier will use in Step 4.",
        "",
        S.round(3).to_markdown(),
        "",
        "## 4. Separability vs. query budget under random probing (proto-E5)",
        "",
        "Balanced accuracy of a cross-validated linear classifier separating each"
        " oracle pair from summary statistics alone, using *random* probes."
        " What matters is not the level but the **headroom**: pairs that random"
        " probing already solves at the smallest budget leave nothing for active"
        " design to win (RQ2); pairs never solved at the largest budget are the"
        " identifiability findings (RQ5).",
        "",
        "### 4a. Aggregate curve",
        "",
        curve.round(3).to_markdown(index=False),
        "",
        f"### 4b. Pairwise matrix at the largest budget ({budgets[-1]} queries)",
        "",
        Sep.round(3).to_markdown(),
        "",
        "### 4c. Queries needed to reach 0.80 balanced accuracy (inf = never)",
        "",
        Q80.to_markdown(),
        "",
        "### 4d. Parameter resolution under random probing (the real headroom)",
        "",
        "Each family's primary parameter is perturbed by +/- delta/2 in relative"
        " terms and the two settings are separated from slate statistics alone."
        " `separability` is already adjusted against a label-permutation null"
        " (`perm_p`, a permutation p-value), because with 16 features and few replicates an"
        " unadjusted classifier reports a meaningless 1.0.",
        "",
        "Values near 0.5 mean the parameter is *not* recoverable by uninformed"
        " probing -- precisely the gap information-gain-driven probe design has"
        " to close in Step 4. Values near 1.0 at the small budget mean that"
        " family is too easy and should be hardened.",
        "",
        (wf.to_markdown(index=False) if not wf.empty else "_not computed_"),
        "",
        "## 5. Ledger",
        "",
        "| table | rows |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in counts.items()],
        "",
        "Example analytic query (`observations` grouped by round):",
        "",
        ex.round(1).to_markdown(index=False),
        "",
        "## 6. Local LLM benchmark",
        "",
        "```json",
        json.dumps(bench, indent=2, default=str),
        "```",
        "",
        "## 7. Interpretation and next actions",
        "",
        *[f"{i+1}. {t}" for i, t in enumerate(
            interpret(curve, wf, bench, budgets, cfg))],
        "",
        "## 8. What to paste back",
        "",
        "Paste sections 0, 2, 4 and 6 of this file. Section 4 sets the difficulty"
        " knobs and Section 6 sets the population size, round count and escalation"
        " threshold for Step 2.",
        "",
    ]
    (out / "STEP1_REPORT.md").write_text("\n".join(lines))
    print("\n".join(f"  {st:5s}  {n}: {d}" for n, st, d in checks))
    print(f"\nwrote {out/'STEP1_REPORT.md'}  ({elapsed:.1f}s)")
    return 0 if not any(st == BAD for _, st, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
