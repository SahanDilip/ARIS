"""Benchmark the local LLM and turn throughput into an experiment budget.

This is the single most useful number Step 1 produces. Everything about the
population tier -- how many personas, how many rounds, what escalation
threshold eta -- is downstream of how many Level-1 calls your laptop can afford.
The numbers here also become the paper's cost table (T6), which your survey's
Section 9 cost argument makes a genuine contribution rather than an apology.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .client import OllamaLLM, StubLLM, make_llm

_SYS = ("You are simulating one social media user. Reply with JSON only: "
        '{"choice_index": int, "action": one of ["skip","dwell","like","share"], '
        '"dwell_ms": int, "reason": short string}.')


def _prompt(pad_items: int) -> str:
    lines = [f"{i}. [measured] Report: topic {i % 12} debate widens as new figures land"
             for i in range(pad_items)]
    return ("Profile: 34, analytical, interests: politics 0.7, science 0.4. "
            "Recently liked 2 political explainers; skipped sports.\n"
            "Feed:\n" + "\n".join(lines) + "\nPick one item and an action.")


def run_bench(cfg: dict) -> dict[str, Any]:
    n_calls = int(cfg["llm"]["bench"]["n_calls"])
    llm = make_llm(cfg, role="population")
    out: dict[str, Any] = {
        "backend": "stub" if isinstance(llm, StubLLM) else "ollama",
        "model": getattr(llm, "model", "stub"),
        "host": cfg["llm"]["host"],
        "installed_models": llm.models() if isinstance(llm, OllamaLLM) else [],
    }
    lat, toks = [], []
    for i in range(n_calls):
        t0 = time.time()
        try:
            llm.chat(_SYS, _prompt(10) + f"\n(bench call {i})", json_mode=True)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            break
        lat.append(time.time() - t0)
    st = llm.stats.as_dict()
    if lat:
        lat_sorted = sorted(lat)
        out.update({
            "n_calls": len(lat),
            "latency_s_median": round(lat_sorted[len(lat_sorted) // 2], 3),
            "latency_s_mean": round(sum(lat) / len(lat), 3),
            "latency_s_max": round(max(lat), 3),
        })
    out["stats"] = st
    out["recommendation"] = recommend(out)
    return out


def recommend(bench: dict) -> dict[str, Any]:
    """Translate measured latency into a feasible Step-2 configuration.

    Model: total L1 calls ~= N_personas * T_rounds * eta_effective, and we want
    a single probe batch to finish inside ~20 minutes of wall-clock so that a
    full experiment sweep is an overnight job rather than a week.
    """
    lat = bench.get("latency_s_median")
    if bench.get("backend") == "stub" or not lat:
        return {"status": "no local model detected",
                "action": "install Ollama and pull a model, then rerun Step 1",
                "commands": ["ollama pull qwen2.5:3b-instruct", "ollama pull llama3.2:3b"],
                "note": "the rest of Step 1 is fully valid without an LLM; only this section is blocked"}

    budget_s = 20 * 60
    calls_per_batch = int(budget_s / lat)
    tiers = []
    for n, t in ((30, 20), (60, 20), (60, 40), (100, 40)):
        for eta_rate in (0.05, 0.10, 0.20):
            total = n * t * eta_rate
            tiers.append({"n_personas": n, "rounds": t, "l1_rate": eta_rate,
                          "l1_calls": int(total),
                          "est_minutes": round(total * lat / 60, 1),
                          "feasible_per_probe": total <= calls_per_batch})
        # keep the list short
    feasible = [t for t in tiers if t["feasible_per_probe"]]
    pick = max(feasible, key=lambda t: (t["n_personas"], t["rounds"], t["l1_rate"])) if feasible \
        else min(tiers, key=lambda t: t["est_minutes"])
    verdict = "comfortable" if lat < 3 else "workable" if lat < 8 else "tight"
    return {
        "status": verdict,
        "median_latency_s": lat,
        "l1_calls_per_20min": calls_per_batch,
        "suggested_step2_config": {
            "population.n_personas": pick["n_personas"],
            "population.rounds": pick["rounds"],
            "population.escalation_target_rate": pick["l1_rate"],
            "llm.population_model": bench.get("model"),
        },
        "grid": tiers,
        "note": ("If verdict is 'tight', drop to a 1.5B population model and raise eta; "
                 "record the substitution in docs/DEVIATIONS.md and the paper's limitations."),
    }


def main():  # pragma: no cover
    from ..config import load
    print(json.dumps(run_bench(load("base")), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
