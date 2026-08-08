"""Local LLM access via Ollama, plus a deterministic stub.

Design constraints for this project (laptop, no paid API):
  * Both tiers run locally. The investigator tier normally wants a frontier
    model; we substitute a larger local model and *narrow its job* so a 7B can
    do it -- it proposes structure over a schema, it never estimates numbers.
    Recorded in docs/DEVIATIONS.md as D2 for the paper's limitations section.
  * Every call is cached on disk keyed by (model, messages, options). Reruns are
    free and byte-identical, which is what makes "one command regenerates every
    number" true in the presence of an LLM.
  * A hard call budget raises rather than silently truncating.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["tokens_per_s"] = round(self.completion_tokens / self.wall_s, 2) if self.wall_s else 0.0
        return d


class LLMClient(Protocol):
    def chat(self, system: str, user: str, *, json_mode: bool = False,
             model: str | None = None) -> str: ...


@dataclass
class StubLLM:
    """Deterministic, offline. Used in CI, in tiny-mode, and as the -LLM ablation."""

    seed: int = 0
    stats: LLMStats = field(default_factory=LLMStats)

    def chat(self, system: str, user: str, *, json_mode: bool = False, model=None) -> str:
        self.stats.calls += 1
        h = int(hashlib.sha256((system + user + str(self.seed)).encode()).hexdigest()[:8], 16)
        if json_mode:
            acts = ["skip", "dwell", "like", "share"]
            return json.dumps({"choice_index": h % 10, "action": acts[h % 4],
                               "dwell_ms": 400 + (h % 20) * 150, "reason": "stub"})
        return f"stub-response-{h % 100000}"


class OllamaLLM:
    """Thin, cached client for a local Ollama server."""

    def __init__(self, cfg: dict, role: str = "population", max_calls: int | None = None):
        c = cfg["llm"]
        self.host = c["host"].rstrip("/")
        self.model = c["population_model"] if role == "population" else c["investigator_model"]
        self.fallback = c.get("fallback_model")
        self.opts = {
            "temperature": float(c.get("temperature", 0.7)),
            "seed": int(c.get("seed", 0)),
            "num_predict": int(c.get("num_predict", 160)),
            "num_ctx": int(c.get("num_ctx", 2048)),
        }
        self.timeout = int(c.get("timeout_s", 180))
        self.cache_dir = Path(c.get("cache_dir", ".llm_cache")) / role
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.stats = LLMStats()

    # ------------------------------------------------------------------ health
    def available(self) -> bool:
        try:
            requests.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
            return True
        except Exception:
            return False

    def models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def resolve_model(self) -> str | None:
        """Pick the configured model, else a same-family tag, else the fallback."""
        have = self.models()
        if not have:
            return None
        for want in (self.model, self.fallback):
            if not want:
                continue
            if want in have:
                return want
            stem = want.split(":")[0]
            for h in have:
                if h.split(":")[0] == stem:
                    return h
        return have[0]

    # -------------------------------------------------------------------- chat
    def _key(self, model: str, msgs: list[dict], json_mode: bool) -> Path:
        blob = json.dumps([model, msgs, self.opts, json_mode], sort_keys=True).encode()
        return self.cache_dir / f"{hashlib.sha256(blob).hexdigest()[:32]}.json"

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             model: str | None = None) -> str:
        model = model or self.model
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ck = self._key(model, msgs, json_mode)
        if ck.exists():
            self.stats.cache_hits += 1
            return json.loads(ck.read_text())["content"]
        if self.max_calls is not None and self.stats.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget {self.max_calls} exhausted")

        payload: dict[str, Any] = {"model": model, "messages": msgs,
                                   "stream": False, "options": self.opts}
        if json_mode:
            payload["format"] = "json"
        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        dt = time.time() - t0
        content = d.get("message", {}).get("content", "")
        self.stats.calls += 1
        self.stats.wall_s += dt
        self.stats.prompt_tokens += int(d.get("prompt_eval_count", 0))
        self.stats.completion_tokens += int(d.get("eval_count", 0))
        ck.write_text(json.dumps({"content": content, "model": model, "wall_s": dt}))
        return content

    def chat_json(self, system: str, user: str, schema_model=None, retries: int = 2):
        """JSON-mode call validated against a pydantic model. Invalid -> retry -> None."""
        for attempt in range(retries + 1):
            raw = self.chat(system, user + ("" if attempt == 0 else "\nReturn valid JSON only."),
                            json_mode=True)
            try:
                obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            except Exception:
                continue
            if schema_model is None:
                return obj
            try:
                return schema_model(**obj)
            except Exception:
                continue
        return None


def make_llm(cfg: dict, role: str = "population", max_calls: int | None = None):
    """Return an Ollama client if reachable, else the deterministic stub."""
    if cfg["llm"].get("backend", "ollama") == "stub":
        return StubLLM(seed=int(cfg["llm"].get("seed", 0)))
    c = OllamaLLM(cfg, role, max_calls)
    if not c.available():
        return StubLLM(seed=int(cfg["llm"].get("seed", 0)))
    m = c.resolve_model()
    if m is None:
        return StubLLM(seed=int(cfg["llm"].get("seed", 0)))
    c.model = m
    return c
