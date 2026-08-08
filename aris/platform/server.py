"""FastAPI wrapper: enforces the information barrier over HTTP.

    POST /session   {persona_id, seed}                        -> {session_id}
    GET  /feed      ?session_id=&k=                           -> {slate:[{item_id,text,ts}]}
    POST /interact  {session_id, item_id, action, dwell_ms}    -> {ack}
    POST /end_round {session_id}                              -> {round}
    GET  /health                                              -> {policy_version}

Run:  python -m aris.platform.server --oracle O2 --port 8000
The route set is deliberately minimal: it exposes exactly what a scraped feed
would. Nothing here returns scores, features, parameters or the policy family.
"""
from __future__ import annotations

import argparse

from pydantic import BaseModel

from ..config import load
from ..corpus import build_corpus
from .env import BlackBoxPlatform
from .oracles import make_oracle


# Request models must live at module scope: `from __future__ import annotations`
# turns every annotation into a string, and FastAPI resolves those against the
# *module* globals. Models defined inside build_app() would be invisible and
# silently demoted to query parameters.
class SessionReq(BaseModel):
    persona_id: int
    seed: int = 0


class InteractReq(BaseModel):
    session_id: str
    item_id: int
    action: str
    dwell_ms: int = 0


class RoundReq(BaseModel):
    session_id: str


def build_app(oracle: str = "O2", conf: str = "base", seed: int | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install fastapi uvicorn  # needed only for the HTTP tier") from e

    cfg = load(conf)
    seed = int(cfg["run"]["seed"]) if seed is None else seed
    corpus = build_corpus(cfg, seed)
    plat = BlackBoxPlatform(corpus, make_oracle(oracle, corpus, cfg), cfg, seed)
    app = FastAPI(title="ARIS black-box platform", version="0.1")

    @app.post("/session")
    def open_session(r: SessionReq):
        return {"session_id": plat.session(r.persona_id, r.seed)}

    @app.get("/feed")
    def feed(session_id: str, k: int | None = None):
        if session_id not in plat.sessions:
            raise HTTPException(404, "unknown session")
        items = plat.get_feed(session_id, k)
        return {"slate": [i.__dict__ for i in items], "served_at": items[0].ts if items else None}

    @app.post("/interact")
    def interact(r: InteractReq):
        if r.session_id not in plat.sessions:
            raise HTTPException(404, "unknown session")
        return plat.interact(r.session_id, r.item_id, r.action, r.dwell_ms)

    @app.post("/end_round")
    def end_round(r: RoundReq):
        plat.end_round(r.session_id)
        return {"round": plat.sessions[r.session_id].state.round}

    @app.get("/health")
    def health():
        return plat.health()

    app.state.platform = plat  # analysis-side handle; not routed
    return app


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="O2")
    ap.add_argument("--conf", default="base")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(build_app(a.oracle, a.conf), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
