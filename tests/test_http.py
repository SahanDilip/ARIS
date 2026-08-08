"""HTTP tier tests. Skipped cleanly if fastapi/httpx are not installed.

These matter because the paper claims the information barrier is architectural.
An in-process test proves the Python object hides its internals; only an HTTP
test proves the *interface* does.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from aris.platform.server import build_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app("O2", "realistic"))


def test_full_http_round_trip(client):
    sid = client.post("/session", json={"persona_id": 1, "seed": 0}).json()["session_id"]
    feed = client.get(f"/feed?session_id={sid}&k=8").json()
    assert len(feed["slate"]) == 8
    item = feed["slate"][0]
    assert set(item) == {"item_id", "text", "ts"}, "HTTP layer leaked a field"
    assert client.post("/interact", json={"session_id": sid, "item_id": item["item_id"],
                                          "action": "like", "dwell_ms": 900}).json()["ack"]
    assert client.post("/end_round", json={"session_id": sid}).json()["round"] == 1


def test_interaction_outside_last_slate_is_rejected(client):
    sid = client.post("/session", json={"persona_id": 2, "seed": 0}).json()["session_id"]
    client.get(f"/feed?session_id={sid}")
    r = client.post("/interact", json={"session_id": sid, "item_id": 10 ** 9, "action": "like"})
    assert r.json()["ack"] is False


def test_health_exposes_only_an_opaque_version(client):
    h = client.get("/health").json()
    assert set(h) == {"policy_version", "slate_size"}
    assert "O2" not in h["policy_version"]
    assert len(h["policy_version"]) == 12


def test_unknown_session_is_404(client):
    assert client.get("/feed?session_id=nope").status_code == 404
    assert client.post("/interact", json={"session_id": "nope", "item_id": 1,
                                          "action": "like"}).status_code == 404


def test_no_route_reveals_the_policy(client):
    paths = {r.path for r in client.app.routes}
    assert paths == {"/session", "/feed", "/interact", "/end_round", "/health",
                     "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
