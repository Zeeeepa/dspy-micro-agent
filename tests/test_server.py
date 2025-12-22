import importlib

from fastapi.testclient import TestClient


def _client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    import micro_agent.server as server
    importlib.reload(server)
    return TestClient(server.app)


def test_ask_rejects_empty_question(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/ask", json={"question": "   ", "max_steps": 2})
    assert resp.status_code == 400


def test_ask_max_steps_bounds(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/ask", json={"question": "hi", "max_steps": 0})
    assert resp.status_code == 400
    resp = client.post("/ask", json={"question": "hi", "max_steps": 100})
    assert resp.status_code == 400


def test_trace_id_validation(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/trace/not-a-uuid")
    assert resp.status_code == 400
