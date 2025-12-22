import datetime
import json
import re
import pytest

from micro_agent.config import configure_lm
from micro_agent.agent import MicroAgent
from micro_agent.tools import safe_eval_math
from micro_agent import runtime
from micro_agent.tools import run_tool


def test_no_false_time_trigger_on_update(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    configure_lm()
    agent = MicroAgent(max_steps=2)
    pred = agent("Please update the docs.")
    assert not any(step.get("tool") == "now" for step in (pred.trace or []))


def test_factorial_rejects_non_integer():
    with pytest.raises(ValueError):
        safe_eval_math("fact(3.5)")


def test_complex_results_rejected():
    with pytest.raises(ValueError):
        safe_eval_math("(-1)^(0.5)")


def test_times_is_math_not_time(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    configure_lm()
    agent = MicroAgent(max_steps=3)
    pred = agent("What is 3 times 4?")
    assert "12" in pred.answer
    assert any(step.get("tool") == "calculator" for step in (pred.trace or []))
    assert not any(step.get("tool") == "now" for step in (pred.trace or []))


def test_dump_trace_serializes_non_json(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "TRACES_DIR", str(tmp_path))
    trace_id = runtime.new_trace_id()
    steps = [{"tool": "now", "args": {}, "observation": {"when": datetime.datetime(2020, 1, 1)}}]
    path = runtime.dump_trace(trace_id, "q", steps, "a")
    with open(path, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert rec["steps"][0]["observation"]["when"].startswith("2020-01-01")


def test_result_magnitude_limit():
    with pytest.raises(ValueError):
        safe_eval_math("1000000*10000000")


def test_unicode_multiply():
    assert safe_eval_math("3\u00d74") == 12


def test_now_local_has_offset():
    obs = run_tool("now", {"timezone": "local"})
    assert "iso" in obs
    assert re.search(r"[+-]\d\d:\d\d$", obs["iso"])


def test_now_invalid_timezone_validation():
    obs = run_tool("now", {"timezone": "pst"})
    assert "error" in obs and "validation" in obs["error"]
