import pytest

from micro_agent.config import configure_lm
from micro_agent.agent import MicroAgent
from micro_agent.tools import safe_eval_math


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
