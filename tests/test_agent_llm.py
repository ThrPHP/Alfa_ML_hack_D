import pytest

from agent.llm import LLMBadResponse, LLMClient, LLMUnavailable, extract_json_object, scripted_transport
from agent.schemas import HypothesisBatch, TaskUnderstanding


def test_extracts_json_from_reasoning_fences_and_prose():
    text = '<think>maybe {"not": "this"}</think>Sure! ```json\n{"metric_name": "rmse", "summary": "x"}\n``` done'
    assert extract_json_object(text) == {"metric_name": "rmse", "summary": "x"}
    assert extract_json_object('Answer: {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    with pytest.raises(LLMBadResponse):
        extract_json_object("no json here")


def test_lenient_schema_tolerates_extra_keys_and_nulls():
    batch = HypothesisBatch.model_validate(
        {"analysis": "a", "hypotheses": [{"title": "t", "kind": "weird", "hyperparameters": None, "extra": 1}], "foo": 2}
    )
    assert batch.hypotheses[0].kind == "other"
    assert batch.hypotheses[0].hyperparameters == {}


def test_repair_round_fixes_a_bad_answer_and_counts_usage():
    client = LLMClient(scripted_transport(["I think rmse.", '{"metric_name": "rmse"}']), max_calls=5)
    result = client.complete_json("sys", "user", TaskUnderstanding)
    assert result.metric_name == "rmse"
    assert client.usage.calls == 2 and client.usage.total_tokens > 0


def test_budget_and_errors_make_the_client_unavailable():
    client = LLMClient(scripted_transport(['{"a": 1}'] * 3), max_calls=1)
    client.complete_json("s", "u", TaskUnderstanding)
    with pytest.raises(LLMUnavailable):
        client.complete_json("s", "u", TaskUnderstanding)
    assert not client.available

    def broken(messages):
        raise ConnectionError("down")

    client = LLMClient(broken, max_consecutive_errors=2)
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            client.complete_json("s", "u", TaskUnderstanding)
    assert not client.available and "consecutive" in client.disabled_reason


def test_from_env_returns_none_without_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert LLMClient.from_env() is None
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.setenv("AGENT_DISABLE_LLM", "1")
    assert LLMClient.from_env() is None
