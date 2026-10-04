"""Request handling of the provider clients, exercised with fake SDK objects (no network)."""

from types import SimpleNamespace

import pytest

from precomem.llm.client import GeminiClient, MeteredLLM, OpenAIClient, parse_json_object


class _BadRequest(Exception):
    status_code = 400


class _FakeCompletions:
    """Rejects the parameters listed in ``reject`` the way OpenAI-compatible servers do."""

    def __init__(self, reject=()):
        self.reject = set(reject)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "max_tokens" in kwargs and "max_tokens" in self.reject:
            raise _BadRequest("Unsupported parameter: 'max_tokens' is not supported with this model.")
        if "temperature" in kwargs and "temperature" in self.reject:
            raise _BadRequest("Unsupported value: 'temperature' does not support 0.0 with this model.")
        if "response_format" in kwargs and "response_format" in self.reject:
            raise _BadRequest("response_format is not supported by this server")
        message = SimpleNamespace(content='{"ok": true}')
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=3)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _openai(reject=(), **kw):
    completions = _FakeCompletions(reject)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIClient(model="m", client=client, **kw), completions


def test_openai_request_shape_and_usage():
    llm, calls = _openai(extra_params={"seed": 7})
    out = llm.complete("hi", system="sys", json_mode=True)
    sent = calls.calls[-1]
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["max_tokens"] == 1024 and sent["temperature"] == 0.0 and sent["seed"] == 7
    assert sent["response_format"] == {"type": "json_object"}
    assert out.text == '{"ok": true}' and llm.usage.total_tokens == 14 and llm.n_calls == 1


def test_openai_adapts_to_reasoning_model_parameters_once():
    llm, calls = _openai(reject={"max_tokens", "temperature"})
    llm.complete("hi")
    final = calls.calls[-1]
    assert "max_tokens" not in final and final["max_completion_tokens"] == 1024 and "temperature" not in final
    n = len(calls.calls)
    llm.complete("again")  # the adaptation is remembered: no further rejected requests
    assert len(calls.calls) == n + 1


def test_openai_drops_json_mode_when_unsupported_and_reraises_other_errors():
    llm, calls = _openai(reject={"response_format"})
    llm.complete("hi", json_mode=True)
    assert "response_format" not in calls.calls[-1]

    class Boom(Exception):
        status_code = 401

    def fail(**kwargs):
        raise Boom("invalid api key")

    bad = OpenAIClient(
        model="m", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    )
    with pytest.raises(Boom):
        bad.complete("hi")


def test_metered_llm_counts_only_its_own_calls():
    shared, _ = _openai()
    a, b = MeteredLLM(shared), MeteredLLM(shared)
    a.complete("x")
    a.complete("y")
    b.complete("z")
    assert (a.n_calls, b.n_calls, shared.n_calls) == (2, 1, 3)


def test_gemini_disables_thinking_on_flash_and_caps_it_elsewhere():
    pytest.importorskip("google.genai")
    seen = {}

    def generate_content(model, contents, config):
        seen["config"] = config
        usage = SimpleNamespace(prompt_token_count=9, candidates_token_count=2, thoughts_token_count=5)
        return SimpleNamespace(text='{"label": "CORRECT"}', usage_metadata=usage)

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    flash = GeminiClient("gemini-2.5-flash", client=fake)
    out = flash.complete("judge", system="s", max_tokens=64, json_mode=True)
    cfg = seen["config"]
    assert cfg.thinking_config.thinking_budget == 0 and cfg.max_output_tokens == 64
    assert cfg.response_mime_type == "application/json" and cfg.system_instruction == "s"
    assert out.usage.prompt_tokens == 9 and out.usage.completion_tokens == 7

    GeminiClient("gemini-2.5-pro", client=fake).complete("verify", max_tokens=512)
    assert seen["config"].thinking_config.thinking_budget == 1024 and seen["config"].max_output_tokens == 1536


def test_parse_json_object_handles_fences_and_trailing_text():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert parse_json_object("not json") is None
