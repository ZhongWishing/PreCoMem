"""LLM client wrappers shared by the sub-agents, the response model and the judges.

Two providers are supported:

* ``openai``: the OpenAI Chat Completions API and any OpenAI-compatible
  endpoint (vLLM, SGLang, ...) reached through ``base_url``. This is how the
  Qwen backbones of the backbone-generalization study are served.
* ``gemini``: Google Gemini through the ``google-genai`` SDK, used for the
  auxiliary judges and the DiMoBench verifier.

Clients return :class:`LLMResponse` objects with the text and the token usage.
:meth:`LLMClient.complete_json` adds the one-shot re-prompt that the
sub-agents rely on when a model returns malformed JSON.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..config import BackboneConfig
from ..utils.logging import get_logger

_LOGGER = get_logger("llm")


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        return self

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Optional[Any] = field(default=None, repr=False)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model reply (tolerates code fences)."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    for candidate in (cleaned, _first_balanced_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class LLMClient:
    """Provider-agnostic interface with thread-safe token accounting."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.usage = TokenUsage()
        self.n_calls = 0
        self._usage_lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        raise NotImplementedError

    def complete_json(
        self,
        prompt: str,
        system: str = "Return strict JSON only. No prose, no markdown fences.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call the model and parse a JSON object; re-prompt once on failure.

        Returns an empty dict if the second attempt is also malformed, so
        callers can degrade gracefully (operators then make no changes).
        """
        response = self.complete(prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        parsed = parse_json_object(response.text)
        if parsed is not None:
            return parsed
        _LOGGER.warning("Malformed JSON from %s; re-prompting once.", self.model)
        repair = (
            f"{prompt}\n\nYour previous reply was not valid JSON:\n{response.text[:800]}\n\n"
            "Reply again with a single valid JSON object and nothing else."
        )
        response = self.complete(repair, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        parsed = parse_json_object(response.text)
        if parsed is None:
            _LOGGER.error("Giving up on JSON parse for %s; the caller receives an empty payload.", self.model)
            return {}
        return parsed

    def _account(self, usage: TokenUsage) -> None:
        with self._usage_lock:
            self.usage += usage
            self.n_calls += 1

    def reset_usage(self) -> TokenUsage:
        with self._usage_lock:
            snapshot = TokenUsage(self.usage.prompt_tokens, self.usage.completion_tokens)
            self.usage = TokenUsage()
            self.n_calls = 0
        return snapshot


class MeteredLLM(LLMClient):
    """Per-owner token meter over a shared client.

    Several agents (or threads) can share one provider client; each wraps it
    in its own ``MeteredLLM`` so that ``usage`` reflects only its own calls.
    """

    def __init__(self, inner: LLMClient) -> None:
        super().__init__(model=inner.model)
        self.inner = inner

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        response = self.inner.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        self._account(response.usage)
        return response


class OpenAIClient(LLMClient):
    """Chat Completions client for OpenAI and OpenAI-compatible servers.

    Transient failures (timeouts, 429, 5xx) are retried by the OpenAI SDK with
    exponential backoff. A 400 response that rejects an optional request
    parameter is handled by adapting the request and remembering the change
    for all later calls:

    * ``max_tokens`` becomes ``max_completion_tokens`` (reasoning models),
    * an unsupported ``temperature`` is omitted, and
    * ``response_format`` is dropped (servers without JSON mode).

    ``extra_params`` is merged into every request, for example
    ``{"reasoning_effort": "minimal"}`` for reasoning models.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 60.0,
        max_retries: int = 6,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
        json_mode: bool = True,
        extra_params: Optional[Mapping[str, Any]] = None,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(model=model)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError("OpenAI SDK not installed. Run `pip install openai`.") from exc
            client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                timeout=request_timeout,
                max_retries=max(0, int(max_retries)),
            )
        self.client = client
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.extra_params: Dict[str, Any] = dict(extra_params or {})
        self._use_json_mode = bool(json_mode)
        self._token_param = "max_tokens"
        self._send_temperature = True

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = None
        for _ in range(4):  # the original request plus at most one adaptation per optional parameter
            kwargs = self._request(messages, temperature, max_tokens, json_mode)
            try:
                completion = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if not self._adapt(exc, kwargs):
                    raise
        if completion is None:  # pragma: no cover - every optional parameter was already adapted
            raise RuntimeError(f"{self.model}: request rejected after adapting all optional parameters")

        text = completion.choices[0].message.content or ""
        usage = TokenUsage()
        if getattr(completion, "usage", None) is not None:
            usage = TokenUsage(
                prompt_tokens=int(completion.usage.prompt_tokens or 0),
                completion_tokens=int(completion.usage.completion_tokens or 0),
            )
        self._account(usage)
        return LLMResponse(text=text, usage=usage, raw=completion)

    def _request(
        self, messages: List[Dict[str, str]], temperature: Optional[float], max_tokens: Optional[int], json_mode: bool
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self._token_param: max_tokens or self.default_max_tokens,
        }
        if self._send_temperature:
            kwargs["temperature"] = self.default_temperature if temperature is None else temperature
        if json_mode and self._use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self.extra_params)
        return kwargs

    def _adapt(self, exc: Exception, kwargs: Mapping[str, Any]) -> bool:
        """Adjust the request after a 400 that names an optional parameter. True if a retry makes sense."""
        if getattr(exc, "status_code", None) != 400:
            return False
        message = str(exc).lower()
        if self._token_param == "max_tokens" and "max_tokens" in message:
            self._token_param = "max_completion_tokens"
        elif self._send_temperature and "temperature" in message and "temperature" in kwargs:
            self._send_temperature = False
        elif self._use_json_mode and "response_format" in message and "response_format" in kwargs:
            self._use_json_mode = False
        else:
            return False
        _LOGGER.warning("%s rejected a request parameter; adapting (%s).", self.model, str(exc)[:160])
        return True


class GeminiClient(LLMClient):
    """Google Gemini client (``google-genai`` SDK).

    Gemini 2.5 models think before they answer, and thinking tokens count
    against ``max_output_tokens``. A short judge budget can therefore be spent
    entirely on thinking, which returns an empty reply. For 2.5 models the
    client disables thinking on Flash variants and caps it at
    ``thinking_budget`` tokens on the others; the output budget is raised by
    the same amount so that the visible answer keeps its full allowance.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        request_timeout: float = 60.0,
        max_retries: int = 6,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
        thinking_budget: Optional[int] = None,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(model=model)
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise ImportError("google-genai not installed. Run `pip install google-genai`.") from exc
        self._types = types
        if client is None:
            from google import genai

            client = genai.Client(
                api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                http_options=types.HttpOptions(
                    timeout=int(request_timeout * 1000),
                    retry_options=types.HttpRetryOptions(attempts=max(1, int(max_retries))),
                ),
            )
        self.client = client
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        if thinking_budget is None and "2.5" in model:
            thinking_budget = 0 if "flash" in model.lower() else 1024
        self.thinking_budget = thinking_budget

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        types = self._types
        answer_budget = max_tokens or self.default_max_tokens
        config: Dict[str, Any] = {
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_output_tokens": answer_budget + max(self.thinking_budget or 0, 0),
        }
        if system:
            config["system_instruction"] = system
        if json_mode:
            config["response_mime_type"] = "application/json"
        if self.thinking_budget is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=types.GenerateContentConfig(**config)
        )
        text = response.text or ""
        usage = TokenUsage()
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            usage = TokenUsage(
                prompt_tokens=int(meta.prompt_token_count or 0),
                completion_tokens=int((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)),
            )
        self._account(usage)
        return LLMResponse(text=text, usage=usage, raw=response)


def infer_provider(model: str) -> str:
    return "gemini" if "gemini" in model.lower() else "openai"


def build_llm_client(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    request_timeout: float = 60.0,
    max_retries: int = 6,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = True,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> LLMClient:
    provider = (provider or infer_provider(model)).lower()
    if provider == "openai":
        return OpenAIClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            request_timeout=request_timeout,
            max_retries=max_retries,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
            json_mode=json_mode,
            extra_params=extra_params,
        )
    if provider in {"gemini", "google"}:
        return GeminiClient(
            model=model,
            api_key=api_key,
            request_timeout=request_timeout,
            max_retries=max_retries,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def build_backbone_client(cfg: BackboneConfig) -> LLMClient:
    return build_llm_client(
        model=cfg.model,
        provider=cfg.provider,
        base_url=cfg.base_url,
        request_timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        json_mode=cfg.json_mode,
        extra_params=cfg.extra_params,
    )
