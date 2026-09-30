"""LLM provider abstraction.

    ollama    -> local Ollama server (/api/generate)
    openai    -> any OpenAI-compatible /chat/completions endpoint
    fallback  -> no LLM; the decomposer and answer generator use their
                 deterministic, extractive paths

Providers use urllib from the standard library, so no HTTP dependency is
required. If the configured provider is unreachable at startup, the app logs
why and runs with the fallback provider instead of failing.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from ..config import Settings

log = logging.getLogger(__name__)


def _post_json(url: str, body: dict, headers: dict | None, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class FallbackProvider:
    name = "deterministic-fallback"
    is_llm = False
    status = "active"

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        raise RuntimeError("fallback provider does not generate text")


class OllamaProvider:
    is_llm = True

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        self.name = f"ollama/{model}"
        self.last_usage: dict = {}

    def healthcheck(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2.5) as resp:
            tags = json.loads(resp.read().decode())
        names = [m.get("name", "") for m in tags.get("models", [])]
        if not any(n.startswith(self.model.split(":")[0]) for n in names):
            raise RuntimeError(f"model {self.model} not pulled (available: {names})")

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        if json_mode:
            body["format"] = "json"
        data = _post_json(f"{self.base_url}/api/generate", body, None, self.timeout)
        self.last_usage = {"prompt_tokens": data.get("prompt_eval_count"), "completion_tokens": data.get("eval_count")}
        return data["response"]


class OpenAICompatibleProvider:
    is_llm = True

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout
        self.name = f"openai/{model}"
        self.last_usage: dict = {}

    def healthcheck(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        body = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = _post_json(f"{self.base_url}/chat/completions", body,
                          {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
        self.last_usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"]


def make_provider(cfg: Settings):
    choice = cfg.llm_provider.lower()
    provider = None
    try:
        if choice == "ollama":
            provider = OllamaProvider(cfg.ollama_base_url, cfg.ollama_model, cfg.llm_timeout_s)
        elif choice == "openai":
            provider = OpenAICompatibleProvider(cfg.openai_base_url, cfg.openai_api_key, cfg.openai_model,
                                                cfg.llm_timeout_s)
        if provider is not None:
            provider.healthcheck()
            provider.status = "active"
            return provider
    except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
        log.warning("LLM provider '%s' unavailable (%s); using deterministic fallback", choice, exc)
        fb = FallbackProvider()
        fb.status = f"fallback ({choice} unavailable: {exc})"
        return fb
    return FallbackProvider()
