"""Calls the pinned target model.

Two backends are supported: OpenAI and Anthropic. Both are imported
lazily (inside the functions that need them) so installing idem does
not require both SDKs — only the one for the provider you actually use.

Provider SDKs are treated as a thin transport: this module's job is
just to send the fixed prompt and return the raw text response. All
comparison logic lives in checks.py, deliberately kept away from any
network code so it stays pure and unit-testable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

from idem.config import ModelConfig, is_floating_alias

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class TargetError(RuntimeError):
    """Raised when the target model cannot be called (auth, network, etc.)."""


class ModelTarget(Protocol):
    def call(self, prompt: str) -> str:
        """Send `prompt` to the model and return its raw text response."""
        ...


class OpenAITarget:
    """Backend for OpenAI itself, or any OpenAI-compatible server.

    Passing `base_url` points this at a self-hosted server (vLLM, Ollama,
    Text Generation Inference, LM Studio, etc.) that speaks the same
    `/chat/completions` protocol instead of api.openai.com. Many such
    servers don't check the API key at all, so one is not required when
    `base_url` is set.
    """

    def __init__(self, model_id: str, api_key: str | None = None, base_url: str | None = None):
        self.model_id = model_id
        self.base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            if base_url:
                self._api_key = "not-needed"
            else:
                raise TargetError(
                    "OPENAI_API_KEY is not set. Export it or pass an api key explicitly."
                )

    def call(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TargetError(
                "the 'openai' package is required for provider=openai; "
                "install it with `pip install idem-check[openai]`"
            ) from exc

        client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # SDK raises various API/network errors
            raise TargetError(f"OpenAI API call failed: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise TargetError("OpenAI API returned an empty response")
        return content


class AnthropicTarget:
    def __init__(self, model_id: str, api_key: str | None = None, max_tokens: int = 1024):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise TargetError(
                "ANTHROPIC_API_KEY is not set. Export it or pass an api key explicitly."
            )

    def call(self, prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise TargetError(
                "the 'anthropic' package is required for provider=anthropic; "
                "install it with `pip install idem-check[anthropic]`"
            ) from exc

        client = Anthropic(api_key=self._api_key)
        try:
            response = client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # SDK raises various API/network errors
            raise TargetError(f"Anthropic API call failed: {exc}") from exc

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise TargetError("Anthropic API returned no text content")
        return "".join(text_blocks)


def substitute_prompt(template: Any, prompt: str) -> Any:
    """Recursively replace the literal token "{{prompt}}" with `prompt`
    inside a request template (at any nesting depth, in any string).
    """
    if isinstance(template, str):
        return template.replace("{{prompt}}", prompt)
    if isinstance(template, dict):
        return {key: substitute_prompt(value, prompt) for key, value in template.items()}
    if isinstance(template, list):
        return [substitute_prompt(item, prompt) for item in template]
    return template


def interpolate_env_vars(value: str, context: str) -> str:
    """Replace ${VAR_NAME} in `value` with the environment variable's value.

    Raises TargetError (rather than silently substituting an empty string)
    if a referenced variable is not set.
    """

    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name not in os.environ:
            raise TargetError(f"{context} references environment variable '{var_name}', which is not set")
        return os.environ[var_name]

    return _ENV_VAR_PATTERN.sub(replace, value)


def extract_response_path(data: Any, path: str) -> str:
    """Walk a dotted path (e.g. "choices.0.message.content") through a
    parsed JSON response and return the string found there.
    """
    current = data
    consumed: list[str] = []
    for segment in path.split("."):
        consumed.append(segment)
        walked = ".".join(consumed)
        if isinstance(current, list):
            try:
                index = int(segment)
                current = current[index]
            except ValueError as exc:
                raise TargetError(
                    f"response_path '{path}' expects a list index at '{walked}', got {segment!r}"
                ) from exc
            except IndexError as exc:
                raise TargetError(f"response_path '{path}' index out of range at '{walked}'") from exc
        elif isinstance(current, dict):
            if segment not in current:
                raise TargetError(
                    f"response_path '{path}' has no key '{segment}' at '{walked}' "
                    f"(available keys: {sorted(current.keys())})"
                )
            current = current[segment]
        else:
            raise TargetError(
                f"response_path '{path}' cannot descend into {type(current).__name__} at '{walked}'"
            )

    if not isinstance(current, str):
        raise TargetError(
            f"response_path '{path}' resolved to a {type(current).__name__}, expected a string"
        )
    return current


class CustomHTTPTarget:
    """Backend for an arbitrary, internally-hosted model behind its own
    REST API (own request/response JSON shape — not OpenAI-compatible).

    Configured entirely from `model.endpoint` in the golden-set YAML:

        endpoint:
          url: "https://internal-api.example.com/v1/generate"
          method: POST                              # default: POST
          headers:
            Authorization: "Bearer ${CUSTOM_API_KEY}"  # env vars interpolated at call time
          request_template:
            model: "support-bot-2024-09-01"
            messages:
              - role: user
                content: "{{prompt}}"                # replaced with the question's prompt
          response_path: "choices.0.message.content"  # dotted path into the JSON response
          timeout: 30                                # seconds, default 30
    """

    def __init__(self, endpoint: dict[str, Any]):
        self.url = endpoint["url"]
        self.method = endpoint.get("method", "POST")
        self.headers = endpoint.get("headers") or {}
        self.request_template = endpoint.get("request_template", {})
        self.response_path = endpoint["response_path"]
        self.timeout = endpoint.get("timeout", 30)

    def call(self, prompt: str) -> str:
        body = substitute_prompt(self.request_template, prompt)
        headers = {
            key: interpolate_env_vars(value, context=f"header '{key}'")
            for key, value in self.headers.items()
        }
        headers.setdefault("Content-Type", "application/json")

        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method=self.method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TargetError(f"custom endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise TargetError(f"custom endpoint request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TargetError(f"custom endpoint request timed out after {self.timeout}s") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TargetError(f"custom endpoint did not return valid JSON: {exc}") from exc

        return extract_response_path(parsed, self.response_path)


def create_target(model_config: ModelConfig) -> ModelTarget:
    """Build the appropriate backend for `model_config`.

    Config validation (idem.config.validate_raw_config) is the primary
    gate against floating model aliases and should always run before
    this is called from the CLI. This is a defense-in-depth check for
    callers that construct a target directly.
    """
    if is_floating_alias(model_config.model_id):
        logger.warning(
            "model_id '%s' looks like a floating alias rather than a dated, "
            "pinned version — drift results may reflect the vendor moving the "
            "alias rather than a real change.",
            model_config.model_id,
        )

    if model_config.provider == "openai":
        return OpenAITarget(model_config.model_id, base_url=model_config.base_url)
    if model_config.provider == "anthropic":
        return AnthropicTarget(model_config.model_id)
    if model_config.provider == "custom":
        assert model_config.endpoint is not None  # guaranteed by validate_raw_config
        return CustomHTTPTarget(model_config.endpoint)
    raise TargetError(f"unsupported provider: {model_config.provider!r}")
