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

import logging
import os
from typing import Protocol

from idem.config import ModelConfig, is_floating_alias

logger = logging.getLogger(__name__)


class TargetError(RuntimeError):
    """Raised when the target model cannot be called (auth, network, etc.)."""


class ModelTarget(Protocol):
    def call(self, prompt: str) -> str:
        """Send `prompt` to the model and return its raw text response."""
        ...


class OpenAITarget:
    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
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

        client = OpenAI(api_key=self._api_key)
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
        return OpenAITarget(model_config.model_id)
    if model_config.provider == "anthropic":
        return AnthropicTarget(model_config.model_id)
    raise TargetError(f"unsupported provider: {model_config.provider!r}")
