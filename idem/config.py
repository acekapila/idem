"""YAML config loading and schema validation for golden-set files.

Validation is deliberately strict and produces a full list of problems
in one pass (not just the first one hit) so `idem validate` can tell a
user everything wrong with their file at once, without calling any API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from idem.checks import CHECK_REGISTRY

SUPPORTED_PROVIDERS = {"openai", "anthropic", "custom"}

# Required fields on model.endpoint when provider is 'custom'.
REQUIRED_ENDPOINT_FIELDS = ["url", "request_template", "response_path"]
VALID_ENDPOINT_METHODS = {"GET", "POST", "PUT"}

# Required fields per check type, used for clear "missing field" errors
# during validation (before any check ever runs against a live response).
REQUIRED_CHECK_FIELDS: dict[str, list[str]] = {
    "contains": ["value"],
    "not_contains": ["value"],
    "contains_all": ["values"],
    "contains_any": ["values"],
    "regex_match": ["pattern"],
    "regex_extract_match": ["pattern", "expected"],
    "min_length": ["value"],
    "max_length": ["value"],
    "format_is_json": [],
    "exact_match": ["value"],
}

# Patterns that indicate a floating/unpinned model alias rather than a
# dated, reproducible version string. This is a heuristic, not a
# guarantee — providers change their aliasing schemes over time.
_FLOATING_ALIAS_PATTERNS = [
    r"^gpt-4o$",
    r"^gpt-4o-mini$",
    r"^gpt-4$",
    r"^gpt-4-turbo$",
    r"^gpt-3\.5-turbo$",
    r"-latest$",
    r"^claude-latest$",
]


class ConfigError(ValueError):
    """Raised when a golden-set config fails validation.

    Carries the full list of problems found, so callers can print all
    of them at once rather than stopping at the first.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    base_url: str | None = None  # openai only: point at a self-hosted, OpenAI-compatible server
    endpoint: dict[str, Any] | None = None  # custom only: how to call the REST API
    system_prompt: str | None = None  # openai/anthropic only: pin the agent's persona/policy text


@dataclass
class Question:
    id: str
    prompt: str
    checks: list[dict[str, Any]]


@dataclass
class Config:
    model: ModelConfig
    questions: list[Question]


def is_floating_alias(model_id: str) -> bool:
    """Heuristic check for un-pinned/floating model identifiers."""
    return any(re.match(pat, model_id) for pat in _FLOATING_ALIAS_PATTERNS)


def _validate_endpoint(endpoint: Any) -> list[str]:
    errors = []
    if not isinstance(endpoint, dict):
        return ["'model.endpoint' is required and must be a mapping when provider is 'custom'"]

    for field in REQUIRED_ENDPOINT_FIELDS:
        if not endpoint.get(field):
            errors.append(f"'model.endpoint.{field}' is required for provider 'custom'")

    method = endpoint.get("method", "POST")
    if method not in VALID_ENDPOINT_METHODS:
        errors.append(
            f"'model.endpoint.method' must be one of {sorted(VALID_ENDPOINT_METHODS)}, got {method!r}"
        )

    headers = endpoint.get("headers")
    if headers is not None and not isinstance(headers, dict):
        errors.append("'model.endpoint.headers' must be a mapping of header name to value")

    timeout = endpoint.get("timeout")
    if timeout is not None and not isinstance(timeout, (int, float)):
        errors.append("'model.endpoint.timeout' must be a number (seconds)")

    if "response_path" in endpoint and not isinstance(endpoint["response_path"], str):
        errors.append("'model.endpoint.response_path' must be a dotted string, e.g. 'choices.0.message.content'")

    return errors


def _validate_check(check: Any, question_idx: int, check_idx: int) -> list[str]:
    errors = []
    prefix = f"questions[{question_idx}].checks[{check_idx}]"

    if not isinstance(check, dict):
        return [f"{prefix}: must be a mapping, got {type(check).__name__}"]

    check_type = check.get("type")
    if not check_type:
        return [f"{prefix}: missing required field 'type'"]

    if check_type not in CHECK_REGISTRY:
        known = ", ".join(sorted(CHECK_REGISTRY))
        return [f"{prefix}: unknown check type '{check_type}' (known types: {known})"]

    for field in REQUIRED_CHECK_FIELDS[check_type]:
        if field not in check:
            errors.append(
                f"{prefix} (type={check_type}): missing required field '{field}'"
            )

    if check_type in ("regex_match", "regex_extract_match") and "pattern" in check:
        try:
            compiled = re.compile(check["pattern"])
        except re.error as exc:
            errors.append(
                f"{prefix} (type={check_type}): malformed regex pattern "
                f"{check['pattern']!r}: {exc}"
            )
        else:
            if check_type == "regex_extract_match" and compiled.groups == 0:
                errors.append(
                    f"{prefix} (type=regex_extract_match): pattern "
                    f"{check['pattern']!r} has no capture group — wrap the part "
                    "you want to extract in parentheses"
                )

    for list_field in ("values",):
        if list_field in check and not isinstance(check[list_field], list):
            errors.append(
                f"{prefix} (type={check_type}): field '{list_field}' must be a list"
            )

    return errors


def validate_raw_config(raw: Any) -> list[str]:
    """Validate a raw (already YAML-parsed) config. Returns a list of
    human-readable error strings; empty list means the config is valid.
    """
    errors: list[str] = []

    if not isinstance(raw, dict):
        return ["config root must be a mapping with 'model' and 'questions' keys"]

    model = raw.get("model")
    if model is None:
        errors.append("missing required top-level field 'model'")
    elif not isinstance(model, dict):
        errors.append("'model' must be a mapping")
    else:
        provider = model.get("provider")
        model_id = model.get("model_id")
        if not provider:
            errors.append(
                f"'model.provider' is required (must be one of {sorted(SUPPORTED_PROVIDERS)})"
            )
        elif provider not in SUPPORTED_PROVIDERS:
            errors.append(
                f"'model.provider' must be one of {sorted(SUPPORTED_PROVIDERS)}, "
                f"got {provider!r}"
            )
        if not model_id:
            errors.append("'model.model_id' is required and must be a dated/pinned version string")
        elif is_floating_alias(str(model_id)):
            errors.append(
                f"'model.model_id' value {model_id!r} looks like a floating alias, "
                "not a dated/pinned version (e.g. use 'gpt-4o-2024-08-06' instead "
                "of 'gpt-4o'). Pin an exact, dated model version."
            )

        if provider == "openai" and "base_url" in model and not isinstance(model["base_url"], str):
            errors.append("'model.base_url' must be a string")
        elif provider == "custom":
            errors.extend(_validate_endpoint(model.get("endpoint")))

        if provider in ("openai", "anthropic") and "system_prompt" in model:
            if not isinstance(model["system_prompt"], str):
                errors.append("'model.system_prompt' must be a string")
        elif provider == "custom" and "system_prompt" in model:
            errors.append(
                "'model.system_prompt' is not used with provider 'custom' — "
                "build the system message directly into 'model.endpoint.request_template'"
            )

    questions = raw.get("questions")
    if questions is None:
        errors.append("missing required top-level field 'questions'")
    elif not isinstance(questions, list) or not questions:
        errors.append("'questions' must be a non-empty list")
    else:
        seen_ids: set[str] = set()
        for i, q in enumerate(questions):
            prefix = f"questions[{i}]"
            if not isinstance(q, dict):
                errors.append(f"{prefix}: must be a mapping")
                continue

            qid = q.get("id")
            if not qid:
                errors.append(f"{prefix}: missing required field 'id'")
            elif qid in seen_ids:
                errors.append(f"{prefix}: duplicate question id '{qid}'")
            else:
                seen_ids.add(qid)

            if not q.get("prompt"):
                errors.append(f"{prefix} (id={qid}): missing required field 'prompt'")

            checks = q.get("checks")
            if not checks:
                errors.append(f"{prefix} (id={qid}): missing required field 'checks' (must be a non-empty list)")
            elif not isinstance(checks, list):
                errors.append(f"{prefix} (id={qid}): 'checks' must be a list")
            else:
                for j, check in enumerate(checks):
                    errors.extend(_validate_check(check, i, j))

    return errors


def load_raw_yaml(path: str | Path) -> Any:
    """Load and parse a YAML file, raising ConfigError on I/O or parse errors."""
    p = Path(path)
    if not p.exists():
        raise ConfigError([f"config file not found: {p}"])
    try:
        text = p.read_text()
    except OSError as exc:
        raise ConfigError([f"could not read config file {p}: {exc}"]) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError([f"invalid YAML syntax in {p}: {exc}"]) from exc


def load_config(path: str | Path) -> Config:
    """Load, validate, and parse a golden-set YAML file into a Config.

    Raises ConfigError (with the full list of problems) if validation fails.
    """
    raw = load_raw_yaml(path)
    errors = validate_raw_config(raw)
    if errors:
        raise ConfigError(errors)

    raw_model = raw["model"]
    model = ModelConfig(
        provider=raw_model["provider"],
        model_id=str(raw_model["model_id"]),
        base_url=raw_model.get("base_url"),
        endpoint=raw_model.get("endpoint"),
        system_prompt=raw_model.get("system_prompt"),
    )
    questions = [
        Question(id=q["id"], prompt=q["prompt"], checks=q["checks"])
        for q in raw["questions"]
    ]
    return Config(model=model, questions=questions)
