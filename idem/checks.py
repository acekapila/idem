"""Deterministic check implementations.

Every check is a pure function: ``(response_text, check_config) -> CheckResult``.
No network calls, no randomness, no hidden state — given the same two
inputs, a check always returns the same result. That's the entire
point of v1: every verdict traces back to an explicit rule that a
non-technical auditor can read in ``detail`` and verify by eye.

To add a new check type: write a pure function with this signature,
then register it in ``CHECK_REGISTRY`` at the bottom of this file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


class CheckConfigError(ValueError):
    """Raised when a check's configuration is missing required fields."""


@dataclass
class CheckResult:
    """The outcome of running a single check against a response.

    ``detail`` is written for a human auditor, not a developer — it
    should read as a complete sentence explaining exactly why the
    check passed or failed.
    """

    check_type: str
    passed: bool
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, check_config: dict[str, Any]) -> dict[str, Any]:
        """Merge the original check config with the result for logging."""
        record = {k: v for k, v in check_config.items() if k != "type"}
        record["type"] = self.check_type
        record.update(self.extra)
        record["passed"] = self.passed
        record["detail"] = self.detail
        return record


def _require(check_config: dict[str, Any], key: str, check_type: str) -> Any:
    if key not in check_config:
        raise CheckConfigError(f"check type '{check_type}' requires a '{key}' field")
    return check_config[key]


def check_contains(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    value = _require(check_config, "value", "contains")
    passed = value in response_text
    detail = (
        f"Found {value!r} in response."
        if passed
        else f"Expected to find {value!r} in response, but it was not present."
    )
    return CheckResult("contains", passed, detail)


def check_not_contains(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    value = _require(check_config, "value", "not_contains")
    passed = value not in response_text
    detail = (
        f"Confirmed {value!r} is absent from response."
        if passed
        else f"Found forbidden text {value!r} in response."
    )
    return CheckResult("not_contains", passed, detail)


def check_contains_all(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    values = _require(check_config, "values", "contains_all")
    missing = [v for v in values if v not in response_text]
    passed = not missing
    detail = (
        f"All {len(values)} required strings were found in response."
        if passed
        else f"Missing {len(missing)} of {len(values)} required strings: {missing!r}."
    )
    return CheckResult("contains_all", passed, detail)


def check_contains_any(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    values = _require(check_config, "values", "contains_any")
    found = [v for v in values if v in response_text]
    passed = bool(found)
    detail = (
        f"Found at least one of the expected strings: {found!r}."
        if passed
        else f"None of the expected strings {values!r} were found in response."
    )
    return CheckResult("contains_any", passed, detail)


def check_regex_match(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    pattern = _require(check_config, "pattern", "regex_match")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise CheckConfigError(f"invalid regex pattern {pattern!r}: {exc}") from exc
    match = compiled.search(response_text)
    passed = match is not None
    detail = (
        f"Pattern {pattern!r} matched response at position {match.start()}."
        if passed
        else f"Pattern {pattern!r} did not match anywhere in response."
    )
    return CheckResult("regex_match", passed, detail)


def check_regex_extract_match(
    response_text: str, check_config: dict[str, Any]
) -> CheckResult:
    pattern = _require(check_config, "pattern", "regex_extract_match")
    expected = _require(check_config, "expected", "regex_extract_match")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise CheckConfigError(f"invalid regex pattern {pattern!r}: {exc}") from exc

    match = compiled.search(response_text)
    if match is None:
        return CheckResult(
            "regex_extract_match",
            False,
            f"Pattern {pattern!r} did not match anywhere in response — "
            f"expected to extract {expected!r}.",
            extra={"extracted": None},
        )

    if compiled.groups == 0:
        raise CheckConfigError(
            f"regex_extract_match pattern {pattern!r} has no capture group; "
            "wrap the part you want to extract in parentheses"
        )

    extracted = match.group(1)
    passed = extracted == expected
    detail = (
        f"Extracted {extracted!r}, matching expected value {expected!r}."
        if passed
        else f"Extracted {extracted!r} but expected {expected!r} — VALUE MISMATCH."
    )
    return CheckResult(
        "regex_extract_match", passed, detail, extra={"extracted": extracted}
    )


def check_min_length(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    value = _require(check_config, "value", "min_length")
    length = len(response_text)
    passed = length >= value
    detail = (
        f"Response length {length} meets minimum of {value} characters."
        if passed
        else f"Response length {length} is below minimum of {value} characters."
    )
    return CheckResult("min_length", passed, detail)


def check_max_length(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    value = _require(check_config, "value", "max_length")
    length = len(response_text)
    passed = length <= value
    detail = (
        f"Response length {length} is within maximum of {value} characters."
        if passed
        else f"Response length {length} exceeds maximum of {value} characters."
    )
    return CheckResult("max_length", passed, detail)


def check_format_is_json(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    try:
        json.loads(response_text)
    except json.JSONDecodeError as exc:
        return CheckResult(
            "format_is_json", False, f"Response is not valid JSON: {exc}"
        )
    return CheckResult("format_is_json", True, "Response is valid, parseable JSON.")


def check_exact_match(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    value = _require(check_config, "value", "exact_match")
    passed = response_text == value
    detail = (
        "Response is exactly equal to the expected string."
        if passed
        else "Response does not exactly equal the expected string "
        f"(expected {len(value)} chars, got {len(response_text)} chars)."
    )
    return CheckResult("exact_match", passed, detail)


CHECK_REGISTRY: dict[str, Callable[[str, dict[str, Any]], CheckResult]] = {
    "contains": check_contains,
    "not_contains": check_not_contains,
    "contains_all": check_contains_all,
    "contains_any": check_contains_any,
    "regex_match": check_regex_match,
    "regex_extract_match": check_regex_extract_match,
    "min_length": check_min_length,
    "max_length": check_max_length,
    "format_is_json": check_format_is_json,
    "exact_match": check_exact_match,
}


def run_check(response_text: str, check_config: dict[str, Any]) -> CheckResult:
    """Dispatch a single check config to its implementation."""
    check_type = check_config.get("type")
    if check_type not in CHECK_REGISTRY:
        raise CheckConfigError(f"unknown check type: {check_type!r}")
    return CHECK_REGISTRY[check_type](response_text, check_config)
