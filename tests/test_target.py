"""Tests for idem.target: the OpenAI base_url override and the generic
CustomHTTPTarget backend for internally-hosted models.

No real network calls are made — urllib.request.urlopen is monkeypatched
with a fake response, and env-var-dependent behavior is driven through
monkeypatch.setenv/delenv.
"""

import json
from unittest.mock import MagicMock

import pytest

from idem.target import (
    CustomHTTPTarget,
    OpenAITarget,
    TargetError,
    extract_response_path,
    interpolate_env_vars,
    substitute_prompt,
)


class TestSubstitutePrompt:
    def test_replaces_in_plain_string(self):
        assert substitute_prompt("Q: {{prompt}}", "hello") == "Q: hello"

    def test_replaces_nested_in_dict_and_list(self):
        template = {
            "model": "x",
            "messages": [{"role": "user", "content": "{{prompt}}"}],
        }
        result = substitute_prompt(template, "what is the rate?")
        assert result["messages"][0]["content"] == "what is the rate?"

    def test_non_string_values_untouched(self):
        assert substitute_prompt(42, "hello") == 42
        assert substitute_prompt(None, "hello") is None

    def test_no_placeholder_returns_unchanged(self):
        assert substitute_prompt("no placeholder here", "hello") == "no placeholder here"


class TestInterpolateEnvVars:
    def test_substitutes_set_variable(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        result = interpolate_env_vars("Bearer ${MY_TOKEN}", context="header 'Authorization'")
        assert result == "Bearer secret123"

    def test_missing_variable_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        with pytest.raises(TargetError, match="MISSING_TOKEN"):
            interpolate_env_vars("Bearer ${MISSING_TOKEN}", context="header 'Authorization'")

    def test_no_placeholder_returns_unchanged(self):
        assert interpolate_env_vars("application/json", context="header 'Content-Type'") == "application/json"


class TestExtractResponsePath:
    def test_simple_dict_path(self):
        assert extract_response_path({"status": "ok"}, "status") == "ok"

    def test_nested_dict_path(self):
        data = {"choices": [{"message": {"content": "hello"}}]}
        assert extract_response_path(data, "choices.0.message.content") == "hello"

    def test_missing_key_raises(self):
        with pytest.raises(TargetError, match="no key 'missing'"):
            extract_response_path({"status": "ok"}, "missing")

    def test_list_index_out_of_range_raises(self):
        with pytest.raises(TargetError, match="out of range"):
            extract_response_path({"choices": []}, "choices.0")

    def test_non_integer_list_index_raises(self):
        with pytest.raises(TargetError, match="expects a list index"):
            extract_response_path({"choices": [1]}, "choices.first")

    def test_non_string_leaf_raises(self):
        with pytest.raises(TargetError, match="expected a string"):
            extract_response_path({"code": 200}, "code")

    def test_cannot_descend_into_scalar(self):
        with pytest.raises(TargetError, match="cannot descend"):
            extract_response_path({"status": "ok"}, "status.nested")


class TestOpenAITargetBaseUrl:
    def test_requires_api_key_without_base_url(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(TargetError, match="OPENAI_API_KEY"):
            OpenAITarget("gpt-4o-2024-08-06")

    def test_base_url_allows_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        target = OpenAITarget("my-self-hosted-model", base_url="http://localhost:8000/v1")
        assert target._api_key == "not-needed"
        assert target.base_url == "http://localhost:8000/v1"

    def test_explicit_api_key_still_used_with_base_url(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        target = OpenAITarget("m", api_key="real-key", base_url="http://localhost:8000/v1")
        assert target._api_key == "real-key"


class TestCustomHTTPTarget:
    def _mock_urlopen(self, monkeypatch, response_body: dict, status: int = 200):
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps(response_body).encode("utf-8")
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        mock_urlopen = MagicMock(return_value=fake_response)
        monkeypatch.setattr("idem.target.urllib.request.urlopen", mock_urlopen)
        return mock_urlopen

    def test_call_extracts_response_text(self, monkeypatch):
        self._mock_urlopen(monkeypatch, {"choices": [{"message": {"content": "The rate is 5.00%."}}]})
        target = CustomHTTPTarget(
            {
                "url": "https://internal.example.com/generate",
                "request_template": {"messages": [{"role": "user", "content": "{{prompt}}"}]},
                "response_path": "choices.0.message.content",
            }
        )
        result = target.call("What is the rate?")
        assert result == "The rate is 5.00%."

    def test_request_body_has_prompt_substituted(self, monkeypatch):
        mock_urlopen = self._mock_urlopen(monkeypatch, {"text": "ok"})
        target = CustomHTTPTarget(
            {
                "url": "https://internal.example.com/generate",
                "request_template": {"input": "{{prompt}}"},
                "response_path": "text",
            }
        )
        target.call("What is the rate?")
        sent_request = mock_urlopen.call_args[0][0]
        sent_body = json.loads(sent_request.data)
        assert sent_body == {"input": "What is the rate?"}

    def test_headers_interpolate_env_vars(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_API_KEY", "secret123")
        mock_urlopen = self._mock_urlopen(monkeypatch, {"text": "ok"})
        target = CustomHTTPTarget(
            {
                "url": "https://internal.example.com/generate",
                "headers": {"Authorization": "Bearer ${CUSTOM_API_KEY}"},
                "request_template": {"input": "{{prompt}}"},
                "response_path": "text",
            }
        )
        target.call("hi")
        sent_request = mock_urlopen.call_args[0][0]
        assert sent_request.headers["Authorization"] == "Bearer secret123"

    def test_missing_env_var_raises_before_request(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        mock_urlopen = self._mock_urlopen(monkeypatch, {"text": "ok"})
        target = CustomHTTPTarget(
            {
                "url": "https://internal.example.com/generate",
                "headers": {"Authorization": "Bearer ${MISSING_KEY}"},
                "request_template": {"input": "{{prompt}}"},
                "response_path": "text",
            }
        )
        with pytest.raises(TargetError, match="MISSING_KEY"):
            target.call("hi")
        mock_urlopen.assert_not_called()

    def test_malformed_json_response_raises(self, monkeypatch):
        fake_response = MagicMock()
        fake_response.read.return_value = b"not json"
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = False
        monkeypatch.setattr("idem.target.urllib.request.urlopen", MagicMock(return_value=fake_response))
        target = CustomHTTPTarget(
            {
                "url": "https://internal.example.com/generate",
                "request_template": {"input": "{{prompt}}"},
                "response_path": "text",
            }
        )
        with pytest.raises(TargetError, match="valid JSON"):
            target.call("hi")

    def test_default_method_and_timeout(self):
        target = CustomHTTPTarget(
            {
                "url": "https://internal.example.com/generate",
                "request_template": {},
                "response_path": "text",
            }
        )
        assert target.method == "POST"
        assert target.timeout == 30
