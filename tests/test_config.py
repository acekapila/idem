"""Tests for idem.config: YAML loading and schema validation."""

import textwrap

import pytest

from idem.config import ConfigError, is_floating_alias, load_config, validate_raw_config


def write_yaml(tmp_path, content):
    path = tmp_path / "golden_set.yaml"
    path.write_text(textwrap.dedent(content))
    return path


VALID_CONFIG = """
model:
  provider: openai
  model_id: gpt-4o-2024-08-06

questions:
  - id: q1
    prompt: "What is the rate?"
    checks:
      - type: contains
        value: "5.00%"
"""


class TestValidateRawConfig:
    def test_valid_config_has_no_errors(self):
        import yaml

        raw = yaml.safe_load(VALID_CONFIG)
        assert validate_raw_config(raw) == []

    def test_missing_model(self):
        errors = validate_raw_config({"questions": []})
        assert any("model" in e for e in errors)

    def test_missing_questions(self):
        errors = validate_raw_config({"model": {"provider": "openai", "model_id": "gpt-4o-2024-08-06"}})
        assert any("questions" in e for e in errors)

    def test_unsupported_provider(self):
        raw = {
            "model": {"provider": "cohere", "model_id": "command-r-2024"},
            "questions": [{"id": "q1", "prompt": "hi", "checks": [{"type": "contains", "value": "x"}]}],
        }
        errors = validate_raw_config(raw)
        assert any("provider" in e for e in errors)

    def test_floating_alias_rejected(self):
        raw = {
            "model": {"provider": "openai", "model_id": "gpt-4o"},
            "questions": [{"id": "q1", "prompt": "hi", "checks": [{"type": "contains", "value": "x"}]}],
        }
        errors = validate_raw_config(raw)
        assert any("floating alias" in e for e in errors)

    def test_claude_latest_rejected(self):
        raw = {
            "model": {"provider": "anthropic", "model_id": "claude-latest"},
            "questions": [{"id": "q1", "prompt": "hi", "checks": [{"type": "contains", "value": "x"}]}],
        }
        errors = validate_raw_config(raw)
        assert any("floating alias" in e for e in errors)

    def test_dated_model_id_accepted(self):
        assert is_floating_alias("gpt-4o-2024-08-06") is False
        assert is_floating_alias("claude-3-5-sonnet-20241022") is False

    def test_unknown_check_type(self):
        raw = {
            "model": {"provider": "openai", "model_id": "gpt-4o-2024-08-06"},
            "questions": [{"id": "q1", "prompt": "hi", "checks": [{"type": "semantic_similarity"}]}],
        }
        errors = validate_raw_config(raw)
        assert any("unknown check type" in e for e in errors)

    def test_missing_required_check_field(self):
        raw = {
            "model": {"provider": "openai", "model_id": "gpt-4o-2024-08-06"},
            "questions": [{"id": "q1", "prompt": "hi", "checks": [{"type": "contains"}]}],
        }
        errors = validate_raw_config(raw)
        assert any("value" in e for e in errors)

    def test_malformed_regex(self):
        raw = {
            "model": {"provider": "openai", "model_id": "gpt-4o-2024-08-06"},
            "questions": [
                {
                    "id": "q1",
                    "prompt": "hi",
                    "checks": [{"type": "regex_match", "pattern": "[unclosed"}],
                }
            ],
        }
        errors = validate_raw_config(raw)
        assert any("malformed regex" in e for e in errors)

    def test_regex_extract_match_without_capture_group(self):
        raw = {
            "model": {"provider": "openai", "model_id": "gpt-4o-2024-08-06"},
            "questions": [
                {
                    "id": "q1",
                    "prompt": "hi",
                    "checks": [
                        {"type": "regex_extract_match", "pattern": r"\d+\.\d{2}%", "expected": "5.00"}
                    ],
                }
            ],
        }
        errors = validate_raw_config(raw)
        assert any("capture group" in e for e in errors)

    def test_duplicate_question_ids(self):
        raw = {
            "model": {"provider": "openai", "model_id": "gpt-4o-2024-08-06"},
            "questions": [
                {"id": "q1", "prompt": "hi", "checks": [{"type": "contains", "value": "x"}]},
                {"id": "q1", "prompt": "hi again", "checks": [{"type": "contains", "value": "y"}]},
            ],
        }
        errors = validate_raw_config(raw)
        assert any("duplicate question id" in e for e in errors)

    def test_root_not_a_mapping(self):
        errors = validate_raw_config(["not", "a", "mapping"])
        assert len(errors) == 1

    def test_empty_questions_list(self):
        raw = {"model": {"provider": "openai", "model_id": "gpt-4o-2024-08-06"}, "questions": []}
        errors = validate_raw_config(raw)
        assert any("non-empty list" in e for e in errors)


class TestLoadConfig:
    def test_load_valid_config(self, tmp_path):
        path = write_yaml(tmp_path, VALID_CONFIG)
        config = load_config(path)
        assert config.model.provider == "openai"
        assert config.model.model_id == "gpt-4o-2024-08-06"
        assert len(config.questions) == 1
        assert config.questions[0].id == "q1"

    def test_load_invalid_config_raises(self, tmp_path):
        path = write_yaml(tmp_path, "model:\n  provider: openai\nquestions: []\n")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_load_malformed_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("model: [unterminated\n")
        with pytest.raises(ConfigError):
            load_config(path)
