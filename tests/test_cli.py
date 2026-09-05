"""Integration tests for the idem CLI, with the model backend mocked out.

These tests never make a real network call — `idem.cli.create_target` is
monkeypatched to return a fake target with canned responses, so the
whole run/validate/init pipeline is exercised deterministically.
"""

import json
import textwrap

import pytest

from idem import cli, exit_codes


class FakeTarget:
    def __init__(self, responses: dict[str, str], raise_for: set[str] | None = None):
        self.responses = responses
        self.raise_for = raise_for or set()

    def call(self, prompt: str) -> str:
        if prompt in self.raise_for:
            from idem.target import TargetError

            raise TargetError(f"simulated failure for prompt: {prompt}")
        return self.responses[prompt]


PASSING_CONFIG = """
model:
  provider: openai
  model_id: gpt-4o-2024-08-06

questions:
  - id: rate_check
    prompt: "What is the rate?"
    checks:
      - type: contains
        value: "5.00%"
      - type: min_length
        value: 5
"""

FAILING_CONFIG = """
model:
  provider: openai
  model_id: gpt-4o-2024-08-06

questions:
  - id: rate_check
    prompt: "What is the rate?"
    checks:
      - type: contains
        value: "5.00%"
"""

INVALID_CONFIG = """
model:
  provider: openai
questions: []
"""


def write_config(tmp_path, content):
    path = tmp_path / "golden_set.yaml"
    path.write_text(textwrap.dedent(content))
    return path


class TestValidateCommand:
    def test_valid_config_exits_zero(self, tmp_path, capsys):
        path = write_config(tmp_path, PASSING_CONFIG)
        code = cli.main(["validate", "--config", str(path)])
        assert code == exit_codes.OK
        assert "valid" in capsys.readouterr().out

    def test_invalid_config_exits_error(self, tmp_path, capsys):
        path = write_config(tmp_path, INVALID_CONFIG)
        code = cli.main(["validate", "--config", str(path)])
        assert code == exit_codes.ERROR
        assert "INVALID" in capsys.readouterr().err

    def test_missing_file_exits_error(self, tmp_path):
        code = cli.main(["validate", "--config", str(tmp_path / "nope.yaml")])
        assert code == exit_codes.ERROR


class TestInitCommand:
    def test_creates_file(self, tmp_path):
        output = tmp_path / "new_golden_set.yaml"
        code = cli.main(["init", "--output", str(output)])
        assert code == exit_codes.OK
        assert output.exists()
        assert "model:" in output.read_text()

    def test_refuses_to_overwrite(self, tmp_path):
        output = tmp_path / "existing.yaml"
        output.write_text("do not touch me")
        code = cli.main(["init", "--output", str(output)])
        assert code == exit_codes.ERROR
        assert output.read_text() == "do not touch me"

    def test_scaffolded_file_validates(self, tmp_path):
        output = tmp_path / "scaffolded.yaml"
        cli.main(["init", "--output", str(output)])
        code = cli.main(["validate", "--config", str(output)])
        assert code == exit_codes.OK


class TestRunCommand:
    def test_all_checks_pass(self, tmp_path, monkeypatch):
        config_path = write_config(tmp_path, PASSING_CONFIG)
        output_dir = tmp_path / "reports"

        fake = FakeTarget({"What is the rate?": "The standard rate is 5.00% p.a."})
        monkeypatch.setattr(cli, "create_target", lambda model_config: fake)

        code = cli.main(["run", "--config", str(config_path), "--output-dir", str(output_dir)])

        assert code == exit_codes.OK
        jsonl_path = output_dir / "audit_log.jsonl"
        assert jsonl_path.exists()
        record = json.loads(jsonl_path.read_text().strip().splitlines()[0])
        assert record["question_id"] == "rate_check"
        assert record["overall_passed"] is True
        assert (output_dir / "report.md").exists()
        assert "ALL CHECKS PASSED" in (output_dir / "report.md").read_text()

    def test_check_failure_exits_one(self, tmp_path, monkeypatch):
        config_path = write_config(tmp_path, FAILING_CONFIG)
        output_dir = tmp_path / "reports"

        fake = FakeTarget({"What is the rate?": "The standard rate is 9.25% p.a."})
        monkeypatch.setattr(cli, "create_target", lambda model_config: fake)

        code = cli.main(["run", "--config", str(config_path), "--output-dir", str(output_dir)])

        assert code == exit_codes.FAIL
        record = json.loads((output_dir / "audit_log.jsonl").read_text().strip().splitlines()[0])
        assert record["overall_passed"] is False
        assert "FAILED" in (output_dir / "report.md").read_text()

    def test_invalid_config_exits_error_without_calling_target(self, tmp_path, monkeypatch):
        config_path = write_config(tmp_path, INVALID_CONFIG)
        called = False

        def fail_if_called(model_config):
            nonlocal called
            called = True
            raise AssertionError("should not be called for invalid config")

        monkeypatch.setattr(cli, "create_target", fail_if_called)
        code = cli.main(["run", "--config", str(config_path), "--output-dir", str(tmp_path / "reports")])

        assert code == exit_codes.ERROR
        assert called is False

    def test_target_call_failure_exits_error(self, tmp_path, monkeypatch):
        config_path = write_config(tmp_path, PASSING_CONFIG)
        output_dir = tmp_path / "reports"

        fake = FakeTarget({}, raise_for={"What is the rate?"})
        monkeypatch.setattr(cli, "create_target", lambda model_config: fake)

        code = cli.main(["run", "--config", str(config_path), "--output-dir", str(output_dir)])

        assert code == exit_codes.ERROR
        record = json.loads((output_dir / "audit_log.jsonl").read_text().strip().splitlines()[0])
        assert record["overall_passed"] is False
        assert "error" in record

    def test_appends_to_existing_jsonl_across_runs(self, tmp_path, monkeypatch):
        config_path = write_config(tmp_path, PASSING_CONFIG)
        output_dir = tmp_path / "reports"
        fake = FakeTarget({"What is the rate?": "The standard rate is 5.00% p.a."})
        monkeypatch.setattr(cli, "create_target", lambda model_config: fake)

        cli.main(["run", "--config", str(config_path), "--output-dir", str(output_dir)])
        cli.main(["run", "--config", str(config_path), "--output-dir", str(output_dir)])

        lines = (output_dir / "audit_log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
