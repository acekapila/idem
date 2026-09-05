"""Unit tests for every check type in idem.checks, including edge cases."""

import pytest

from idem.checks import CheckConfigError, run_check


class TestContains:
    def test_pass(self):
        result = run_check("The rate is 5.00%", {"type": "contains", "value": "5.00%"})
        assert result.passed is True
        assert "5.00%" in result.detail

    def test_fail(self):
        result = run_check("The rate is 9.25%", {"type": "contains", "value": "5.00%"})
        assert result.passed is False

    def test_empty_response(self):
        result = run_check("", {"type": "contains", "value": "anything"})
        assert result.passed is False

    def test_missing_value_raises(self):
        with pytest.raises(CheckConfigError):
            run_check("text", {"type": "contains"})


class TestNotContains:
    def test_pass_when_absent(self):
        result = run_check("The complaint process is simple.", {"type": "not_contains", "value": "cannot help"})
        assert result.passed is True

    def test_fail_when_present(self):
        result = run_check("I cannot provide financial advice.", {"type": "not_contains", "value": "cannot provide"})
        assert result.passed is False


class TestContainsAll:
    def test_pass(self):
        result = run_check(
            "Contact us by email, phone, or written complaint.",
            {"type": "contains_all", "values": ["email", "phone", "written complaint"]},
        )
        assert result.passed is True

    def test_fail_partial(self):
        result = run_check(
            "Contact us by email only.",
            {"type": "contains_all", "values": ["email", "phone", "written complaint"]},
        )
        assert result.passed is False
        assert "phone" in result.detail

    def test_empty_values_list_passes(self):
        result = run_check("anything", {"type": "contains_all", "values": []})
        assert result.passed is True


class TestContainsAny:
    def test_pass_one_match(self):
        result = run_check("Call our phone line.", {"type": "contains_any", "values": ["email", "phone"]})
        assert result.passed is True

    def test_fail_no_match(self):
        result = run_check("Visit our office.", {"type": "contains_any", "values": ["email", "phone"]})
        assert result.passed is False


class TestRegexMatch:
    def test_pass(self):
        result = run_check("The rate is 5.00% p.a.", {"type": "regex_match", "pattern": r"\d+\.\d{2}%"})
        assert result.passed is True

    def test_fail(self):
        result = run_check("No numbers here.", {"type": "regex_match", "pattern": r"\d+\.\d{2}%"})
        assert result.passed is False

    def test_malformed_pattern_raises(self):
        with pytest.raises(CheckConfigError):
            run_check("text", {"type": "regex_match", "pattern": "[unclosed"})


class TestRegexExtractMatch:
    def test_pass_exact_extract(self):
        result = run_check(
            "The rate is 5.00% p.a. standard variable.",
            {"type": "regex_extract_match", "pattern": r"(\d+\.\d{2})%\s*p\.a\.", "expected": "5.00"},
        )
        assert result.passed is True
        assert result.extra["extracted"] == "5.00"

    def test_fail_value_mismatch(self):
        result = run_check(
            "The rate is 15.00% p.a. standard variable.",
            {"type": "regex_extract_match", "pattern": r"(\d+\.\d{2})%\s*p\.a\.", "expected": "5.00"},
        )
        assert result.passed is False
        assert result.extra["extracted"] == "15.00"
        assert "MISMATCH" in result.detail

    def test_fail_no_match_at_all(self):
        result = run_check(
            "No rate mentioned here.",
            {"type": "regex_extract_match", "pattern": r"(\d+\.\d{2})%\s*p\.a\.", "expected": "5.00"},
        )
        assert result.passed is False
        assert result.extra["extracted"] is None

    def test_missing_capture_group_raises(self):
        with pytest.raises(CheckConfigError):
            run_check(
                "5.00% p.a.",
                {"type": "regex_extract_match", "pattern": r"\d+\.\d{2}%", "expected": "5.00"},
            )

    def test_malformed_pattern_raises(self):
        with pytest.raises(CheckConfigError):
            run_check(
                "text",
                {"type": "regex_extract_match", "pattern": "(unclosed", "expected": "x"},
            )


class TestMinLength:
    def test_pass(self):
        result = run_check("a" * 60, {"type": "min_length", "value": 50})
        assert result.passed is True

    def test_fail(self):
        result = run_check("short", {"type": "min_length", "value": 50})
        assert result.passed is False

    def test_exact_boundary_passes(self):
        result = run_check("a" * 50, {"type": "min_length", "value": 50})
        assert result.passed is True

    def test_empty_response(self):
        result = run_check("", {"type": "min_length", "value": 1})
        assert result.passed is False


class TestMaxLength:
    def test_pass(self):
        result = run_check("short", {"type": "max_length", "value": 500})
        assert result.passed is True

    def test_fail(self):
        result = run_check("a" * 501, {"type": "max_length", "value": 500})
        assert result.passed is False

    def test_exact_boundary_passes(self):
        result = run_check("a" * 500, {"type": "max_length", "value": 500})
        assert result.passed is True

    def test_empty_response_passes(self):
        result = run_check("", {"type": "max_length", "value": 0})
        assert result.passed is True


class TestFormatIsJson:
    def test_pass_object(self):
        result = run_check('{"status": "ok", "code": 200}', {"type": "format_is_json"})
        assert result.passed is True

    def test_pass_array(self):
        result = run_check("[1, 2, 3]", {"type": "format_is_json"})
        assert result.passed is True

    def test_fail_malformed(self):
        result = run_check('{"status": "ok",}', {"type": "format_is_json"})
        assert result.passed is False

    def test_fail_empty_string(self):
        result = run_check("", {"type": "format_is_json"})
        assert result.passed is False

    def test_fail_plain_text(self):
        result = run_check("this is not json", {"type": "format_is_json"})
        assert result.passed is False


class TestExactMatch:
    def test_pass(self):
        result = run_check("exact text", {"type": "exact_match", "value": "exact text"})
        assert result.passed is True

    def test_fail_whitespace_sensitive(self):
        result = run_check("exact text ", {"type": "exact_match", "value": "exact text"})
        assert result.passed is False

    def test_fail_case_sensitive(self):
        result = run_check("Exact Text", {"type": "exact_match", "value": "exact text"})
        assert result.passed is False

    def test_empty_vs_empty(self):
        result = run_check("", {"type": "exact_match", "value": ""})
        assert result.passed is True


class TestUnknownCheckType:
    def test_raises(self):
        with pytest.raises(CheckConfigError):
            run_check("text", {"type": "semantic_similarity"})
