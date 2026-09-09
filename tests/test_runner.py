"""Tests for idem.runner: the shared call-and-check loop used by both
the CLI and the GUI.
"""

from idem.checks import CheckConfigError
from idem.config import Config, ModelConfig, Question
from idem.runner import run_questions
from idem.target import TargetError


class FakeTarget:
    def __init__(self, responses: dict[str, str], raise_for: set[str] | None = None):
        self.responses = responses
        self.raise_for = raise_for or set()

    def call(self, prompt: str) -> str:
        if prompt in self.raise_for:
            raise TargetError(f"simulated failure for prompt: {prompt}")
        return self.responses[prompt]


def make_config(questions: list[Question]) -> Config:
    model = ModelConfig(provider="openai", model_id="gpt-4o-2024-08-06")
    return Config(model=model, questions=questions)


class TestRunQuestions:
    def test_all_pass(self):
        config = make_config(
            [Question(id="q1", prompt="rate?", checks=[{"type": "contains", "value": "5.00%"}])]
        )
        target = FakeTarget({"rate?": "The rate is 5.00%."})

        results, had_runtime_error = run_questions(config, target)

        assert had_runtime_error is False
        assert len(results) == 1
        assert results[0].overall_passed is True
        assert results[0].raw_response == "The rate is 5.00%."

    def test_check_failure_is_not_a_runtime_error(self):
        config = make_config(
            [Question(id="q1", prompt="rate?", checks=[{"type": "contains", "value": "5.00%"}])]
        )
        target = FakeTarget({"rate?": "The rate is 9.25%."})

        results, had_runtime_error = run_questions(config, target)

        assert had_runtime_error is False
        assert results[0].overall_passed is False

    def test_target_error_marks_runtime_error(self):
        config = make_config(
            [Question(id="q1", prompt="rate?", checks=[{"type": "contains", "value": "5.00%"}])]
        )
        target = FakeTarget({}, raise_for={"rate?"})

        results, had_runtime_error = run_questions(config, target)

        assert had_runtime_error is True
        assert results[0].overall_passed is False
        assert results[0].error is not None
        assert results[0].check_results == []

    def test_invalid_check_config_marks_runtime_error(self):
        config = make_config(
            [Question(id="q1", prompt="rate?", checks=[{"type": "not_a_real_check_type"}])]
        )
        target = FakeTarget({"rate?": "some response"})

        results, had_runtime_error = run_questions(config, target)

        assert had_runtime_error is True
        assert results[0].error is not None
        assert results[0].raw_response == "some response"

    def test_on_result_called_once_per_question(self):
        config = make_config(
            [
                Question(id="q1", prompt="p1", checks=[{"type": "contains", "value": "a"}]),
                Question(id="q2", prompt="p2", checks=[{"type": "contains", "value": "b"}]),
            ]
        )
        target = FakeTarget({"p1": "a", "p2": "b"})
        seen_ids = []

        run_questions(config, target, on_result=lambda r: seen_ids.append(r.question_id))

        assert seen_ids == ["q1", "q2"]

    def test_on_result_called_for_errored_questions_too(self):
        config = make_config(
            [Question(id="q1", prompt="rate?", checks=[{"type": "contains", "value": "5.00%"}])]
        )
        target = FakeTarget({}, raise_for={"rate?"})
        seen = []

        run_questions(config, target, on_result=lambda r: seen.append(r))

        assert len(seen) == 1
        assert seen[0].error is not None

    def test_mixed_pass_and_fail_across_questions(self):
        config = make_config(
            [
                Question(id="q1", prompt="p1", checks=[{"type": "contains", "value": "correct"}]),
                Question(id="q2", prompt="p2", checks=[{"type": "contains", "value": "correct"}]),
            ]
        )
        target = FakeTarget({"p1": "correct answer", "p2": "wrong answer"})

        results, had_runtime_error = run_questions(config, target)

        assert had_runtime_error is False
        assert results[0].overall_passed is True
        assert results[1].overall_passed is False
