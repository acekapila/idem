"""Shared "call the target for every question and run its checks" loop.

Extracted out of the CLI so the GUI (idem/gui/app.py) can drive the exact
same run logic instead of re-implementing it — there is only one place
that decides what counts as a pass, a fail, or a runtime error.
"""

from __future__ import annotations

from typing import Callable

from idem.audit_log import QuestionRunResult
from idem.checks import CheckConfigError, run_check
from idem.config import Config
from idem.target import ModelTarget, TargetError

# Called once per question, right after its result is computed, so a
# caller can report progress incrementally instead of waiting for the
# whole run to finish.
ProgressCallback = Callable[[QuestionRunResult], None]


def run_questions(
    config: Config,
    target: ModelTarget,
    on_result: ProgressCallback | None = None,
) -> tuple[list[QuestionRunResult], bool]:
    """Call `target` for every question in `config` and run its checks.

    Returns (results, had_runtime_error) — had_runtime_error is True if
    any question failed to call the model or hit an invalid check
    (distinct from a question whose checks simply failed).
    """
    results: list[QuestionRunResult] = []
    had_runtime_error = False

    for question in config.questions:
        try:
            raw_response = target.call(question.prompt)
        except TargetError as exc:
            had_runtime_error = True
            result = QuestionRunResult(
                question_id=question.id,
                model_id=config.model.model_id,
                provider=config.model.provider,
                prompt=question.prompt,
                raw_response="",
                check_results=[],
                check_configs=question.checks,
                error=str(exc),
            )
            results.append(result)
            if on_result:
                on_result(result)
            continue

        try:
            check_results = [run_check(raw_response, check) for check in question.checks]
        except CheckConfigError as exc:
            # Should be unreachable if `idem validate` passed on this
            # config, but treated as a hard runtime error rather than a
            # check failure.
            had_runtime_error = True
            result = QuestionRunResult(
                question_id=question.id,
                model_id=config.model.model_id,
                provider=config.model.provider,
                prompt=question.prompt,
                raw_response=raw_response,
                check_results=[],
                check_configs=question.checks,
                error=str(exc),
            )
            results.append(result)
            if on_result:
                on_result(result)
            continue

        result = QuestionRunResult(
            question_id=question.id,
            model_id=config.model.model_id,
            provider=config.model.provider,
            prompt=question.prompt,
            raw_response=raw_response,
            check_results=check_results,
            check_configs=question.checks,
        )
        results.append(result)
        if on_result:
            on_result(result)

    return results, had_runtime_error
