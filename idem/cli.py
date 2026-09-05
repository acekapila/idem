"""The `idem` command-line interface.

Three subcommands:

- `idem run`      — call the pinned model and check its responses.
- `idem validate` — validate a golden-set YAML file without calling any API.
- `idem init`      — scaffold an example golden-set YAML file.

Exit codes (see idem.exit_codes): 0 = all checks passed, 1 = at least
one check failed, 2 = a configuration or runtime error prevented the
run from producing a trustworthy result.
"""

from __future__ import annotations

import argparse
import importlib.resources
import sys
from pathlib import Path

from idem import exit_codes
from idem.audit_log import QuestionRunResult, append_jsonl, write_markdown_report
from idem.checks import CheckConfigError, run_check
from idem.config import ConfigError, load_config, load_raw_yaml, validate_raw_config
from idem.target import TargetError, create_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idem",
        description="Detect when an LLM's response to a fixed set of test "
        "questions has drifted from an approved baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the golden set against the target model")
    run_parser.add_argument("--config", required=True, help="Path to the golden-set YAML file")
    run_parser.add_argument(
        "--output-dir",
        default="./reports",
        help="Directory to write audit_log.jsonl and report.md into (default: ./reports)",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a golden-set YAML file without calling any API"
    )
    validate_parser.add_argument("--config", required=True, help="Path to the golden-set YAML file")

    init_parser = subparsers.add_parser("init", help="Scaffold an example golden_set.yaml")
    init_parser.add_argument(
        "--output",
        default="golden_set.yaml",
        help="Path to write the example config to (default: ./golden_set.yaml)",
    )

    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        raw = load_raw_yaml(args.config)
    except ConfigError as exc:
        for error in exc.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return exit_codes.ERROR

    errors = validate_raw_config(raw)
    if errors:
        print(f"Config is INVALID: {len(errors)} problem(s) found in {args.config}\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return exit_codes.ERROR

    print(f"Config is valid: {args.config}")
    return exit_codes.OK


def cmd_init(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    if output_path.exists():
        print(f"ERROR: {output_path} already exists; refusing to overwrite", file=sys.stderr)
        return exit_codes.ERROR

    template = importlib.resources.files("idem.templates").joinpath("golden_set.yaml").read_text()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template)
    print(f"Wrote example golden set to {output_path}")
    return exit_codes.OK


def cmd_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Config is INVALID: {len(exc.errors)} problem(s) found in {args.config}\n", file=sys.stderr)
        for error in exc.errors:
            print(f"  - {error}", file=sys.stderr)
        return exit_codes.ERROR

    try:
        target = create_target(config.model)
    except TargetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exit_codes.ERROR

    output_dir = Path(args.output_dir)
    jsonl_path = output_dir / "audit_log.jsonl"
    report_path = output_dir / "report.md"

    results: list[QuestionRunResult] = []
    had_runtime_error = False

    for question in config.questions:
        try:
            raw_response = target.call(question.prompt)
        except TargetError as exc:
            print(f"ERROR: question '{question.id}' failed to call model: {exc}", file=sys.stderr)
            had_runtime_error = True
            results.append(
                QuestionRunResult(
                    question_id=question.id,
                    model_id=config.model.model_id,
                    provider=config.model.provider,
                    prompt=question.prompt,
                    raw_response="",
                    check_results=[],
                    check_configs=question.checks,
                    error=str(exc),
                )
            )
            continue

        try:
            check_results = [run_check(raw_response, check) for check in question.checks]
        except CheckConfigError as exc:
            # Should be unreachable if `idem validate` passed on this file,
            # but treated as a hard runtime error rather than a check failure.
            print(f"ERROR: question '{question.id}' has an invalid check: {exc}", file=sys.stderr)
            had_runtime_error = True
            results.append(
                QuestionRunResult(
                    question_id=question.id,
                    model_id=config.model.model_id,
                    provider=config.model.provider,
                    prompt=question.prompt,
                    raw_response=raw_response,
                    check_results=[],
                    check_configs=question.checks,
                    error=str(exc),
                )
            )
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

        status = "PASS" if result.overall_passed else "FAIL"
        print(f"[{status}] {question.id}")

    for result in results:
        append_jsonl(jsonl_path, result.to_record())
    write_markdown_report(report_path, results)

    print(f"\nAudit log: {jsonl_path}")
    print(f"Report:    {report_path}")

    if had_runtime_error:
        return exit_codes.ERROR
    if not all(r.overall_passed for r in results):
        return exit_codes.FAIL
    return exit_codes.OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "init":
        return cmd_init(args)

    parser.print_help()
    return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
