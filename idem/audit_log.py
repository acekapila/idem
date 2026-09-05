"""Writers for the two run artifacts: the append-only JSONL audit log
and the human-readable Markdown summary.

The JSONL log is the source of truth for auditors and tooling — one
line per question per run, never rewritten. The Markdown report is a
convenience view generated fresh each run, meant to be skimmed by a
non-technical reviewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from idem.checks import CheckResult


@dataclass
class QuestionRunResult:
    question_id: str
    model_id: str
    provider: str
    prompt: str
    raw_response: str
    check_results: list[CheckResult]
    check_configs: list[dict[str, Any]]
    timestamp: str | None = None
    error: str | None = None

    @property
    def overall_passed(self) -> bool:
        if self.error is not None:
            return False
        return all(r.passed for r in self.check_results)

    def to_record(self) -> dict[str, Any]:
        ts = self.timestamp or utc_now_iso()
        record: dict[str, Any] = {
            "timestamp": ts,
            "question_id": self.question_id,
            "model_id": self.model_id,
            "provider": self.provider,
        }
        if self.error is not None:
            record["error"] = self.error
            record["overall_passed"] = False
            return record

        record["checks"] = [
            result.to_dict(config)
            for result, config in zip(self.check_results, self.check_configs)
        ]
        record["overall_passed"] = self.overall_passed
        record["raw_response"] = self.raw_response
        return record


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record as a single line to the audit log file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown_report(path: str | Path, results: list[QuestionRunResult]) -> None:
    """Write (overwrite) a Markdown summary of this run's results."""
    total = len(results)
    passed = sum(1 for r in results if r.overall_passed)
    failed = total - passed

    lines: list[str] = []
    lines.append("# Idem Drift Check Report")
    lines.append("")
    lines.append(f"- Generated: {utc_now_iso()}")
    if results:
        lines.append(f"- Model: `{results[0].provider}` / `{results[0].model_id}`")
    lines.append(f"- Questions checked: {total}")
    lines.append(f"- Passed: {passed}")
    lines.append(f"- Failed: {failed}")
    lines.append("")

    status = "✅ ALL CHECKS PASSED" if failed == 0 else f"❌ {failed} QUESTION(S) FAILED"
    lines.append(f"## {status}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Question ID | Result |")
    lines.append("|---|---|")
    for r in results:
        mark = "✅ pass" if r.overall_passed else "❌ FAIL"
        lines.append(f"| `{r.question_id}` | {mark} |")
    lines.append("")

    failing = [r for r in results if not r.overall_passed]
    if failing:
        lines.append("## Failed Checks (detail)")
        lines.append("")
        for r in failing:
            lines.append(f"### `{r.question_id}`")
            lines.append("")
            if r.error is not None:
                lines.append(f"- **Error:** {r.error}")
                lines.append("")
                continue
            lines.append("| Check type | Passed | Detail |")
            lines.append("|---|---|---|")
            for result in r.check_results:
                mark = "✅" if result.passed else "❌"
                detail = result.detail.replace("|", "\\|")
                lines.append(f"| `{result.check_type}` | {mark} | {detail} |")
            lines.append("")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
