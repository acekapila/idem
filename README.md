# Idem

**Idem** (Latin: "the same") detects when an LLM's response to a fixed set of
test questions has changed from an approved baseline.

## The problem

LLM APIs are usually called by a stable name — a model ID, an endpoint, an
alias like "our production model." But the thing behind that name is not
guaranteed to stay the same over time. Vendors retrain, quietly swap in a
new checkpoint, or route a legacy name to a newer model, and the API
contract (same URL, same model string) gives you no signal that anything
changed underneath it.

If your product relies on an LLM to state a specific interest rate, follow
a specific disclosure format, or reproduce a specific piece of policy
wording — and that fact silently changes because the vendor updated the
model — you find out from a customer complaint or a regulator, not from
your monitoring.

Existing tooling doesn't cover this gap:

- **Adversarial security testing** (PyRIT, Garak) probes for jailbreaks,
  prompt injection, and unsafe outputs. It's not designed to ask "did this
  specific fact still hold."
- **General-purpose LLM observability** (Arize, Galileo, and similar)
  tracks quality, cost, and drift at a broad, statistical level. It's built
  for product teams optimizing behavior, not for producing a yes/no
  answer an auditor can defend.

Idem is narrower on purpose: a fixed golden set of prompts, each with
explicit, deterministic checks, run against a pinned model version, logged
to an append-only audit trail. Every result traces back to a rule a human
wrote and can read — "the response no longer contains the string
`5.00%`" — not a similarity score or a second model's judgment call.

This is v1 of a three-part system, and it is intentionally the simplest,
fully rule-based layer. See [What this tool does NOT do](#what-this-tool-does-not-do)
below.

## Install

```bash
pip install idem-check
```

The `openai` and `anthropic` SDKs are optional, installed based on which
provider(s) you use:

```bash
pip install "idem-check[openai]"
pip install "idem-check[anthropic]"
pip install "idem-check[all]"       # both
```

The installed command is `idem`.

## Quickstart

Scaffold an example golden set:

```bash
idem init
```

This writes `golden_set.yaml` with a few worked examples. Edit it to match
facts, policies, or formats you want to protect, then validate the file
without calling any API:

```bash
idem validate --config golden_set.yaml
```

Set your provider's API key and run the golden set against the live model:

```bash
export OPENAI_API_KEY=sk-...
idem run --config golden_set.yaml --output-dir ./reports
```

This writes two files to `./reports`:

- `audit_log.jsonl` — append-only, one JSON line per question per run.
  This is the durable record; every run adds to it, nothing is ever
  rewritten.
- `report.md` — a human-readable summary of the latest run: pass/fail
  counts and a table of any failed checks with their explanations,
  meant to be read by a non-technical reviewer.

Exit codes are distinct so this works cleanly in CI/CD or a scheduled job:

| Code | Meaning |
|---|---|
| `0` | Every check passed. |
| `1` | At least one check failed — this is the drift signal. |
| `2` | A configuration or runtime error prevented a trustworthy result (bad YAML, missing API key, network failure). Distinct from `1` on purpose: a `2` means "we don't actually know," not "it failed." |

## The golden-set format

A golden set is a YAML file with a pinned model and a list of questions,
each with one or more checks against the response:

```yaml
model:
  provider: openai              # openai | anthropic | custom
  model_id: gpt-4o-2024-08-06   # must be a dated/pinned version string

questions:
  - id: interest_rate_disclosure
    prompt: "What is the current standard variable interest rate disclosure wording?"
    checks:
      - type: contains
        value: "5.00%"
      - type: regex_extract_match
        pattern: '(\d+\.\d{2})%\s*p\.a\.'
        expected: "5.00"
      - type: max_length
        value: 500
```

`model_id` must be a dated/pinned version string, not a floating alias.
`idem validate` and `idem run` both reject aliases like `gpt-4o` or
`claude-latest` — if the alias itself is what silently changes, there is
nothing stable left to compare against.

See [`examples/golden_set.yaml`](examples/golden_set.yaml) for a complete,
commented example.

## Testing a custom or self-hosted model

Idem isn't limited to the public OpenAI/Anthropic APIs — it can also point
at a model you trained and host yourself (e.g. a fine-tuned customer-support
model behind an internal API). Which setup you use depends on how your
model is served:

**If your server speaks the OpenAI chat-completions protocol** — true for
vLLM, Ollama, Text Generation Inference, LM Studio, and most self-hosting
frameworks — point the `openai` provider at it with `base_url`:

```yaml
model:
  provider: openai
  model_id: support-bot-2024-09-01     # the name/tag your server expects
  base_url: "http://localhost:8000/v1"
```

`OPENAI_API_KEY` is not required when `base_url` is set, since most
self-hosted servers don't check it — set one anyway if yours does.

**If it's a fully custom REST API** with its own request/response shape,
use the `custom` provider. You describe the whole HTTP call declaratively —
no code to write:

```yaml
model:
  provider: custom
  model_id: support-bot-2024-09-01   # your own internal name/version for this checkpoint
  endpoint:
    url: "https://internal-api.example.com/v1/generate"
    method: POST                                 # optional, default: POST
    headers:
      Authorization: "Bearer ${CUSTOM_API_KEY}"  # ${VAR} read from the environment at call time
    request_template:
      # Sent as the JSON body. "{{prompt}}" is replaced with the question's
      # prompt text wherever it appears, at any nesting depth.
      messages:
        - role: user
          content: "{{prompt}}"
    response_path: "choices.0.message.content"   # dotted path to the response text in the JSON reply
    timeout: 30                                  # optional, seconds, default: 30
```

`response_path` walks the parsed JSON response with dot-separated segments;
a numeric segment indexes into a list (`choices.0...`). If the model's
response comes back somewhere idem can't reach with a dotted path, or in a
format other than JSON, that's a sign the response needs to be normalized
before it reaches idem — this stays a thin, declarative HTTP client, not a
place for per-vendor parsing logic.

See [`examples/custom_endpoint_golden_set.yaml`](examples/custom_endpoint_golden_set.yaml)
for both options side by side.

## Testing an agent (base model + system prompt)

Most production "customer-facing AI agents" aren't fine-tuned models at
all — they're a general-purpose model plus a fixed system prompt that
encodes the persona, policies, and facts. Idem supports this directly with
`system_prompt` on the `openai` or `anthropic` provider:

```yaml
model:
  provider: openai
  model_id: gpt-4o-mini-2024-07-18
  system_prompt: |
    You are a customer service representative for Acme Bank. Answer
    questions using only these facts:
    - Standard variable interest rate: 5.00% p.a. (comparison rate 5.24% p.a.)
    - Complaints: email complaints@acmebank.example, phone 1800-000-000,
      or written complaint to PO Box 1, Sydney NSW 2000.

questions:
  - id: interest_rate_disclosure
    prompt: "What is the current standard variable interest rate?"
    checks:
      - type: contains
        value: "5.00%"
```

This is arguably the more realistic drift scenario to test for: the system
prompt in your YAML never changes, but if the model underneath
`gpt-4o-mini-2024-07-18` is ever silently retrained or swapped, its
adherence to that same fixed prompt can still shift — which is exactly
what idem is built to catch. (`system_prompt` isn't used with `provider:
custom` — build the system message directly into `endpoint.request_template`
there, since you already control the full request shape.)

## Check-type reference

Every check is a pure, independently-tested function:
`(response_text, check_config) -> CheckResult`. `CheckResult` carries a
`detail` string — this is the sentence an auditor actually reads, so it
always explains exactly why the check passed or failed.

| Type | Required fields | What it checks |
|---|---|---|
| `contains` | `value` | `value` is a substring of the response. |
| `not_contains` | `value` | `value` is **not** a substring of the response. |
| `contains_all` | `values` (list) | Every string in `values` is present. |
| `contains_any` | `values` (list) | At least one string in `values` is present. |
| `regex_match` | `pattern` | `pattern` matches somewhere in the response. |
| `regex_extract_match` | `pattern` (with a capture group), `expected` | Extracts the first capture group via `pattern` and compares it **exactly** to `expected`. This is the check for catching a specific fact drifting (e.g. "5% became 15%") while everything else about the response looks fine. |
| `min_length` / `max_length` | `value` | Response character count is at least / at most `value`. |
| `format_is_json` | — | Response is valid, parseable JSON. |
| `exact_match` | `value` | The full response equals `value` exactly (case- and whitespace-sensitive). Rare — use for highly deterministic, fully-scripted responses. |

`idem validate` catches (without calling any API): missing required
fields, unknown check types, malformed regex patterns, a missing or
unpinned `model_id`, and an unset `provider`.

## CLI reference

```
idem run --config golden_set.yaml --output-dir ./reports
idem validate --config golden_set.yaml
idem init [--output golden_set.yaml]
```

## What this tool does NOT do

This is a deliberate scope boundary, not a gap to apologize for:

- **No semantic or embedding comparison.** Idem does not measure how
  "similar" two responses are. Similarity scores are exactly the kind of
  fuzzy, hard-to-defend number this tool exists to avoid.
- **No LLM-as-judge.** Idem does not ask a second model to decide whether
  the first model's answer is acceptable. That introduces a second
  non-deterministic system into what's supposed to be a deterministic
  check.
- **No probabilistic scoring.** Every result is a boolean, derived from an
  explicit rule. There is no confidence interval, no threshold to tune, no
  "mostly passed."

If you need any of the above, that's a different (and valid) layer of the
problem — v1 is the rule-based floor underneath it, built to be the one
part of the system whose verdicts are 100% reproducible and don't require
trusting another model's judgment.

## Contributing

Issues and pull requests are welcome. Run the test suite with:

```bash
pip install -e ".[dev]"
pytest
```

Every check type in `idem/checks.py` is a pure function with no I/O — if
you add one, add unit tests covering its edge cases (empty response,
missing regex groups, malformed input) in `tests/test_checks.py`.

## License

MIT — see [LICENSE](LICENSE).
