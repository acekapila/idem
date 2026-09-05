"""Process exit codes used by the `idem` CLI.

The distinction between FAIL and ERROR matters for CI/CD: a FAIL means
idem ran successfully and found a real drift (or check failure) — the
thing you're watching for. An ERROR means idem itself could not
complete the run (bad config, network failure, etc.) and the result
should not be trusted either way.
"""

OK = 0
FAIL = 1
ERROR = 2
