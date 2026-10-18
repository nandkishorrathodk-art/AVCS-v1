# AVCS Milestone 6 — Controlled Windows Pilot Runbook

## Purpose and status

This is a pilot harness and proposed operating procedure, not a claim that the live AVCS runtime has passed the pilot. The initial workflow cohort is intentionally low-risk and file-based, using Windows Notepad in a disposable test sandbox. It does not authorize mail sending, purchases, deletion of user files, credential access, elevated actions, or unattended control of a customer's live workspace.

## Pre-flight gates (operator must attest)

1. Use a dedicated Windows test account and disposable VM or resettable test machine.
2. Pin the Windows build, Notepad version, AVCS commit, SDK version, adapter version, model/provider versions and configuration.
3. Configure the AVCS adapter plugin. The harness intentionally refuses to run actual trials without an adapter.
4. Confirm a dedicated output directory has no user data. The harness creates a unique subdirectory per trial.
5. Keep human supervision enabled. Configure an operator kill switch and verify that cancellation stops further actions.
6. Disable high-risk actions. Use only synthetic text fixtures and the exact sandbox directory.
7. Decide whether any text, screenshots, diagnostics or prompts can leave the machine. For local-only mode, audit outbound traffic; a setting alone is not proof of zero egress.
8. Capture signed-off attestations in `config/pilot_release_attestations.template.json` (copy to a run-specific file before filling in).

## Scenarios

- `notepad_create_save_exact_text`: create a new text file and save it to a run-specific path; independently verify exact bytes/text.
- `notepad_append_single_line`: append a single sentinel line to a harness-created fixture; verify complete final content.
- `notepad_find_replace_token`: replace a unique synthetic token twice and preserve all other lines; verify exact final content.

The runner passes a `PublicTask` containing only task text, app name, timeout, action budget and risk level to the executor. Expected final content and evaluator configuration stay in the harness process and are not passed to `adapter.execute`.

## Dry run and execution

From the bundle root:

```powershell
python -m avcs_pilot.runner --dry-run --runs-per-workflow 100
```

Configure a repository-owned adapter factory implementing `async execute(PublicTask) -> AgentExecution | Mapping`. `examples/adapter_skeleton.py` intentionally fails closed until wired to the existing AVCS task planner; it does not guess at internal methods.

Pin versions in the run environment before starting:

```powershell
$env:AVCS_COMMIT = "<git-commit-sha>"
$env:AVCS_SDK_VERSION = "0.2.0"
$env:AVCS_ADAPTER_VERSION = "<adapter-version>"
$env:AVCS_WINDOWS_BUILD = "<winver-build>"
$env:AVCS_APP_VERSION = "<notepad-version>"
$env:AVCS_PILOT_ADAPTER = "your_package.avcs_pilot_adapter:create_adapter"
python -m avcs_pilot.runner --runs-per-workflow 100 --seed 20261009 --output-dir pilot-results
```

The harness records these version fields in each trial and the summary. A value of `UNSET` means the run's environment metadata is incomplete and must be corrected before interpreting results for a release decision.

The planned initial suite contains 300 runs total (100 per workflow). A run requires a dedicated, interactive Windows session. Do not treat a dry run or mocked adapter as pilot evidence.

## Go/no-go gates

Automated gates are necessary but not sufficient:

- At least 300 completed trial records, with all three workflows represented.
- At least 95% independently verified final-state completion overall. Inspect per-workflow rates; a strong aggregate must not hide a broken workflow.
- Zero false confirmations (agent status `CONFIRMED` while the independent verifier says the task failed).
- No unauthorized or out-of-sandbox action, no uncontrolled retry of ambiguous/non-idempotent actions, no unresolved critical/high security issue.
- p50/p95/p99 latency and all terminal statuses reviewed.
- Operator cancellation, rollback/reset, data retention and network egress audit pass.
- Sponsor approval and the manual attestation file are complete.

Statistical caution: 100 runs per workflow is a pilot screen, not a universal reliability claim. Report Wilson confidence intervals and do not pool distinct workflows without also showing per-workflow results.

## Evidence and privacy

Each run writes one JSONL record with task ID, workflow, environment, action status/count, duration, independent verifier outcome and a run directory. Avoid screenshots by default. Never log secrets, credentials, real customer content or unrestricted accessibility-tree text. Restrict result-directory ACLs and set an explicit retention/deletion date. The adapter's free-form diagnostics should contain no content copied from the desktop.

## Stop conditions

Stop the pilot immediately on: false confirmation for a consequential action; any out-of-sandbox write; unauthorized action; unexpected external network transmission; repeated verifier disagreement; unbounded resource growth; failed kill switch; or an unresolved high-severity security event. Preserve minimal diagnostic evidence, reset the sandbox and require human review before resuming.
