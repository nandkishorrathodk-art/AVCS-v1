# Single-app live observation loop

`python -m avcs.single_app` connects the real desktop capture path to a
configured text-only System 2 bridge. It is an operator-mediated
observe/reason/re-observe loop, not an autonomous computer-use agent:

1. The operator supplies an app context and task.
2. AVCS captures a fresh desktop frame locally and creates a bounded text
   observation.
3. Before every remote turn, the operator reviews the displayed observation
   and explicitly consents to sending the task and text observation.
4. System 2 returns an advisory goal. AVCS does not parse that text as an
   action and does not dispatch mouse or keyboard input.
5. The operator may perform an approved step manually and request a fresh
   observation for the next turn.

The app context is operator-provided; this loop does not claim that pixel
perception identified the app. No synthetic fallback is allowed. If the real
desktop capture backend is unavailable, startup fails instead of silently
switching to synthetic frames.

## Run

The `TextOnlySystem2Client` is the client adapter; this repository does not
provide the remote System 2 bridge service. Configure the URL of a service
implementing `/v1/system2/goal`. An API key, if required, is read from
`AVCS_SYSTEM2_API_KEY`.

```powershell
python -m pip install -e ".[client,desktop]"
$env:AVCS_SYSTEM2_ENDPOINT = "https://your-system2-bridge.example"
$env:AVCS_SYSTEM2_API_KEY = "your-key"
python -m avcs.single_app --app "Notepad" --task "Describe the current state and suggest a safe next step"
```

Before every request, the command displays the task and serialized allowlisted
observation for review and asks for a separate yes/no confirmation.
The endpoint must implement the JSON contract: a `task` string and
allowlisted text-only `observation` object in, and a non-empty `goal` string
out. No provider-specific endpoint is included in this repository. Review
the task and observation before consenting; sensitive-text masking is
best-effort, not comprehensive.

## Current action limitation

The available grounding training report does not pass all gates, and
`GoalConditionedGrounder.predict()` currently returns
`action_authorized=False`. Therefore this loop deliberately never invokes
grounding for actuation or sends input. A checkpoint loading or a successful
text-only turn does not validate real-app target grounding. Enabling automated
actions requires a separately reviewed action-authorization mechanism and
held-out real-desktop evidence, followed by fresh action/postcondition tests.

Unit tests use stub controller/client objects to validate the consent and
resource-lifecycle mechanics. They do not claim to test live app recognition
or task completion. Run the command above only on a non-sensitive screen and
only after configuring a real text-only bridge.
