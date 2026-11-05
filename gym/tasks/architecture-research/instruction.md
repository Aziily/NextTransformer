# Architecture research

Design and compare native model plugins through the Gym API. This task's shipped
profile is a bounded synthetic integration fixture, not an official benchmark run.

Your deliverable is a native ModelConfig and its plugin source. Gym owns the training profiles.
Do not submit machine settings or training commands.

Read `/run/gym-auth/credential.json` for your session-scoped endpoint and token.
Use the preconfigured `gym` MCP tools or import `GymClient` from
`nexttransformer_gym`. Both use the same HTTP API and session budget.
Call `describe_space`, `get_resources`,
and `list_components` before starting. Candidate packages must follow the
`proposal.json`, `architecture.json`, `plugin/<method>/method.json` and
`plugin/<method>/__init__.py` contract. Use `/opt/gym/make_candidate.py` as an example.

Read `describe_space.candidate_contract` for the native ModelConfig JSON schema,
defaults and merge order, and each profile's `model_config_constraints` for fixed
values. You may submit new configs and new native plugin source. For config-only
changes, re-upload unchanged source with a new `architecture.json`, or use
`calibrate_candidate.grid`: each item replaces the entire architecture object,
not a patch. Calibration is static cost analysis, not training. Pass target steps
and GPU placement to `quote_experiment`, never inside the model configuration.

Use `list_experiments(scope="published")` and `read_experiment` to inspect past
experiments, their results and model configurations. Follow `source_reference_ids`
with `read_reference` for architecture code. Gym exposes published historical
results and current reference files. Check provenance before you treat a reference
as the historical source. `scope="session"` shows your own research
jobs, results and candidate source. You do not need a repository checkout or Git history.

`get_resources` reports the actual local execution node, GPU inventory, configured
allocation, memory allowance and current worker/card availability. On
`quote_experiment`, you may choose an allocated `gpu_id`, the local `server_id` and
`target_update` (absolute step, not additional steps). Omit placement for automatic
selection. A busy GPU keeps the job queued. Use `get_job` to read its waiting reason.

You may use a script to call Gym. Gym rejects arbitrary training commands,
trainer/data/environment overrides and remote shell jobs.
Use `get_budget` and `quote_experiment`: the finite allocation covers all of this
session's Gym numerical work, including checks. CORE-22 testing for your candidates
is unavailable during research. You may use internet research from this container.

Use the native plugin structure: `method.json`,
`__init__.py`, and optional local Python modules. Define native `model`, `mixer`,
`ffn` or `block` hooks. A `model` hook must subclass `ReferenceLM`. `architecture.json`
contains native `ModelConfig` fields and declared plugin `options`. Read the
`plugin-contract` and `reference-model` references for exact interfaces.

Gym does
not parse or translate model expressions. Research and final workers load the
same source without changes.

The platform supplies optimizer, data and scoring.
Do not install or execute candidate code in the Agent workspace.

1. Submit source bytes.
2. Inspect costs.
3. Run charged checks.
4. Run bounded experiments.
5. Resume only from checkpoints returned by this service.
6. Record your hypothesis.
7. Select one candidate.
8. Select the final profile provided by Gym.
9. Call `validate_candidate` with that `profile_id` before `finalize`.

Wait for validation to finish. Check that all checks pass and `admitted` is true.

Write `/workspace/submission.json` containing exactly your `session_id`, the
server-returned `submission_id`, and `candidate_hash`. The separate verifier checks
this receipt against authenticated server evidence. Acceptance means valid submission,
not model performance. The verifier never starts final training. Do not write reward files.
