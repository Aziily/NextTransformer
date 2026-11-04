# Architecture research

Design and compare native NextTransformer model plugins through the Gym API. This task's shipped
profile is a bounded synthetic integration fixture, not an official benchmark run.

Read `/run/gym-auth/credential.json` for your session-scoped endpoint and token.
Import `GymClient` from `nexttransformer_gym`; call `describe_space`, `get_resources`,
and `list_components` before starting. Candidate packages must follow the
`proposal.json`, `architecture.json`, `plugin/<method>/method.json` and
`plugin/<method>/__init__.py` contract. Use `/opt/gym/make_candidate.py` as an example.

Use the existing NextTransformer plugin structure: native `method.json`,
`__init__.py`, and optional local Python modules. Define native `model`, `mixer`,
`ffn` or `block` hooks; `model` subclasses `ReferenceLM`. `architecture.json`
contains native `ModelConfig` fields and declared plugin `options`. Read the
`plugin-contract` and `reference-model` references for exact interfaces. Gym does
not parse or translate model expressions. The same source is loaded unchanged
by research and final workers. The platform supplies optimizer, data and scoring.
Do not install or execute candidate code in the Agent workspace.

Submit source bytes, inspect costs, run charged checks and bounded experiments.
Resume only from checkpoints returned by this service. Record your hypothesis and
select one candidate. Validate it under `final-fixture` before finalizing.

Write `/workspace/submission.json` containing exactly your `session_id`, the
server-returned `submission_id`, and `candidate_hash`. The separate verifier checks
this receipt against authenticated server evidence. Acceptance means valid submission,
not model performance. It never launches final training. Do not write reward files.
