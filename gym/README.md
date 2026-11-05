# NextTransformer Gym

Architecture auto-research with Gym on the host and native Harbor Claude Code
and independent verifier containers. Implementation lives in `gym/`; Harbor and
the repository's shared scientific code remain unchanged.

- [Design and API](DESIGN.md): architecture, all 22 operations, resources, budgets,
  datasets, final training, recovery and supported limits.
- [Candidate inputs](CANDIDATE_API.md): exact native ModelConfig/source inputs,
  defaults, configuration-only changes and errors.
- [Testing](TESTING.md): maintained regression and integration checks.

Examples use small synthetic fixtures with real model updates. They do not produce
production ClimbMix/CORE-22 results. Verifier reward measures deliverable validity.

## Install

From the repository root:

```sh
uv sync --project gym --extra worker
docker build -f gym/Dockerfile --target client -t nt-gym-client:local .
docker build -f gym/Dockerfile --target verifier -t nt-gym-verifier:local .
```

The host needs Docker, bubblewrap and unprivileged user namespaces. The worker
extra installs a CPU runtime. GPU execution uses an existing CUDA-enabled Python
with the pinned worker dependencies; Gym needs no service container or CUDA image.
Pin Docker base images to reviewed digests for deployment. The launcher requires
prebuilt images and resolves their immutable IDs.

## Start research

```sh
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml --dry-run
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml
```

Configure authentication through Harbor/Claude Code's native environment. The
example specifies CLI version, model, Agent turn/time/USD limits and the whole
session's research FLOPs. Final training has a separate allocation. Inference,
web search and Agent shell CPU do not consume Gym's training ledger.

For DeepSeek Flash, use `gym/examples/deepseek-v4.1-flash.toml`, set
`GYM_RESEARCH_MODEL` to the provider's actual model ID, and configure
`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`. The endpoint must support Anthropic
Messages. Credentials are not written into Gym state. Strict billing limits need
provider-side enforcement when gateway pricing is unknown.

For GPUs, copy `gym/examples/resources.gpu.json` to an ignored local config, select
an allocated card index/UUID and set its memory cap. Pass
`--execution-config .agent-work/resources.local.json` and
`--worker-python /path/to/cuda-environment/bin/python` to the launcher. Omit GPU
allocation for CPU fixtures. Run one research trial at a time: each service state
has one numerical worker slot, rather than a machine-wide scheduler. Busy GPUs
queue jobs; Agent APIs cannot expand operator allocation.

The launcher provisions credentials, starts Gym and uses Harbor lifecycle hooks
to settle work and export signed evidence before verification. Retain host state,
execution policy and pinned source/Python environment for final training. Each
service persists its first OS-assigned port; restarts reuse it and fail on conflicts.

## Agent interface

SDK and MCP both use authenticated `POST /v1/call`. Harbor registers the stdio
`gym` MCP server automatically. External clients can run
`python -m nexttransformer_gym.mcp_server --credential <credential.json>`.
HTTP owns schemas, validation, authorization and budgeting.

```python
import json
from pathlib import Path
from nexttransformer_gym import GymClient

binding = json.loads(Path("/run/gym-auth/credential.json").read_text())
gym = GymClient(binding["endpoint"], binding["token"])
space = gym.describe_space()
resources = gym.get_resources()
# Build a native source ZIP using the example and discovered model config schema.
candidate = gym.submit_candidate(Path("candidate.zip").read_bytes(), "idea-1")
quote = gym.quote_experiment(
    candidate_id=candidate["id"], profile_id="research-fixture", target_update=2
)
job = gym.run_experiment(quote_id=quote["id"], idempotency_key="experiment-1")
result = gym.wait_result(job["id"])
```

Use [make_candidate.py](examples/make_candidate.py) for a native multi-file plugin,
and [research_with_placement.py](examples/research_with_placement.py) for a bounded
workflow. Research history/configuration/source references come through Gym;
the Agent needs no repository checkout or Git access.

Run `uv run --project gym python gym/smoke.py` with the prepared images to exercise
native Claude Code against a deterministic local model stub without external model
calls. Full regression and bounded GPU commands are in [TESTING.md](TESTING.md).
