# NextTransformer Gym for Harbor

A repository-local implementation of the [Gym design](../docs/design/gym-design-interface.md).
Harbor is a pinned dependency; its source and the existing NextTransformer scientific
code are unchanged. Read [CONTRACT.md](CONTRACT.md) for the supported source language,
accounting rules, trust boundaries and production release gates.
See [VALIDATION.md](VALIDATION.md) for executed checks and their limits.

## Layout

```
gym/
  src/nexttransformer_gym/       API, client, ledger, workers, Harbor adapter
  tasks/architecture-research/
    task.toml                   Harbor 0.24.0 / task schema 1.4
    instruction.md
    environment/                main Agent container + Gym service
    tests/                      independent verifier container
    solution/                   bounded two-candidate integration oracle
  examples/                     source package builder and CPU fixture profiles
  tests/                        behavioral, isolation and scientific parity checks
```

The shipped task runs genuine NextTransformer model updates on a small synthetic
fixture. It is useful for verifying the complete research/submission lifecycle;
it does **not** produce official benchmark results. Research BPB is labelled dev
feedback. Verifier reward means submission validity only. Formal BPB, CORE and
leaderboard rank are never filled with fixture scores.

## Install and test

From the repository root:

```sh
uv sync --project gym --extra worker
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
```

The client itself has no training imports. The optional `worker` dependency group
installs the CPU numerical runtime for these bounded checks. Production GPU images
need their own pinned, reviewed dependencies and compatibility evidence.

## Native Harbor trial

Build each target from the repository root. Pin `BASE_IMAGE` to a reviewed digest
when preparing a cohort; the run helper resolves all three built images to immutable
local image IDs and never implicitly pulls them.

```sh
docker build -f gym/Dockerfile --target service -t nt-gym-service:local .
docker build -f gym/Dockerfile --target client -t nt-gym-client:local .
docker build -f gym/Dockerfile --target verifier -t nt-gym-verifier:local .

uv run --project gym --extra worker python gym/run_trial.py \
  --research-flops-limit 1000000000000 \
  --jobs-dir .agent-work/harbor-gym \
  --client-image nt-gym-client:local \
  --service-image nt-gym-service:local \
  --verifier-image nt-gym-verifier:local
```

The numerical cap above is an explicit **fixture example**, not a production default.
Omitting the allocation refuses launch. The oracle submits two FFN candidates,
trains them through the API, continues one checkpoint, validates the final choice,
and writes a receipt. Harbor collects actual main-container runtime state and
signed Gym evidence, then starts the separate verifier.

The run helper creates uniquely labelled state/verifier volumes and records their
names alongside Harbor job output. Those volumes intentionally survive task cleanup,
so frozen submissions and separately dispatched final runs survive the Agent.
The helper also retains `nexttransformer-gym/<trial>:client`, `:service`, and
`:verifier` image tags, while execution uses their immutable IDs. Retain these
images with the trial state. Delete only that trial's volumes and image tags when
their retention ends.
No host listener is published by Compose; its private service port is not a host
port allocation. The optional host HTTP server binds port 0 once, retains the
socket, persists its chosen port and fails explicitly if that port is occupied.

For a mediated LLM researcher, select
`--agent nexttransformer_gym.harbor_agent:GymResearchAgent` and explicitly configure
`GYM_MODEL_ENDPOINT` (fixed HTTPS chat-completion endpoint), `GYM_MODEL_NAME`, and
`GYM_MODEL_API_KEY` on the controller. Provider credentials stay outside the
container. The Agent receives only JSON research actions, not a shell tool.
Model calls are not part of the default oracle/test run; inference cost metering
remains deferred. Generic Harbor agents can use the same API but do not inherit
this restricted tool policy automatically.

## Client flow

Inside the Agent container:

```python
import json
from pathlib import Path
from nexttransformer_gym import GymClient

binding = json.loads(Path('/run/gym-auth/credential.json').read_text())
gym = GymClient(binding['endpoint'], binding['token'])
space = gym.describe_space()
candidate = gym.submit_candidate(package_bytes=package, idempotency_key='idea-1')
quote = gym.quote_experiment(candidate_id=candidate['id'],
                             profile_id='research-fixture', target_update=2)
job = gym.run_experiment(quote_id=quote['id'], idempotency_key='experiment-1')
result = gym.wait_result(job['id'])
```

Complete example: `tasks/architecture-research/solution/research.py`. It does not
import platform training code. Use `HarborRuntime(environment).snapshot()` from a
controller holding Harbor's real environment object to inspect the running Agent;
post-run snapshots are also retained as `agent-runtime.json` artifacts.

## Final training and recovery

Finalization and Harbor verification never launch training. Trusted controller
commands live in `nt-gym`, with required `--state`, `--profiles`, and `--platform`:

- `prepare-final`: create a separate staging Git snapshot, lower the selected
  source, re-audit its unchanged configuration and write an admitted pipeline plan.
  It performs no numerical training and leaves the source repository untouched.
- `launch-final`: verify frozen identities, allocate a separate finite budget and
  queue fresh initialization. Agent/verifier HTTP credentials cannot call it.
- `work-once`: dispatch one queued operation through the trusted scheduler.
- `get-final`: read final status and audited result, if complete.
- `reconcile --worker-confirmed-stopped`: conservatively settle a lost worker's
  uncertain reservation. Confirm termination first; unknown work is not refunded.

Profiles and official pipeline plans are operator-owned local configuration.
The official path requires actual resources and compatibility evidence and invokes
native formal training, audit and export in the isolated staging snapshot. Publication
into the upstream leaderboard is a separate process. No full training run is
performed by the tests or by submission acceptance.

Prepare the official plan before freezing a final profile: place the returned plan
under `official_plan`, explicitly set `cuda_visible_devices`, and establish
`official_compatibility_verified` from deployment parity checks. The single-node
launcher requires 8, 16 or 32 distinct allocated GPUs, matching the native protocol;
it uses file rendezvous, fresh initialization and the unchanged formal schedule.
Each rank has a finite share of the separately authorized allocation. A successful
run conservatively consumes that allocation; only the native audited result is
reported as a scientific measurement. Full GPU execution remains a deployment check.

The initial resource pool has one worker slot. Candidate and checkpoint quotas are
checked before dispatch; checkpoints remain pinned until explicit retention cleanup.
The code does not yet automate multi-node provisioning or checkpoint garbage
collection. Production image/hardware, offline-policy and full-run acceptance must
be validated for the selected deployment; bounded CPU fixtures do not establish them.
