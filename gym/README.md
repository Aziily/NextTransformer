# NextTransformer Gym for Harbor

A repository-local implementation of the [Gym design](../docs/design/gym-design-interface.md).
Harbor is a pinned dependency; its source and the existing NextTransformer scientific
code are unchanged. Read [CONTRACT.md](CONTRACT.md) for the supported source language,
accounting rules, trust boundaries and production release gates.
See [VALIDATION.md](VALIDATION.md) for executed checks and their limits.
See [ROADMAP.md](ROADMAP.md) for the revised auto-research target: native Harbor
Claude Code, research history through Gym and Agent internet access, three dataset splits, and a
submission ID backed by source/configuration retained in Gym. Its status table
separates planned changes from the current fixture described below.

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

For ClimbMix, the data adapter follows the native default: training shards precede
the last shard, and `shard_06542.parquet` supplies research validation. Existing
`dev_path`/`dev_bpb` names refer to that validation split. Final submission is a
receipt; Gym retains the full plugin source/configuration and resolves the
submission ID when dispatching final training.

## Install and test

From the repository root:

```sh
uv sync --project gym --extra worker
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
```

Native research workers require bubblewrap on the controller host (the service
image includes it), with unprivileged user namespaces enabled. The client itself has no training imports. The optional `worker` dependency group
installs the CPU numerical runtime for these bounded checks. Production GPU images
need their own pinned, reviewed dependencies and compatibility evidence.

## Native Harbor trial

Build each target from the repository root. Pin `BASE_IMAGE` and `NODE_IMAGE` to reviewed digests
when preparing a cohort; the run helper resolves all three built images to immutable
local image IDs and never implicitly pulls them.

```sh
docker build -f gym/Dockerfile --target service -t nt-gym-service:local .
docker build -f gym/Dockerfile --target client -t nt-gym-client:local .
docker build -f gym/Dockerfile --target verifier -t nt-gym-verifier:local .

uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml --dry-run
# Supply provider authentication through Harbor's native controller environment.
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml
```

The default Agent is Harbor's built-in **Claude Code**, pinned to CLI 2.1.312 in
the examples. Harbor installs and runs it; Gym has no Agent subclass, model broker
or reasoning loop. The Agent can use its shell and internet tools, while numerical
workers remain offline. No repository or Git checkout is mounted into the Agent.

Both examples set the **whole session's research FLOPs allocation**, plus Agent
turn/time limits and Claude Code's separate USD option. The example 10^12 FLOPs
cap is for the CPU fixture, not a production allocation. Numerical checks and all
research experiments share the same durable account. Final training has a separate
allocation; model inference and local shell CPU do not consume the Gym FLOPs ledger.
No retries or extra attempts silently create another research allocation.

For the requested DeepSeek V4.1 Flash setup, use
`gym/examples/deepseek-v4.1-flash.toml`. Set `GYM_RESEARCH_MODEL` to your provider's
actual model ID and configure `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` in the
controller environment. The endpoint must support **Anthropic Messages**; an
OpenAI-only chat-completions endpoint does not work directly with Claude Code.
The example does not assert that a particular provider offers that model ID.
Provider credentials are handled by native Harbor/Claude Code inside the Agent
container; they are not written into the trial binding receipt or Gym service.
For custom gateways, the CLI USD limit depends on its reported usage/pricing;
use provider-side limits when a strict billing cap is required.

Model-free smoke with the actual native Claude Code CLI and a local deterministic
Anthropic stub (no external model calls):

```sh
uv run --project gym python gym/smoke.py
```

The stub first asks Claude Code to call the native `gym` MCP server for resources,
then execute the bounded Gym workflow through its real Bash tool: historical context and resources, two candidates, checkpoint continuation,
final checks and a submission receipt. This is integration coverage, not an AI
architecture-search quality measurement. A prepared client image with the pinned
CLI can be selected using `--client-image`; otherwise Harbor installs the CLI.
For a simpler test without Claude Code, explicitly pass `--agent oracle` to
`run_trial.py` with either example configuration. Harbor collects runtime state
and signed Gym evidence before starting the separate verifier.

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

## Research information and execution resources

`list_experiments(scope="published")` lists repository-published historical runs;
`read_experiment(experiment_id=...)` returns results, model cards/structure descriptions, model configuration and
reference IDs for architecture source. `read_reference` reads only catalogued
files, and `list_components` paginates the catalog. Current source/configuration
is labelled separately from historical run provenance, including whether a
recorded model hash matches. Runs whose code is absent still have their published
model card and an explicit missing configuration. Missing historical source is not silently claimed
to be the exact code from that run. Published historical CORE scores are context;
new candidates cannot access CORE-22 evaluation during research.

`scope="session"` returns the current session's numerical jobs, results and source
packages. Other sessions' private results are inaccessible. The Agent has no need
for filesystem or Git access to the repository.

`get_resources` observes the actual local backend: server identity, CPU/memory,
NVIDIA inventory, configured allocation, currently available cards and the shared
worker slot. `servers` contains one local node. Each GPU reports its UUID/index,
name, total/free memory, utilization, allocation, availability and busy reason.
Jobs record the selected server and actual GPU; results include actual parameter
devices, CUDA/PyTorch versions and peak GPU memory. Discovery failure is reported
as unavailable, not as invented capacity. Remote dispatch is not implemented.

The operator controls allocation with `--execution-config`; the Agent can read it
but cannot expand it. CPU is the default. Start with `examples/resources.gpu.json`,
replace the placeholder card with an allocated physical index or UUID, and set its
memory allowance. The local backend runs one numerical job at a time. A busy GPU
keeps work queued with a waiting reason; no numerical FLOPs are charged until
execution, and its budget reservation remains held. There is no CPU fallback for
GPU jobs. Checks also use the configured allocation. Cooperative leases coordinate
Gym instances sharing the same host/user lease directory; external host jobs still
need operator coordination. Memory limits combine conservative admission bounds
and the PyTorch allocator cap; they are not a hardware partition.

For a host service, pass `--execution-config <operator.json>` and optionally
`--worker-python <CUDA-enabled-python>` to `nt-gym` alongside its required state,
profiles and platform paths. Use the same configuration/interpreter for bootstrap,
serve and later controller commands. The actual worker runtime is recorded in
revision identities. `--server-id` sets a stable operator node name. The existing
HTTP listener retains its first OS-assigned port across restarts.

For Harbor, set `runtime.execution_config` in the example TOML or pass
`--execution-config gym/examples/resources.gpu.json` to `run_trial.py`. The helper
freezes the policy into the trial artifacts and mounts it only into Gym. GPU
containers require a CUDA-enabled service image and the NVIDIA Docker runtime;
preflight checks these before creating trial volumes. The Agent container receives
no GPU devices. See [VALIDATION.md](VALIDATION.md) for the host GPU smoke and the
separate container-runtime limitation of this machine.

An Agent may choose `target_update` (absolute step), `server_id` and an authorized
`gpu_id` on `quote_experiment`; omit placement to let Gym select an idle allocated
card. It can write scripts which orchestrate these APIs, but cannot submit shell
commands, trainer replacements, environment overrides, data paths or SSH jobs.
See `examples/research_with_placement.py` for a complete bounded submission flow.

## HTTP contract and MCP tools

**HTTP is the authoritative interface.** `describe_space.operation_schemas` exposes
all Agent operation names, descriptions, JSON input schemas and read-only hints.
The stdio MCP adapter obtains these definitions from authenticated HTTP discovery
and forwards every tool call through the same `GymClient`. It has no separate
scheduler, ledger, authorization policy or training imports. HTTP results and Gym
error envelopes are preserved; MCP additionally sets `isError` for failed calls.
A new HTTP operation therefore appears in MCP without another tool definition.

The native Harbor task registers server `gym` under `environment.mcp_servers`.
Claude Code receives these tools through Harbor's standard MCP integration, with
no custom Agent implementation. Both SDK and MCP read the same session credential:

```sh
python -m nexttransformer_gym.mcp_server --credential /run/gym-auth/credential.json
```

`examples/mcp.json` supplies the equivalent stdio registration for a client with
the Gym adapter installed. MCP adds no host listener and no independently exposed
credential. SDK scripts and MCP tools can be mixed within the same session, budget,
job lifecycle and idempotency scope.

## Native plugin compatibility

A candidate contains the original native plugin directory and model configuration,
plus Gym submission metadata. `method.json`, `__init__.py` and local Python modules
are preserved unchanged. Use `model = YourReferenceLMSubclass`, `mixer`, `ffn`, or
`block` exactly as documented in `nexttransformer/models/plugins.py`; the platform
does not inspect a model expression tree. `architecture.json` contains native
`ModelConfig` fields and declared `options`. The example is a multi-file `model`
plugin, and tests compare its direct native load with the Gym path exactly.

```sh
python gym/examples/make_candidate.py candidate.zip --activation silu
```

Research cost reports are native trace estimates, not a proof of all arbitrary
Python branches. The OS sandbox separates worker files/processes/networking;
production adversarial accounting and scoring isolation remain a separate gate.

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

- `prepare-final`: create a separate staging Git snapshot, copy the selected
  native plugin unchanged, re-audit its configuration and write an admitted pipeline plan.
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
collection. Production image/hardware, worker isolation and full-run acceptance must
be validated for the selected deployment; bounded CPU fixtures do not establish them.
