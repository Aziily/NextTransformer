# Harbor auto-research roadmap

This is the target design agreed after the original
[Gym proposal](../docs/design/gym-design-interface.md). The
[implemented contract](CONTRACT.md) describes what currently runs. This roadmap
replaces the proposal's mediated Agent, offline research policy and data split
for the new auto-research task; it does not change published NextTransformer
results or the existing official protocol.

## Task and Agent

- Keep the implementation inside this repository's independent `gym/` directory.
  Use native Harbor task layout: `task.toml`, `instruction.md`, `environment/`,
  `tests/`, and an optional `solution/` integration oracle. Harbor remains a pinned
  dependency; do not modify Harbor source or the shared NextTransformer core.
- Harbor owns the Agent lifecycle, container execution, trajectory collection
  and teardown. Use its built-in **Claude Code** Agent (`--agent claude-code`) by
  default. Select the model and provider authentication through Harbor's native
  configuration. Do not add a Gym model broker, custom Agent subclass or separate
  reasoning loop. The oracle is an explicitly selected bounded test only.
- The Agent develops native plugins in its own container, with research information
  supplied by Gym and outbound internet access for research.
  A closed container means filesystem/process isolation, not an offline network.
  It needs no repository checkout or Git history. Gym exposes past experiments,
  results, model configurations and read-only architecture reference code.
- Gym runs this repository's training and evaluation service on the host.
  Harbor contains the Agent and independent verifier; a CUDA Gym container is not
  required. The Agent uses its HTTP SDK or MCP tools projected from that HTTP API;
  neither imports training code. The host service owns
  datasets, checkpoints, budgets and final training. Its private state and the
  verifier's credentials are not mounted into the Agent container.
- Keep numerical workers network-isolated. Record actual Agent container process
  state through Harbor's environment API and retain runtime snapshots and
  trajectories. Do not infer runtime state from a host-side launch flag or depend
  on a custom Agent loop emitting status files.

## Data and scoring

| Set | Contents | Research access |
| --- | --- | --- |
| Training | ClimbMix training shards except the last shard | Training through Gym |
| Validation | The last shard in the pinned ClimbMix manifest | Feedback through Gym for algorithm selection |
| Test | CORE-22 | Final scoring only, after the candidate is frozen |

Pin dataset manifests and shard ordering before the cohort starts. Under the
current full shard numbering, validation is `shard_06542.parquet`; shard 06541
is training data. This matches the repository's default `BestFitStream`:
all preceding shards train, and the last shard validates. All candidates use the same fixed
training/evaluation profiles, tokenizer and data order. The Agent may choose
architecture and plugin options, absolute target updates within profile limits,
and an operator-authorized local GPU. It cannot override the experimental protocol
or provide arbitrary training commands. Gym launches the fixed trainer.

Do not expose CORE-22 data, per-candidate results or a test-evaluation operation
through the research environment/API. Final ranking uses only the agreed CORE-22
aggregate after fresh final training. Training/validation metrics remain
diagnostics. Published historical CORE results may be used as research context.
CORE-22 is public: with web access and published experimental history, this
is an evaluation-access and feedback boundary, not a claim that its contents
have never been seen by the Agent.

The original Gym proposal added a separate research dev shard (06541) and withheld
the native validation shard (06542). The auto-research task instead exposes native
validation feedback for tuning, as requested. Record that selection policy and
CORE-22-only ranking in the cohort configuration; do not silently change the
native official protocol or label results official-compatible without the
corresponding validation.

## Native candidate and ID-based submission

Use the repository's original plugin structure and interfaces. A `model` must
implement the native model contract; its configuration uses native `ModelConfig`
fields and declared plugin options. Gym validates metadata/configuration and
loads the plugin through the native loader. It does not parse an FFN expression
language or translate model source.

Upload a source-only candidate package to Gym containing:

```text
proposal.json
architecture.json
plugin/<method>/method.json
plugin/<method>/__init__.py
plugin/<method>/<optional local Python modules>
```

Preserve native plugin files unchanged. Include no trained weights, research
checkpoints or dependency installers. `finalize` freezes one validated candidate
and its content identity. The final deliverable is a `submission.json` receipt
containing `session_id`, `submission_id` and `candidate_hash`. Gym durably retains
the complete source package and configuration associated with that submission.
No separate final archive export is required.

After acceptance, the platform dispatches Gym final training from fresh
initialization under a separate fixed allocation. The verifier checks the frozen
receipt against authenticated Gym evidence; it does not train models. Gym resolves
the submission ID to retained source/configuration and copies them into the final
run's durable state. Final training does not depend on the Agent workspace remaining
available. Research credentials cannot launch arbitrary
final runs, and research checkpoints are never used for final initialization.

## Gym interface and budget

Read `endpoint` and `token` from `/run/gym-auth/credential.json`. Every call is a
JSON `POST /v1/call` with `schema_version`, `operation`, and `params`, authenticated
with the bearer token. The Python `GymClient` wraps this transport only. HTTP is the source of truth:
`describe_space.operation_schemas` publishes operation definitions, and the stdio
MCP adapter discovers them and forwards calls to the same authenticated HTTP
service. Native Harbor registers this adapter for Claude Code.

| Purpose | Operations |
| --- | --- |
| Discover capabilities and limits | `describe_space`, `list_components`, `read_reference`, `get_resources` |
| Read historical experiments and architectures | `list_experiments`, `read_experiment` |
| Upload candidate source | `submit_candidate` |
| Check loading, correctness and cost | `validate_candidate`, `calibrate_candidate`, `debug_candidate`, `inspect_cost` |
| Price work and inspect allocation | `quote_experiment`, `get_budget` |
| Train under a fixed profile, optionally continuing a compatible checkpoint | `run_experiment` |
| Track or stop work | `get_job`, `get_result`, `cancel_job` |
| Freeze the final candidate | `finalize` |

Prechecks reduce wasted training, but numerical checks also consume research
compute. Reserve a quoted bound before dispatch, charge executed work, retain
uncertain reservations and stop when the fixed allocation is exhausted. Trace
costs and parameter counts are checks for the selected configuration, not a proof
of all arbitrary Python behavior. Agent inference, web search and local shell CPU
use are outside the Gym training ledger and need separate Harbor time/resource
limits. Do not describe this ledger as metering every computation by the Agent.

## Execution scope

Use one local machine and one concurrent numerical job. Additional experiments
queue under the same research budget. The Agent can select an operator-allocated
GPU on that machine. Remote dispatch and parallel workers are outside this scope.

## Current implementation and remaining work

| Area | Current state | Required change |
| --- | --- | --- |
| Harbor task and host service | Native task layout; Gym runs on the host, Agent/verifier remain separate containers; runtime and signed evidence are collected | Retain single-machine host execution as the default |
| Agent | Native Harbor Claude Code default; custom broker/loop removed; model/budget examples supplied | Validate the selected provider and production research configuration |
| Research environment | Agent-only internet access; Gym exposes published results, native model configurations and architecture reference code | Preserve provenance and owner-scoped private history; no Git access is required |
| Execution resources | Local NVIDIA discovery, operator GPU allocation, memory limits, busy queues, cooperative leases and real CUDA workers; one node and one worker slot | Validate the selected host CUDA environment; single-machine execution with one concurrent numerical job is the agreed scope |
| Data | Parquet adapter uses native last-shard validation (06542) and permits training through 06541; shipped task still uses synthetic CPU fixtures | Supply and validate production ClimbMix/tokenizer manifests |
| Candidate/API | Native source packages, authoritative HTTP schemas, thin MCP adapter, checks, jobs, continuation and shared budget ledger exist | Keep native interfaces; finish production execution and accounting validation |
| Final artifact | Candidate source/configuration live in durable Gym state; Harbor collects a `submission.json` receipt; final dispatch resolves retained code | Retain the agreed ID-based delivery and per-trial host service state |
| Final execution | Separate final dispatch and a native formal adapter exist; full GPU training is unverified | Wire post-acceptance dispatch and validate the new cohort's final training and CORE-22-only scoring |

Acceptance requires a native Claude Code trial with Gym research-history access,
bounded Gym experiments, submission receipt verification and retained real
runtime artifacts. Production acceptance additionally requires dataset, GPU,
budget/accounting and final-evaluation checks. Existing CPU oracle results do not
establish those capabilities.
