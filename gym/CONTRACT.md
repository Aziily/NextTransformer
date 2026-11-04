# Implemented contract

The design authority is `../docs/design/gym-design-interface.md`. Harbor 0.24.0
and task schema 1.4 are pinned. This document describes the implementation's
supported subset and release gates, not a change to the scientific protocol.

## Candidate runtime

`reference_plugin_v1` currently supports the FFN hook through the closed
`ffn-expression-v1` language. Source is a single literal assignment to
`FFN_EXPRESSION`. The tree contains `op` and `args`; operators are `input`,
`relu`, `square`, `silu`, `tanh`, `add`, and `mul`. Depth is less than eight.
Expressions are pointwise over the hidden channel projection and cannot index
future tokens, inspect labels, access files, import modules or branch on values.

The trusted runtime supplies the two projections and the unchanged ReferenceLM
shell. Architecture configuration contains `layers` (1–64) and `mlp_ratio`
(3–5). Execution profiles pin all other dimensions, data and schedules. This
allows bounded FFN invention; arbitrary attention, model hooks, Python modules
and kernels are explicitly rejected before dispatch, not executed on trust.

The four-file ZIP layout is exactly:

```
proposal.json
architecture.json
plugin/<method>/method.json
plugin/<method>/__init__.py
```

`proposal.json` uses schema `gym.candidate.v1`, SDK `reference_plugin_v1`,
`method`, optional `parent_ids`, and a human-readable hypothesis. `method.json`
is `{ "name": "<method>", "family": "others", "hook": "ffn",
"language": "ffn-expression-v1" }`. The example builder supplies a complete
package. Packages are at most 1 MiB and 32 entries; this contract accepts exactly
four files. Absolute/traversal/duplicate paths, symlinks, extra modules, binaries,
installers and executable Python are rejected. The control service never imports
candidate source. The source/configuration hash is server computed.

## API and persistence

Every HTTP call uses `/v1/call`, schema `gym.api.v1`, an operation, a JSON
parameter object, and a bearer session credential. The SDK sends only these
messages. Agent and verifier credentials have separate scopes. Sessions and
allocations can only be created through a trusted local controller.

Supported operations follow section 8 of the design: space/resources/catalog,
source submission and reads, cost inspection, charged validation/debugging,
static calibration, quote/run, jobs/events/results, cancellation, budget reads,
finalization and submission reads. Mutations require idempotency keys; a key
cannot be rebound to different content. Unknown parameters are rejected.
Candidate/event pagination uses stable cursors. Calibration is a completed
static-analysis job returning derived immutable candidates and cost reports.

Quotes bind source, profile, runtime revision, checkpoint, resource profile and
absolute target update. A quote expires after five minutes and is consumed once.
Reservations and jobs commit in one SQLite transaction. FLOPs amounts cross the
wire as decimal strings. Source storage has a 128-candidate session limit.
The initial scheduler serializes numerical workers on one declared resource slot;
the account remains concurrency-safe across simultaneous callers.

## Numerical accounting and checkpoints

Static analysis uses the existing shape-only FakeTensor/ATen cost convention,
including separate initialization, forward, backward, optimizer, prediction and
trusted dev-scoring paths. Only branch-free supported source can reach this audit.
Each numerical operation requires a durable intent and controller acknowledgement.
The worker then reports completion using a unique charge identity. Unknown work
retains its reservation. A stopped crashed worker can be explicitly reconciled
by charging its remaining conservative bound; there is no automatic refund.
Known numerical failures consume the current operation's conservative bound and
report `numerical_failure`; cancellation stops at a reconciled operation boundary.
Final allocation exhaustion reports `budget_stopped`. Unknown worker failures stay
in `reconciliation`, with their reservations retained until termination is confirmed.

Checkpoint artifacts retain model/optimizer/RNG/routing/data-buffer/schedule/stage
and cumulative training state. They are private to the service, hash checked on
continuation, and bound to exact candidate/profile/runtime identities. Source
changes start over. Dev scoring is performed by trusted code from predictions,
labels and token-byte lengths; candidate-returned metrics are never accepted.

The parquet adapter selects an explicit contiguous training prefix excluding
`shard_06541.parquet` and `shard_06542.parquet`. Dev selects only shard 06541.
It reuses upstream best-fit row generation and refuses training-prefix exhaustion.
The shipped profile uses deterministic synthetic data, explicitly labelled as a
fixture; it is never presented as ClimbMix or as a formal comparison.

## Harbor and agent runtime

Harbor owns `main`, the Gym sidecar, execution, collect hooks and teardown. The
agent image contains only the SDK, package example and observation helper. No
platform source, dataset, checkpoint root, verifier secret or Docker socket is
mounted there. The Gym has private durable state. Agent and verifier identities
use different volumes. The verifier runs in its own network-disabled container.

The supplied `GymResearchAgent` exposes JSON actions only: Gym API calls,
source-byte submission, bounded notes and final receipt writing. It offers no
shell/code execution tool. Its model broker runs on the controller, uses only an
explicit fixed HTTPS endpoint/model, and never transfers provider credentials
into the container. Agent inference metering remains deferred, as in the design.
Other Harbor agents can use the SDK but cannot claim this mediated-tool policy.
The oracle is a bounded integration test, not an enforced research competitor.

`HarborRuntime(environment).snapshot()` uses Harbor's real environment `exec`
API to read the main container's `/proc` state. It separately labels the mediated
agent's reported phase/turn. Collect hooks retain that snapshot and authenticated
Gym evidence as native Harbor artifacts. No host-side launch flag substitutes
for runtime observation.

## Final verification and execution

Finalization requires successful exact-identity reports and final admission,
a reconciled account and no active jobs. It freezes one choice. The separate
Harbor verifier checks a receipt against HMAC-authenticated server evidence and
its trusted per-trial binding. Reward 1 means deliverable validity only. There is
no automatic final launch, model performance reward or leaderboard rank.

`prepare-final` creates a separate Git snapshot, lowers the expression into a
standard plugin, re-audits the selected configuration without searching for a
replacement, runs native admission and pins a source/configuration manifest.
It never changes the source repository or launches training. `launch-final`
requires a verified submission and a distinct finite allocation. The official
adapter invokes native formal training, cumulative audit and result export in
that staging checkout. Single-node dispatch uses 8, 16 or 32 explicitly allocated
GPUs and file rendezvous. Each rank receives a finite allocation share, with exact
rational cost rates rather than floating-point budget arithmetic. Each model call is guarded by a conservative finite
allocation; abnormal termination cannot refund uncertain work. Only a successful
native audit/export yields an official-compatible record. Upstream publication
remains a separate review process.

The shipped fixture exercises separately allocated fresh initialization without
claiming official performance: formal BPB, CORE and rank remain null. A production
final profile needs a reviewed official plan, resources, runtime compatibility
and bounded allocation. The full 49.152B-token run is not part of development tests.

## Remaining release validation

The short CPU path and reference lowering/packing/continuation parity are tested.
A production GPU deployment still requires its image/hardware/runtime parity
validation, dataset/tokenizer manifests, appropriate memory/storage quotas and
full formal-run acceptance. The initial worker pool is single-slot and has no
checkpoint garbage collector. These are explicit deployment/release limits;
this repository does not claim arbitrary Python sandbox safety or completed
production benchmark results.
