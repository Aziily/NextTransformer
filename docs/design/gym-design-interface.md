# NextTransformer Gym: design principles and interface specification

Status: design proposal for review; no gym service is implemented by this document.
Source behavior was inspected at NextTransformer commit `41b6016`. This proposal
does not amend the frozen NextTransformer protocol or Arena v1 rules, and its
policies are not adopted upstream by the existence of this document.

## 1. Purpose and deliverable

Build an offline research environment in which an RSI system designs language-model
architectures, learns from controlled experiments, and submits **one final architecture**.
The gym verifies and freezes that architecture. A separately dispatched platform
pipeline automatically trains and scores it. Final presentation uses
the existing leaderboard's BPB, CORE, compute and provenance fields.

The participant receives a **Gym API/SDK**, a versioned architecture component
contract, read-only reference material and a candidate workspace. It does **not**
receive an executable checkout of the platform repository, a training installation,
a dataset directory or permission to modify the evaluator.

The platform retains NextTransformer, environments, data, execution resources,
training/evaluation code, accounting and publication. Participants submit architecture
source and configuration, never repository patches, training commands, pretrained
weights or authoritative scores.

Success means that different RSI systems can use the same interface, starting
materials and resource limits to conduct research, while the platform determines
what actually ran, how much it consumed, and how the final model performed.
This evaluates architecture research ability in this task family; it is not by itself
a claim about general recursive self-improvement.

## 2. Design principles and their consequences

### 2.1 Separate research decisions from experimental authority

The researcher decides what architecture to try, how much research to allocate,
which results merit continuation and which candidate to submit. The platform owns
execution, measurement and policy enforcement.

This preserves meaningful search freedom without allowing the researcher to improve
its apparent score by changing the optimizer, data, evaluator or accounting. A useful
API exposes experimental decisions, not arbitrary access to the machine.

| Researcher owns | Platform owns |
| --- | --- |
| Hypothesis, plugin implementation and architecture options | Supported plugin contract and component runtime |
| Candidate lineage and selection | Immutable candidate identity and provenance |
| Experiment length, continuation and early stopping | Fixed profile semantics, training and data order |
| Allocation across concurrent experiments | Atomic reservations and resource scheduling |
| One final architecture choice | Submission verification, then separately dispatched final training/evaluation |

### 2.2 Expose a capability, not a repository

Understanding source is valuable; modifying infrastructure is unnecessary. Serve
selected architecture implementations, interfaces and examples as read-only resources.
Accept candidate packages in a separate namespace. Never accept a patch against the
platform tree, an arbitrary command, a filesystem overlay or a participant installer.

An importable client SDK must remain a client: importing it in the researcher process
must not load platform internals or grant in-process access to the trainer.
Architecture components are provided separately inside the candidate runtime, through
a documented versioned contract. API and SDK names below are proposals, not existing
packages.

### 2.3 Keep exploration flexible and final comparison controlled

Research architectures may differ in FLOPs/token. The researcher sees this difference
and pays for it. Requiring every exploratory idea to match the final anchor before
any experiment would unnecessarily constrain search.

Final official admission is different: the selected architecture/configuration must
meet the fixed protocol, including the cost constraint. Calibration can produce a
new architecture configuration during research, but final submission must explicitly
select that configuration. The platform never silently changes the submitted design.

A better score from a more expensive or longer experiment is useful feedback, not
automatically evidence of a matched improvement. Return the relevant cost, tokens,
steps, data and profile with every observation.

### 2.4 Budget the entire search, not just its winner

A system that tries many failed ideas has consumed research resources even if it
submits only one successful architecture. Training, dev evaluation, dynamic checks,
profiling, failures and cancellation all contribute executed model computation.

Use theoretical experiment FLOPs as the v1 hard research account. This matches the
project's architectural-compute emphasis across hardware, while throughput, memory,
GPU-hours and elapsed time remain useful separate observations. Equal FLOPs do not
imply equal latency or monetary cost. Agent inference cost is explicitly deferred.

### 2.5 Feedback should support diagnosis, not only ranking

A scalar score does not explain whether a candidate is invalid, unstable, too costly,
undertrained or promising. Return structured diagnostics, training curves, dev BPB,
resource measurements and accounting. Let RSI systems decide how to use them.

Do not leak formal evaluation through logs, downloadable artifacts or job events while
research remains open. Published baseline results are explicitly public starting
context, not new feedback on a participant's current candidate.

### 2.6 Immutable artifacts make decisions reproducible

Freeze the source, configuration, profile and environment that determine an experiment.
Give source candidates, executions and checkpoints distinct identities. A code change
creates a new candidate, and a changed architecture/configuration starts from scratch.

State belongs to the gym rather than an agent conversation. Controller restarts and
network retries must not reset budgets, duplicate jobs or lose experiment lineage.

### 2.7 Offline isolation and accounting are requirements, not trust assumptions

Treat all candidate Python as untrusted. Read-only files, hashes, AST checks and
containers each help, but none alone proves safe execution or complete accounting.
A participant cannot gain permissions by naming a path, claiming a cost or supplying
a result file. Unsupported computational paths fail closed.

The permitted executable language/runtime must be narrow enough to enforce these
requirements. General unrestricted Python plus a tracing pass is not an acceptable
substitute. Selecting and validating that runtime is an implementation gate, not a
capability this document claims already exists.

## 3. Proposed v1 scope and decisions

| Area | V1 decision |
| --- | --- |
| Architecture freedom | Existing plugin-compatible composition and local invention; no arbitrary replacement of the shared model shell |
| Research connectivity | Offline workspace and workers; only a restricted platform inference broker for researcher LLM calls |
| Starting conditions | Same approved references, component versions, templates and baseline records per comparison cohort |
| Participant output | One final source architecture plus declared configuration; no research weights |
| Execution | All candidate execution goes through the gym, including CPU debug computation |
| Search policy | RSI chooses bounded training lengths, stopping, continuation and final selection |
| Concurrency | Allowed, with a configured limit and one shared session account |
| Research budget | Finite, explicit cumulative experiment FLOPs cap; missing cap blocks execution |
| Unaccountable computation | Rejected before real dispatch; not run for free as a diagnostic |
| Resume | Same immutable architecture/configuration and compatible complete checkpoint only |
| Research feedback | Training loss and dev BPB plus cost, diagnostics and resource observations |
| Submission verification | Harbor verifier checks valid submission/evidence only; acceptance is not a performance score |
| Final scoring | Separately started platform pipeline, one frozen candidate, from scratch, common separate allocation |
| Display | Official leaderboard-style result; no required new composite or research-efficiency score |
| Failure accounting | Executed work remains charged, including failed and cancelled attempts |

The numeric research cap, machine allocation, worker limits and sandbox implementation
are deployment settings still to specify. No numeric default or training authorization
is implied by this document.

## 4. What is reusable from NextTransformer

Paths below are relative to the platform checkout, not paths exposed to participants.

| Existing capability | Source | Reuse and limitation |
| --- | --- | --- |
| Model construction/loading | `nexttransformer/models/__init__.py` | Common model API and checkpoint reconstruction |
| Architecture families | `nexttransformer/models/` | Reference components for attention, recurrence, residuals and hybrids |
| Plugin declarations and hooks | `nexttransformer/models/plugins.py` | Metadata discovery and mixer/FFN/block/model hooks; not a security boundary |
| Model shell | `nexttransformer/models/lm.py`, `models/config.py` | Initial compatibility target; model hook remains a `ReferenceLM` subclass |
| Operator accounting and calibration | `nexttransformer/protocol/` | Cost convention and admission; not a session-wide resource ledger |
| Training and restoration | `nexttransformer/train.py`, `runtime/checkpoint.py` | Scientific behavior to preserve and test against |
| Data and evaluation | `runtime/data.py`, `evaluation/formal.py` | Tokenization, packing, BPB and CORE semantics; no ready dev-only runner |
| Final audit and export | `audit_run.py`, `export_run.py` | Final completeness checks and result format |
| Result presentation | `results/`, `website/` | File-backed official-style leaderboard |

The formal shell fixes width 1536, 12 heads, vocabulary 32768 and sequence length 2048;
shared initialization expects particular attention/FFN projections. Advertising a
`model` hook must not imply arbitrary `nn.Module` freedom.

The current `BestFitStream` only accepts `train` and `val`. The current trainer also
performs validation in smoke mode. Wrapping `--research` or `--smoke-steps` directly
would not create the dev-only research backend required here.

The existing trainer and loss API load model code in-process. That is scientific
infrastructure, not an isolation boundary. A safe adapter must preserve the intended
numerics and pass parity checks; do not claim the frozen executable can simply run
unchanged with arbitrary hostile plugins and remain secure. Formal compatibility
must be demonstrated before exporting formal-compatible results.

## 5. System layout and trust boundaries

```mermaid
flowchart TB
    R[Offline RSI workspace: client, candidate drafts, notes]
    K[Read-only reference catalog]
    A[Gym API: authentication and policy]
    S[Control service: sessions, artifacts, jobs]
    L[Budget ledger]
    W[Isolated candidate execution]
    T[Platform training coordinator]
    E[Trusted scoring service]
    D[Controlled data delivery]
    V[Submission verifier]
    P[Authorized final-run dispatcher]
    F[Final training, audit and result export]
    R -->|resource IDs| K
    R -->|typed requests and candidate bytes| A
    A --> S
    S --> L
    S --> T
    T --> W
    D -->|authorized batches| T
    D -->|evaluation inputs| W
    W -->|predictions| E
    D -->|evaluation labels| E
    T -->|trusted progress| S
    E -->|trusted metrics| S
    S -->|immutable submission and evidence| V
    V -->|verified submission| P
    P -->|separate launch and allocation| F
```

- **Research workspace:** contains drafts and client tools, not a runtime checkout.
  Code editing/search tools are permitted. Unrestricted shell execution would defeat
  the all-computation-through-gym rule even without a GPU; the research harness must
  mediate tool execution. An uncontrolled external agent host can use the API but
  cannot claim the same enforced-budget offline cohort.
- **Control plane:** owns authentication, authorization, artifact validation, job
  transitions and ledger writes. It never imports participant modules.
- **Candidate worker:** has only its frozen package, approved component runtime,
  operation-specific inputs and a private output area. No network egress, credentials,
  host socket, writable platform code or other session storage. Training coordination
  and privileged scoring are not participant-controlled Python globals.
- **Data/scoring plane:** controls batch delivery and derives metrics. Participants
  cannot read arbitrary dataset files or declare their own evaluation labels.
- **Inference broker:** allows only the declared researcher-model API, without tools,
  browsing, arbitrary URLs or environment secrets. Model identity and access policy
  are recorded even while inference cost accounting remains TODO.

Container/VM choice, device isolation and the supported execution subset require
validation. The architecture describes required boundaries, not a promise that a
particular container configuration is sufficient.

### 5.1 Organize heavy assets separately from the control service

The API server owns metadata and policy, not a copy of all datasets and checkpoints
in its process or per-session directory. Start with a modular control service, a
transactional database, an artifact store and an isolated worker pool. Logical
separation is required; a fleet of independently deployed microservices is not.

| Asset class | Organization | Sharing and access | Lifecycle |
| --- | --- | --- | --- |
| Runtime images, component packages | Immutable digest-addressed versions | Worker image/layer caches; participant sees approved SDK/reference only | Publish new version; never mutate a pinned version |
| Data, tokenizer, evaluation bundles | Manifested datasets with split/view IDs and checksums | Shared physical storage; read-only operation-specific delivery, never the whole dataset root to a candidate | Retain pinned versions required for reproducibility |
| Architecture references and baseline metadata | Versioned public catalog | Same approved catalog for the cohort | Update only with a new catalog/cohort version |
| Candidate source and reports | Immutable artifacts indexed by session/candidate | Session-authorized reads; no cross-session browsing | Retain final packages and provenance; expire drafts by policy |
| Checkpoints and training logs | Experiment-scoped artifacts, separate from database rows | Platform restore only; research/final namespaces remain distinct | Explicit quota/retention; pin resumable and final artifacts |
| Database and event/charge journal | Sessions, jobs, ownership, hashes, state and accounting | Control plane writes; separate backup/restore plan | Durable; never garbage-collect unresolved accounting |
| Worker scratch and staging | Private temporary volume per job | No shared writable working directory | Remove after upload, reconciliation and lease release |

Use shared storage or read-only caches for large assets, rather than copying a data
collection or runtime into each Harbor trial. Physical deduplication must not expose
another session's artifact existence or bypass authorization. Workers must not receive
a broad mount containing formal evaluation data when running research.

A resource manager resolves manifest IDs to internal storage locations, checks asset
readiness, allocates execution slots and fences stale worker leases. The scheduler
only starts jobs whose pinned images/data and requested layout are ready. Asset
staging errors are infrastructure outcomes, not bad architecture scores.

Separate compute reservations from storage quotas and GPU-slot leases. Available
FLOPs do not imply free GPUs or unlimited checkpoint storage. Announce storage limits
and TTLs; do not evict a checkpoint used by a live continuation or a frozen final
artifact. Mark artifacts ready only after an atomic upload/manifest commit. Garbage
collection must respect live references, jobs and unsettled reservations.

### 5.2 Agent workspace and server storage do not overlap

A Harbor agent image contains the client, approved reference material and editing
tools. Writable candidate drafts/notes and a final receipt live in the agent workspace.
There is no shared writable mount into the platform checkout, artifact root, database,
worker scratch or checkpoint directories. Uploads carry bytes; downloads resolve
session-authorized object IDs through the service.

Only a trusted trial controller can allocate a session and bind its resource/budget
policy. The agent receives session-scoped research credentials. The separate verifier
receives a submission-inspection identity, not the agent credential or training-control
credential. Administrative final-run launch credentials never enter either container.

## 6. Participant-facing resources and candidate package

### 6.1 Read-only catalog

Expose versioned resource IDs for:

- component signatures, tensor shapes, hooks, supported operations and restrictions;
- selected architecture source and explanatory documentation;
- complete small templates and valid configuration examples;
- approved baseline records with their original protocol/cost labels;
- research profiles, final-admission requirements and accounting conventions.

`read_reference(resource_id)` cannot read an arbitrary path or URL. Responses are
bounded and identical for participants in the same cohort. Copying reference code
into a candidate changes only that candidate, never the platform original.

### 6.2 Resource information is part of the public contract

A researcher needs enough information to design feasible experiments, not internal
machine inventory. Expose both a versioned capability descriptor and a timestamped,
short-lived availability view, filtered to the caller's allowed resource profiles.

| Information | Examples | Meaning |
| --- | --- | --- |
| Stable hardware/layout capabilities | Accelerator class, usable memory per device, supported GPU counts/distributed layouts, supported precisions | What this resource profile can run; not a promise that it is idle |
| Runtime capabilities | Image digest, dependency/component versions, supported operators/kernels | What source can execute and what requires a new platform version |
| Session limits | Concurrent jobs, worker memory/time bounds, artifact/checkpoint quotas | What this researcher may request |
| Allowed execution profiles | Context, shapes, batch rules and research/final compatibility | What constitutes a valid experiment on this pool |
| Current availability | Allocatable slots, own queued/running jobs, capacity state, snapshot time | Planning information; rechecked at dispatch |
| Candidate fit | Estimated peak memory, compatible profile IDs, measured or estimated throughput and provenance | Advisory until supported preflight has passed |

Do not reveal hostnames, IP addresses, storage paths, other users' jobs, cloud keys or
raw device-management endpoints. Report missing memory/throughput/queue estimates as
null, with reasons; theoretical FLOPs do not determine wall-clock runtime. Distinguish
measured, estimated and unavailable values.

Proposed `get_resources` response shape; quantities are illustrative, not provisioned
hardware or chosen quotas:

```json
{
  "schema_version": "gym.resources.v1",
  "capability_revision": "resources-example-1",
  "observed_at": "2026-11-03T00:00:00Z",
  "profiles": [{
    "resource_profile_id": "research-example",
    "accelerator_class": "example-gpu",
    "usable_memory_bytes_per_device": null,
    "supported_device_counts": [1],
    "supported_precisions": ["float32"],
    "runtime_manifest_id": "runtime-example-1",
    "availability": "unknown",
    "allocatable_slots": null
  }],
  "session_limits": {"max_concurrent_experiments": 2}
}
```

Profiles must be fully configured before dispatch; unknown required capabilities in
an informational response never mean an execution check may be skipped. Add optional
`resource_profile_id` to quotations. The server chooses the actual worker. Bind the
selected capability/runtime revision to the quote; compatible placement may vary,
but silently changing hardware class, numerical path or distributed layout is not
allowed. A capacity query or queue-time estimate does not reserve a GPU. GPU-slot
leases and FLOPs reservations are separate transactional scheduling concerns.

### 6.3 Submission format

Proposed package:

```text
candidate/
  proposal.json
  architecture.json
  plugin/<method>/
    method.json
    __init__.py
    <optional local Python modules>
```

Example metadata; this is a proposed envelope, not an implemented schema:

```json
{
  "schema_version": "gym.candidate.v1",
  "sdk_version": "reference_plugin_v1",
  "method": "example_mixer",
  "parent_ids": [],
  "hypothesis": "Replace selected mixers while retaining the shared shell."
}
```

`architecture.json` contains architecture choices only. Allowed fields and ranges
come from the selected contract; reject optimizer overrides, data paths, seeds,
evaluation options, commands and environment settings. `method.json` retains the
existing declaration structure where supported. The platform resolves hooks itself;
participants do not provide a launcher or arbitrary import target.

Only bounded text/source files and declared metadata are accepted in v1. Reject
absolute paths, traversal, symlinks, duplicate/ambiguous paths, binary payloads,
package-name collisions, oversized submissions and attempts to shadow SDK/runtime
modules. Do not accept pretrained state, participant executables or dependency installs.
These controls do not prove absence of memorized information embedded in source;
keep payload limits, provenance and adversarial validation explicit.

The server computes the canonical content hash. A candidate identity binds source,
architecture configuration and contract version. Metadata-only annotations can live
separately; computational changes always create a new candidate. Calibration produces
a derived candidate/configuration identity with a parent reference, never an in-place
edit. The platform selects the pinned execution environment, not the participant.

## 7. Persistent objects and lifecycle

| Object | Identity and purpose |
| --- | --- |
| Session | Researcher identity/version, cohort, limits, current phase and final choice |
| Candidate | Immutable architecture package, configuration hash, parents and contract version |
| Validation report | Candidate, environment and verifier identities; checks, coverage and findings |
| Cost report | Candidate/profile/accounting identities; components, stages and admissibility |
| Resource profile | Pinned runtime/hardware/layout capabilities and session-visible limits |
| Quote | Immutable request and resource specification, bound on cost, expiry; not yet a reservation |
| Experiment | Candidate, profile, start checkpoint, target update and job state |
| Checkpoint | Owner experiment, compatibility hashes, progress and complete training state |
| Reservation | Account, operation, maximum charge, committed charges and unsettled balance |
| Final submission | One frozen candidate, intended final profile, evidence manifest and verification receipt |
| Final run | Separately authorized allocation, pinned submission, attempts and eventual scored result |

All reads and mutations check session ownership. Guessing an ID or knowing an artifact
hash does not grant access. Version hashes invalidate stale reports and quotes.

Research session lifecycle:

```text
created -> researching -> finalizing -> submitted
                 |
                 +-> budget_exhausted -> finalizing
                 +-> closed_without_submission
```

Exhaustion blocks additional spending but permits selecting an already eligible final
candidate. Finalization checks eligibility before atomically locking the final choice,
rejects new research jobs, and cancels/drains and reconciles outstanding research work.
It produces a frozen submission and evidence manifest; it does **not** launch formal
training. A session cannot return to research after its final choice is accepted.

Submission verification and final scoring have independent lifecycles:

```text
submission: pending_verification -> verified | rejected
final run:  not_scheduled -> queued -> running -> completed | failed | cancelled
```

A verified submission is a valid architecture deliverable, not a measured result.
The final-run transition out of `not_scheduled` requires a separate authorized launch,
allocation and resource profile. Harbor can finish its research trial without waiting
for the final run, which may be scheduled much later. Neither session cleanup nor
Harbor verifier retries should launch, duplicate or destroy final runs.

Job lifecycle:

```text
queued -> running -> succeeded
   |         |----> failed
   |         |----> cancelling -> cancelled
   |         +----> stopping_at_budget -> budget_stopped
   +--------------> cancelled
```

A lost worker enters a reconciliation state; it is not automatically a free failed
job. Cancellation is a request, not evidence of termination. A completed bounded
research job may yield a checkpoint for a new continuation job. `succeeded` means
that the requested operation completed, not that the candidate improved performance.

## 8. Proposed API contract

### 8.1 Common rules

Use versioned JSON requests and responses with an authenticated session context.
Mutations carry an idempotency key: repeating the same key and request returns the
same object; reusing it with different content is a conflict. Long operations return
job IDs, not long-lived model calls. Pagination and event cursors support reconnects.

The client sends architecture bytes, IDs and typed experimental choices. It never
sends host paths, shell commands, arbitrary environment variables or executable job
specifications. Public transports must all enforce the same service policy.

### 8.2 Operations

| Operation | Key inputs | Output / behavior |
| --- | --- | --- |
| `describe_space` | Contract/profile filters | Controls, constraints, profiles, schema versions and final requirements |
| `get_resources` | Optional allowed profile filter | Versioned capabilities/limits and timestamped availability; no capacity guarantee |
| `list_components` | Filters, page cursor | Supported components, signatures and resource IDs |
| `read_reference` | Approved resource ID | Bounded source/documentation/template content |
| `submit_candidate` | Package bytes, idempotency key | Candidate ID, server hash, static metadata findings |
| `get_candidate` / `list_candidates` | IDs / filters, cursor | Own candidates, lineage and report references |
| `inspect_cost` | Candidate ID, profile ID | Cost report or analysis job; no unmetered real candidate execution |
| `validate_candidate` | Candidate ID, profile ID, idempotency key | Static report and, only when covered/reserved, dynamic validation job |
| `calibrate_candidate` | Candidate ID, allowed grid, idempotency key | Analysis job and derived candidate IDs with cost reports |
| `debug_candidate` | Candidate ID, named check, bounded shape, idempotency key | Covered, reserved diagnostic job; not arbitrary user code execution |
| `quote_experiment` | Candidate ID, profile ID, target update, optional checkpoint/resource profile IDs | Bound execution/resource request, cost breakdown, fit findings, expiry and affordability snapshot |
| `run_experiment` | Quote ID, idempotency key | Atomic reservation and experiment/job IDs |
| `get_job` | Job ID | Durable status, progress, result/artifact references |
| `read_events` | Cursor, filters, limit | Ordered events and next cursor; clients deduplicate by event ID |
| `cancel_job` | Job ID, idempotency key | Cancellation requested/current terminal status |
| `get_result` | Job ID | Trusted metrics, curves, costs, diagnostics and checkpoint reference |
| `get_budget` | Authenticated session | Limit, spent, reserved, available and accounting version |
| `finalize` | Candidate ID, intended final profile ID, idempotency key | Frozen submission ID and evidence manifest; no formal training launch |
| `get_submission` | Submission ID | Verification state/receipt and separately linked final-run status, if scheduled |

Administrative/verifier operations are separate from the agent API:

| Operation | Caller | Behavior |
| --- | --- | --- |
| `verify_submission` | Trusted separate verifier | Read frozen server evidence; emit idempotent valid/rejected receipt, never launch training |
| `launch_final_run` | Authorized platform scheduler/operator | Bind verified submission, pinned runtime/resources and separate allocation; queue fresh training |
| `get_final_run` | Authorized read-only client | Read scheduling/evaluation status and eventual official-style result |

The target update is absolute, not an ambiguous number of additional steps. A quote
binds candidate, profile, resource capability revision, checkpoint and target update; `run_experiment` cannot change
those values. Quotation alone does not reserve funds, so dispatch rechecks available
budget and quote validity atomically. Dynamic validation/debug operations use the same
internal quotation/reservation machinery even when exposed as convenience calls.

A blocking `evaluate(...)` client adapter can wait on this job API for frameworks
expecting a callback. It preserves full feedback and accounting. RSIHub can integrate
through its evaluator script/result record; FlashRSI through its snapshot evaluation
boundary. Neither code snapshot substitutes for a model checkpoint.

### 8.3 Example client flow

Illustrative API usage only:

```python
space = gym.describe_space()
resources = gym.get_resources()
components = gym.list_components()
reference = gym.read_reference(resource_id=components.items[0].reference_id)

candidate = gym.submit_candidate(package_bytes=package, idempotency_key="proposal-1")
cost = gym.inspect_cost(candidate_id=candidate.id, profile_id=research_profile)
checks = gym.validate_candidate(
    candidate_id=candidate.id, profile_id=research_profile, idempotency_key="check-1"
)
# Wait for required reports to pass; the SDK does not train locally.
quote = gym.quote_experiment(
    candidate_id=candidate.id, profile_id=research_profile, target_update=100
)
job = gym.run_experiment(quote_id=quote.id, idempotency_key="train-1")
result = gym.wait_result(job_id=job.id)

continuation = gym.quote_experiment(
    candidate_id=candidate.id, profile_id=research_profile,
    checkpoint_id=result.checkpoint_id, target_update=500
)
# Launch continuation only if its cost and the earlier feedback justify it.
# After research, select a candidate that passes the final-profile admission gates.
submission = gym.finalize(
    candidate_id=selected_candidate_id, profile_id=final_profile,
    idempotency_key="final-choice"
)
# This freezes a submission only; a privileged platform workflow launches final training.
```

### 8.4 Structured errors

```json
{
  "schema_version": "gym.api.v1",
  "error": {
    "code": "unsupported_operator",
    "phase": "cost_coverage",
    "retryable": false,
    "details": {"operator": "example::custom_op"},
    "hint": "Use a supported operation or request platform-reviewed support."
  }
}
```

Other stable codes should cover invalid artifact, incompatible contract, incomplete
coverage, stale quote, insufficient budget, incompatible checkpoint, denied access,
resource unavailability, non-finite predictions and infrastructure failure. Diagnostic
messages identify actionable causes without leaking other sessions, labels or host
internals. Never encode an invalid experiment as a plausible numeric score.

## 9. Data, evaluation and feedback

| Data role | Existing source | Gym behavior |
| --- | --- | --- |
| Train | ClimbMix training shards | Platform-controlled batches for weight updates |
| Dev | `shard_06541.parquet` | Candidate selection and research BPB feedback |
| Validation | `shard_06542.parquet` | Formal evaluation after final choice is frozen |
| CORE-22 | Separate public collection of 22 tasks | Formal capability evaluation |

These are roles, not four independent dataset sources. In the current implementation,
dev is still in the train file inventory; the official finite stream does not reach
it. The gym must explicitly implement dev selection and prevent a research profile
from training on its dev shard. Preserve the official training prefix/order for
formal-compatible profiles; reject horizons that invalidate that guarantee rather
than silently extending into dev or shuffling data.

Research profiles pin tokenizer, packing, train/dev identities, evaluation token count,
optimizer/schedule, supported dimensions and seed policy. User-selected stop steps
truncate a declared schedule; continuation does not reinterpret that schedule. The
service validates compatibility before restoring optimizer, RNG, data cursor/buffers,
stage, schedule and cumulative compute.

The scorer derives BPB from predictions, token-byte lengths and its labels. It does
not trust a candidate's loss or output JSON. Prediction workers receive evaluation
inputs, not the target-label array or unrestricted dataset files. Still, next-token
teacher-forcing inputs contain later tokens: withholding labels alone does not prove
causality. Enforce the causal model contract and test supported computation paths for
future-token access; sampled tests alone are not a complete proof against malicious
Python. This is part of the executable-contract release gate.

Return training loss, dev BPB, actual evaluated tokens, checkpoint step, training
tokens, cost components, timing, memory and errors. Tag every result with candidate,
profile, data, environment and accounting hashes. Formal labels, predictions,
per-example outputs and evaluation logs are not research-download artifacts. Any
post-final public artifact policy must remain consistent with official publication.

Public validation/CORE sets and published baseline scores are not secret. Offline
controls prevent this session from downloading them but cannot undo prior knowledge
in the researcher model. V1 claims controlled execution on a public benchmark,
not contamination-free hidden-test generalization.

## 10. Cost model and budget enforcement

### 10.1 Two different budgets

- **Research:** cumulative executed candidate-model FLOPs across all attempts and
  checks, capped by required `research_flops_limit`.
- **Final scoring:** a separately configured common allocation for fresh training,
  complete evaluation and bounded operational retries. It does not grant more search.

No numeric cap is selected here. Limits must be finite and explicit. Missing settings
block execution. Agent inference fees are TODO; CPU time for static analysis, metadata
and orchestration is tracked separately. Worker CPU/memory/time limits prevent
non-model workloads from becoming an unbounded resource channel. Unsupported numerical
model work cannot be relabeled as orchestration to escape accounting.

### 10.2 Report cost before spending

A cost report includes forward, backward, optimizer, auxiliary/recomputation and
applicable state-operation components; per-stage costs; the averaging horizon;
coverage status; and ratio to the SSSL anchor of 4.67239832544785 GFLOPs/input-token.
SSSL is the cost anchor, distinct from the full-attention Dense comparison baseline.

For a constant-cost training path, training cost is approximately
`training_flops_per_token * training_input_tokens`. The experiment also includes
checks and evaluation, so this product alone is not a total quote. Evaluation uses
its applicable forward path, not training FLOPs/token. Staged/dynamic paths require
stage-specific or dispatch-specific charges and a safe upper bound for reservations.

Displayed rates may be floating-point summaries. Authoritative ledger amounts use
exact integer units under the declared accounting convention; transmit large totals
as decimal strings to avoid JSON client precision loss. Static symbolic/FakeTensor
analysis is not permission to execute unmetered numerical candidate code.

### 10.3 Coverage before dispatch

Inventory allowed operations, dependencies and execution effects. A supported runtime
must check coverage and budget before dispatching actual work, including constructors,
initializers, debug checks and accelerated paths. Unknown operations fail before they
execute. Calibration cannot introduce a free dynamic benchmark loop.

An auditable reference path and numerical equivalence are necessary for proposed
custom kernels but are not sufficient to prove the actual kernel's cost or behavior.
V1 uses platform-vetted kernels; adding new native extensions is a reviewed runtime
capability, not a participant upload feature. All candidate branches permitted by the
contract must remain covered at execution. Hashes/traces from one sample input cannot
justify unchecked Python or alternate paths on another input.

If these controls cannot be enforced for a proposed plugin, reject that plugin contract
rather than relaxing the requirement. Exact implementation of the allowed Python
subset, operation mediation and causal constraints is a prerequisite for untrusted
participant access.

### 10.4 Reserve, charge, reconcile

Maintain `available = limit - spent - reserved >= 0`. Reserve atomically before
scheduling. Convert reservation into spent charge as trusted execution reports work;
checkpoints do not reset either value. Reserve a conservative bound for the next
indivisible operation/update, evaluation batch and work already in flight. Do not
start work that cannot fit.

Only trusted accounting can charge or release reservations. Use unique charge IDs
and transaction boundaries so duplicate events/retries cannot double-charge or refund.
Release unspent amounts after termination and reconciliation. In a crash, keep the
uncertain portion reserved. If exact recovery is impossible, a predeclared conservative
settlement policy must apply; do not fabricate an exact measurement or refund unknown
work. Publish infrastructure-failure status separately from candidate failure.

Cancellation stops at a bounded safe point or terminates the worker; already executed
work is charged. Differentiate `budget_stopped`, cancellation, numerical failure and
infrastructure failure. A session cannot acquire a fresh budget by creating another
job, candidate or checkpoint. Session allocation itself is an administrator operation.

## 11. Submission verification, separate final training and publication

### 11.1 Harbor verifier accepts a deliverable, not a performance score

One Harbor trial corresponds to a research session. The task's agent environment
contains only client/reference/workspace assets; a separate verifier receives an
explicit final receipt artifact and a trusted trial-to-session binding. The receipt
contains IDs, not authoritative scores. It is untrusted input until checked against
the server. Pin and test the Harbor task schema/runtime version used by the adapter.

The verifier checks:

- a single final choice exists for this trial's session and the caller owns it;
- candidate/configuration hashes and the declared contract match the frozen package;
- packaging, supported hooks and cost coverage are valid;
- required bounded construction, numerical, causal and checkpoint checks have valid
  server-issued reports for these exact candidate/runtime versions;
- intended final-profile admission evidence is present, not a claimed candidate cost;
- research jobs/charges are reconciled and no budget or integrity violation is recorded.

This is more than a syntax check, but it is not full training or validation/CORE
scoring. Required dynamic checks should have been performed through the research gym
and charged before freezing. The verifier reuses the immutable evidence rather than
importing untrusted Python or launching unaccounted checks. Missing evidence makes
verification fail with an actionable reason; there is no free post-budget validation
loop. Infrastructure failures are distinguished from an invalid artifact.

The verifier writes a submission-validity result and report. A numeric Harbor reward,
if required by the task adapter, means validity only, not architecture performance.
No placeholder BPB/CORE or successful leaderboard rank is generated. A verified
architecture may later train poorly; that does not retroactively make its package
invalid. Verifier retries are idempotent and cannot start expensive final training.

### 11.2 Final training is explicitly and separately dispatched

After verification, the submission waits in `not_scheduled`. An authorized operator
or platform scheduling policy invokes `launch_final_run` with the submission ID,
final profile, runtime/resource profile and separate finite allocation. The researcher
and Harbor verifier have no permission to call this mutation. Submission acceptance,
resource availability and a valid quote are not implicit launch authorization.

Once launched, the platform automatically provisions the runtime, initializes from
scratch, trains, evaluates, audits and exports. No participant-written launcher,
research checkpoint, manual weight transfer or self-reported metric is used. Recheck
that the approved environment and admission evidence remain compatible; if not,
report the problem rather than silently changing the architecture or scientific profile.

Operational retries retain exactly the frozen artifact and are recorded under a
finite retry/allocation policy. Failure never silently selects a different candidate
or becomes unlimited free research. Harbor/agent cleanup cannot delete the pinned
submission or cancel a separately authorized final run.

### 11.3 Publish measured results only

For official-compatible scoring, preserve:

- the fixed 49.152B input tokens, 46,875 updates, tokenizer, data order and seed 42;
- admission within 1% of the cost anchor and accepted cumulative deviation within 2%;
- all 189 periodic validation events and six complete CORE-22 events;
- the final checkpoint and separately reported independent final BPB;
- audit, complete provenance and result export checks.

Leaderboard values use the measured final-update periodic validation BPB and CORE,
with the independent final BPB retained as a separate field. Preserve official sorting
choices rather than inventing a composite score. Incomplete or mismatched runs retain
failure records and do not receive a successful rank.

A gym-produced result is not automatically accepted by upstream. Reuse the result
format/display locally; publication into the official upstream leaderboard still
requires its admission, review and result process. Short research profiles must never
be represented as official full-run results. Existing results and proofs are immutable.

## 12. Isolation verification and acceptance criteria

These are required validations, not completed security claims:

| Boundary | Acceptance evidence |
| --- | --- |
| Repository ownership | Candidate submissions cannot patch, shadow or replace platform/SDK modules |
| Filesystem and session separation | Traversal, symlinks and cross-session artifact/checkpoint access are rejected |
| Network and credentials | No worker/workspace arbitrary egress, host socket, instance credential or secret access |
| Workspace execution | Candidate computation cannot bypass gym via CPU, shell or an alternate tool |
| Scoring | Forged score files/losses cannot change metrics; labels and formal feedback do not leak through outputs |
| Causality and accounting | Supported paths reject future-token leakage and unknown work; evaluation-specific and alternate paths are covered |
| Scientific compatibility | Baseline checks reproduce reference loss, gradients, costs, packing and checkpoint continuation under declared tolerances |
| Budget concurrency | Simultaneous launches cannot overspend; duplicate requests/events do not duplicate launches or charges |
| Failure recovery | Crash/cancel/restart leaves no free computation, premature refund or lost running worker |
| Final selection | One frozen choice, fresh initialization, no research checkpoint reuse, no post-score reopening |
| Submission/final-run separation | Verifier can accept/reject a deliverable but cannot launch training; acceptance creates no performance score |
| Resource/asset handling | Read-only shared assets, session-scoped artifacts, fenced worker leases and no premature checkpoint eviction |

Correctness checks include finite outputs/gradients, parameter coverage, stage
boundaries and checkpoint round trips. Source-aware cost re-audits are required when
executed runtime/model behavior changes. Isolation and scientific equivalence must
both pass; success at one does not imply the other.

## 13. Implementation plan

1. **Schemas and read-only catalog.** Define candidate/session/profile/report formats,
   SDK signatures, resource IDs, artifact ingestion and immutable identities. Provide
   component templates. No numerical participant execution at this milestone.
2. **Trusted execution minimum.** Select and validate the offline sandbox, mediated
   workspace tools, supported operation/effect contract and accounting-before-dispatch.
   Add the ledger before exposing dynamic candidate checks. Unsupported paths refuse.
3. **Candidate validation and cost reports.** Adapt plugin hooks, coverage checks,
   named diagnostics and calibration. Reproduce known baseline costs and report repairable
   errors; distinguish static cost analysis from charged numerical checks.
4. **Dev-only research runner.** Implement explicit dev data selection, trusted scoring,
   bounded training and complete resume. Verify reference parity without altering
   historical proofs or silently using formal validation for selection.
5. **Persistent research service and task adapter.** Integrate resource descriptors,
   shared-asset manifests, quotes, reservations, leases, concurrency, events, cancellation,
   reconciliation and finalization. Demonstrate a two-candidate research loop and a
   separate Harbor verifier accepting its frozen submission without launching training.
6. **Separately dispatched final runner and publication.** Implement privileged final-run
   launch and automatic fresh scoring, audit and compatible export. Keep upstream
   acceptance separate. Exercise the pipeline with bounded fixtures before any
   independently authorized full run.
7. **Framework adapters.** Add CLI/MCP or framework-specific convenience adapters over
   the same application service. No transport-specific budgeting or scoring policy.

Start without a host listener when a mediated local transport suffices. If HTTP is
needed, bind port 0 on first launch while retaining the socket, persist the actual
service port, reuse it thereafter and report an occupied saved port explicitly.
Do not migrate an existing published address without instruction.

## 14. Open deployment settings and deferred work

Required before numerical execution:

- finite research/final FLOPs allocations, retry policy and conservative crash settlement;
- hardware allocation, concurrency, worker memory/time limits and supported dependencies;
- research profiles, evaluation cadence/token budgets and fixed schedule semantics;
- validated sandbox, workspace tool policy and supported executable/operation subset;
- artifact size limits, shared-asset manifests, cache/checkpoint retention rules and
  versioned comparison-cohort identity;
- pinned Harbor adapter/task schema, submission-validity reporting and separately
  authorized final-run scheduling policy.

These are implementation/configuration decisions, not reasons to obscure the proposed
product boundaries. No machine access or full training launch is authorized merely
by this document.

Deferred:

- arbitrary whole-model architecture contract beyond the shared shell, with parameter
  roles, initialization, checkpoint schema, calibration and new comparability rules;
- agent inference cost metering and hard limits;
- cross-architecture weight transfer, distillation and parameter inheritance;
- an explicitly separate internet-enabled research setting;
- optional genuinely private evaluation sets and additional task families.

## 15. Evidence and current implementation status

Existing behavior is grounded in:

- `nexttransformer/models/plugins.py`, `models/config.py` and `tests/plugins/`;
- `nexttransformer/protocol/spec.py`, `calibrate.py` and `fingerprint.py`;
- `nexttransformer/runtime/data.py`, `train.py` and `evaluation/formal.py`;
- `nanochat/loss_eval.py`, `audit_run.py` and `export_run.py`;
- `docs/src/content/docs/arena/rules.md` and
  `docs/src/content/docs/protocol/data-and-evaluation.md`;
- `docs/src/content/docs/guides/submit-a-result.md`.

RSI adapter seams were inspected in RSIHub's `evaluation/execution.py` and result
records, and FlashRSI's `core/algorithm.py` and snapshot/trajectory objects. Those
interfaces support candidate/result integration, but do not supply this gym's
training checkpoint semantics or experiment FLOPs enforcement.

This revision changes documentation only. API names, SDK packages, service objects,
sandbox guarantees and automated workflows above are proposed contracts, not runnable
features. No experiment, model evaluation or service deployment was performed.
