# Validation record

Executed on 2026-11-05 against Harbor 0.24.0, task schema 1.4 and
PyTorch 2.9.1+cpu. Local tests used Python 3.12.15; container trials used
Python 3.12.13. Native research workers use unprivileged user/mount/PID/network
namespaces with no controller procfs mount. Runtime revisions bind the actual dependency and Python versions.

## Live Cosmos / DeepSeek Flash / Claude Code trial

A real provider trial on 2026-11-05 used Cosmos's advertised `deepseek-flash`
model ID with native Harbor Claude Code 2.1.312 and the prepared MCP client image.
Unlike the deterministic smoke below, the model autonomously read Gym resources,
budget, published experiment history and architecture sources through MCP, then
wrote native plugin packages and orchestrated experiments with `GymClient` from
its container. No oracle script or model stub directed its choices.

The model submitted two four-layer native block candidates with parallel and
sequential layouts and trained each for eight updates. Their synthetic validation
BPB values were 4.376051083992528 and 4.376077913286062 respectively. It selected
the parallel candidate, passed all five final-fixture checks and froze a valid
submission receipt. These tiny synthetic differences do not establish architecture
quality or formal benchmark performance.

Claude Code reported success after 178.368 seconds. Harbor reported no trial
exception, independent verifier reward **1**, and zero tool-result errors in the
native transcript. The training ledger consumed **756,568,941 of 1,000,000,000
FLOPs**, leaving **243,431,059 available** and **zero reserved**. The run used one
local CPU worker. It does not add GPU-container or production-data evidence.

The configured Agent timeout was 900 seconds, with `max_turns=32` and a requested
CLI USD limit of 3. Claude's terminal report recorded `num_turns=44`; both the
requested setting and the native reported counter are retained in the local run
report rather than treating the latter as an exact request cap. Claude reported
USD 2.175823 with pricing basis `unknown`; this is not a verified Cosmos bill.
The requested/reported model alias was `deepseek-flash`; the gateway's internal
upstream route was not independently inspected.

The retained local report is `.agent-work/cosmos-live/report.json`, alongside the
native trajectory, runtime snapshot, signed Gym evidence and submission receipt.
Candidate source/configuration and checkpoints remain in the trial's retained Gym
volume. This run needed no implementation fix. The original checkout was untouched.

## HTTP, MCP and CUDA follow-up

The final Gym suite passes **53 tests**. The repository suite passes **532 tests**,
with 131 skipped and one existing tensor conversion warning. Ruff checks, example
dry runs, whitespace checks and required change classification pass.

HTTP publishes the canonical operation schemas. Official MCP SDK stdio integration
checks that tool discovery matches those schemas and that HTTP/MCP share session
scope, budget, result and error behavior. GPU tests distinguish discovered,
allocated and available cards, reject unauthorized placement, recheck occupancy
after leasing, fail closed on discovery/runtime failure and preserve reservations
while busy jobs wait. Cancellation releases unexecuted reservations.

A native Harbor **Claude Code 2.1.312** trial completed with zero exceptions and
verifier reward **1**. The local Anthropic stub caused a native ToolSearch request,
a successful `mcp__gym__get_resources` call and a Bash-based research workflow
(two candidates, continuation, validation and submission). There were **four local
stub requests and zero external model calls**. This validates the native MCP
registration and HTTP path, not model research quality or provider routing.

Separately, an explicitly allocated idle **A100-SXM4-80GB** completed a bounded
host HTTP/MCP CUDA run under bubblewrap isolation. The controller used Python
3.12.15 and the read-only numerical environment used Python 3.10.12,
PyTorch **2.9.1+cu128**, CUDA runtime 12.8 and NVIDIA driver 535.54.03. The run
performed four optimizer updates plus numerical validation. The resumed two-step
checkpoint matched uninterrupted training bit-for-bit, including validation BPB.
All five numerical checks passed; this is not formal protocol admission.
The ledger charged **113,845,484 FLOPs**, with **zero reserved FLOPs** afterward.
Results recorded actual `cuda:0` model parameters and about 17.85 MB peak allocated
GPU memory, under a 2 GiB configured allowance. Resource discovery observed eight
cards, of which only one was authorized for this smoke; other jobs were untouched.

Reproduce with a deliberately allocated idle card and a pinned CUDA interpreter:

```sh
PYTHONPATH=gym/src:. gym/.venv/bin/python gym/gpu_smoke.py \
  --state-dir .agent-work/gpu-smoke --gpu <allocated-index-or-uuid> \
  --worker-python <cuda-python>
```

The native Harbor smoke used a CPU service image and a prepared client image with
the cached MCP 1.30.0 dependencies and pinned Claude executable. The ordinary client
Docker build was stopped after a stalled network wheel download; the prepared
image passed offline import/version checks and the full trial. No credentials
were copied into it. This host has no NVIDIA Docker runtime, so GPU sidecar execution
was **not** tested in Docker. The runner refuses GPU trials without that runtime
or a CUDA-enabled service image. No multi-node scheduling, paid model call,
production ClimbMix run or full CORE-22 final evaluation was performed.

## Native Claude Code and research-context follow-up

The preceding research-context Gym suite passed **33 tests**; the repository suite passes **532 tests**
with 131 skipped and one existing tensor conversion warning. Added checks cover
published history pagination, model cards and model/config provenance, exact source
references, session isolation, actual execution-node recording, cross-session
worker availability, native Claude Code options and explicit example budgets.
Both example configurations pass a dry run without creating containers or calling
a provider. Ruff formatting and undefined/unused-name checks pass.

The native Harbor **Claude Code 2.1.312** smoke completed with no trial exception
and verifier reward **1**. A deterministic local Anthropic Messages stub drove
one real Bash tool call through Claude Code. The tool read historical research
context and local resources, trained two candidates, continued a checkpoint,
validated the final choice and wrote the submission receipt. Harbor retained the
native trajectory, actual container runtime snapshot and signed Gym evidence.
Two stub model requests were made, with **zero external model calls**. The initial
run consumed 91,610,516 of its 1,000,000,000,000 research FLOPs, with zero reserved
FLOPs at verification. Published context contains 35 runs and 21 source references;
five historical methods have model cards but no configuration in the current tree.

Client/service/verifier images were built from locally available bases. The client
combines the Python runtime with the Node Bookworm base's CLI dependencies. For
this smoke, the already-installed pinned Claude executable was copied into a
separate client image; no host credentials were copied. This tests native CLI
execution, not the network installer or a real DeepSeek/gateway connection. The
service used the prepared CPU dependency base described below. At that earlier stage, only one local CPU server with one worker slot was tested.
The CUDA follow-up above adds bounded local GPU evidence. The local model stub persists and reuses its port.

Reproduce with `python gym/smoke.py --client-image <prepared-client-image>`;
the normal client image lets Harbor install the pinned CLI instead.

## Earlier native-plugin baseline

| Check | Observed result |
| --- | --- |
| Repository test suite, with two xdist workers | 531 passed, 131 skipped; one existing tensor conversion warning |
| Gym behavioral and integration tests | 26 passed |
| Ruff formatting and undefined/unused-name checks | Passed |
| Native Harbor Docker trial | One trial, zero exceptions, verifier reward 1 |
| Post-cleanup final fixture | Separately allocated fresh initialization succeeded using retained state and the original image |
| Native static final preparation | A 23-layer candidate passed calibration and admission; 125 source/config/proof files pinned |

After aligning Gym with native last-shard validation and confirming ID-based
delivery, the Gym suite was rerun: **29 passed**. The added coverage compares
training and validation batches with native `BestFitStream`, checks continuation,
allows shard 06541 in the full training prefix and rejects validation/training
overlap. Final-dispatch coverage recreates the service/backend from durable state
and completes a fresh CPU fixture run using the submission receipt, with matching
candidate identity. Formatting, undefined/unused-name checks and documentation
links also passed. After rebasing onto the current upstream main for PR submission,
the Gym suite again passed all 29 tests and the repository suite passed 532 tests
with 131 skipped and one existing tensor conversion warning. Change classification
against both origin/main and upstream/main reported no training-contract or
published-method fingerprint changes. The Docker results above are from the
earlier native-plugin baseline; no production ClimbMix or GPU run was performed
for this follow-up.

Reproduce the maintained checks from the repository root:

```sh
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
PYTEST_XDIST_AUTO_NUM_WORKERS=2 gym/.venv/bin/python -m pytest -q -n auto
gym/.venv/bin/python -m nexttransformer.checks.changes --base origin/main
uvx --from ruff==0.14.10 ruff check --select F gym
```

The Gym tests cover exact checkpoint continuation, native loss/gradient/cost and
packing parity, direct native model loading and serialized-config parity, atomic concurrent budget use,
idempotency, cross-session denial, stale quotes, checkpoint modification/removal,
queued/running cancellation, conservative crash recovery, signed evidence,
separate final dispatch, stable host ports, native Claude configuration, provider
credential exclusion, original source bytes (including CRLF), worker filesystem/network
isolation, and the actual Harbor task schema. The distributed adapter
test uses a two-rank Gloo CPU protocol probe to exercise file rendezvous and budget
hooks; it is not a GPU scientific acceptance test.

The Docker trial loads a native multi-file `ResearchModel` subclass with native
`method.json`, options and local modules. It performs bounded NextTransformer updates on explicitly
synthetic data: two candidates, checkpoint continuation, final checks, one frozen
submission, actual main-container process observation and separate offline
verification. Signed evidence showed zero reserved research budget and
`final_run_state = not_scheduled` before the independent controller dispatched the
final fixture. Afterward the fixture reported `fresh_initialization = true`,
`official_compatible = false`, and null formal BPB, CORE and rank. Verifier reward
means submission validity only.

The static 23-layer preparation used the native cost convention and measured
4,674,727,229.46476 theoretical training FLOPs/token, a relative deviation of
0.0004984386721109679 from the pinned dense anchor. Preparation ran no numerical
training and wrote only to a disposable staging checkout. No source translation or method-name rewriting was used.

Container builds used an offline prepared dependency base containing the pinned
installed worker packages, and bubblewrap 0.6.1, with `INSTALL_WORKER_DEPS=0` and
`INSTALL_SANDBOX_DEPS=0`; service/client/verifier
targets were built with networking disabled. The ordinary networked dependency
installation path is provided but was not the successful build path in this
environment. Runtime revisions include the actual installed packages.

No full 49.152B-token formal run, production GPU parity validation, upstream
result publication or arbitrary-Python sandbox certification was performed.
The separately recorded Cosmos follow-up above adds a real provider trial. The repository's shared trainer, model shell, data pipeline, protocol,
published results and Harbor dependency source were not modified. Production
deployment gates are listed in [CONTRACT.md](CONTRACT.md).
