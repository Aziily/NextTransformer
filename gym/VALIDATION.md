# Validation record

Executed on 2026-11-05 against Harbor 0.24.0, task schema 1.4 and
PyTorch 2.9.1+cpu. Local tests used Python 3.12.15; container trials used
Python 3.12.13. Native research workers use unprivileged user/mount/PID/network
namespaces with no controller procfs mount. Runtime revisions bind the actual dependency and Python versions.

## Native Claude Code and research-context follow-up

The current Gym suite passes **33 tests**; the repository suite passes **532 tests**
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
service used the prepared CPU dependency base described below. No remote training
server or GPU research capacity is claimed: the observed backend is one local
server with one worker slot. The local model stub persists and reuses its port.

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

No full 49.152B-token formal run, production GPU parity validation, live paid LLM
call, upstream result publication or arbitrary-Python sandbox certification was
performed. The repository's shared trainer, model shell, data pipeline, protocol,
published results and Harbor dependency source were not modified. Production
deployment gates are listed in [CONTRACT.md](CONTRACT.md).
