# Validation record

Executed on 2026-11-04 against Harbor 0.24.0, task schema 1.4 and
PyTorch 2.9.1+cpu. Local tests used Python 3.12.15; container trials used
Python 3.12.13. Runtime revisions bind the actual dependency and Python versions.

| Check | Observed result |
| --- | --- |
| Repository test suite, with two xdist workers | 531 passed, 131 skipped; one existing tensor conversion warning |
| Gym behavioral and integration tests | 25 passed |
| Ruff formatting and undefined/unused-name checks | Passed |
| Native Harbor Docker trial | One trial, zero exceptions, verifier reward 1 |
| Post-cleanup final fixture | Separately allocated fresh initialization succeeded using retained state and the original image |
| Native static final preparation | A 23-layer candidate passed calibration and admission; 123 source/config/proof files pinned |
| Over-budget final preparation | A 24-layer candidate was rejected without silently changing its architecture |

Reproduce the maintained checks from the repository root:

```sh
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
PYTEST_XDIST_AUTO_NUM_WORKERS=2 gym/.venv/bin/python -m pytest -q -n auto
gym/.venv/bin/python -m nexttransformer.checks.changes --base origin/main
uvx --from ruff==0.14.10 ruff check --select F gym
```

The Gym tests cover exact checkpoint continuation, native loss/gradient/cost and
packing parity, trusted generated-plugin parity, atomic concurrent budget use,
idempotency, cross-session denial, stale quotes, checkpoint modification/removal,
queued/running cancellation, conservative crash recovery, signed evidence,
separate final dispatch, stable host ports, mediated action rejection, provider
credential exclusion, and the actual Harbor task schema. The distributed adapter
test uses a two-rank Gloo CPU protocol probe to exercise file rendezvous and budget
hooks; it is not a GPU scientific acceptance test.

The Docker trial performs real bounded NextTransformer updates on explicitly
synthetic data: two candidates, checkpoint continuation, final checks, one frozen
submission, actual main-container process observation and separate offline
verification. Signed evidence showed zero reserved research budget and
`final_run_state = not_scheduled` before the independent controller dispatched the
final fixture. Afterward the fixture reported `fresh_initialization = true`,
`official_compatible = false`, and null formal BPB, CORE and rank. Verifier reward
means submission validity only.

The static 23-layer preparation used the native cost convention and measured
4,674,661,693.46476 theoretical training FLOPs/token, a relative deviation of
0.00048441247069686 from the pinned dense anchor. Preparation ran no numerical
training and wrote only to a disposable staging checkout. The rejected 24-layer
example exceeded the anchor by approximately 4.12%.

Container builds used an offline prepared dependency base containing the pinned
installed worker packages, with `INSTALL_WORKER_DEPS=0`; service/client/verifier
targets were built with networking disabled. The ordinary networked dependency
installation path is provided but was not the successful build path in this
environment. Runtime revisions include the actual installed packages.

No full 49.152B-token formal run, production GPU parity validation, live paid LLM
call, upstream result publication or arbitrary-Python sandbox certification was
performed. The repository's shared trainer, model shell, data pipeline, protocol,
published results and Harbor dependency source were not modified. Production
deployment gates are listed in [CONTRACT.md](CONTRACT.md).
