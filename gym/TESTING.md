# Testing the Gym

Run the maintained checks from the repository root:

```sh
uv sync --project gym --extra worker
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
PYTEST_XDIST_AUTO_NUM_WORKERS=2 gym/.venv/bin/python -m pytest -q -n auto
gym/.venv/bin/python -m nexttransformer.checks.changes --base origin/main
uvx --from ruff==0.14.10 ruff check --select F gym
uvx --from ruff==0.14.10 ruff format --check gym
```

The Gym suite covers native plugin compatibility, checkpoint continuation,
HTTP/MCP parity, session isolation, budget accounting, GPU allocation policy,
host listener persistence, Harbor mounts and signed submission verification.

For a bounded integration check using native Claude Code and a deterministic
local model stub, prepare the client and verifier images described in
[README.md](README.md), then run:

```sh
uv run --project gym python gym/smoke.py
```

This check makes no external model calls and uses synthetic CPU data. To test
numerical execution on an explicitly allocated GPU with a CUDA-enabled worker:

```sh
PYTHONPATH=gym/src:. gym/.venv/bin/python gym/gpu_smoke.py \
  --state-dir .agent-work/gpu-smoke \
  --gpu <allocated-index-or-uuid> \
  --worker-python <cuda-python>
```

GPU checks require the configured card to satisfy the allocation policy. Synthetic
fixtures validate integration, not production ClimbMix or CORE-22 performance.
Verifier reward measures submission validity. See [CONTRACT.md](CONTRACT.md) for
execution and accounting boundaries.
