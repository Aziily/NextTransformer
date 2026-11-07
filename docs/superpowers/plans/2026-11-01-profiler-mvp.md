# Profiler MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a protocol-preserving profiling launcher for the reference `nexttransformer.train` workload that records steady-state timing, GPU telemetry, optional Nsight artifacts, and a machine-readable summary.

**Architecture:** Add one standalone launcher under `launch/` that starts the existing trainer as a bounded research smoke, samples `nvidia-smi` by GPU UUID, and parses the trainer's existing `metrics.jsonl` after completion. The launcher writes a profile directory containing the child run, telemetry, command metadata, and `summary.json`; Nsight is optional and runs as an external wrapper, so the trainer and formal protocol remain unchanged.

**Tech Stack:** Python 3.10+, standard library (`argparse`, `json`, `subprocess`, `threading`, `csv`, `statistics`, `pathlib`), PyTorch distributed launcher invoked as a subprocess, `nvidia-smi`, optional `nsys`, pytest.

---

### Task 1: Define profile summary and failing tests

**Files:**
- Create: `launch/profile_train.py`
- Create: `tests/test_profile_train.py`

- [ ] **Step 1: Write tests for metrics parsing and steady-state summary**

Create pure tests that do not start GPUs. The parser must:

```python
rows = [
    {"event": "validation", "step": 0, "seconds": 0.7},
    {"event": "train", "step": 1, "step_seconds": 19.0,
     "data_seconds": 0.2, "optimizer_seconds": 18.0,
     "tokens_per_second": 800.0, "peak_memory_bytes": 10_000},
    {"event": "train", "step": 2, "step_seconds": 1.4,
     "data_seconds": 0.001, "optimizer_seconds": 0.4,
     "tokens_per_second": 12_000, "peak_memory_bytes": 11_000},
    {"event": "train", "step": 3, "step_seconds": 1.3,
     "data_seconds": 0.001, "optimizer_seconds": 0.39,
     "tokens_per_second": 12_300, "peak_memory_bytes": 11_100},
]
```

Assert that `summarize_metrics(rows, warmup_steps=1)` returns:

```python
{
    "warmup": {"steps": 1, "step_seconds": [19.0]},
    "steady": {
        "steps": 2,
        "step_seconds": [1.4, 1.3],
        "median_step_seconds": 1.35,
        "median_tokens_per_second": 12150.0,
        "data_seconds_total": 0.002,
        "optimizer_seconds_total": 0.79,
        "peak_memory_bytes": 11100,
    },
}
```

Add tests for missing train events and for malformed JSONL raising `ValueError` with the line number.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest -q tests/test_profile_train.py
```

Expected: FAIL because `launch.profile_train` and `summarize_metrics` do not exist.

- [ ] **Step 3: Commit the failing tests**

```bash
git add launch/profile_train.py tests/test_profile_train.py
git commit -m "test: define profiler summary contract"
```

### Task 2: Implement the pure summary and telemetry helpers

**Files:**
- Modify: `launch/profile_train.py`
- Test: `tests/test_profile_train.py`

- [ ] **Step 1: Implement JSONL parsing and steady-state summary**

Add these functions with stable signatures:

```python
def read_metrics(path: Path) -> list[dict]:
    """Read trainer metrics JSONL and report malformed line numbers."""


def summarize_metrics(rows: list[dict], warmup_steps: int) -> dict:
    """Split train events into warmup and steady windows and aggregate timing fields."""
```

`read_metrics` must ignore blank lines, parse each nonblank line with `json.loads`, and raise `ValueError(f"invalid metrics JSON at line {line_no}")` on malformed input. `summarize_metrics` must select rows where `event == "train"`, require at least `warmup_steps + 1` rows, use the first `warmup_steps` as warmup and the remaining rows as steady, preserve per-step arrays, calculate median step time/throughput, total data and optimizer time, and take the maximum steady peak memory. It must reject `warmup_steps < 0` and missing required numeric fields with a clear `ValueError` naming the field.

- [ ] **Step 2: Implement UUID telemetry sampling**

Add:

```python
def sample_nvidia_smi(output_path: Path, stop: threading.Event, period_ms: int = 200) -> None:
    """Sample GPU UUID, memory, utilization, power, temperature, and SM clock until stop is set."""
```

Run `nvidia-smi` with query fields `index,uuid,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm`, write one JSON object per line with a Unix timestamp, and terminate the child process when `stop` is set. If `nvidia-smi` is unavailable, raise a clear `RuntimeError` before starting the trainer. Add:

```python
def summarize_telemetry(path: Path, interval_start: float, interval_end: float) -> dict:
    """Aggregate samples during the steady interval by GPU UUID."""
```

The summary must include per-GPU sample count, mean/min/max utilization, peak memory MiB and percent, mean power, and mean SM clock. Use UUID as the stable key and return an empty list when no samples fall inside the interval.

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest -q tests/test_profile_train.py
```

Expected: PASS.

- [ ] **Step 4: Commit the pure helper implementation**

```bash
git add launch/profile_train.py tests/test_profile_train.py
git commit -m "feat: add profiler summary and telemetry helpers"
```

### Task 3: Implement bounded trainer launcher and optional Nsight wrapper

**Files:**
- Modify: `launch/profile_train.py`
- Test: `tests/test_profile_train.py`

- [ ] **Step 1: Add CLI arguments and command construction**

Support:

```text
--site-config PATH                 required
--config PATH                      required
--output PATH                      required profile directory
--micro-batch INT                  required
--warmup-steps INT                 default 1
--steady-steps INT                 default 2
--smoke-accumulation INT           default 1
--telemetry-period-ms INT          default 200
--nsys                            optional flag
--nsys-output PATH                 optional, defaults to <output>/trace
--keep-run                         keep the child run directory, default true
```

Construct the child trainer command exactly as:

```text
python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node <world_size>
  -m nexttransformer.train
  --site-config <site_config>
  --config <config>
  --output <output>/run
  --micro-batch <micro_batch>
  --smoke-steps <warmup_steps + steady_steps>
  --smoke-accumulation <smoke_accumulation>
  --log-every 1
  --rolling-every 1
  --wandb-mode disabled
```

Read `world_size` from `--world-size` with default 8, and pass `CUDA_VISIBLE_DEVICES` unchanged unless the caller supplies it. Record the exact command, environment fields relevant to reproducibility (`CUDA_VISIBLE_DEVICES`, `NANOCHAT_DTYPE`, `OMP_NUM_THREADS`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`), Git commit, Python/Torch/CUDA versions, config path, and warmup/steady counts in `metadata.json`.

- [ ] **Step 2: Add process lifecycle and output files**

The launcher must:

1. Refuse an existing nonempty profile directory unless `--overwrite` is supplied.
2. Create `<output>/run`, `<output>/metadata.json`, `<output>/telemetry.jsonl`, and `<output>/profile.log`.
3. Start telemetry before the trainer.
4. Start the trainer in a new process group and tee combined stdout/stderr to `profile.log`.
5. Stop telemetry after the trainer exits.
6. Return the trainer exit code; write `summary.json` only when the trainer succeeds and `metrics.jsonl` exists.
7. Mark `summary.json` with `status: "passed"` or `status: "failed"`, the child exit code, `metrics_summary`, `telemetry_summary`, and `nsys` artifact paths.
8. Never invoke `--formal`; reject a user-supplied formal flag because this tool is research-only.

- [ ] **Step 3: Add optional Nsight wrapping**

When `--nsys` is set, validate `nsys` with `shutil.which` and prepend:

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
  --cuda-memory-usage=true --stats=true --force-overwrite=true
  --output <output>/trace
```

Do not request `--trace=nccl`; the installed Nsight version rejects it, while NCCL CUDA kernels still appear in `cuda_gpu_kern_sum`. Record `.nsys-rep`, `.sqlite`, and `profile.log` paths in the summary when they exist.

- [ ] **Step 4: Add CLI-level tests with mocked subprocesses**

Mock `subprocess.Popen`, `shutil.which`, `nvidia-smi`, and filesystem paths to assert:

- the child command contains `--smoke-steps 3` for warmup 1 + steady 2;
- `--formal` is never generated;
- `--nsys` uses only `cuda,nvtx,osrt`;
- a nonzero child exit produces failed summary metadata;
- existing output is refused without `--overwrite`.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
pytest -q tests/test_profile_train.py
```

Expected: PASS.

```bash
git add launch/profile_train.py tests/test_profile_train.py
git commit -m "feat: add bounded training profiler launcher"
```

### Task 4: Document and run the MVP on 3090

**Files:**
- Create: `docs/src/content/docs/guides/profiling.md`
- Modify: `docs/src/content/docs/index.mdx` or the existing guides navigation file that lists guides
- Test: `tests/test_profile_train.py`

- [ ] **Step 1: Document the workflow**

Document the exact command for the reference workload:

```bash
python launch/profile_train.py \
  --site-config site.local.json \
  --config configs/baseline/dense_full.json \
  --output /path/to/runtime/runs/profile-dense-full \
  --world-size 8 \
  --micro-batch 1 \
  --warmup-steps 1 \
  --steady-steps 2 \
  --nsys
```

Explain that the output is research-only, uses smoke steps, does not produce formal results, and that `.nsys-rep`/`.sqlite` open in Nsight Systems. Explain the summary fields and why warmup and steady windows are separated.

- [ ] **Step 2: Run the complete local tests**

Run:

```bash
pytest -q -n auto
```

Expected: existing fast suite remains green and the profiler tests pass.

- [ ] **Step 3: Copy the MVP commit series to 3090 and run it**

On 3090, fetch the branch and run the documented command with the existing ignored `site.local.json`, using a fresh runtime output directory. Do not run `--formal`; do not overwrite the existing smoke or Nsight directories.

Expected artifacts:

```text
summary.json
metadata.json
metrics.jsonl or run/metrics.jsonl
telemetry.jsonl
profile.log
trace.nsys-rep
trace.sqlite
```

- [ ] **Step 4: Validate the generated summary**

Check:

```bash
python -c 'import json; print(json.load(open("summary.json"))["status"])'
```

Expected: `passed`.

Verify `metrics_summary.steady.steps == 2`, at least one GPU appears in `telemetry_summary.per_gpu`, and the Nsight files exist when `--nsys` is used. Record the observed median step time, optimizer time, throughput, peak memory, and whether NCCL kernels appear in the Nsight stats.

- [ ] **Step 5: Commit documentation and final verification**

```bash
git add docs/src/content/docs/guides/profiling.md docs/src/content/docs/index.mdx tests/test_profile_train.py
git commit -m "docs: document training profiler workflow"
```

Run:

```bash
git status --short
```

Expected: clean working tree. Keep runtime traces, checkpoints, datasets, local configs, and machine paths outside the repository.
