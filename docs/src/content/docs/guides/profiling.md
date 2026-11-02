---
title: Profile a run
description: Find where a training update spends its time, from stages down to operators and kernels, and estimate a formal run.
---

The profiler runs a few bounded research updates of the reference trainer and writes one self-contained HTML report
and a machine-readable `summary.json`. It answers three questions: where an update spends its time, what each GPU was
doing meanwhile, and how long a formal run would take on this machine.

The profiler is research-only. It always runs smoke updates, refuses `--formal`, and changes no training math: the
stage timer only records clocks and CUDA events around the trainer's existing functions.

## Run the profiler

From the repository root, point `--site-config` at the ignored machine configuration:

```bash
python launch/profile_train.py \
  --site-config site.local.json \
  --config configs/baseline/dense_full.json \
  --output /path/to/runtime/runs/profile-dense-full \
  --world-size 8 \
  --micro-batch 1 \
  --warmup-steps 1 \
  --steady-steps 2 \
  --trace torch
```

By default each update has the formal shape for the chosen micro-batch: with micro-batch 1 on 8 ranks that is 64
micro-batches and 1,048,576 tokens. Pass `--smoke-accumulation N` for a quicker, smaller update; the report then
warns that gradient communication and the optimizer, which run once per update, take a larger share than in formal
training. Only the final update writes a checkpoint.

| Option | Effect |
|---|---|
| `--warmup-steps` | Updates excluded from measurements (compilation, communicator setup) |
| `--steady-steps` | Updates whose time and throughput the report measures |
| `--trace torch` | Traces `--trace-steps` extra updates after the steady ones with the PyTorch profiler on `--trace-ranks` (rank 0 by default): stages, operators with input shapes, and kernels |
| `--trace nsys` | Traces the same updates with Nsight Systems on every rank: stages and kernels per GPU (`--nsys` is an alias) |
| `--smoke-accumulation` | Micro-batches per update (default: the formal update) |
| `--validation-tokens` | Tokens of each smoke validation (default: one batch). Set it to 41943040, the formal size, to include validation in the formal run estimate |

Traced updates run after the steady ones, so tracing overhead never enters the measured step time.

## Read the report

Open `report.html` in a browser. It needs no server or network. Sections, top to bottom:

- **What to look at first** lists findings with the evidence behind each: an update shape that differs from formal
  training, communication not hidden behind compute, idle GPUs, data stalls, ranks that disagree on a stage, many tiny
  kernels, low utilization on some GPUs, or memory close to full.
- **Steady state** shows median update time, throughput, peak memory and the update shape, with the audited model
  FLOPs per second and GPU.
- **Formal run estimate** appears when the profile ran the formal update shape: training updates, validation events
  (when `--validation-tokens` measured at least an eighth of the formal size), and rolling checkpoints. CORE-22 evaluation is not measured.
- **Where an update spends its time** splits a steady update into data, forward, backward, gradient norm, host wait
  and the optimizer's reduce, compute and gather phases, timed with CUDA events on every rank. Expand a stage for
  per-rank times.
- **Wall time of this profile** separates startup, warmup, steady and traced updates, evaluation and checkpoints.
- **GPU activity in traced updates** shows how much of each traced GPU's time ran compute only, compute overlapped
  with communication, communication alone, memory copies, or nothing.
- **Operators and kernels** is a collapsible tree: stage, kernel category, operator (with input shapes, from the
  PyTorch trace), kernel. Each kernel belongs to the stage that launched it, so an asynchronous collective appears
  where it was issued.
- **GPU × time heatmap** shows sampled utilization, memory, power or idle ratio for every GPU over the whole profile,
  with a strip marking each phase. Per-GPU means and peaks cover the steady updates only.

## Output files

```text
summary.json       everything the report shows, for scripts and agents
report.html        the self-contained report
metadata.json      command, environment, versions, and Git commit
stages/            per-rank stage timelines; stages/torch/ holds PyTorch traces (open in Perfetto)
telemetry.jsonl    sampled GPU telemetry keyed by UUID
profile.log        trainer and profiler output
run/               bounded trainer artifacts and metrics.jsonl
trace.nsys-rep     Nsight Systems report, with --trace nsys
trace.sqlite       its SQLite export, which the report reads
```

Compare steady-state values across runs of the same shape and trace mode only. Keep profile directories, traces,
checkpoints, datasets and local site files outside Git, and use a fresh output directory for each run unless you pass
`--overwrite`.
