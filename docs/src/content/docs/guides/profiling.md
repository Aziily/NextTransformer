---
title: Profile a training run
description: Measure reference training warmup, steady-state timing, GPU telemetry, and optional Nsight traces.
---

Use the profiler for bounded research runs on a prepared machine. It wraps the existing reference trainer, keeps compiler warmup separate from steady-state steps, samples each GPU by UUID, and can save an Nsight Systems trace.

The profiler is research-only. It always launches smoke steps and never adds `--formal`; its output cannot become a leaderboard result.

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
  --nsys
```

The command runs three bounded updates: one warmup update and two measured updates. It uses the same trainer, data, tokenizer, optimizer, and protocol configuration as a normal research smoke. It does not write a formal result.

The output directory contains:

```text
summary.json       timing and telemetry summary
report.html        static human-readable report generated from summary.json
metadata.json      command, environment, versions, and Git commit
telemetry.jsonl    sampled GPU telemetry keyed by UUID
profile.log        trainer and profiler output
run/               bounded trainer artifacts and metrics.jsonl
trace.nsys-rep     Nsight Systems report, when --nsys is available
trace.sqlite       SQLite export of the Nsight report, when --nsys is available
```

Open `.nsys-rep` in Nsight Systems. The installed Nsight versions may reject `--trace=nccl`; the profiler requests CUDA, NVTX, and OS runtime tracing, and NCCL kernels still appear in the CUDA kernel report.

## Read the summary

`summary.json` separates `metrics_summary.warmup` from `metrics_summary.steady`. The steady section reports median step time and throughput, total data and optimizer time, and peak allocated memory. The `telemetry_summary.per_gpu` entries report utilization, memory, power, temperature, and SM clock for each GPU UUID during the profile interval. Heatmap v2 also stores a compact `telemetry_summary.samples` list with relative time offsets, so `report.html` can show a GPU-by-time grid. The report has a metric switch for GPU utilization, memory used percentage, power, and the derived wait/idle ratio; cells with no sample remain unavailable. A low-utilization warning is based on the captured telemetry, not on inferred kernel behavior.

The report includes a compact training timeline. Data, forward/backward, gradient clipping, optimizer reduce/compute/gather, and synchronization stages are shown as `unavailable` unless the trainer emits stage timings. GPU telemetry alone does not identify optimizer or NCCL phases. The page is a self-contained local file with inline HTML, CSS, and JavaScript; it has no server, JavaScript bundle, or external assets.

Compare steady-state values across runs. The first update often includes Inductor/Triton compilation and communicator setup, so it should not be used as the training throughput baseline.

Keep profile directories, traces, checkpoints, datasets, and local site files outside Git. Use a fresh output directory for each run unless you explicitly pass `--overwrite`.
