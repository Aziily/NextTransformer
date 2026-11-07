---
title: Training metrics
description: What the trainer writes during a run, and optional experiment tracking.
---

Every run directory is self-contained. The trainer writes JSON files that the audit and the exporter read; nothing
depends on a tracking service.

| File | Contents |
|---|---|
| `metrics.jsonl` | One event per line: `train`, `validation`, `core`, `final_bpb`, `budget_anomaly` |
| `provenance.json`, `resume_<step>.json` | The configuration, fingerprints, software and batch layout of each training segment |
| `run_status.json`, `budget.json` | Phase, final status and the cumulative compute ledger |
| `val_<step>.json`, `core_<step>.json`, `core_final.json`, `bpb_final.json` | Evaluation results at the scheduled points |
| `checkpoint_<step>/`, `latest.json` | Checkpoints every 5,000 updates and at the end |

`train` events carry the optimizer step, input tokens, loss and auxiliary loss, gradient norm, learning-rate
multiplier and the cumulative theoretical FLOPs with their deviation from the admitted proof; methods add their own
diagnostics (sampled depth, router statistics, sparse selection). Training logs every 10 updates by default
(`--log-every`), plus the first and the last.

## Optional tracking

Weights & Biases logging is off by default (`--wandb-mode disabled`). `offline` and `online` modes log numbers,
the scientific configuration and provenance from rank 0; models, data, source and console text are never uploaded.
`--wandb-metrics core` uploads a compact profile and `full` every numerical field. Keys carry event prefixes such as
`train/loss`, `validation/val_bpb`, `core/core_metric` and `final_bpb/val_bpb`. Credentials come from the usual
`WANDB_API_KEY` or a mounted netrc file, never from configuration files.

## Watch runs locally

`launch/run_monitor.py` serves a live view of run directories on this machine. It reads the files above again on every
request, never writes into a run directory and never takes its lock, and imports neither torch nor the trainer, so it
can run beside training or on another machine that sees the same files.

```bash
python launch/run_monitor.py /path/to/runtime/runs            # open http://127.0.0.1:8765
python launch/run_monitor.py /path/to/runtime/runs --once site # static pages instead of a server
```

Each argument is a run directory or a folder that contains run directories. The overview lists every run with its
status, progress, latest loss, throughput and estimated time left, and selected runs can be compared on one page. A run
page shows progress against the run's own validation and CORE schedule, the training curves, evaluations next to the
published curve of the same method, throughput, memory, the compute ledger against its tolerance, and health findings:
non-finite values, ledger drift, budget anomalies, loss or gradient-norm spikes, slowing updates and runs that stopped
writing events.

The trainer reports MFU only for GPUs in its peak table; pass `--peak-tflops` with the dense BF16 peak per GPU to see it
for other GPUs. The server listens on `127.0.0.1` by default. Reach a remote machine through an SSH tunnel
(`ssh -L 8765:127.0.0.1:8765 HOST`) rather than `--host 0.0.0.0`: the pages show run paths and configurations.
The time-left estimate covers updates, validations and measured CORE events, not checkpoints.
