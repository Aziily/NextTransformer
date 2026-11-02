---
title: Training on 24 GiB GPUs
description: What one formal run costs on eight 24 GiB Ampere GPUs, where the time goes, and which settings help.
---

Eight 24 GiB Ampere GPUs (sm86, 71 TFLOPS dense BF16 with FP32 accumulation) run a full-context forward and
backward for all but two registered configurations, and at least one more does not train (see
[method coverage](#method-coverage-on-24-gib-ampere)). These numbers come from bounded research-mode runs of
`configs/baseline/dense_full.json` on one node with NVLink pairs (0-1, 2-3, 4-5, 6-7) and PCIe between pairs,
PyTorch 2.9.1, CUDA 12.8 and NCCL 2.27.5. They are estimates, not a formal run.

## Measure the formal update

A formal update is 1,048,576 tokens: micro-batch 1 needs 64 micro-batches per rank, micro-batch 2 needs 32. Smoke runs
with one micro-batch per update overstate communication, because DDP synchronizes once per update:

| Micro-batch × accumulation | Tokens per update | Update | Tokens/s |
|---|---:|---:|---:|
| 1 × 1 | 16,384 | 1.35 s | 12,137 |
| 1 × 8 | 131,072 | 2.97 s | 44,111 |
| 1 × 64 (formal) | 1,048,576 | 16.12 s | 65,032 |
| 2 × 32 (formal) | 1,048,576 | 14.54 s | 72,135 |

Measure with `launch/mapping_trials.py --smoke-accumulation 64` (or `--micro-batch 2 --smoke-accumulation 32`); it
writes only the final checkpoint unless `--rolling-every` asks for more.

## Estimated cost of one formal run

| Part | Micro-batch 1 | Micro-batch 2, eval batch 8 |
|---|---:|---:|
| 46,875 updates | 210.0 h | 189.3 h |
| 189 validation events (41.9M tokens each) | 10.8 h | 8.6 h |
| 6 CORE-22 events (about 14.4 min each) | 1.4 h | 1.4 h |
| 94 checkpoints every 500 updates (9.2 GB, about 18 s each) | 0.5 h | 0.5 h |
| **Total** | **222.8 h (9.3 days)** | **199.9 h (8.3 days)** |

With the audited 4.6747 GFLOPs per input token, micro-batch 2 reaches about 42 TFLOPS per GPU, 59% of the 71 TFLOPS
dense BF16 peak (54% at micro-batch 1).

## Where the time goes

An Nsight Systems trace of one micro-batch-1 formal update (16.07 s on the traced rank):

| Part | Time | Share |
|---|---:|---:|
| Compute kernels | 14.51 s | 90% |
| NCCL kernels | 1.26 s, of which 1.03 s not overlapped | 8% |
| GPU idle | 0.53 s | 3% |

- GEMMs (CUTLASS and cuBLAS BF16) take about 8.5 s; FlashAttention forward and backward about 1.3 s.
- Unfused elementwise kernels take about 3.9 s: residual additions, scalar multiplications, BF16 casts and copies.
  Each micro-batch launches about 3,000 kernels (195,000 per update), but the GPU idles only 3% of the update, so CPU
  launch overhead and CUDA graphs are not the bottleneck. Fusing these kernels would help, but it changes rounding.
- The last micro-batch's gradient all-reduce is 0.71 s that backward does not hide; the optimizer step takes 0.41 s,
  mostly parameter all-gathers (0.39 s) around 0.08 s of Muon and AdamW compute.
- GPUs draw 325-345 W of their 350 W limit at about 1.9 GHz, below the 2.1 GHz maximum.

## Settings

| Setting | Effect | Note |
|---|---|---|
| `--micro-batch 2` | +11% training throughput | Needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.9.1 warns that the name is deprecated, but the new `PYTORCH_ALLOC_CONF` does not enable expandable segments there); peak memory is 23.4 of 23.6 GiB, and without the setting update 3 runs out of memory |
| `--eval-batch 8` | Validation 207 s → 165 s | Same rows and tokens; BPB agrees to 5e-7 |
| `NCCL_PROTO=Simple`, `NCCL_ALGO=allreduce:tree` | No measurable change | `NCCL_ALGO=Tree` fails: NCCL has no Tree all-gather |
| GPU order, `--numa-binding node` | No change or slower | |

Micro-batch and evaluation batch are per-GPU layout choices of the [training protocol](../../protocol/training/);
they change reduction order, not the update. Record them with the run.

## Attention backend and acceleration packages

Most methods use the SSSL window pattern: 18 of 24 layers attend to a 512-token window. Without FlashAttention-3,
`nanochat/flash_attention.py` computes those layers with SDPA and a dense boolean mask, about 6 times slower than full
causal attention (6.3 ms against 1.1 ms per layer, forward and backward, at 2,048 tokens). The
`kernels-community/flash-attn3` build for sm80 runs on sm86, but the locked `kernels` 0.17.2 requires a version
argument that `flash_attention.py` does not pass, so the trainer always falls back to SDPA. Enabling it is a
[shared-code change](../changing-shared-code/): it changes the recorded `native_attention` and the numerics.

Measured at the formal update (research mode, bounded runs):

| Candidate | `configs/others/dense.json` (SSSL), micro-batch 1 | `dense_full`, micro-batch 2 | Protocol impact |
|---|---:|---:|---|
| SDPA (as installed) | 22.7 s | 14.5-14.7 s | Baseline |
| FlashAttention-3 hub build | 16.2 s (-29%) | 14.5 s (no change) | Attention numerics; `native_attention: fa3`; backward not deterministic |
| `torch.compile` of the model | | 12.8 s (-13%), 1.6 GiB less memory | Fused numerics: loss differs by 0.16 after 8 updates; trainer change |
| NCCL 2.32.3 | | No change | Reduction order only |
| PyTorch 2.14.1 + CUDA 13.0 | | No change | Different PyTorch build; stale cost evidence |
| fla-core 0.6.0 | | | Fails: the linear backend imports an API that 0.6.0 removed |

SSSL methods do not fit micro-batch 2 in 24 GiB with either attention backend. Liger's fused linear cross-entropy
supports the logit softcap and matches the loss to 5e-7, but it is slower than `torch.compile` for the head and needs a
model change. FP8 paths (Transformer Engine) need Hopper or newer.

## Method coverage on 24 GiB Ampere

A full-context forward and backward (`nexttransformer.checks.validate --mode target --sequence-length 2048`) passes
for every registered configuration except:

| Methods | Problem | Workaround |
|---|---|---|
| kda, kda_no_ve, kda_hybrid, retnet, gdn, gdn_hybrid, gdn_hybrid_swa, deltanet_hybrid, qsa_hybrid | `No module named 'fla'` | Install `requirements/linear.txt` (fla-core 0.5.2); all nine then pass |
| dsa, moba, moba_hybrid, nsa, qsa, seer with `--sparse-backend kernel` | FlexAttention template needs 114,688 bytes of shared memory; sm86 has 101,376 | Use the default reference backend. The FlexAttention option `fwd_num_stages=2` fits sm86: for qsa_hybrid at micro-batch 1 it takes 21.1 s per update in burn-in and 20.7 s in joint (18x and 5.2x faster than the reference backend, about 11 days per run), with the same loss and gradient norm in each stage. The trainer does not expose it; runs past update 2 in burn-in and the other five methods are not checked |
| gla, deltanet | Out of memory at micro-batch 1 | Not runnable on 24 GiB without a memory-saving kernel |
| qsa_hybrid in training (sparse reference backend) | Passes the check at 11.9 GiB, but a micro-batch-1 formal run in the burn-in stage (the first 512 updates) runs out of memory at update 3 with expandable segments: 22.5 GiB allocated and 0.2 GiB reserved but unused, so the cause is real growth, not fragmentation. One update takes 387 s in burn-in and 107 s in the joint stage; a full run would take about 60 days | Not runnable on 24 GiB |

The check holds no optimizer state or DDP buckets: `dense_full` peaks at 9.5 GiB there and at 17.7 GiB in training,
and qsa_hybrid passes the check at 11.9 GiB but does not train. Passing the check is not evidence that a method trains;
run a bounded formal-shape smoke (`--smoke-accumulation 64`, a few updates) before a formal run.
