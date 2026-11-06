---
title: Protocol overview
description: The fixed training contract every NextTransformer method is trained and evaluated under.
---

NextTransformer asks one question: **when the training budget is fixed, how should a language model spend its
computation?** Every method trains on the same data stream, for the same number of input tokens, at the same
theoretical training FLOPs per token, and is evaluated at the same checkpoints. This section is the contract. The
English text is normative; the machine-readable values live in
[`nexttransformer/protocol/spec.py`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/nexttransformer/protocol/spec.py),
and every configuration and run record carries the protocol's fingerprint.

## The lifecycle of a method

```text
 implement            calibrate             admit               train                  evaluate              audit             publish
 a plugin or   ──▶  compute to within  ──▶  the config  ──▶  46,875 updates     ──▶  189 × val BPB   ──▶  cumulative     ──▶  run record
 built-in method     1% of the SSSL         (cost proof      49.152B tokens          6 × CORE-22          compute ≤ 2%       + audit
                     anchor                 matches source)  seed 42, --formal       final BPB            of the anchor      on the site
```

1. **Implement** the method in the shared shell ([model shell](../model-shell/)).
2. **Calibrate** its structure so that its audited training FLOPs per input token are within 1% of the SSSL anchor
   ([compute accounting](../compute/), [calibration and admission](../admission/)).
3. **Admit** the configuration: its cost proof must match the code it runs and the PyTorch build it was audited with.
4. **Train** under the fixed data, batch, schedule and seed ([training](../training/)).
5. **Evaluate** at the fixed points ([data and evaluation](../data-and-evaluation/)).
6. **Audit** the finished run: every event present and the cumulative compute within 2% ([validation and resume](../validation/)).
7. **Publish** the run record and its audit ([results](../results/)).

## What is controlled

| Control | Value |
|---|---|
| Data | ClimbMix at a pinned revision; identical files, content and order; one fixed validation shard |
| Tokenizer | One prepared 32,768-token vocabulary and byte table, pinned by SHA-256 |
| Context | 2,048 tokens, best-fit packing with BOS |
| Budget | **49,152,000,000 input tokens**, including warmup, burn-in and every training stage |
| Batch and updates | **1,048,576 tokens per update** (512 × 2,048-token rows), **46,875 updates** |
| Randomness | One seed, **42** |
| Compute | **4.67239832544785 GFLOPs per input token** (the SSSL anchor); ≤ 1% at admission, ≤ 2% cumulative |
| Optimizer | Muon for block matrices, AdamW for embeddings, head and scalars; one shared schedule |
| Evaluation | Validation BPB at step 0, every 250 updates and the final update (**189 events**); full CORE-22 at 8K, 16K, 24K, 32K, 40K and 46,875 (**6 events**) |

Parameters, memory, throughput and GPU-hours are **outcomes**, not controls. A sparse model is not required to have
more parameters, and nothing assumes it must be faster at a 2K context.

## Two baselines with different roles

- **SSSL** is the historical `dense` model: 24 blocks, 18 short-window and 6 global layers. It defines the frozen
  **compute anchor**; its configuration ID, checkpoint keys and historical scores are preserved.
- **Dense** (`dense_full`) is full causal multi-head attention in every layer, independently calibrated, trained and
  audited. It is the only **quality baseline**: every Δ on the leaderboard is measured against it, at the same
  checkpoint. SSSL scores are never reused as Dense scores.

## Research mode and formal mode

`python -m nexttransformer.train` runs in **research mode** by default: you can change the code or use another PyTorch
build and still train; the trainer records stale or out-of-budget cost evidence, and the run can never be published.
Leaderboard runs pass **`--formal`**: the configuration's cost proof must match the source and PyTorch build it was
audited with, and every training segment must be formal.

The opt-in execution speedups (`--speedups`, off by default) fuse the shared shell's kernels without changing what a
method computes. Formal runs accept only **`--speedups exact`** or a selection of its parts, which reproduce the
default path bit for bit; `all`, `cudnn` and `merge` change rounding and stay in research mode. Every segment of a
run uses the same selection, and the exported result records it.

## Where to go next

- [Compute accounting](../compute/) explains exactly what is counted.
- [Results and statuses](../results/) explains what a published run must contain.
- [FAQ](../faq/) answers the common "why" questions.
