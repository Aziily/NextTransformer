# 3090 Training Optimization Campaign

## Goal

Increase steady-state throughput for the reference `dense_full` training workload on one node with eight RTX 3090 GPUs while preserving numerical behavior, the published training protocol, and bounded-run reproducibility.

## Fixed task contract

- Workload: `configs/baseline/dense_full.json`
- Data: pinned ClimbMix revision from `site.local.json`
- Hardware: eight RTX 3090 GPUs on the same host
- Compute dtype: BF16
- World size: 8
- Micro-batch: 1 per GPU for the initial campaign
- Smoke length: 1 warmup step plus 10 steady steps
- Accumulation: 1
- Seed: 42
- W&B: disabled
- Formal training: forbidden for this campaign
- Protocol: unchanged; no data, token budget, evaluation schedule, FLOP ledger, or model math changes

## Correctness gate

Every candidate must:

1. Exit successfully with `run_status.phase == "smoke_complete"`.
2. Produce finite loss and gradient norm events for every steady step.
3. Preserve the protocol hash and the expected step/token progression.
4. Match the baseline loss trajectory within the bounded-run tolerance agreed for the candidate. A candidate that changes numerical behavior must be rejected or isolated as a separate research question.
5. Avoid OOM, deadlock, NCCL errors, or incomplete checkpoints.

## Promotion gate

A candidate is promoted only when it passes correctness and improves the median steady tokens/s by at least 2% over the matched baseline. A smaller change is recorded as inconclusive unless it addresses a reliability or memory regression. Every candidate is benchmarked under the same command shape, warmup, steady steps, and output parser.

## Campaign rounds

1. Baseline variance across repeated runs.
2. Explicit rank/device initialization and NCCL device mapping.
3. Islands versus interleaved GPU order confirmation.
4. CPU NUMA affinity and process placement.
5. NCCL algorithm/ring/tree comparison.
6. NCCL buffer/channel configuration comparison.
7. Optimizer communication and compute phase attribution.
8. Small-tensor communication batching candidate.
9. Ampere Triton shared-memory/block/stage candidate.
10. FLA/linear-attention Ampere candidate after the dependency and baseline path are verified.
11. Input prefetch and pinned-memory candidate.
12. End-to-end confirmation of the best surviving candidate.

Each round changes one variable, keeps the previous best as baseline, stores profile/benchmark evidence outside Git, and appends a decision to `runs/3090-campaign/candidates.jsonl`.
