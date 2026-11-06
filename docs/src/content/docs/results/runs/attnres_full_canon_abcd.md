---
title: AttnRes-Canon
description: Arena v1 entry. Full AttnRes with Canon layers A–D, two formal segments on 32 ranks, and the comparison with the Residual family.
---

AttnRes-Canon (`attnres_full_canon_abcd`) is Full AttnRes with Canon layers A–D. It is an
[arena v1 entry](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/attnres-canon.json):
an LLM agent found and selected it with NextT-Gym on the dev split (involvement L1). At 46,875 updates (49.152B
tokens) and seed 42, it reached **35.9921 CORE ×100**. The agent had read the published results in `results/runs/`;
it chose every candidate with dev-split values only, and never used the validation shard or CORE-22 for selection.

The protocol's validation measurements are **0.660520 periodic validation BPB** and **0.657754 independent final
validation BPB** (0.659834 on the training split). The model has 23 layers, FFN width 6144 and **1,357,154,568
parameters**: Full AttnRes over 23 depth groups and 138 Canon layers.

Δ is the CORE ×100 / periodic validation BPB difference from the Dense Baseline:

| Run | FFN | CORE ×100 | Periodic val BPB | Final val BPB | Δ vs Dense |
|---|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 6144 | 33.0952 | 0.667172 | 0.664460 | — |
| Standard Residual | 6144 | 31.9513 | 0.673959 | 0.671214 | −1.14 / +0.00679 |
| Block AttnRes (`attnres`) | 6144 | 33.9971 | 0.662963 | 0.660217 | +0.90 / −0.00421 |
| HC | 6144 | 35.0943 | 0.664400 | 0.661683 | +2.00 / −0.00277 |
| mHC | 6016 | 35.3278 | 0.661665 | 0.658908 | +2.23 / −0.00551 |
| AttnRes-Canon | 6144 | 35.9921 | 0.660520 | 0.657754 | +2.90 / −0.00665 |

Compared with Block AttnRes, the closest built-in method, CORE ×100 is 2.00 points higher. It improved or matched 18
of the 22 CORE tasks. The largest gains were BoolQ (+28.41 centered points), SQuAD (+11.59) and COPA (+6.00). The
largest losses were Winograd (−5.86), BIG-bench CS Algorithms (−5.83) and AGI Eval LSAT-AR (−4.89). BoolQ alone
contributes 1.29 of the 2.00 points; over the other 21 tasks, the mean is 0.74 points higher. Compared with mHC, CORE
×100 is 0.66 points higher, and 0.47 points over the 21 tasks without BoolQ.

Periodic validation BPB is 0.00244 lower than Block AttnRes's and 0.00115 lower than mHC's. At selection, after 2,000
updates, NextT-Gym had measured −0.00380 dev BPB against Block AttnRes. Every value comes from one seed and is not a
statistical ranking.

The formal run trained the preregistered configuration from fork commit `bf54f0d` (code bundle SHA-256
`c9d3f98c…`). It ran on 32 ranks (four nodes of eight GPUs) with micro-batch 16 and no gradient accumulation, in two
segments:

| Updates retained | Restored from | Training source |
|---|---|---|
| 1–39,000 | — | `bf54f0d` |
| 39,001–46,875 | Checkpoint 39,000 | `bf54f0d` |

The cluster's 37-hour job limit stopped the first segment after it logged update 39,500, as planned. Its checkpoint
at 39,500 was incomplete, so the second segment restored checkpoint 39,000 and re-ran updates 39,001–39,500. The
trainer archived the replayed events, and the curve uses the second segment's. Both segments have the same model,
protocol, data, tokenizer, configuration, cost-proof and source hashes, and both ran in formal mode. Neither launch
failed; the card counts two formal attempts.

The cumulative training charge is **4.695110 GFLOPs per token** (+0.486% versus the SSSL anchor). It equals the cost
proof at every logged update, with no budget anomalies. The read-only audit covers all 189 scheduled validations, the
six full CORE-22 evaluations and the final checkpoint. Training took 143,842 s (3.07 s per update) and evaluation took
9,794 s. Gradient norms stayed finite: after update 500, the median was 0.48 and the maximum 2.49.

The configuration, fork commit and cost proof were preregistered in
[pull request #10](https://github.com/Interactive-Active-AI/NextTransformer/pull/10), opened on 2026-11-03 at 06:36 UTC.
It was closed four minutes later at the operator's request, and the formal run was launched from the fork at 11:00
UTC. `bf54f0d` remains the head of that pull request. Since 2026-11-04, the arena preregisters entries in proposal
issues instead. [Proposal issue #30](https://github.com/Interactive-Active-AI/NextTransformer/issues/30), filed on
2026-11-06 after the formal run, records the same commit, configuration and cost proof.

Before the launch, a bounded four-node smoke check ran on the dev split with 32 ranks: 80 updates, with a resume at
update 40. The step-1 loss equalled an eight-GPU run's. The resumed run matched the uninterrupted one to a mean
\|Δloss\| of 0.030 over updates 41–80, consistent to rounding but not bitwise. The check used neither the validation
shard nor CORE.

The Canon layers are eager PyTorch depthwise convolutions without fused kernels, so the update time reflects this
implementation rather than the method.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/attnres_full_canon_abcd.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/attnres_full_canon_abcd.json) · [Disclosure card](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/attnres-canon.json) · [Final checkpoint and training logs](https://huggingface.co/GY2233/nextt-arena-attnres-full-canon-abcd)
