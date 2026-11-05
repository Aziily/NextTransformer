---
title: Hybrid MoBA + MHA
description: Single-segment training of the 22-MoBA + 3-MHA hybrid, and the comparison with standalone MoBA and Dense.
---

At 46,875 updates (49.152B tokens), Hybrid MoBA + MHA (`moba_hybrid`) reached **0.670240 periodic validation BPB**, **0.667620 independent final validation BPB** (0.669544 on the training split) and **32.7422 CORE ×100** at seed 42. The 25-layer stack uses MoBA (128-token blocks, 256-token budget including the current block) in the first 22 layers and full causal MHA in the final three layers (layers 23–25), calibrated to FFN width 6016 with **1,452,935,068 parameters** (including alternating value embeddings on the 13 one-based odd layers).

| Run | Layers | FFN | Parameters | Periodic val BPB | Final val BPB | CORE ×100 | Δ vs Dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| MoBA (`moba`) | 25 | 6144 | 1,462,765,468 | 0.674019 | 0.671396 | 31.0429 | +0.00685 / -2.05 |
| Hybrid MoBA + MHA (`moba_hybrid`) | 25 | 6016 | 1,452,935,068 | 0.670240 | 0.667620 | 32.7422 | +0.00307 / -0.35 |

Compared with standalone MoBA, replacing the final three of 25 MoBA layers (12% of the stack) with full causal MHA lowered periodic validation BPB by 0.00378 (closing 55.2% of standalone MoBA's validation BPB gap to Dense), lowered independent final validation BPB by 0.00378, and raised CORE ×100 by 1.70 points (closing 82.8% of the CORE gap to Dense, within 0.35 points of `dense_full`). Across the 22 CORE tasks, the hybrid improved or matched 19 tasks over standalone MoBA, led by CommonsenseQA (+19.96 centered points), BoolQ (+11.03), Winograd (+4.40), BIG-bench CS Algorithms (+2.95), WinoGrande (+2.84), ARC Challenge (+1.71), OpenBookQA (+1.07) and SQuAD (+1.06), while trailing mainly on COPA (−10.00) and PIQA (−1.52).

The formal run completed in one uninterrupted segment on eight ranks with microbatch eight and eight accumulation steps (`fab8c032` parent source with launch-environment patch `08668a33` and science hash `e1b5a94c`). MoBA layers trained with the local block-attention `kernel` backend (CORE evaluation always uses the reference path) and the three MHA layers used deterministic SDPA without FlashAttention.

The cumulative training charge is **4.659989 GFLOPs per token** (−0.266% versus the SSSL anchor), equal to the cost proof at every logged update with zero budget anomalies. The read-only audit covers all 189 scheduled validations, six full CORE-22 evaluations and 94 verified checkpoints. Gradient norms stayed smooth and finite throughout training: after update 500 the median was 0.33 and the maximum was 1.04.

Distributed eight-rank 12-update target-batch smoke and fresh-process resume acceptance passed (`weights_equal`, `losses_equal`, `all_rank_states_equal`). A separate 512-update pilot and the launch-patch diff were not archived, so final acceptance remains subject to maintainer review.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/moba_hybrid.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/moba_hybrid.json)
