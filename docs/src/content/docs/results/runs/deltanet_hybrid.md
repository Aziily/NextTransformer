---
title: Hybrid DeltaNet + MHA
description: Single-segment training of the 22-DeltaNet + 2-MHA global hybrid, and the comparison with standalone DeltaNet and Dense.
---

At 46,875 updates (49.152B tokens), Hybrid DeltaNet + MHA (`deltanet_hybrid`) reached **0.673079 periodic validation BPB**, **0.670325 independent final validation BPB** (0.672545 on the training split) and **29.9012 CORE ×100** at seed 42. The 24-layer stack places full causal MHA at one-based layers 2 and 13 and DeltaNet in the other 22 layers, calibrated to FFN width 6400 with **1,403,810,314 parameters** (including alternating value embeddings on the 12 one-based even layers).

| Run | Layers | FFN | Parameters | Periodic val BPB | Final val BPB | CORE ×100 | Δ vs Dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline (`dense_full`) | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| DeltaNet (`deltanet`) | 25 | 6144 | 1,463,690,268 | 0.678675 | 0.675966 | 28.3357 | +0.01150 / -4.76 |
| Hybrid DeltaNet + MHA (`deltanet_hybrid`) | 24 | 6400 | 1,403,810,314 | 0.673079 | 0.670325 | 29.9012 | +0.00591 / -3.19 |

Compared with standalone DeltaNet, inserting just two full causal MHA layers at positions 2 and 13 (8.3% of the 24 layers) lowered periodic validation BPB by 0.00560 (closing 48.7% of DeltaNet's validation BPB gap to Dense), lowered independent final validation BPB by 0.00564, and raised CORE ×100 by 1.57 points (closing 32.9% of the CORE gap to Dense). Across the 22 CORE tasks, the hybrid improved 16 tasks over standalone DeltaNet, led by COPA (+8.00 centered points), CommonsenseQA (+6.96), BIG-bench Operators (+6.19), BoolQ (+5.79), BIG-bench Dyck Languages (+4.60), CoQA (+3.56) and BIG-bench Repeat Copy Logic (+3.12), while trailing on Winograd (−6.59), AGIEval LSAT-AR (−2.17) and WinoGrande (−2.05).

The formal run completed in one uninterrupted segment on eight ranks with microbatch eight and eight accumulation steps (`fab8c032` parent source with launch-environment patch `08668a33` and science hash `e1b5a94c`). DeltaNet layers used the FLA `chunk64` backend (`chunk_delta_rule`) and the two MHA layers used deterministic SDPA without FlashAttention.

The cumulative training charge is **4.675326 GFLOPs per token** (+0.063% versus the SSSL anchor), equal to the cost proof at every logged update with zero budget anomalies. The read-only audit covers all 189 scheduled validations, six full CORE-22 evaluations and 94 verified checkpoints. Gradient norms stayed smooth and finite throughout training: after update 500 the median was 0.25 and the maximum was 0.94.

Distributed eight-rank 12-update target-batch smoke and fresh-process resume acceptance passed (`weights_equal`, `losses_equal`, `all_rank_states_equal`). A separate 512-update pilot and the launch-patch diff were not archived, so final acceptance remains subject to maintainer review.

[Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/deltanet_hybrid.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/deltanet_hybrid.json)
